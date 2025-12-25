
import os, random, argparse
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import vgg19, VGG19_Weights

from torch.amp import autocast, GradScaler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_train_test(data_root: str) -> Tuple[str, Optional[str]]:
    # Prefer data_root/train and data_root/test
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
            return train_dir, test_dir
        return train_dir, None

    # Fallback: user passes the train dir directly
    if os.path.isdir(data_root) and len(os.listdir(data_root)) > 0:
        return data_root, None

    raise FileNotFoundError(f"Cannot resolve dataset directory from: {data_root}")


def stratified_split_indices(targets: List[int], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    y = np.asarray(targets)

    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        t = idx[:n_train].tolist()
        v = idx[n_train:n_train + n_val].tolist()
        te = idx[n_train + n_val:].tolist()

        # tiny-class protection
        if len(v) == 0 and len(t) > 1:
            v = [t.pop()]
        if len(te) == 0 and len(t) > 2:
            te = [t.pop()]

        train_idx += t
        val_idx += v
        test_idx += te

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def build_weighted_sampler_sqrt(train_targets: List[int], num_classes: int) -> WeightedRandomSampler:
    y = np.asarray(train_targets)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_w = 1.0 / np.sqrt(counts)
    sample_w = class_w[y]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )


def compute_class_weights_invcount(train_targets: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    y = np.asarray(train_targets)
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=k, dim=1).indices
    return (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_p, all_y = [], []
    t1=t3=t5=0.0; nb=0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        t1 += topk_acc(logits, y, 1)
        t3 += topk_acc(logits, y, 3)
        t5 += topk_acc(logits, y, 5)
        nb += 1

        all_p.append(torch.argmax(logits, 1).cpu().numpy())
        all_y.append(y.cpu().numpy())

    yp = np.concatenate(all_p) if all_p else np.array([])
    yt = np.concatenate(all_y) if all_y else np.array([])
    acc = float((yp == yt).mean()) if len(yt) else 0.0
    macro_f1 = float(f1_score(yt, yp, average="macro")) if len(yt) else 0.0
    w_f1 = float(f1_score(yt, yp, average="weighted")) if len(yt) else 0.0

    return {
        "top1": t1/max(nb,1),
        "top3": t3/max(nb,1),
        "top5": t5/max(nb,1),
        "report_acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": w_f1,
    }


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, max_retries=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.bad_count = 0

    def __getitem__(self, index):
        for _ in range(self.max_retries):
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target
            except (FileNotFoundError, OSError, Image.DecompressionBombError):
                self.bad_count += 1
                index = random.randint(0, len(self.samples) - 1)
        raise FileNotFoundError(f"Too many bad samples. Last tried: {path}")


def freeze_all_features(model: nn.Module):
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_n_feature_layers(model: nn.Module, n: int):
    # Unfreeze last n layers in model.features (works reliably across torchvision VGG variants)
    layers = list(model.features.children())
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def make_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    head_params, bb_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            bb_params.append(p)

    groups = []
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, device, optimizer, scaler, amp: bool, criterion, grad_clip: float):
    model.train()
    use_amp = amp and device.type == "cuda"
    losses = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def run_stage(model, train_loader, val_loader, device, stage_name: str,
              epochs: int, lr_head: float, lr_backbone: float, weight_decay: float,
              amp: bool, warmup_epochs: int, patience: int, criterion, grad_clip: float,
              save_path: str):
    optimizer = make_optimizer(model, lr_head, lr_backbone, weight_decay)
    scheduler = cosine_with_warmup(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs)
    scaler = GradScaler("cuda") if (amp and device.type == "cuda") else None

    best = {"val_top1": -1.0}
    bad = 0

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, scaler, amp, criterion, grad_clip)
        val = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"[{stage_name}] epoch {ep}/{epochs} | train_loss={tr_loss:.4f} | val_top1={val['top1']:.4f}")

        if val["top1"] > best["val_top1"]:
            best = {"val_top1": val["top1"], **val, "epoch": ep}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{stage_name}] early stop (patience={patience})")
                break

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)  # VGG is heavy
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true", default=True)

    ap.add_argument("--split_train", type=float, default=0.64)
    ap.add_argument("--split_val", type=float, default=0.16)
    ap.add_argument("--split_test", type=float, default=0.20)

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout_p", type=float, default=0.5)

    # Stages
    ap.add_argument("--epochs_head", type=int, default=6)
    ap.add_argument("--epochs_ft", type=int, default=14)
    ap.add_argument("--unfreeze_last_n", type=int, default=10)  # last n feature layers
    ap.add_argument("--lr_head0", type=float, default=8e-4)
    ap.add_argument("--lr_head1", type=float, default=4e-4)
    ap.add_argument("--lr_bb1", type=float, default=2e-5)

    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir, test_dir = resolve_train_test(args.data_root)
    print("Resolved train_dir:", train_dir)
    print("Resolved test_dir:", test_dir)

    # Use ImageNet normalization from weights
    mean = VGG19_Weights.DEFAULT.transforms().mean
    std = VGG19_Weights.DEFAULT.transforms().std

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_train = SafeImageFolder(train_dir, transform=train_tf)
    full_train_eval = SafeImageFolder(train_dir, transform=eval_tf)
    num_classes = len(full_train.classes)
    print("num_classes:", num_classes)

    targets = full_train.targets

    if test_dir is not None:
        test_ds = SafeImageFolder(test_dir, transform=eval_tf)
        train_idx, val_idx, _ = stratified_split_indices(targets, 0.80, 0.20, 0.0, args.seed)
    else:
        train_idx, val_idx, test_idx = stratified_split_indices(
            targets, args.split_train, args.split_val, args.split_test, args.seed
        )
        test_ds = Subset(full_train_eval, test_idx)

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    train_targets = [targets[i] for i in train_idx]

    sampler = build_weighted_sampler_sqrt(train_targets, num_classes) if args.use_weighted_sampler else None
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    print("train batches per epoch:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    # Model
    model = vgg19(weights=VGG19_Weights.DEFAULT)
    # Replace classifier output
    in_f = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_f, num_classes)
    # Add stronger dropout (VGG tends to overfit)
    model.classifier[2] = nn.Dropout(args.dropout_p)
    model.classifier[5] = nn.Dropout(args.dropout_p)
    model = model.to(device)

    # Loss
    if args.use_class_weights:
        cw = compute_class_weights_invcount(train_targets, num_classes, device)
        print("Using class-weighted CE + label_smoothing.")
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    else:
        print("Using CE + label_smoothing.")
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_path = os.path.join(args.out_dir, "best_vgg19.pt")

    # Stage 1: train classifier only
    freeze_all_features(model)
    for p in model.classifier.parameters():
        p.requires_grad = True

    best0 = run_stage(
        model, train_loader, val_loader, device,
        stage_name="head", epochs=args.epochs_head,
        lr_head=args.lr_head0, lr_backbone=0.0, weight_decay=args.weight_decay,
        amp=args.amp, warmup_epochs=args.warmup_epochs, patience=args.patience,
        criterion=criterion, grad_clip=args.grad_clip,
        save_path=best_path
    )

    # Stage 2: unfreeze last N feature layers + keep classifier trainable
    freeze_all_features(model)
    unfreeze_last_n_feature_layers(model, args.unfreeze_last_n)
    for p in model.classifier.parameters():
        p.requires_grad = True

    best1 = run_stage(
        model, train_loader, val_loader, device,
        stage_name="finetune_last_layers", epochs=args.epochs_ft,
        lr_head=args.lr_head1, lr_backbone=args.lr_bb1, weight_decay=args.weight_decay,
        amp=args.amp, warmup_epochs=args.warmup_epochs, patience=args.patience,
        criterion=criterion, grad_clip=args.grad_clip,
        save_path=best_path
    )

    # Final test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    best_stage = best0 if best0["val_top1"] >= best1["val_top1"] else best1

    print("\n========== FINAL RESULT (VGG19) ==========")
    print(f"best_val_top1: {best_stage.get('top1', 0.0):.4f}")
    print(f"best_val_top3: {best_stage.get('top3', 0.0):.4f}")
    print(f"best_val_top5: {best_stage.get('top5', 0.0):.4f}")
    print(f"best_val_report_acc: {best_stage.get('report_acc', 0.0):.4f}")
    print(f"best_val_macro_f1: {best_stage.get('macro_f1', 0.0):.4f}")
    print(f"best_val_weighted_f1: {best_stage.get('weighted_f1', 0.0):.4f}")
    print("---- TEST ----")
    print(f"test_top1: {test_metrics['top1']:.4f}")
    print(f"test_top3: {test_metrics['top3']:.4f}")
    print(f"test_top5: {test_metrics['top5']:.4f}")
    print(f"test_report_acc: {test_metrics['report_acc']:.4f}")
    print(f"test_macro_f1: {test_metrics['macro_f1']:.4f}")
    print(f"test_weighted_f1: {test_metrics['weighted_f1']:.4f}")
    print("=========================================\n")

    print(f"Skipped bad samples (train scan): {getattr(full_train, 'bad_count', 0)}")


if __name__ == "__main__":
    main()
