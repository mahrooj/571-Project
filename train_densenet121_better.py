
import os, random, argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights

from PIL import Image
from torch.amp import autocast, GradScaler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_train_test(data_root: str) -> Tuple[str, Optional[str]]:
    # Prefer data_root/train and data_root/test if they exist
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
            return train_dir, test_dir
        return train_dir, None

    # fallback: user passes train dir directly
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



def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros((y.size(0), num_classes), device=y.device).scatter_(1, y.unsqueeze(1), 1.0)


def rand_bbox(W: int, H: int, lam: float):
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_mixup_cutmix(x, y, num_classes: int, mixup_alpha: float, cutmix_alpha: float):
    # returns x_mix, soft_targets
    use_mix = mixup_alpha > 0 and np.random.rand() < 0.5
    use_cut = cutmix_alpha > 0 and not use_mix

    y1 = one_hot(y, num_classes)

    if not (use_mix or use_cut):
        return x, y1

    idx = torch.randperm(x.size(0), device=x.device)
    y2 = y1[idx]

    if use_mix:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = x * lam + x[idx] * (1 - lam)
        soft = y1 * lam + y2 * (1 - lam)
        return x, soft

    # cutmix
    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    B, C, H, W = x.size()
    x1b, y1b, x2b, y2b = rand_bbox(W, H, lam)
    x[:, :, y1b:y2b, x1b:x2b] = x[idx, :, y1b:y2b, x1b:x2b]
    lam_adj = 1.0 - ((x2b - x1b) * (y2b - y1b) / (W * H))
    soft = y1 * lam_adj + y2 * (1 - lam_adj)
    return x, soft


def soft_target_ce(logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: Optional[torch.Tensor], label_smoothing: float):
    # label smoothing on soft targets: blend with uniform
    if label_smoothing > 0:
        K = soft_targets.size(1)
        soft_targets = soft_targets * (1 - label_smoothing) + (label_smoothing / K)

    logp = torch.log_softmax(logits, dim=1)
    per_sample = -(soft_targets * logp).sum(dim=1)

    if class_weights is not None:
        # weight per sample = sum_k soft * w_k
        w = (soft_targets * class_weights.unsqueeze(0)).sum(dim=1)
        per_sample = per_sample * w

    return per_sample.mean()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        sd = model.state_dict()
        for k, v in sd.items():
            self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1 - self.decay))
            else:
                self.shadow[k] = v.detach().clone()

    def apply(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module):
        model.load_state_dict(self.backup, strict=True)
        self.backup = {}


class DenseNet121Head(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float):
        super().__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_f = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_f, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def set_trainable(model: nn.Module, trainable_prefixes: List[str]):
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pref) for pref in trainable_prefixes)


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, wd: float):
    head_params, bb_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone.classifier"):
            head_params.append(p)
        else:
            bb_params.append(p)

    groups = []
    if bb_params:
        groups.append({"params": bb_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(groups, weight_decay=wd)


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine from 1 -> 0
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@dataclass
class StageCfg:
    name: str
    epochs: int
    lr_head: float
    lr_backbone: float
    trainable_prefixes: List[str]


def train_one_epoch(model, loader, device, optimizer, scaler, amp: bool,
                    num_classes: int, class_weights: Optional[torch.Tensor],
                    mixup_alpha: float, cutmix_alpha: float, label_smoothing: float,
                    grad_clip: float, ema: Optional[EMA]):
    model.train()
    use_amp = amp and device.type == "cuda"
    losses = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha > 0 or cutmix_alpha > 0:
            x, soft = apply_mixup_cutmix(x, y, num_classes, mixup_alpha, cutmix_alpha)
        else:
            soft = one_hot(y, num_classes)

        if use_amp:
            with autocast(device_type="cuda"):
                logits = model(x)
                loss = soft_target_ce(logits, soft, class_weights, label_smoothing)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = soft_target_ce(logits, soft, class_weights, label_smoothing)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def run_stage(model, train_loader, val_loader, device,
              stage: StageCfg, wd: float, amp: bool, patience: int,
              num_classes: int, class_weights: Optional[torch.Tensor],
              mixup_alpha: float, cutmix_alpha: float, label_smoothing: float,
              grad_clip: float, warmup_epochs: int, save_path: str,
              ema: Optional[EMA]):
    optimizer = build_optimizer(model, stage.lr_head, stage.lr_backbone, wd)
    scheduler = cosine_with_warmup(optimizer, warmup_epochs=warmup_epochs, total_epochs=stage.epochs)
    scaler = GradScaler("cuda") if (amp and device.type == "cuda") else None

    best = {"val_top1": -1.0}
    bad = 0

    for ep in range(1, stage.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, device, optimizer, scaler, amp,
            num_classes, class_weights, mixup_alpha, cutmix_alpha, label_smoothing,
            grad_clip, ema
        )

        if ema is not None:
            ema.apply(model)
            val = evaluate(model, val_loader, device)
            ema.restore(model)
        else:
            val = evaluate(model, val_loader, device)

        scheduler.step()

        print(f"[{stage.name}] epoch {ep}/{stage.epochs} | train_loss={tr_loss:.4f} | val_top1={val['top1']:.4f}")

        if val["top1"] > best["val_top1"]:
            best = {"val_top1": val["top1"], **val, "epoch": ep}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # save EMA weights if enabled (usually better)
            if ema is not None:
                ema.apply(model)
                torch.save(model.state_dict(), save_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), save_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{stage.name}] early stop (patience={patience})")
                break

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true", default=True)

    ap.add_argument("--split_train", type=float, default=0.64)
    ap.add_argument("--split_val", type=float, default=0.16)
    ap.add_argument("--split_test", type=float, default=0.20)

    # training tricks
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--cutmix", type=float, default=0.2)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout_p", type=float, default=0.4)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # stages
    ap.add_argument("--epochs_head", type=int, default=4)
    ap.add_argument("--epochs_last", type=int, default=18)
    ap.add_argument("--epochs_last2", type=int, default=12)

    ap.add_argument("--lr_head0", type=float, default=5e-4)
    ap.add_argument("--lr_bb0", type=float, default=0.0)

    ap.add_argument("--lr_head1", type=float, default=3e-4)
    ap.add_argument("--lr_bb1", type=float, default=3e-5)

    ap.add_argument("--lr_head2", type=float, default=2e-4)
    ap.add_argument("--lr_bb2", type=float, default=1e-5)

    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--warmup_epochs", type=int, default=2)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir, test_dir = resolve_train_test(args.data_root)
    print("Resolved train_dir:", train_dir)
    print("Resolved test_dir:", test_dir)

    mean = DenseNet121_Weights.DEFAULT.transforms().mean
    std = DenseNet121_Weights.DEFAULT.transforms().std

    # Stronger aug than basic, but still stable
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.10),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
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

    model = DenseNet121Head(num_classes=num_classes, dropout_p=args.dropout_p).to(device)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_invcount(train_targets, num_classes, device)
        print("Using class-weighted soft CE + label_smoothing (supports MixUp/CutMix).")
    else:
        print("Using soft CE + label_smoothing (supports MixUp/CutMix).")

    ema = None if args.no_ema else EMA(model, decay=args.ema_decay)

    save_path = os.path.join(os.getcwd(), "best_densenet121_better.pt")

    stages = [
        StageCfg(
            name="head",
            epochs=args.epochs_head,
            lr_head=args.lr_head0,
            lr_backbone=args.lr_bb0,
            trainable_prefixes=["backbone.classifier"],
        ),
        StageCfg(
            name="finetune_last_block",
            epochs=args.epochs_last,
            lr_head=args.lr_head1,
            lr_backbone=args.lr_bb1,
            trainable_prefixes=["backbone.classifier", "backbone.features.denseblock4", "backbone.features.norm5"],
        ),
        StageCfg(
            name="finetune_last2_blocks",
            epochs=args.epochs_last2,
            lr_head=args.lr_head2,
            lr_backbone=args.lr_bb2,
            trainable_prefixes=[
                "backbone.classifier",
                "backbone.features.denseblock3", "backbone.features.transition3",
                "backbone.features.denseblock4", "backbone.features.norm5"
            ],
        ),
    ]

    best_all = []
    for st in stages:
        set_trainable(model, st.trainable_prefixes)
        best = run_stage(
            model, train_loader, val_loader, device,
            st, wd=args.weight_decay, amp=args.amp, patience=args.patience,
            num_classes=num_classes, class_weights=class_weights,
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            warmup_epochs=args.warmup_epochs,
            save_path=save_path,
            ema=ema
        )
        best_all.append(best)

    # final load best & test
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    best_stage = max(best_all, key=lambda d: d.get("val_top1", -1.0))

    print("\n========== FINAL RESULT (DenseNet121 Better) ==========")
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
    print("======================================================\n")
    # show how many bad samples got skipped
    try:
        print(f"Skipped bad samples (train scan): {getattr(full_train, 'bad_count', 0)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
