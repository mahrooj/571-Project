
import os
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    with torch.no_grad():
        topk = torch.topk(logits, k=k, dim=1).indices
        return (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()

def stratified_split_indices(
    targets: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)

    train_idx, val_idx, test_idx = [], [], []
    classes = np.unique(targets)

    for c in classes:
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n = len(idx)

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        t = idx[:n_train].tolist()
        v = idx[n_train:n_train + n_val].tolist()
        te = idx[n_train + n_val:].tolist()

        if len(v) == 0 and len(t) > 1:
            v = [t.pop()]
        if len(te) == 0 and len(t) > 2:
            te = [t.pop()]

        train_idx.extend(t)
        val_idx.extend(v)
        test_idx.extend(te)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def compute_class_weights_from_targets(targets: List[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

def build_weighted_sampler(train_targets: List[int], num_classes: int) -> WeightedRandomSampler:
    targets = np.asarray(train_targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_w = 1.0 / class_counts
    sample_w = class_w[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )

def get_data_dirs(data_root: str) -> Tuple[str, Optional[str]]:
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    if os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0:
        if os.path.isdir(test_dir) and len(os.listdir(test_dir)) > 0:
            return train_dir, test_dir
        return train_dir, None

    return data_root, None

@dataclass
class StageCfg:
    epochs: int
    lr: float
    weight_decay: float

def freeze_all_except(model: nn.Module, trainable_prefixes: List[str]) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pref) for pref in trainable_prefixes)

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_y = [], []
    top1_sum, top3_sum, top5_sum, n_batches = 0.0, 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        top1_sum += accuracy_topk(logits, y, 1)
        top3_sum += accuracy_topk(logits, y, 3)
        top5_sum += accuracy_topk(logits, y, 5)
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_y.append(y.cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])

    report_acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted")) if len(y_true) else 0.0

    return {
        "top1": top1_sum / max(n_batches, 1),
        "top3": top3_sum / max(n_batches, 1),
        "top5": top5_sum / max(n_batches, 1),
        "report_acc": report_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp: bool
) -> float:
    model.train()
    losses = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0

def run_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    stage_cfg: StageCfg,
    amp: bool,
    patience: int,
    save_path: str
) -> dict:
    optimizer = build_optimizer(model, lr=stage_cfg.lr, weight_decay=stage_cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None

    best = {"val_top1": -1.0, "epoch": -1}
    bad = 0

    for epoch in range(1, stage_cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, amp)
        val_metrics = evaluate(model, val_loader, device)
        val_top1 = val_metrics["top1"]

        print(f"[{stage_name}] epoch {epoch}/{stage_cfg.epochs} | train_loss={train_loss:.4f} | val_top1={val_top1:.4f}")

        if val_top1 > best["val_top1"]:
            best = {"val_top1": val_top1, "epoch": epoch, **val_metrics}
            torch.save(model.state_dict(), save_path)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{stage_name}] early stop (patience={patience})")
                break

    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_weighted_sampler", action="store_true", default=True)

    parser.add_argument("--split_train", type=float, default=0.64)
    parser.add_argument("--split_val", type=float, default=0.16)
    parser.add_argument("--split_test", type=float, default=0.20)

    parser.add_argument("--epochs_head", type=int, default=6)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--epochs_last", type=int, default=15)
    parser.add_argument("--lr_last", type=float, default=1e-4)
    parser.add_argument("--epochs_last2", type=int, default=10)
    parser.add_argument("--lr_last2", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--target_acc", type=float, default=0.80)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, test_dir = get_data_dirs(args.data_root)
    print("Resolved train_dir:", train_dir)
    print("Resolved test_dir:", test_dir)

    mean = DenseNet121_Weights.DEFAULT.transforms().mean
    std = DenseNet121_Weights.DEFAULT.transforms().std

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    full_train_eval = datasets.ImageFolder(train_dir, transform=eval_tf)

    num_classes = len(full_train.classes)
    print("num_classes:", num_classes)

    if test_dir is not None:
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
        targets = full_train.targets
        train_idx, val_idx, _ = stratified_split_indices(
            targets, train_ratio=0.80, val_ratio=0.20, test_ratio=0.00, seed=args.seed
        )
    else:
        targets = full_train.targets
        train_idx, val_idx, test_idx = stratified_split_indices(
            targets, args.split_train, args.split_val, args.split_test, seed=args.seed
        )
        test_ds = Subset(full_train_eval, test_idx)

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    train_subset_targets = [targets[i] for i in train_idx]
    sampler = build_weighted_sampler(train_subset_targets, num_classes) if args.use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("train batches per epoch:", len(train_loader))
    print("val batches:", len(val_loader))
    print("test batches:", len(test_loader))

    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)

    if args.use_class_weights:
        class_w = compute_class_weights_from_targets(train_subset_targets, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
        print("Using class-weighted CE.")
    else:
        criterion = nn.CrossEntropyLoss()

    # Stage 0: classifier only
    freeze_all_except(model, ["classifier"])
    best_path = "best_densenet121.pt"
    best0 = run_stage(
        "head",
        model, train_loader, val_loader, device, criterion,
        StageCfg(args.epochs_head, args.lr_head, args.weight_decay),
        amp=args.amp, patience=args.patience, save_path=best_path
    )

    # Stage 1: last dense block + classifier
    # DenseNet naming uses 'features.denseblock4' etc.
    freeze_all_except(model, ["features.denseblock4", "features.norm5", "classifier"])
    best1 = run_stage(
        "finetune_last_block",
        model, train_loader, val_loader, device, criterion,
        StageCfg(args.epochs_last, args.lr_last, args.weight_decay),
        amp=args.amp, patience=args.patience, save_path=best_path
    )

    # Stage 2: last 2 dense blocks + classifier
    freeze_all_except(model, ["features.denseblock3", "features.transition3",
                             "features.denseblock4", "features.norm5", "classifier"])
    best2 = run_stage(
        "finetune_last2_blocks",
        model, train_loader, val_loader, device, criterion,
        StageCfg(args.epochs_last2, args.lr_last2, args.weight_decay),
        amp=args.amp, patience=args.patience, save_path=best_path
    )

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    all_best = [best0, best1, best2]
    best_stage = max(all_best, key=lambda d: d.get("val_top1", -1.0))

    print("\n========== FINAL RESULT (DenseNet121) ==========")
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
    print("==============================================\n")

if __name__ == "__main__":
    main()
