# train_resnet50v0.py the very first resnet model I had

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scan_dermnet_images(data_root: str) -> Tuple[List[Path], List[int], List[str]]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")


    candidates: List[Path] = []
    if (root / "train").exists() and (root / "test").exists():
        candidates = [root / "train", root / "test"]
    else:
        candidates = [root]

    class_name_set = set()
    for base in candidates:
        for d in base.iterdir():
            if d.is_dir():
                class_name_set.add(d.name)
    class_names = sorted(class_name_set)
    if not class_names:
        raise RuntimeError(f"No class folders found under {root}.")

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths: List[Path] = []
    labels: List[int] = []
    for base in candidates:
        for c in class_names:
            class_dir = base / c
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    paths.append(p)
                    labels.append(class_to_idx[c])

    if not paths:
        raise RuntimeError(f"No images found under {root}.")
    return paths, labels, class_names


def stratified_split_indices(
    labels: Sequence[int],
    test_ratio: float = 0.20,
    val_ratio_within_trainval: float = 0.20,  # 0.2 of 0.8 => 0.16 overall
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels, dtype=np.int64)
    idx = np.arange(len(y))

    train_list, val_list, test_list = [], [], []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)

        n_test = int(round(len(cls_idx) * test_ratio))
        test_cls = cls_idx[:n_test]
        trainval_cls = cls_idx[n_test:]

        n_val = int(round(len(trainval_cls) * val_ratio_within_trainval))
        val_cls = trainval_cls[:n_val]
        train_cls = trainval_cls[n_val:]

        train_list.append(train_cls)
        val_list.append(val_cls)
        test_list.append(test_cls)

    train_idx = np.concatenate(train_list) if train_list else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_list) if val_list else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_list) if test_list else np.array([], dtype=np.int64)

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def compute_f1_macro_weighted(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    support = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp[c] = int(np.sum((y_true == c) & (y_pred == c)))
        fp[c] = int(np.sum((y_true != c) & (y_pred == c)))
        fn[c] = int(np.sum((y_true == c) & (y_pred != c)))
        support[c] = int(np.sum(y_true == c))

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * support) / max(1, np.sum(support)))
    return macro_f1, weighted_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--target_acc", type=float, default=0.80, help="val top-1 target, reach then early stop")
    ap.add_argument("--use_class_weights", action="store_true")
    # staged training
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_last", type=int, default=10)
    ap.add_argument("--epochs_last2", type=int, default=5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_last", type=float, default=1e-4)
    ap.add_argument("--lr_last2", type=float, default=3e-5)
    ap.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    args = ap.parse_args()

    seed_everything(args.seed)

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels, class_names = scan_dermnet_images(args.data_root)
    num_classes = len(class_names)
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)

    # Datasets
    class DermnetDS(Dataset):
        def __init__(self, indices: np.ndarray, train: bool):
            self.indices = indices
            self.train = train
            if train:
                self.tfms = T.Compose([
                    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.2),
                    T.RandomRotation(25),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            else:
                self.tfms = T.Compose([
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            idx = int(self.indices[i])
            img = Image.open(paths[idx]).convert("RGB")
            x = self.tfms(img)
            y = int(labels[idx])
            return x, y

    train_loader = DataLoader(DermnetDS(train_idx, True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(DermnetDS(val_idx, False), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(DermnetDS(test_idx, False), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"num_classes: {num_classes}")
    print(f"train batches per epoch: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"test batches: {len(test_loader)}")

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    def set_requires_grad_all(flag: bool) -> None:
        for p in model.parameters():
            p.requires_grad = flag

    def set_trainable(policy: str) -> None:
        set_requires_grad_all(False)
        # head only
        for p in model.fc.parameters():
            p.requires_grad = True
        if policy == "head":
            return
        # unfreeze layer4
        for p in model.layer4.parameters():
            p.requires_grad = True
        if policy == "last":
            return
        # unfreeze layer3 + layer4
        for p in model.layer3.parameters():
            p.requires_grad = True
        if policy == "last2":
            return
        raise ValueError(f"Unknown policy: {policy}")

    if args.use_class_weights:
        train_labels = np.array([labels[i] for i in train_idx], dtype=np.int64)
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
        w = counts.sum() / np.maximum(1.0, counts)
        w = w / w.mean()
        weight_t = torch.tensor(w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
        k = min(k, logits.shape[1])
        topk = torch.topk(logits, k=k, dim=1).indices
        correct = (topk == y.view(-1, 1)).any(dim=1).float().mean().item()
        return float(correct)

    @torch.no_grad()
    def evaluate(loader) -> Dict[str, float]:
        model.eval()
        y_true, y_pred = [], []
        top1_sum = top3_sum = top5_sum = 0.0
        nb = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

            top1_sum += topk_acc(logits, y, 1)
            top3_sum += topk_acc(logits, y, 3)
            top5_sum += topk_acc(logits, y, 5)
            nb += 1

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = compute_accuracy(y_true, y_pred)
        macro_f1, weighted_f1 = compute_f1_macro_weighted(y_true, y_pred, num_classes)
        return {
            "report_acc": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "top1": float(top1_sum / max(1, nb)),
            "top3": float(top3_sum / max(1, nb)),
            "top5": float(top5_sum / max(1, nb)),
        }

    def train_stage(epochs: int, lr: float, policy: str) -> Tuple[float, float]:
        set_trainable(policy)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        best_val_top1 = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            acc_sum = 0.0
            nb = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                    logits = model(x)
                    loss = criterion(logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_sum += float(loss.item())
                acc_sum += topk_acc(logits, y, 1)
                nb += 1

            val_m = evaluate(val_loader)
            train_loss = loss_sum / max(1, nb)
            train_top1 = acc_sum / max(1, nb)

            print(f"[{policy}] epoch {ep:03d}/{epochs} | lr={lr:.2e} | train_loss={train_loss:.4f} "
                  f"| train_top1={train_top1:.4f} | val_top1={val_m['top1']:.4f} | val_top5={val_m['top5']:.4f}")

            if val_m["top1"] > best_val_top1:
                best_val_top1 = val_m["top1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if val_m["top1"] >= args.target_acc:
                print(f"Reached target val_top1 >= {args.target_acc:.2f}. Early stopping this stage.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return best_val_top1, lr

    best1, _ = train_stage(args.epochs_head, args.lr_head, "head")
    best2, _ = train_stage(args.epochs_last, args.lr_last, "last")
    best3, _ = train_stage(args.epochs_last2, args.lr_last2, "last2")

    best_val = max(best1, best2, best3)
    test_m = evaluate(test_loader)

    print("\n======== FINAL RESULT (ResNet50) ========")
    print(f"best_val_top1: {best_val:.4f}")
    print(f"test_top1:     {test_m['top1']:.4f}")
    print(f"test_top3:     {test_m['top3']:.4f}")
    print(f"test_top5:     {test_m['top5']:.4f}")
    print(f"report_acc:    {test_m['report_acc']:.4f}")
    print(f"macro_f1:      {test_m['macro_f1']:.4f}")
    print(f"weighted_f1:   {test_m['weighted_f1']:.4f}")
    print("========================================\n")


if __name__ == "__main__":
    main()
