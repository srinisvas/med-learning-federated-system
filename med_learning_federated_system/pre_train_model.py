"""
pre_train_model.py — Train EfficientNet-B0 to stable 70% on ISIC 2019.

No plots, no metrics, no CSV. Just training + a stable checkpoint.

"Stable" means accuracy stays >= TARGET_ACC for PATIENCE consecutive epochs.
This avoids saving a checkpoint during a lucky spike — the model has to
genuinely hold the target before stopping.

Output: isic_pretrained_70pct.pth

Usage:
    export ISIC_DATA_ROOT="/dev/shm/isic2019_full"
    python med_learning_federated_system/pre_train_model.py
"""

import math
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from med_learning_federated_system.task import (
    DATA_ROOT,
    TRAIN_TRANSFORMS,
    TEST_TRANSFORMS,
    NUM_CLASSES,
    _TEST_SPLIT_RATIO,
    _DIRICHLET_SEED,
    get_isic_model,
    _TransformSubset,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_ACC    = 0.70   # accuracy threshold to reach
PATIENCE      = 3      # must stay >= TARGET_ACC for this many consecutive epochs
MAX_EPOCHS    = 200    # hard cap
BATCH_SIZE    = 128
BACKBONE_LR   = 4e-4
HEAD_LR       = 4e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
MIXUP_ALPHA   = 0.2
AMP_ENABLED   = torch.cuda.is_available()
CHECKPOINT    = "isic_pretrained_70pct.pth"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loaders(batch_size: int = BATCH_SIZE):
    full_ds    = ImageFolder(root=DATA_ROOT, transform=TEST_TRANSFORMS)
    all_labels = np.array(full_ds.targets)
    rng        = np.random.default_rng(_DIRICHLET_SEED)

    train_idx, test_idx = [], []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(all_labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * _TEST_SPLIT_RATIO))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())

    train_labels = all_labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    sampler      = WeightedRandomSampler(
        weights=torch.from_numpy((1.0 / class_counts)[train_labels]).float(),
        num_samples=len(train_idx),
        replacement=True,
    )

    nw = 4 if torch.cuda.is_available() else 2
    pm = torch.cuda.is_available()

    train_loader = DataLoader(
        _TransformSubset(full_ds, train_idx, TRAIN_TRANSFORMS),
        batch_size=batch_size, sampler=sampler,
        num_workers=nw, pin_memory=pm,
        drop_last=True, persistent_workers=True, prefetch_factor=2,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(full_ds, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=pm, persistent_workers=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    head_names = {"classifier.1.weight", "classifier.1.bias", "fc.weight", "fc.bias"}
    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n not in head_names], "lr": BACKBONE_LR},
        {"params": [p for n, p in model.named_parameters() if n in head_names],     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def set_lr(optimizer, epoch, warmup_epochs, total_epochs):
    base_lrs = [BACKBONE_LR, HEAD_LR]
    min_lr   = 1e-7
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        scale    = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = max(min_lr, base_lr * scale)


def mixup(images, labels, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Eval — returns accuracy only (no sklearn, no extra overhead)
# ---------------------------------------------------------------------------

def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Data root  : {DATA_ROOT}")
    print(f"Target     : {TARGET_ACC*100:.0f}% stable for {PATIENCE} consecutive epochs")
    print(f"Max epochs : {MAX_EPOCHS}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_loaders()
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model     = get_isic_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    # Phase 1: freeze backbone — train head only during warmup
    for name, param in model.named_parameters():
        if "classifier.1" not in name and "fc" not in name:
            param.requires_grad = False
    optimizer = get_optimizer(model)
    backbone_unfrozen = False

    consecutive_above = 0  # patience counter

    for epoch in range(MAX_EPOCHS):

        # Unfreeze backbone after warmup
        if epoch == WARMUP_EPOCHS and not backbone_unfrozen:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = get_optimizer(model)
            backbone_unfrozen = True
            print(f"[Epoch {epoch+1}] Backbone unfrozen.")

        set_lr(optimizer, epoch, WARMUP_EPOCHS, MAX_EPOCHS)
        lrs = [pg["lr"] for pg in optimizer.param_groups]

        # Train
        model.train()
        running_loss, steps = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mixed, la, lb, lam = mixup(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                out  = model(mixed)
                loss = lam * criterion(out, la) + (1 - lam) * criterion(out, lb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        acc      = accuracy(model, test_loader, device)

        print(
            f"Epoch [{epoch+1:3d}/{MAX_EPOCHS}] | loss={avg_loss:.4f} | "
            f"acc={acc*100:.2f}% | lr=[{lrs[0]:.2e}, {lrs[1]:.2e}]"
        )

        # Patience check
        if acc >= TARGET_ACC:
            consecutive_above += 1
            print(f"  >= {TARGET_ACC*100:.0f}% ({consecutive_above}/{PATIENCE})")
        else:
            consecutive_above = 0

        if consecutive_above >= PATIENCE:
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"\nStable at {TARGET_ACC*100:.0f}% for {PATIENCE} epochs.")
            print(f"Checkpoint saved: {CHECKPOINT}")
            print(f"\nNext steps:")
            print(f"  Centralized: python med_learning_federated_system/central_train_model.py")
            print(f"  FL:          ISIC_PRETRAINED_PATH={CHECKPOINT} flwr run .")
            return

    # Fallback — hit MAX_EPOCHS without stabilizing at target
    torch.save(model.state_dict(), CHECKPOINT)
    final_acc = accuracy(model, test_loader, device)
    print(f"\nMax epochs reached. Final acc={final_acc*100:.2f}%")
    print(f"Checkpoint saved: {CHECKPOINT}")


if __name__ == "__main__":
    main()