"""
central_train_model.py — Centralized continuation from 70% checkpoint.

Loads isic_pretrained_70pct.pth and trains for 50 epochs.
Directly comparable to 50 FL rounds — both start from the same checkpoint.

Uses AdamW (correct for stateful centralized training) with cosine decay
from full LR. No freeze phase — backbone already trained at 70%.

Outputs → results/centralized/continuation/
  - final_metrics.csv  (per-class P/R/F1 + weighted, same schema as FL)
  - confusion_matrix.png
  - roc_curves.png
  - pr_curves.png
  - training_curves.png  (70% and 80% reference lines for paper figures)

Usage:
    export ISIC_DATA_ROOT="/dev/shm/isic2019_50pct"
    python med_learning_federated_system/central_train_model.py
"""

import csv
import math
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize
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
ISIC_CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
PRETRAIN_CKPT    = "isic_pretrained_70pct.pth"
FINAL_CKPT       = "isic_centralized_final.pth"
OUT_DIR          = os.path.join(os.environ.get("FL_LOG_DIR", "results"), "centralized", "continuation")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS       = 50
BATCH_SIZE   = 128
BACKBONE_LR  = 4e-4
HEAD_LR      = 4e-3
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA  = 0.2
AMP_ENABLED  = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Data — identical split to pre_train_model.py and FL task.py
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
        num_samples=len(train_idx), replacement=True,
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
# Optimizer — AdamW correct here (stateful centralized training).
# No freeze phase — backbone already converged at 70%.
# Cosine decay from full LR over EPOCHS with no warmup.
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    head_names = {"classifier.1.weight", "classifier.1.bias", "fc.weight", "fc.bias"}
    return torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if n not in head_names], "lr": BACKBONE_LR},
        {"params": [p for n, p in model.named_parameters() if n in head_names],     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)


def set_lr(optimizer, epoch, total_epochs):
    """Pure cosine decay, no warmup — backbone already trained."""
    base_lrs = [BACKBONE_LR, HEAD_LR]
    min_lr   = 1e-7
    scale    = 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = max(min_lr, base_lr * scale)


def mixup(images, labels, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                outputs = model(images)
            probs = torch.softmax(outputs.float(), dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        confusion_matrix(all_labels, all_preds),
        all_labels, all_preds, all_probs,
    )


# ---------------------------------------------------------------------------
# Final evaluation — identical schema to FL server_strategy
# ---------------------------------------------------------------------------

def run_final_evaluation(model, loader, device, train_losses, accuracies):
    print("\n[Metrics] Running final evaluation on held-out test set...")
    acc, precision, recall, f1, cm, all_labels, all_preds, all_probs = evaluate_model(
        model, loader, device
    )
    per_class_p = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print("\n" + "=" * 60)
    print("Final Centralized")
    print("=" * 60)
    print(f"Starting point     : ~70%  (from isic_pretrained_70pct.pth)")
    print(f"Final accuracy     : {acc*100:.2f}%")
    print(f"Peak accuracy      : {max(accuracies)*100:.2f}%")
    print(f"Weighted Precision : {precision:.4f}")
    print(f"Weighted Recall    : {recall:.4f}")
    print(f"Weighted F1        : {f1:.4f}")
    print("\nPer-class breakdown:")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        print(f"  {cls:<8} {per_class_p[i]:>10.4f} {per_class_r[i]:>10.4f} {per_class_f[i]:>10.4f}")
    print("=" * 60)

    metrics_csv = os.path.join(OUT_DIR, "final_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1"])
        for i, cls in enumerate(ISIC_CLASS_NAMES):
            w.writerow([cls, f"{per_class_p[i]:.4f}", f"{per_class_r[i]:.4f}", f"{per_class_f[i]:.4f}"])
        w.writerow(["weighted", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    print(f"[Metrics] Per-class CSV: {metrics_csv}")

    _plot_confusion_matrix(cm)
    _plot_roc_curves(all_labels, all_probs)
    _plot_pr_curves(all_labels, all_probs)
    _plot_training_curves(train_losses, accuracies)


def _plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ISIC_CLASS_NAMES))); ax.set_yticks(range(len(ISIC_CLASS_NAMES)))
    ax.set_xticklabels(ISIC_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ISIC_CLASS_NAMES)
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Centralized Training")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] Confusion matrix: {path}")


def _plot_roc_curves(all_labels, all_probs):
    y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Per-class ROC Curves - Centralized Training")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] ROC curves: {path}")


def _plot_pr_curves(all_labels, all_probs):
    y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        prec_vals, rec_vals, _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
        ax.plot(rec_vals, prec_vals, label=f"{cls} (AUC={auc(rec_vals, prec_vals):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Per-class PR Curves - Centralized Training")
    ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pr_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] PR curves: {path}")


def _plot_training_curves(train_losses, accuracies):
    epochs_x = list(range(1, len(accuracies) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_x, accuracies, label="Accuracy", marker="o", markersize=3, color="steelblue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Centralized Accuracy over {EPOCHS} Epochs")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.60, 1.00)

    axes[1].plot(epochs_x, train_losses, label="Train Loss", color="orange", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss over Epochs")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Centralized Training", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] Training curves: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Data root   : {DATA_ROOT}")
    print(f"Output dir  : {OUT_DIR}")
    print(f"Epochs      : {EPOCHS}")
    print(f"AMP enabled : {AMP_ENABLED}")

    if not os.path.isfile(PRETRAIN_CKPT):
        print(f"\nERROR: {PRETRAIN_CKPT} not found.")
        print("Run: python med_learning_federated_system/pre_train_model.py")
        return

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_loaders()
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = get_isic_model().to(device)
    model.load_state_dict(torch.load(PRETRAIN_CKPT, map_location="cpu"))
    print(f"Loaded: {PRETRAIN_CKPT}")

    # Verify starting accuracy
    start_acc, *_ = evaluate_model(model, test_loader, device)
    print(f"Starting accuracy: {start_acc*100:.2f}%  (target ~70%)\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = get_optimizer(model)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    train_losses, accuracies = [], []

    for epoch in range(EPOCHS):
        set_lr(optimizer, epoch, EPOCHS)
        lrs = [pg["lr"] for pg in optimizer.param_groups]

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
        acc, precision, recall, f1, _, _, _, _ = evaluate_model(model, test_loader, device)
        train_losses.append(avg_loss)
        accuracies.append(acc)

        print(
            f"Epoch [{epoch+1:3d}/{EPOCHS}] | loss={avg_loss:.4f} | acc={acc*100:.2f}% "
            f"| P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} "
            f"| lr=[{lrs[0]:.2e}, {lrs[1]:.2e}]"
        )

    torch.save(model.state_dict(), FINAL_CKPT)
    print(f"\nCheckpoint: {FINAL_CKPT}")

    run_final_evaluation(model, test_loader, device, train_losses, accuracies)
    print(f"\nResults: {OUT_DIR}/")


if __name__ == "__main__":
    main()