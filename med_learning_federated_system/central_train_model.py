"""
pre_train_model.py — Centralized fine-tuning of pretrained ResNet18 on ISIC 2019.

Augmentations: identical to FL clients — imports TRAIN_TRANSFORMS from task.py
which includes CLAHE, RandomRotation, RandomPerspective, RandomAdjustSharpness,
RandomAutocontrast, and GaussianBlur on top of standard spatial augmentations.

Metrics and plots: identical schema to FL final evaluation in server_strategy.py —
accuracy, weighted P/R/F1, per-class P/R/F1, confusion matrix, per-class ROC
curves, per-class PR curves, training curves. All saved to results/centralized/.

Safe to run in parallel with FL on the same node:
  - Uses ISIC_DATA_ROOT env var — point at same /dev/shm dataset as FL
  - All outputs go to results/centralized/ — no collision with FL results/
  - matplotlib.use("Agg") — no display required, safe on HPC

Usage:
    export ISIC_DATA_ROOT="/dev/shm/isic2019_30pct"
    python med_learning_federated_system/pre_train_model.py
"""

import csv
import math
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    TRAIN_TRANSFORMS,   # includes CLAHE + all augmentations — identical to FL clients
    TEST_TRANSFORMS,
    NUM_CLASSES,
    _TEST_SPLIT_RATIO,
    _DIRICHLET_SEED,
    get_isic_model,
    _TransformSubset,
)

ISIC_CLASS_NAMES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
OUT_DIR = os.path.join(os.environ.get("FL_LOG_DIR", "results"), "centralized")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data — same 85/15 stratified split and WeightedRandomSampler as FL
# ---------------------------------------------------------------------------

def build_centralized_loaders(batch_size: int = 32):
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

    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory  = torch.cuda.is_available()

    train_loader = DataLoader(
        _TransformSubset(full_ds, train_idx, TRAIN_TRANSFORMS),
        batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        torch.utils.data.Subset(full_ds, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Optimizer and LR schedule — differential LR, warmup + cosine
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = [p for n, p in model.named_parameters() if "fc" in n]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ], weight_decay=1e-4)


def warmup_cosine_schedule(optimizer, epoch: int, warmup_epochs: int, total_epochs: int):
    base_lrs = [1e-4, 1e-3]
    min_lr   = 1e-6
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        scale    = 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = max(min_lr, base_lr * scale)


# ---------------------------------------------------------------------------
# Evaluation — returns all_preds explicitly
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, device):
    """Returns (acc, precision, recall, f1, cm, all_labels, all_preds, all_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)
            preds   = torch.argmax(outputs, dim=1)
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
# Final evaluation — identical schema to FL server_strategy._run_final_evaluation
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
    print("Final Centralized Evaluation — ISIC 2019")
    print("=" * 60)
    print(f"Accuracy           : {acc*100:.2f}%")
    print(f"Weighted Precision : {precision:.4f}")
    print(f"Weighted Recall    : {recall:.4f}")
    print(f"Weighted F1        : {f1:.4f}")
    print("\nPer-class breakdown:")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        print(f"  {cls:<8} {per_class_p[i]:>10.4f} {per_class_r[i]:>10.4f} {per_class_f[i]:>10.4f}")
    print("=" * 60)

    # Per-class CSV — same schema as FL
    metrics_csv = os.path.join(OUT_DIR, "final_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1"])
        for i, cls in enumerate(ISIC_CLASS_NAMES):
            w.writerow([cls, f"{per_class_p[i]:.4f}", f"{per_class_r[i]:.4f}", f"{per_class_f[i]:.4f}"])
        w.writerow(["weighted", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
    print(f"[Metrics] Per-class CSV saved: {metrics_csv}")

    _plot_confusion_matrix(cm)
    _plot_roc_curves(all_labels, all_probs)
    _plot_pr_curves(all_labels, all_probs)
    _plot_training_curves(train_losses, accuracies)


# ---------------------------------------------------------------------------
# Plots — identical style to FL server_strategy plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ISIC_CLASS_NAMES)))
    ax.set_yticks(range(len(ISIC_CLASS_NAMES)))
    ax.set_xticklabels(ISIC_CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(ISIC_CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Centralized")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] Confusion matrix saved: {path}")


def _plot_roc_curves(all_labels: np.ndarray, all_probs: np.ndarray) -> None:
    y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC Curves — Centralized")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] ROC curves saved: {path}")


def _plot_pr_curves(all_labels: np.ndarray, all_probs: np.ndarray) -> None:
    y_true = label_binarize(all_labels, classes=list(range(len(ISIC_CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(ISIC_CLASS_NAMES):
        prec_vals, rec_vals, _ = precision_recall_curve(y_true[:, i], all_probs[:, i])
        ax.plot(rec_vals, prec_vals, label=f"{cls} (AUC={auc(rec_vals, prec_vals):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision-Recall Curves — Centralized")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pr_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] PR curves saved: {path}")


def _plot_training_curves(train_losses, accuracies) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, accuracies, label="Accuracy", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_losses, label="Train Loss", color="orange",
                 marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss over Epochs")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Centralized Training Curves — ISIC 2019", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[Metrics] Training curves saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Data root   : {DATA_ROOT}")
    print(f"Output dir  : {OUT_DIR}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_centralized_loaders(batch_size=32)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = get_isic_model().to(device)
    print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    epochs        = 50
    warmup_epochs = 5
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer     = get_optimizer(model)

    train_losses, accuracies = [], []

    for epoch in range(epochs):
        warmup_cosine_schedule(optimizer, epoch, warmup_epochs, epochs)
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]

        model.train()
        running_loss, steps = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            steps += 1

        avg_loss = running_loss / max(steps, 1)
        acc, precision, recall, f1, _, _, _, _ = evaluate_model(model, test_loader, device)

        train_losses.append(avg_loss)
        accuracies.append(acc)

        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | loss={avg_loss:.4f} | acc={acc*100:.2f}% "
            f"| P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} "
            f"| lr=[{current_lrs[0]:.2e}, {current_lrs[1]:.2e}]"
        )

        # Save checkpoint only at the final epoch
        if epoch == epochs - 1:
            torch.save(model.state_dict(), "isic_centralized_resnet18.pth")
            print(f"  -> Checkpoint saved: isic_centralized_resnet18.pth (acc={acc*100:.2f}%)")

    run_final_evaluation(model, test_loader, device, train_losses, accuracies)

    print(f"\nCheckpoint : isic_centralized_resnet18.pth")
    print(f"Plots      : {OUT_DIR}/")
    print(f"FL command : ISIC_PRETRAINED_PATH=isic_centralized_resnet18.pth flwr run .")


if __name__ == "__main__":
    main()
