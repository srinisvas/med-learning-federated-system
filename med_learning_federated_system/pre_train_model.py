import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import numpy as np

from med_learning_federated_system.task import (
    DATA_ROOT,
    TRAIN_TRANSFORMS,
    TEST_TRANSFORMS,
    NUM_CLASSES,
    _TEST_SPLIT_RATIO,
    _DIRICHLET_SEED,
    get_isic_model,
    test,
    _TransformSubset,
)


def build_centralized_loaders(batch_size: int = 32):

    full_ds = ImageFolder(root=DATA_ROOT, transform=TEST_TRANSFORMS)
    all_labels = np.array(full_ds.targets)
    rng = np.random.default_rng(_DIRICHLET_SEED)

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
    weights = (1.0 / class_counts)[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(train_idx),
        replacement=True,
    )

    train_subset = _TransformSubset(full_ds, train_idx, TRAIN_TRANSFORMS)
    test_subset  = torch.utils.data.Subset(full_ds, test_idx)

    num_workers = 2 if torch.cuda.is_available() else 1
    pin_memory  = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=True,
    )
    return train_loader, test_loader


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:

    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = [p for n, p in model.named_parameters() if "fc" in n]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},   # backbone: conservative
        {"params": head_params,     "lr": 1e-3},   # head: faster adaptation
    ], weight_decay=1e-4)


def warmup_cosine_schedule(optimizer, epoch: int, warmup_epochs: int, total_epochs: int):

    base_lrs = [1e-4, 1e-3]   # must match get_optimizer() order
    min_lr   = 1e-6

    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = max(min_lr, base_lr * scale)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_centralized_loaders(batch_size=32)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = get_isic_model().to(device)
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    epochs        = 75
    warmup_epochs = 5
    criterion     = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer     = get_optimizer(model)

    best_acc = 0.0
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
        _, acc = test(model, test_loader, device)
        print(
            f"Epoch [{epoch+1:3d}/{epochs}] | loss={avg_loss:.4f} | acc={acc*100:.2f}% "
            f"| lr=[{current_lrs[0]:.2e}, {current_lrs[1]:.2e}]"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "isic_pretrained_resnet18.pth")
            print(f"  -> Saved best checkpoint (acc={best_acc*100:.2f}%)")

    print(f"\nPretraining complete. Best accuracy: {best_acc*100:.2f}%")
    print("Checkpoint: isic_pretrained_resnet18.pth")
    print("FL training: ISIC_PRETRAINED_PATH=isic_pretrained_resnet18.pth flwr run .")


if __name__ == "__main__":
    main()
