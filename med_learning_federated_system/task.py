import os
from collections import OrderedDict
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from med_learning_federated_system.models.resnet_cnn_model import med_tiny_resnet18
from med_learning_federated_system.utils.dirichlet_partition import dirichlet_indices

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT: str = os.environ.get(
    "ISIC_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "data", "isic2019"),
)

ISIC_CLASSES: List[str] = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
NUM_CLASSES: int = 8
IMG_SIZE:    int = 224

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_full_dataset:   Optional[ImageFolder]    = None
_train_indices:  Optional[List[List[int]]] = None
_test_indices:   Optional[List[int]]       = None

_TEST_SPLIT_RATIO: float = 0.15
_DIRICHLET_ALPHA:  float = 0.5
_DIRICHLET_SEED:   int   = 42


def _load_and_partition(num_partitions: int, alpha: float) -> None:
    """
    Loads the full ISIC ImageFolder dataset exactly once, stratified 85/15
    train/test split, then Dirichlet partition across num_partitions clients.
    Cached in module globals — subsequent calls are free.
    """
    global _full_dataset, _train_indices, _test_indices

    if _train_indices is not None:
        return

    if not os.path.isdir(DATA_ROOT):
        raise RuntimeError(
            f"ISIC data root not found: {DATA_ROOT}\n"
            "Set ISIC_DATA_ROOT to the path containing the 8 class subdirectories."
        )

    _full_dataset = ImageFolder(root=DATA_ROOT, transform=TEST_TRANSFORMS)
    all_labels = np.array(_full_dataset.targets)

    rng = np.random.default_rng(_DIRICHLET_SEED)
    train_idx_all, test_idx_all = [], []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(all_labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * _TEST_SPLIT_RATIO))
        test_idx_all.extend(cls_idx[:n_test].tolist())
        train_idx_all.extend(cls_idx[n_test:].tolist())

    _test_indices = test_idx_all
    train_labels  = all_labels[train_idx_all]

    raw_partitions = dirichlet_indices(
        labels=train_labels,
        num_partitions=num_partitions,
        alpha=alpha,
        seed=_DIRICHLET_SEED,
    )
    train_idx_arr  = np.array(train_idx_all)
    _train_indices = [train_idx_arr[part].tolist() for part in raw_partitions]


def _make_weighted_sampler(dataset: ImageFolder, indices: List[int]) -> WeightedRandomSampler:
    targets       = np.array(dataset.targets)[indices]
    class_counts  = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    class_counts  = np.where(class_counts == 0, 1.0, class_counts)
    sample_weights = (1.0 / class_counts)[targets]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(indices),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_isic_model(num_classes: int = NUM_CLASSES, base_width: int = 32) -> nn.Module:
    return med_tiny_resnet18(num_classes=num_classes, base_width=base_width)


def load_data(
    partition_id: int,
    num_partitions: int,
    alpha_val: float = _DIRICHLET_ALPHA,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for a given client partition.

    num_workers=0 and pin_memory=False — clients run as Ray actor subprocesses.
    Forking DataLoader workers inside Ray actors causes memory bloat and
    instability. pin_memory only helps when the target device is GPU and
    the DataLoader runs in the same process context, which is not the case here.
    """
    _load_and_partition(num_partitions, alpha_val)

    train_indices = _train_indices[partition_id]
    train_subset  = _TransformSubset(_full_dataset, train_indices, TRAIN_TRANSFORMS)
    val_subset    = Subset(_full_dataset, _test_indices)
    sampler       = _make_weighted_sampler(_full_dataset, train_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader


def load_test_data_for_eval(batch_size: int = 32) -> DataLoader:
    """
    Global held-out test DataLoader for server-side centralized evaluation.

    Uses FL_NUM_CLIENTS env var to match the partition count used during
    client load_data() calls — ensures the train/test split is identical.
    num_workers=2 and pin_memory=True are safe here because this runs in
    the server's main process which has legitimate GPU access.
    """
    if _test_indices is None:
        num_partitions = int(os.environ.get("FL_NUM_CLIENTS", "10"))
        _load_and_partition(num_partitions=num_partitions, alpha=_DIRICHLET_ALPHA)
    return DataLoader(
        Subset(_full_dataset, _test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.01,
) -> Tuple[float, torch.Tensor]:
    from torch.nn.utils import parameters_to_vector

    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss  = total_loss / max(steps, 1)
    final_vec = parameters_to_vector(net.parameters()).detach().cpu()
    return avg_loss, final_vec


def test(net: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / max(len(test_loader), 1), correct / max(total, 1)


def get_weights(net: nn.Module) -> list:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: list) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class _TransformSubset(torch.utils.data.Dataset):
    """
    Subset with per-sample transform override. Loads PIL images directly
    from the base dataset's loader, bypassing the base transform, then
    applies the provided transform. Allows train/test transforms to differ
    while sharing a single ImageFolder instance in memory.
    """

    def __init__(self, dataset: ImageFolder, indices: List[int], transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label