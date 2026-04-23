import os
from collections import OrderedDict
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image

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
# CLAHE contrast enhancement
# ---------------------------------------------------------------------------

class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if len(arr.shape) == 3:
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            arr = self.clahe.apply(arr)
        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORMS = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomAutocontrast(p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
_full_dataset:  Optional[ImageFolder]     = None
_train_indices: Optional[List[List[int]]] = None
_test_indices:  Optional[List[int]]       = None

_TEST_SPLIT_RATIO: float = 0.15
_DIRICHLET_SEED:   int   = 42

# α=1.0 — near-IID. With only 10 clients, α=0.5 creates severe class skew
# per client that causes gradient conflicts across the 5 sampled per round,
# producing the oscillation pattern (loss drops, MTA stays flat).
# α=1.0 keeps class proportions close to global distribution — each client's
# gradient points in roughly the same direction, aggregation is coherent.
_DIRICHLET_ALPHA:  float = 1.0


def _load_and_partition(num_partitions: int, alpha: float) -> None:
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
    targets        = np.array(dataset.targets)[indices]
    class_counts   = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    class_counts   = np.where(class_counts == 0, 1.0, class_counts)
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
    _load_and_partition(num_partitions, alpha_val)

    train_indices = _train_indices[partition_id]
    train_subset  = _TransformSubset(_full_dataset, train_indices, TRAIN_TRANSFORMS)
    val_subset    = Subset(_full_dataset, _test_indices)
    sampler       = _make_weighted_sampler(_full_dataset, train_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, val_loader


def load_test_data_for_eval(batch_size: int = 32) -> DataLoader:
    if _test_indices is None:
        num_partitions = int(os.environ.get("FL_NUM_CLIENTS", "10"))
        _load_and_partition(num_partitions=num_partitions, alpha=_DIRICHLET_ALPHA)
    return DataLoader(
        Subset(_full_dataset, _test_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )


def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.0005,
) -> Tuple[float, torch.Tensor]:
    """
    SGD with Nesterov momentum and differential LRs for FL fine-tuning.

    Why SGD over AdamW in FL:
    AdamW maintains per-parameter moment estimates (m, v) that accumulate
    across thousands of steps in centralized training. In FL the optimizer
    is re-initialized from scratch every round — those estimates never build
    up, so every round starts with unregulated gradient steps. With 5 clients
    pulling in different directions this compounds into the divergence pattern
    (loss drops, MTA oscillates). SGD has no per-parameter state to corrupt
    between rounds and behaves predictably with a fixed LR.

    Nesterov momentum adds look-ahead which helps convergence on smooth
    loss surfaces typical of fine-tuning.

    Differential LRs:
      backbone = lr * 0.1 = 0.00005  (very conservative — preserve features)
      head     = lr       = 0.0005   (adapts to local class distribution)
    """
    from torch.nn.utils import parameters_to_vector

    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()

    head_param_names = {"classifier.1.weight", "classifier.1.bias", "fc.weight", "fc.bias"}
    backbone_params  = [p for n, p in net.named_parameters() if n not in head_param_names]
    head_params      = [p for n, p in net.named_parameters() if n in head_param_names]

    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": lr * 0.1},  # backbone: very conservative
        {"params": head_params,     "lr": lr},         # head: full LR
    ], momentum=0.9, weight_decay=1e-4, nesterov=True)

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