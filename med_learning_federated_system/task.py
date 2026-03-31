"""med-learning-federated-system: task.py — ISIC 2019 data, model, train/test."""
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import parameters_to_vector
from torchvision import transforms

from med_learning_federated_system.models.resnet_cnn_model import tiny_resnet18
from med_learning_federated_system.utils.dirichlet_partition import dirichlet_indices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 8

# Canonical ISIC 2019 label ordering — alphabetical by diagnosis code.
# These must match the one-hot column order in the ground truth CSV.
ISIC_CLASSES = [
    "AK",    # 0 — Actinic Keratosis
    "BCC",   # 1 — Basal Cell Carcinoma
    "BKL",   # 2 — Benign Keratosis-like Lesions
    "DF",    # 3 — Dermatofibroma
    "MEL",   # 4 — Melanoma
    "NV",    # 5 — Melanocytic Nevi
    "SCC",   # 6 — Squamous Cell Carcinoma
    "VASC",  # 7 — Vascular Lesions
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

base_dir = os.path.dirname(__file__)
DATA_DIR  = os.path.join(base_dir, "data", "isic2019")

# ---------------------------------------------------------------------------
# Module-level caches — built once per process
# ---------------------------------------------------------------------------

_metadata_cache  = None   # pandas DataFrame of all samples
_partition_cache = None   # list[list[int]] — Dirichlet split indices

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_resnet_cnn_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    return tiny_resnet18(num_classes=num_classes, base_width=8)

# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ISICDataset(Dataset):
    """Thin wrapper: list of image paths + integer labels -> (tensor, int)."""

    def __init__(self, image_paths: list, labels: list, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------

def _load_metadata():
    """
    Parse ISIC_2019_Training_GroundTruth.csv and return a DataFrame with
    columns  path (str)  and  label (int).  Result is cached globally.
    """
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    import pandas as pd

    csv_path = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
    img_dir  = os.path.join(
        DATA_DIR,
        "ISIC_2019_Training_Input",
        "ISIC_2019_Training_Input",
    )

    if not os.path.isfile(csv_path):
        raise RuntimeError(
            f"ISIC metadata CSV not found at {csv_path}. "
            "Run setup_data.py first."
        )

    df = pd.read_csv(csv_path)

    # One-hot columns -> integer label
    label_cols = [c for c in ISIC_CLASSES if c in df.columns]
    if not label_cols:
        raise RuntimeError(
            f"None of the expected class columns {ISIC_CLASSES} found in CSV. "
            f"Columns present: {df.columns.tolist()}"
        )
    df["label"] = df[label_cols].values.argmax(axis=1)
    df["path"]  = df["image"].apply(
        lambda x: os.path.join(img_dir, f"{x}.jpg")
    )

    # Drop rows whose image file is missing on disk
    df = df[df["path"].apply(os.path.isfile)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No valid image files found under {img_dir}. "
            "Check that setup_data.py completed successfully."
        )

    _metadata_cache = df
    return df

# ---------------------------------------------------------------------------
# Transforms
# ISIC images are 450x600 px; resize then crop to 224x224.
# Mean/std estimated from the ISIC 2019 training distribution.
# ---------------------------------------------------------------------------

_MEAN = (0.7012, 0.5517, 0.5714)
_STD  = (0.1517, 0.1703, 0.1814)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ---------------------------------------------------------------------------
# load_data — per-client federated split
# ---------------------------------------------------------------------------

def load_data(
    partition_id: int,
    num_partitions: int,
    alpha_val: float = 0.9,
    batch_size: int = 32,
):
    """
    Return (train_loader, test_loader) for a single federated client.

    The full dataset is split non-IID across clients using a Dirichlet draw
    on integer labels (alpha_val controls heterogeneity).  The partition map
    is built once and reused across all subsequent calls in the same process.

    Each client's slice is then divided 80/20 into local train and test sets.
    """
    global _partition_cache

    df         = _load_metadata()
    all_paths  = df["path"].tolist()
    all_labels = df["label"].tolist()

    if _partition_cache is None:
        _partition_cache = dirichlet_indices(
            labels=all_labels,
            num_partitions=num_partitions,
            alpha=alpha_val,
            seed=42,
        )

    indices = _partition_cache[partition_id]

    split      = int(0.8 * len(indices))
    train_idx  = indices[:split]
    test_idx   = indices[split:]

    train_ds = ISICDataset(
        [all_paths[i]  for i in train_idx],
        [all_labels[i] for i in train_idx],
        transform=TRAIN_TRANSFORM,
    )
    test_ds = ISICDataset(
        [all_paths[i]  for i in test_idx],
        [all_labels[i] for i in test_idx],
        transform=TEST_TRANSFORM,
    )

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    return train_loader, test_loader

# ---------------------------------------------------------------------------
# load_test_data_for_eval — global held-out set for server-side evaluation
# ---------------------------------------------------------------------------

def load_test_data_for_eval(batch_size: int = 64):
    """
    Return a DataLoader over a fixed 10% held-out slice of the full dataset
    for centralised server-side evaluation.

    This sample is drawn with a fixed random state so it is deterministic
    across runs.  It may overlap with client partitions — acceptable for a
    baseline; strict separation can be enforced later if needed.
    """
    df      = _load_metadata()
    eval_df = df.sample(frac=0.10, random_state=0).reset_index(drop=True)

    dataset = ISICDataset(
        eval_df["path"].tolist(),
        eval_df["label"].tolist(),
        transform=TEST_TRANSFORM,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(net, training_data, epochs: int, device, lr: float = 0.005):
    """
    SGD + CosineAnnealingLR with label smoothing.
    Expects plain (image_tensor, label_int) batches from ISICDataset.
    Returns (avg_loss, final_param_vector).
    """
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(training_data) * epochs
    )

    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in training_data:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

    avg_loss  = running_loss / max(1, len(training_data))
    final_vec = parameters_to_vector(net.parameters()).detach().cpu().clone()
    return avg_loss, final_vec

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

def test(net, test_data, device):
    """
    Return (avg_loss, accuracy) over test_data.
    Expects plain (image_tensor, label_int) batches from ISICDataset.
    """
    net.to(device)
    net.eval()

    criterion   = nn.CrossEntropyLoss()
    correct     = 0
    total       = 0
    total_loss  = 0.0

    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / max(1, len(test_data)), correct / max(1, total)