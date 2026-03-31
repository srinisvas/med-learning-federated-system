"""Dirichlet-based non-IID partition utility."""
import numpy as np
from typing import List


def dirichlet_indices(
    labels: List[int],
    num_partitions: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition a dataset into `num_partitions` non-IID subsets using a
    Dirichlet distribution over class labels.

    Args:
        labels:          Integer label for every sample in the dataset.
        num_partitions:  Number of federated clients.
        alpha:           Dirichlet concentration parameter.
                         Lower -> more heterogeneous.  Typical range: 0.1 – 1.0.
        seed:            Random seed for reproducibility.

    Returns:
        List of length `num_partitions`, where each element is a list of
        integer indices into the original dataset.
    """
    rng    = np.random.default_rng(seed)
    labels = np.array(labels)
    classes = np.unique(labels)

    # Indices grouped by class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in classes}

    # Each partition accumulates indices here
    partitions: List[List[int]] = [[] for _ in range(num_partitions)]

    for c in classes:
        idx = class_indices[c]
        rng.shuffle(idx)

        # Dirichlet proportions for this class across all partitions
        proportions = rng.dirichlet(alpha=np.full(num_partitions, alpha))

        # Convert proportions to split points
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, proportions)

        for p, split in enumerate(splits):
            partitions[p].extend(split.tolist())

    # Shuffle each partition so samples aren't sorted by class
    for p in partitions:
        rng.shuffle(p)

    return partitions
