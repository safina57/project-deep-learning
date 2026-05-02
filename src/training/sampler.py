"""WeightedRandomSampler for the 4-class ICBHI imbalance."""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler


def make_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Return a WeightedRandomSampler that equalizes class frequency.

    Args:
        labels: 1-D long tensor of class indices (0-3) for the training split.
    """
    labels = labels.long()
    num_classes = int(labels.max().item()) + 1
    class_counts = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        class_counts[c] = (labels == c).sum().float()

    # Weight per class: inverse frequency. Classes with 0 samples get weight 0.
    class_weights = torch.zeros(num_classes, dtype=torch.float)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero]

    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
