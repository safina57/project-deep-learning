"""Waveform dataset for models that process raw audio (CNN14).

Returns (waveform, label) where waveform is the raw float32 tensor (128000,)
from the cache — no feature extractor needed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class WaveformDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_waveform_datasets(
    cache_path: Path | str,
) -> tuple[WaveformDataset, WaveformDataset, dict]:
    """Load cache and return (train_ds, val_ds, raw_cache_dict)."""
    cache = torch.load(Path(cache_path), weights_only=False)
    train_ds = WaveformDataset(cache["x_train"], cache["y_train"])
    val_ds   = WaveformDataset(cache["x_val"],   cache["y_val"])
    return train_ds, val_ds, cache
