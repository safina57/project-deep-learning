"""Wrap the preprocessed cache into a torch Dataset that yields AST inputs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ICBHIDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        processor,
        sample_rate: int = 16000,
    ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        self.x = x
        self.y = y
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav = self.x[idx].numpy()
        feats = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = feats.input_values.squeeze(0)  # (1024, 128)
        label = self.y[idx]
        return input_values, label


def load_processor(checkpoint: str = AST_CHECKPOINT):
    from transformers import ASTFeatureExtractor

    return ASTFeatureExtractor.from_pretrained(checkpoint)


def build_datasets(
    cache_path: Path | str,
    processor=None,
    checkpoint: str = AST_CHECKPOINT,
) -> tuple[ICBHIDataset, ICBHIDataset, dict]:
    """Load a cache .pt and return (train_ds, test_ds, raw_cache_dict)."""
    cache = torch.load(Path(cache_path), weights_only=False)
    if processor is None:
        processor = load_processor(checkpoint)
    sr = int(cache.get("sample_rate", 16000))
    train_ds = ICBHIDataset(cache["x_train"], cache["y_train"], processor, sr)
    test_ds = ICBHIDataset(cache["x_test"], cache["y_test"], processor, sr)
    return train_ds, test_ds, cache
