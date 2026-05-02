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
        augment: bool = False,
        augment_kwargs: dict | None = None,
    ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        self.x = x
        self.y = y
        self.processor = processor
        self.sample_rate = sample_rate
        self.augment = augment
        self.augment_kwargs = augment_kwargs or {}

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav = self.x[idx].numpy()
        feats = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = feats.input_values.squeeze(0)  # (1024, 128)

        if self.augment:
            from src.augment.specaugment import spec_augment
            input_values = spec_augment(input_values, **self.augment_kwargs)

        return input_values, self.y[idx]


def load_processor(checkpoint: str = AST_CHECKPOINT):
    from transformers import ASTFeatureExtractor

    return ASTFeatureExtractor.from_pretrained(checkpoint)


def build_datasets(
    cache_path: Path | str,
    processor=None,
    checkpoint: str = AST_CHECKPOINT,
    augment: bool = False,
    augment_kwargs: dict | None = None,
) -> tuple[ICBHIDataset, ICBHIDataset, dict]:
    """Load a cache .pt and return (train_ds, val_ds, raw_cache_dict).

    augment applies SpecAugment to the train split only; val is never augmented.
    """
    cache = torch.load(Path(cache_path), weights_only=False)
    if processor is None:
        processor = load_processor(checkpoint)
    sr = int(cache.get("sample_rate", 16000))
    train_ds = ICBHIDataset(cache["x_train"], cache["y_train"], processor, sr,
                            augment=augment, augment_kwargs=augment_kwargs)
    val_ds = ICBHIDataset(cache["x_val"], cache["y_val"], processor, sr, augment=False)
    return train_ds, val_ds, cache
