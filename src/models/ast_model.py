"""AST backbone with a 4-class head, initialized from AudioSet pretraining."""

from __future__ import annotations

import torch.nn as nn
from transformers import ASTForAudioClassification

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 4


def build_model(
    checkpoint: str = CHECKPOINT,
    num_classes: int = NUM_CLASSES,
) -> ASTForAudioClassification:
    """Load AST from checkpoint and swap in a num_classes head."""
    return ASTForAudioClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
