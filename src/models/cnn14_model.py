"""CNN14 classifier (PANNs — Pretrained Audio Neural Networks).

CNN14 is a 14-layer CNN pretrained on AudioSet for audio tagging (Kong et al. 2020).

Input:  (batch, 128000)  raw float32 waveform at 16 kHz
Output: (batch, 4)       logits

Loaded via panns-inference which downloads the AudioSet-pretrained weights.
The original 527-class AudioSet head is replaced with a 4-class head.

Install: pip install panns-inference
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES = 4


class CNN14Classifier(nn.Module):
    """CNN14 backbone (panns-inference) + 4-class head. forward(x) returns logits."""

    def __init__(self, backbone: nn.Module, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = backbone
        # Replace the AudioSet 527-class head with a 4-class head.
        # CNN14 from panns-inference has fc_audioset: Linear(2048, 527)
        in_features = backbone.fc_audioset.in_features
        backbone.fc_audioset = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # panns CNN14 forward returns a dict {'clipwise_output': ..., 'embedding': ...}
        out = self.backbone(x)
        return out["clipwise_output"]  # (B, num_classes)


def build_cnn14_model(
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> CNN14Classifier:
    """Download pretrained CNN14 weights via panns-inference and attach a new head."""
    from panns_inference import AudioTagging

    at = AudioTagging(checkpoint_path=None, device=device)
    backbone = at.model
    if isinstance(backbone, nn.DataParallel):
        backbone = backbone.module
    return CNN14Classifier(backbone, num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
