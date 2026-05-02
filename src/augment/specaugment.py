"""SpecAugment for log-Mel spectrograms (Park et al. 2019).

Applies random time masking and frequency masking to a (time, freq) tensor.
Applied during training only — never at inference.

Conservative defaults for ICBHI:
  - Crackles are 5-15 ms bursts (~1-2 frames at 10 ms hop): time masks kept short.
  - Wheezes are >80 ms (~8+ frames): time_mask_max=80 still preserves most of the event.
  - Crackle/wheeze frequencies are 350-650 Hz out of 8 kHz Nyquist.
    128 mel bins cover 0-8 kHz, so diagnostic bands are roughly bins 5-10.
    freq_mask_max=27 (~21% of 128) is safe but should not cover the full band.
"""

from __future__ import annotations

import random

import torch


def spec_augment(
    spec: torch.Tensor,
    time_mask_max: int = 80,
    freq_mask_max: int = 27,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment to a (time, freq) log-Mel spectrogram tensor.

    Masks are filled with the mean value of the spectrogram (not zero) to
    avoid introducing artificial silence that would confuse the model.

    Args:
        spec:           (T, F) float tensor.
        time_mask_max:  max width of each time mask in frames.
        freq_mask_max:  max width of each frequency mask in bins.
        n_time_masks:   number of time masks to apply.
        n_freq_masks:   number of frequency masks to apply.
    """
    spec = spec.clone()
    T, F = spec.shape
    fill = spec.mean().item()

    for _ in range(n_time_masks):
        t = random.randint(0, time_mask_max)
        t0 = random.randint(0, max(T - t, 0))
        spec[t0: t0 + t, :] = fill

    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_max)
        f0 = random.randint(0, max(F - f, 0))
        spec[:, f0: f0 + f] = fill

    return spec
