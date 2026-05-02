"""Resample, segment, and cyclic-pad ICBHI respiratory cycles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

from .annotations import label_4class, parse_annotation
from .splits import Recording, load_recordings, train_val_split

TARGET_SR = 16000
TARGET_DURATION_S = 8
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION_S  # 128000
MIN_CHUNK_SAMPLES = 100  # ~6 ms; drops annotation-noise micro-cycles

DEVICE_TO_ID = {"AKGC417L": 0, "LittC2SE": 1, "Litt3200": 2, "Meditron": 3}


def cyclic_pad(wav: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    n = wav.shape[0]
    if n >= target_len:
        return wav[:target_len]
    repeats = target_len // n + 1
    return np.tile(wav, repeats)[:target_len]


def segment_cycles(
    audio: np.ndarray,
    cycles: Iterable[tuple[float, float, int, int]],
    sr: int = TARGET_SR,
) -> Iterator[tuple[np.ndarray, int]]:
    for start_s, end_s, c, w in cycles:
        start = int(start_s * sr)
        end = int(end_s * sr)
        chunk = audio[start:end]
        if chunk.shape[0] < MIN_CHUNK_SAMPLES:
            continue
        yield cyclic_pad(chunk).astype(np.float32, copy=False), label_4class(c, w)


def process_recording(rec: Recording) -> list[tuple[np.ndarray, int]]:
    audio, _ = librosa.load(str(rec.wav_path), sr=TARGET_SR, mono=True)
    cycles = parse_annotation(rec.txt_path)
    return list(segment_cycles(audio, cycles))


def _empty_split() -> dict:
    return {"x": [], "y": [], "device": [], "stem": []}


def build_cache(
    recordings: list[Recording],
    out_path: Path | str | None = None,
    limit: int | None = None,
    progress: bool = True,
    val_ratio: float = 0.2,
    val_seed: int = 17,
) -> dict:
    """Process recordings into per-split tensor arrays. Returns the cache dict.

    If out_path is given, also torch.saves the dict to that path.
    """
    if limit is not None:
        recordings = recordings[:limit]

    train_recs = [r for r in recordings if r.split == "train"]
    test_recs = [r for r in recordings if r.split == "test"]
    train_recs, val_recs = train_val_split(train_recs, val_ratio=val_ratio, seed=val_seed)

    from dataclasses import replace
    labeled: list[Recording] = (
        [replace(r, split="train") for r in train_recs]
        + [replace(r, split="val") for r in val_recs]
        + [replace(r, split="test") for r in test_recs]
    )

    buckets = {"train": _empty_split(), "val": _empty_split(), "test": _empty_split()}
    iterable: Iterable[Recording] = labeled
    if progress:
        iterable = tqdm(list(iterable), desc="preprocess")

    for rec in iterable:
        dev_id = DEVICE_TO_ID.get(rec.device, -1)
        for wav, label in process_recording(rec):
            b = buckets[rec.split]
            b["x"].append(wav)
            b["y"].append(label)
            b["device"].append(dev_id)
            b["stem"].append(rec.stem)

    cache: dict = {
        "sample_rate": TARGET_SR,
        "target_samples": TARGET_SAMPLES,
        "device_to_id": DEVICE_TO_ID,
        "val_ratio": val_ratio,
        "val_seed": val_seed,
    }
    for split, b in buckets.items():
        if b["x"]:
            x = torch.from_numpy(np.stack(b["x"], axis=0))  # (N, 128000) float32
        else:
            x = torch.empty((0, TARGET_SAMPLES), dtype=torch.float32)
        cache[f"x_{split}"] = x
        cache[f"y_{split}"] = torch.tensor(b["y"], dtype=torch.long)
        cache[f"device_{split}"] = torch.tensor(b["device"], dtype=torch.long)
        cache[f"stem_{split}"] = b["stem"]

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, out_path)

    return cache


def summarize(cache: dict) -> str:
    from .annotations import CLASS_NAMES

    lines = []
    for split in ("train", "val", "test"):
        y = cache[f"y_{split}"]
        x = cache[f"x_{split}"]
        counts = {name: int((y == k).sum()) for k, name in enumerate(CLASS_NAMES)}
        lines.append(
            f"{split}: x={tuple(x.shape)} {x.dtype}  "
            + "  ".join(f"{n}={c}" for n, c in counts.items())
        )
    return "\n".join(lines)


def build_default(
    out_path: Path | str | None = None,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Convenience: load recordings via splits + run build_cache."""
    return build_cache(load_recordings(), out_path=out_path, limit=limit, progress=progress)
