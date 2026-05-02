"""Parse the ICBHI official train/test split into Recording objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import find_audio_dir, find_split_file, get_icbhi_root

DEVICES = {"AKGC417L", "LittC2SE", "Litt3200", "Meditron"}


@dataclass(frozen=True)
class Recording:
    stem: str          # filename without extension (as it exists on disk)
    wav_path: Path
    txt_path: Path     # paired annotation file
    patient_id: int    # leading numeric prefix
    device: str        # one of DEVICES
    split: str         # "train" or "test"


def _parse_stem(stem: str) -> tuple[int, str]:
    """Extract (patient_id, device) from a filename stem.

    ICBHI format: <pid>_<rec_idx>_<location>_<mode>_<device>.
    """
    parts = stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected filename format: {stem}")
    return int(parts[0]), parts[-1]


def _resolve_stem(split_stem: str, available: dict[str, Path]) -> Path:
    """Map a stem from train_test.txt to an actual .wav file."""
    if split_stem in available:
        return available[split_stem]
    prefix = "_".join(split_stem.split("_")[:-1]) + "_"
    candidates = [p for s, p in available.items() if s.startswith(prefix)]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot resolve {split_stem!r}: {len(candidates)} prefix candidates."
    )


def load_recordings(root: Path | None = None) -> list[Recording]:
    """Return every recording listed in train_test.txt as a Recording."""
    base = root if root is not None else get_icbhi_root()
    audio_dir = find_audio_dir(base)
    split_path = find_split_file(base)

    available = {p.stem: p for p in audio_dir.glob("*.wav")}
    recordings: list[Recording] = []
    for line in split_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stem, split = line.split()
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split label {split!r} for {stem}")
        wav_path = _resolve_stem(stem, available)
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing annotation: {txt_path}")
        actual_stem = wav_path.stem
        pid, device = _parse_stem(actual_stem)
        recordings.append(Recording(actual_stem, wav_path, txt_path, pid, device, split))
    return recordings


def train_test(root: Path | None = None) -> tuple[list[Recording], list[Recording]]:
    """Split recordings into (train, test) lists."""
    recs = load_recordings(root)
    train = [r for r in recs if r.split == "train"]
    test = [r for r in recs if r.split == "test"]
    return train, test


def train_val_split(
    train_recordings: list[Recording],
    val_ratio: float = 0.2,
    seed: int = 17,
) -> tuple[list[Recording], list[Recording]]:
    """Carve a patient-disjoint val set out of the official train recordings."""
    import random

    rng = random.Random(seed)

    patients: dict[int, list[Recording]] = {}
    for rec in train_recordings:
        patients.setdefault(rec.patient_id, []).append(rec)

    patient_ids = sorted(patients.keys())
    rng.shuffle(patient_ids)

    n_val = max(1, round(len(patient_ids) * val_ratio))
    val_patients = set(patient_ids[-n_val:])

    train_split = [r for r in train_recordings if r.patient_id not in val_patients]
    val_split = [r for r in train_recordings if r.patient_id in val_patients]
    return train_split, val_split
