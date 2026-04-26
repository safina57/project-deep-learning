"""Resolve ICBHI dataset paths across runtimes (local / Kaggle / Colab)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input/icbhi-2017-challenge-respiratory-sound-database")
COLAB_DRIVE = Path("/content/drive/MyDrive/icbhi")
LOCAL_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "icbhi"

# Expected file counts after full setup: 920 .wav + 920 paired annotation .txt
EXPECTED_WAV = 920
EXPECTED_TXT = 923


def detect_runtime() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or KAGGLE_INPUT.exists():
        return "kaggle"
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


def get_icbhi_root(override: str | os.PathLike[str] | None = None) -> Path:
    if override is not None:
        root = Path(override)
        if not root.exists():
            raise FileNotFoundError(f"ICBHI override path does not exist: {root}")
        return root

    env = os.environ.get("ICBHI_ROOT")
    if env:
        root = Path(env)
        if not root.exists():
            raise FileNotFoundError(f"ICBHI_ROOT points to a missing path: {root}")
        return root

    runtime = detect_runtime()
    candidate = {"kaggle": KAGGLE_INPUT, "colab": COLAB_DRIVE, "local": LOCAL_DEFAULT}[runtime]
    if not candidate.exists() or not any(candidate.iterdir()):
        raise FileNotFoundError(f"ICBHI dataset not found at {candidate} (runtime={runtime}).")
    return candidate


def find_audio_dir(root: Path | None = None) -> Path:
    """Directory holding the .wav + paired .txt annotation files."""
    base = root if root is not None else get_icbhi_root()
    for path in [base, *base.rglob("*")]:
        if path.is_dir() and any(p.suffix == ".wav" for p in path.iterdir() if p.is_file()):
            return path
    raise FileNotFoundError(f"No directory with .wav files under {base}")


def find_split_file(root: Path | None = None) -> Path:
    """Official ICBHI patient-disjoint split file (train_test.txt)."""
    base = root if root is not None else get_icbhi_root()
    matches = list(base.rglob("train_test.txt"))
    if not matches:
        raise FileNotFoundError(f"train_test.txt not found under {base}")
    return matches[0]
