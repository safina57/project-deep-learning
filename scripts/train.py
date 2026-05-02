"""CLI: train from a YAML config.

    uv run python scripts/train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_loop import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(args.config.read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  config: {args.config.name}")

    results = train(config, cache_path=config["cache_path"], device=device)
    print("best checkpoint:", results["best_checkpoint"])


if __name__ == "__main__":
    main()
