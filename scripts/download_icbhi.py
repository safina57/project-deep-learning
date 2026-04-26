"""Download the ICBHI 2017 dataset from Kaggle into ``data/icbhi/``.

Requires KAGGLE_USERNAME and KAGGLE_KEY (loaded from `.env` if present).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

KAGGLE_DATASET = "nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/icbhi"))
    parser.add_argument("--force", action="store_true", help="Re-download even if non-empty.")
    args = parser.parse_args()

    load_dotenv()
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        sys.exit("KAGGLE_USERNAME / KAGGLE_KEY must be set (see .env.example).")

    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()) and not args.force:
        existing = sum(1 for p in args.out.rglob("*") if p.is_file())
        print(f"{args.out} already has {existing} files. Pass --force to re-download.")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} → {args.out} (~2 GB; will unzip in place)…")
    api.dataset_download_files(KAGGLE_DATASET, path=str(args.out), unzip=True, quiet=False)
    print("Done.")


if __name__ == "__main__":
    main()
