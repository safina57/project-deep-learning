"""CLI: resample + segment + cyclic-pad ICBHI cycles into a .pt cache.

Examples
--------
    uv run python scripts/preprocess.py --out data/cache/smoke.pt --limit 5
    uv run python scripts/preprocess.py --out data/cache/icbhi_16k_8s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import build_default, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output .pt cache path")
    ap.add_argument("--limit", type=int, default=None, help="process first N recordings (smoke)")
    args = ap.parse_args()

    cache = build_default(out_path=args.out, limit=args.limit)
    print(summarize(cache))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
