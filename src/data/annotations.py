"""Parse ICBHI per-cycle annotation files (one .txt per recording)."""

from __future__ import annotations

from pathlib import Path

# (0,0)=Normal, (1,0)=Crackle, (0,1)=Wheeze, (1,1)=Both
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def parse_annotation(path: str | Path) -> list[tuple[float, float, int, int]]:
    """Return [(start_s, end_s, crackle, wheeze), ...] for one recording."""
    cycles: list[tuple[float, float, int, int]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cycles.append((float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
    return cycles


def label_4class(crackle: int, wheeze: int) -> int:
    """0=Normal, 1=Crackle, 2=Wheeze, 3=Both."""
    return crackle + 2 * wheeze
