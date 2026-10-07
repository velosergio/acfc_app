from __future__ import annotations

import csv
from pathlib import Path

MANIFEST = "manifest.csv"
SIDES = ("arriba", "abajo", "ninguno")


def write_row(folder: Path, name: str, side: str, cut: int) -> None:
    path = Path(folder) / MANIFEST
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["archivo", "lado", "corte"])
        w.writerow([name, side, int(cut)])


def read_manifest(folder: Path) -> dict[str, tuple[str, int]]:
    path = Path(folder) / MANIFEST
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["archivo"]: (r["lado"], int(r["corte"])) for r in csv.DictReader(fh)}
