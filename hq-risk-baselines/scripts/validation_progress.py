"""How far the runs are: distinct run_ids done per sweep versus runs its YAML asks for."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sweep import SweepConfig


def done_ids() -> set[str]:
    ids = set()
    for path in (Path("results/raw/runs.csv"), Path("results/raw/combined.csv")):
        if path.exists():
            with path.open(newline="") as handle:
                ids.update(row["run_id"] for row in csv.DictReader(handle))
    return ids


def main() -> None:
    done = done_ids()
    total_done = total_all = 0
    for config in sorted(Path("configs").glob("*.yaml")):
        runs = SweepConfig(config).runs()
        finished = sum(1 for r in runs if r.run_id() in done)
        total_done += finished
        total_all += len(runs)
        print(f"{config.stem:26s} {finished:5d}/{len(runs):<5d} {'done' if finished == len(runs) else ''}")
    print(f"{'TOTAL':26s} {total_done:5d}/{total_all:<5d}  ({100 * total_done / total_all:.1f}%)")


if __name__ == "__main__":
    main()
