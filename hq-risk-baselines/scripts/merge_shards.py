"""Append every HPC shard CSV into the single append-only runs.csv."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import ResultWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default="results/raw/shards")
    parser.add_argument("--output", default="results/raw/runs.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = ResultWriter(args.output)
    appended = 0
    for shard in sorted(Path(args.shards).glob("*.csv")):
        with shard.open(newline="") as handle:
            for row in csv.DictReader(handle):
                writer.append(row)
                appended += 1
    print(f"appended {appended} rows to {args.output}")


if __name__ == "__main__":
    main()
