"""Run every configuration in one YAML sweep. --shard/--num-shards lets an HPC array split the work."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import ExperimentRunner
from src.storage import ResultWriter
from src.sweep import SweepConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", default=None, help="override the CSV path in the YAML")
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    parser.add_argument("--skip-existing", action="store_true", help="skip runs whose run_id is already in the output CSV")
    return parser.parse_args()


def existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {row["run_id"] for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    sweep = SweepConfig(args.config)
    output = Path(args.output or sweep.output)
    runs = sweep.runs()[args.shard :: args.num_shards]
    if args.skip_existing:
        done = existing_run_ids(output)
        runs = [config for config in runs if config.run_id() not in done]
    print(f"{sweep.path.name}: {len(runs)} runs in shard {args.shard}/{args.num_shards}")

    if args.dry_run:
        for config in runs:
            print(f"  {config.dataset:14s} {config.partition or '-':8s} intensity={config.intensity:<5} seed={config.seed}")
        return

    writer = ResultWriter(output)
    runner = ExperimentRunner()
    for index, config in enumerate(runs, start=1):
        row = runner.run(config)
        writer.append(row)
        print(
            f"  [{index}/{len(runs)}] {config.dataset:14s} {config.partition or '-':8s} "
            f"intensity={config.intensity:<5} seed={config.seed} "
            f"clean={row['clean_accuracy']:.3f} attacked={row['attacked_accuracy']:.3f} "
            f"drop={row['accuracy_drop_pp']:+.1f}pp"
        )


if __name__ == "__main__":
    main()
