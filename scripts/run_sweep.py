"""Run every configuration in one YAML sweep. --shard/--num-shards lets an HPC array split the work."""

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep = SweepConfig(args.config)
    runs = sweep.runs()[args.shard :: args.num_shards]
    print(f"{sweep.path.name}: {len(runs)} runs in shard {args.shard}/{args.num_shards}")

    if args.dry_run:
        for config in runs:
            print(f"  {config.dataset:14s} {config.partition or '-':8s} intensity={config.intensity:<5} seed={config.seed}")
        return

    writer = ResultWriter(args.output or sweep.output)
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
