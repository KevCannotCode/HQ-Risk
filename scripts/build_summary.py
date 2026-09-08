"""Aggregate runs.csv over seeds into summary.csv: mean, sd, min, max, n per configuration."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUP_KEYS = ["scenario", "dataset", "model", "attack", "attack_mode", "partition", "intensity"]
METRICS = ["clean_accuracy", "attacked_accuracy", "accuracy_drop_pp", "attack_success_rate", "balanced_accuracy", "positive_recall", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--output", default="results/summary/summary.csv")
    return parser.parse_args()


def load_latest_per_run(path: str) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs["partition"] = runs["partition"].fillna("-")
    return runs.sort_values("timestamp").drop_duplicates("run_id", keep="last")


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(GROUP_KEYS, dropna=False)
    summary = grouped[METRICS].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary["n"] = grouped["seed"].nunique()
    summary["seeds"] = grouped["seed"].agg(lambda seeds: ",".join(str(s) for s in sorted(seeds)))
    return summary.reset_index().sort_values(GROUP_KEYS)


def main() -> None:
    args = parse_args()
    runs = load_latest_per_run(args.runs)
    summary = summarise(runs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, float_format="%.6f")
    incomplete = summary[summary["n"] < 5]
    print(f"{len(runs)} distinct runs -> {len(summary)} configurations -> {args.output}")
    if len(incomplete):
        print(f"WARNING: {len(incomplete)} configurations have fewer than 5 seeds")


if __name__ == "__main__":
    main()
