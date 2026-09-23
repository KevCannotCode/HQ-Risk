"""Export summary.csv and the per-seed runs into explorer/data.js for the static results explorer (explorer/index.html)."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_summary import GROUP_KEYS, METRICS, load_latest_per_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--output", default="explorer/data.js")
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=6))


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary["partition"] = summary["partition"].fillna("-")
    runs = load_latest_per_run(args.runs)[GROUP_KEYS + ["seed"] + METRICS]
    payload = {"summary": records(summary), "runs": records(runs), "commit": str(pd.read_csv(args.runs)["git_commit"].iloc[-1])}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("window.HQ_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{len(summary)} configurations, {len(runs)} runs -> {args.output}")


if __name__ == "__main__":
    main()
