"""Export summary.csv, the per-seed runs and the paper tables into explorer/data.js for the static results explorer
(explorer/index.html), and copy every source CSV into explorer/data/ so the site serves them."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_summary import GROUP_KEYS, METRICS, load_latest_per_run
from scripts.build_tables import build_all
from scripts.sweep_catalog import SWEEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--tables", default="results/tables")
    parser.add_argument("--output", default="explorer/data.js")
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=6))


def publish(sources: list[Path], site: Path) -> list[dict]:
    """Copy each source into <site>/data/ and describe it: served path, repo path, rows, sha256."""
    folder = site / "data"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    files = []
    for source in sources:
        shutil.copyfile(source, folder / source.name)
        rows = sum(1 for _ in source.open()) - 1 if source.suffix == ".csv" else None
        files.append({"path": f"data/{source.name}", "source": source.as_posix(), "rows": rows,
                      "bytes": source.stat().st_size, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()[:12]})
    return files


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary["partition"] = summary["partition"].fillna("-")
    runs = load_latest_per_run(args.runs)[["run_id"] + GROUP_KEYS + ["seed"] + METRICS]
    tables = build_all(summary)
    site = Path(args.output).parent
    sources = [Path(args.runs), Path(args.summary)] + [Path(args.tables) / t["file"] for t in tables] + [Path(args.tables) / "tables.md"]
    payload = {
        "summary": records(summary), "runs": records(runs), "sweeps": SWEEPS, "tables": tables,
        "files": publish(sources, site), "commit": str(pd.read_csv(args.runs)["git_commit"].iloc[-1]),
    }
    Path(args.output).write_text("window.HQ_DATA = " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"{len(summary)} configurations, {len(runs)} runs, {len(tables)} tables -> {args.output}")


if __name__ == "__main__":
    main()
