"""Check that every derived file still matches runs.csv: summary, tables, explorer data. Exits 1 on the first mismatch.
Run after build_summary / build_tables / build_explorer and before deploying."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_summary import GROUP_KEYS, METRICS, load_latest_per_run, summarise
from scripts.build_tables import build_all, flat_columns
from scripts.sweep_catalog import SWEEPS

RUNS, SUMMARY, TABLES, EXPLORER = Path("results/raw/runs.csv"), Path("results/summary/summary.csv"), Path("results/tables"), Path("explorer")


def check(ok: bool, message: str) -> None:
    print(("ok    " if ok else "FAIL  ") + message)
    if not ok:
        sys.exit(1)


def main() -> None:
    runs = load_latest_per_run(RUNS)
    stored = pd.read_csv(SUMMARY)
    stored["partition"] = stored["partition"].fillna("-")
    fresh = summarise(runs).reset_index(drop=True)
    check(len(fresh) == len(stored), f"summary.csv has {len(stored)} configurations, runs.csv gives {len(fresh)}")
    diff = max((fresh[c] - stored[c]).abs().max() for c in stored.columns if c.endswith(("_mean", "_std", "_min", "_max")))
    check(diff < 1e-6, f"summary.csv matches runs.csv (max abs difference {diff:.1e})")
    short = stored[stored["n"] < 5]
    check(short.empty, f"every configuration has 5 seeds ({len(short)} short)")

    catalogued = {(s["scenario"], s["attack"], s["mode"]) for s in SWEEPS}
    present = set(map(tuple, stored[["scenario", "attack", "attack_mode"]].drop_duplicates().values))
    check(present == catalogued, f"sweep catalogue covers every sweep in summary.csv ({len(present)})")

    tables = build_all(stored)
    for t in tables:
        on_disk = pd.read_csv(TABLES / t["file"], dtype=str, keep_default_na=False)
        expected = pd.DataFrame([[c["text"] for c in row] for row in t["rows"]], columns=flat_columns(t))
        if not on_disk.equals(expected):
            check(False, f"{t['file']} is stale, run scripts/build_tables.py")
    check(True, f"{len(tables)} table CSVs match summary.csv")

    payload = json.loads((EXPLORER / "data.js").read_text().removeprefix("window.HQ_DATA = ").rstrip().rstrip(";"))
    check(len(payload["runs"]) == len(runs) and len(payload["summary"]) == len(stored), "explorer/data.js has every run and configuration")
    check([t["rows"] for t in payload["tables"]] == [t["rows"] for t in json.loads(json.dumps(tables))], "explorer tables match results/tables")
    missing = [f["path"] for f in payload["files"] if not (EXPLORER / f["path"]).exists()]
    check(not missing, f"explorer serves all {len(payload['files'])} source files {missing or ''}")
    keys = {tuple(r[k] for k in GROUP_KEYS if k != "model") for r in payload["summary"]}
    for t in payload["tables"]:
        for row in t["rows"]:
            for c in row:
                if "key" in c and tuple(c["key"].values()) not in keys:
                    check(False, f"table {t['number']} cell {c['text']} points at a missing configuration")
    check(True, "every table cell drills into an existing configuration")
    check(set(METRICS) <= set(payload["runs"][0]), "explorer runs carry every metric")


if __name__ == "__main__":
    main()
