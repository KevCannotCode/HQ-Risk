"""Check that every derived file still matches runs.csv: summary, tables, explorer data. Exits 1 on the first mismatch.
Run after build_summary / build_tables / build_explorer and before deploying."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_summary import GROUP_KEYS, METRICS, load_latest_per_run, summarise
from scripts.build_tables import build_all, flat_columns, load_combined
from scripts.sweep_catalog import COMBINED, SWEEPS

RUNS, SUMMARY, TABLES, EXPLORER = Path("results/raw/runs.csv"), Path("results/summary/summary.csv"), Path("results/tables"), Path("explorer")
COMBINED_RUNS, COMBINED_SUMMARY, CONFIGS = Path("results/raw/combined.csv"), Path("results/summary/combined_summary.csv"), Path("configs")


def expected_seeds() -> dict[tuple, int]:
    """(scenario, attack, attack_mode) -> number of seeds the YAML asks for."""
    out = {}
    for path in CONFIGS.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        out[(raw["scenario"], raw["attack"], raw["attack_mode"])] = len(raw["seeds"])
    return out


def check_seeds(stored: pd.DataFrame, partial: bool) -> None:
    expected = expected_seeds()
    wanted = stored.apply(lambda r: expected.get((r["scenario"], r["attack"], r["attack_mode"]), 0), axis=1)
    short = stored[stored["n"] < wanted]
    over = stored[stored["n"] > wanted]
    check(over.empty, f"no configuration has more seeds than its YAML ({len(over)} over)")
    if partial and not short.empty:
        print(f"warn  {len(short)} configurations still short of their YAML seed count (--partial)")
        return
    check(short.empty, f"every configuration has every seed its YAML lists ({len(short)} short)")


def check(ok: bool, message: str) -> None:
    print(("ok    " if ok else "FAIL  ") + message)
    if not ok:
        sys.exit(1)


def check_summary(runs_path: Path, summary_path: Path) -> pd.DataFrame:
    runs = load_latest_per_run(runs_path)
    stored = pd.read_csv(summary_path)
    stored["partition"] = stored["partition"].fillna("-")
    fresh = summarise(runs).reset_index(drop=True)
    check(len(fresh) == len(stored), f"{summary_path.name} has {len(stored)} configurations, {runs_path.name} gives {len(fresh)}")
    diff = max((fresh[c] - stored[c]).abs().max() for c in stored.columns if c.endswith(("_mean", "_std", "_min", "_max")))
    check(diff < 1e-6, f"{summary_path.name} matches {runs_path.name} (max abs difference {diff:.1e})")
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="seed shortfalls are warnings, for a deploy while runs are in flight")
    args = parser.parse_args()

    runs = load_latest_per_run(RUNS)
    stored = check_summary(RUNS, SUMMARY)
    combined = None
    if COMBINED_RUNS.exists():
        combined = check_summary(COMBINED_RUNS, COMBINED_SUMMARY)
        blocks = {(b["scenario"], "combined", b["id"].split("_", 1)[1]) for b in COMBINED}
        present = set(map(tuple, combined[["scenario", "attack", "attack_mode"]].drop_duplicates().values))
        check(present <= blocks, f"every combined block in combined_summary.csv is catalogued ({len(present)})")
        check_seeds(combined, args.partial)
    check_seeds(stored, args.partial)

    catalogued = {(s["scenario"], s["attack"], s["mode"]) for s in SWEEPS}
    present = set(map(tuple, stored[["scenario", "attack", "attack_mode"]].drop_duplicates().values))
    check(present == catalogued, f"sweep catalogue covers every sweep in summary.csv ({len(present)})")

    tables = build_all(stored, combined)
    for t in tables:
        on_disk = pd.read_csv(TABLES / t["file"], dtype=str, keep_default_na=False)
        expected = pd.DataFrame([[c["text"] for c in row] for row in t["rows"]], columns=flat_columns(t))
        if not on_disk.equals(expected):
            check(False, f"{t['file']} is stale, run scripts/build_tables.py")
    check(True, f"{len(tables)} table CSVs match summary.csv")

    payload = json.loads((EXPLORER / "data.js").read_text().removeprefix("window.HQ_DATA = ").rstrip().rstrip(";"))
    check(len(payload["runs"]) == len(runs) and len(payload["summary"]) == len(stored), "explorer/data.js has every run and configuration")
    check(len(payload["combined"]) == (0 if combined is None else len(combined)), "explorer/data.js has every combined configuration")
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
