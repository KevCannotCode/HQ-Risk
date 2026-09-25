"""Paper-ready tables from summary.csv: one reference-system table per dataset, then one table per sweep.
Writes results/tables/table_NN_<id>.csv and tables.md (Cureus style: caption below, abbreviations under it).
The explorer embeds the same table objects, so the site and the paper show identical numbers."""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_reference_table import SYSTEMS, sweep_rows
from scripts.sweep_catalog import DATASETS, METRIC_LABELS, PARTITIONS, SWEEPS

SEEDS_NOTE = "Values are mean ± SD over 5 seeds."
ABBREVIATIONS = {
    "SD": "standard deviation", "pp": "percentage points", "IID": "independent and identically distributed",
    "FGSM": "fast gradient sign method", "TSV": "transpiler supply-chain vector", "ARF": "adversarial-AI factor",
    "QTF": "quantum-threat factor", "FL": "federated learning", "ML": "machine learning", "QML": "quantum machine learning",
    "RX": "X-rotation gate", "RZ": "Z-rotation gate", "SWAP": "qubit swap gate", "FedAvg": "federated averaging",
}
REFERENCE_SUCCESS = {
    None: "—", "undefined (D10)": "undefined", "source rows → target class": "source rows → target",
    "correct → wrong": "correct → wrong", "triggered rows → target class": "triggered rows → target",
    "fidelity of stolen copy": "fidelity", "membership accuracy (50 = guess)": "membership accuracy",
    "cosine to class mean (×100)": "cosine × 100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--output", default="results/tables")
    return parser.parse_args()


def value(row: pd.Series, metric: str) -> str:
    mean, sd = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        return "—"
    scale = 1 if METRIC_LABELS[metric][1] == "pp" else 100
    return f"{mean * scale:.1f} ± {sd * scale:.1f}"


def intensity(v: float, x_unit: str) -> str:
    if x_unit == "fraction":
        return f"{v * 100:g}%"
    return f"{v:g}"


def key(row: pd.Series) -> dict:
    return {k: (float(row[k]) if k == "intensity" else row[k]) for k in ["scenario", "dataset", "attack", "attack_mode", "partition", "intensity"]}


def cell(text: str, row: pd.Series | None = None) -> dict:
    return {"text": text, "key": key(row)} if row is not None else {"text": text}


def footnote(*texts: str) -> str:
    used = [a for a in ABBREVIATIONS if any(re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", t) for t in texts)]
    return "; ".join(f"{a}, {ABBREVIATIONS[a]}" for a in used)


def metric_header(metric: str) -> str:
    label, unit = METRIC_LABELS[metric]
    return f"{label} ({unit})"


def sweep_table(summary: pd.DataFrame, sweep: dict, number: int) -> dict:
    rows = summary[(summary["scenario"] == sweep["scenario"]) & (summary["attack"] == sweep["attack"]) & (summary["attack_mode"] == sweep["mode"])]
    partitions = ["iid", "non_iid"] if sweep["scenario"] == "s2" else ["-"]
    datasets = [d for d in DATASETS if d in set(rows["dataset"])]
    metrics = sweep["metrics"]

    lead = (["Partition"] if len(partitions) > 1 else []) + [sweep["x"]]
    groups = [{"text": "", "span": len(lead)}] + [{"text": DATASETS[d], "span": len(metrics)} for d in datasets]
    columns = lead + [metric_header(m) for _ in datasets for m in metrics]

    body = []
    for part in partitions:
        for x in sorted(rows["intensity"].unique()):
            line = ([cell(PARTITIONS[part])] if len(partitions) > 1 else []) + [cell(intensity(x, sweep["x_unit"]))]
            for d in datasets:
                match = rows[(rows["dataset"] == d) & (rows["partition"] == part) & (rows["intensity"] == x)]
                for m in metrics:
                    line.append(cell(value(match.iloc[0], m), match.iloc[0]) if len(match) else cell("not run"))
            body.append(line)

    measures = []
    if "attacked_accuracy" in metrics:
        measures.append(f"Accuracy: {sweep.get('accuracy', 'test accuracy under attack')}.")
    if "attack_success_rate" in metrics:
        measures.append(f"Success: {sweep['success']}.")
    if "accuracy_drop_pp" in metrics:
        measures.append("Drop: clean minus attacked accuracy.")
    note = " ".join([sweep["note"], *measures, SEEDS_NOTE])
    caption = f"{sweep['title']}: {' and '.join(METRIC_LABELS[m][0].lower() for m in metrics)} by {sweep['x'].lower()}"
    return {"number": number, "id": sweep["id"], "sweep": sweep["id"], "caption": caption, "note": note,
            "abbreviations": footnote(caption, note, *columns), "groups": groups, "columns": columns, "rows": body}


def reference_table(summary: pd.DataFrame, dataset: str, number: int, partition: str = "iid") -> dict:
    columns = ["System", "Attack strength", "Clean accuracy (%)", "Accuracy (%)", "Success (%)", "Success measure", "Drop (pp)"]
    body = []
    for label, scenario, attack, mode, success in SYSTEMS:
        sweep = next(s for s in SWEEPS if (s["scenario"], s["attack"], s["mode"]) == (scenario, attack, mode))
        try:
            rows = sweep_rows(summary, dataset, partition, scenario, attack, mode)
        except LookupError:
            body.append([cell(label)] + [cell("not run")] * (len(columns) - 1))
            continue
        clean = rows.iloc[0]
        point = clean if success is None else rows.iloc[-1]
        strength = "none" if success is None else f"{sweep['x']} {intensity(point['intensity'], sweep['x_unit'])}"
        body.append([
            cell(label), cell(strength), cell(value(clean, "clean_accuracy"), clean), cell(value(point, "attacked_accuracy"), point),
            cell("—" if success is None else value(point, "attack_success_rate"), point), cell(REFERENCE_SUCCESS[success]),
            cell(value(point, "accuracy_drop_pp"), point),
        ])
    caption = f"Reference systems on {DATASETS[dataset]}: every attack at the strongest point of its sweep"
    note = (f"FL rows use the {PARTITIONS[partition]} partition; FL clean accuracy is FedAvg with no malicious clients. "
            "QML rows run QuantumNet on Qiskit Aer, 256 shots; their drop is against clean QML. "
            f"Stolen rows: accuracy is the stolen copy's. Membership and inversion leave the model unchanged. {SEEDS_NOTE}")
    labels = " ".join(s[0] for s in SYSTEMS)
    return {"number": number, "id": f"reference_{dataset}", "sweep": None, "caption": caption, "note": note,
            "abbreviations": footnote(caption, note, labels), "groups": None, "columns": columns, "rows": body}


def build_all(summary: pd.DataFrame) -> list[dict]:
    summary = summary.copy()
    summary["partition"] = summary["partition"].fillna("-")
    tables = [reference_table(summary, d, i + 1) for i, d in enumerate(DATASETS)]
    tables += [sweep_table(summary, s, len(tables) + i + 1) for i, s in enumerate(SWEEPS)]
    for t in tables:
        t["file"] = f"table_{t['number']:02d}_{t['id']}.csv"
    return tables


def flat_columns(table: dict) -> list[str]:
    if not table["groups"]:
        return table["columns"]
    owners = [g["text"] for g in table["groups"] for _ in range(g["span"])]
    return [f"{o}: {c}" if o else c for o, c in zip(owners, table["columns"])]


def write_csv(table: dict, folder: Path) -> None:
    with open(folder / table["file"], "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(flat_columns(table))
        writer.writerows([c["text"] for c in row] for row in table["rows"])


def markdown(tables: list[dict]) -> str:
    out = ["# HQ-Risk results tables", "", "Generated by `scripts/build_tables.py` from `results/summary/summary.csv`.", ""]
    for t in tables:
        cols = flat_columns(t)
        out += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(c["text"] for c in row) + " |" for row in t["rows"]]
        out += ["", f"**TABLE {t['number']}:** {t['caption']}", "", f"*{t['note']}*", ""]
        if t["abbreviations"]:
            out += [f"*{t['abbreviations']}*", ""]
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    tables = build_all(pd.read_csv(args.summary))
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("table_*.csv"):
        old.unlink()
    for t in tables:
        write_csv(t, folder)
    (folder / "tables.md").write_text(markdown(tables))
    print(f"{len(tables)} tables -> {folder}")


if __name__ == "__main__":
    main()
