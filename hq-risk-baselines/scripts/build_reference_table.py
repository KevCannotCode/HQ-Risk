"""Fill the project lead's illustrative table with real numbers from summary.csv. HQ-Risk column stays empty -- it is his."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--output", default="results/summary/reference_table.md")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--poisoning-intensity", type=float, default=0.4)
    parser.add_argument("--evasion-intensity", type=float, default=1.0)
    parser.add_argument("--federated-intensity", type=float, default=0.5)
    parser.add_argument("--partition", default="iid")
    return parser.parse_args()


def select(summary: pd.DataFrame, **filters) -> pd.Series:
    rows = summary
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    if len(rows) != 1:
        raise LookupError(f"expected exactly one summary row for {filters}, found {len(rows)}")
    return rows.iloc[0]


def percent(row: pd.Series, metric: str, scale: float) -> str:
    mean, sd = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        return "— (undefined, see D10)"
    return f"{mean * scale:.1f} ± {sd * scale:.1f}"


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)

    clean = select(summary, scenario="s1", dataset=args.dataset, attack="poisoning", attack_mode="symmetric", intensity=0.0)
    poisoned = select(summary, scenario="s1", dataset=args.dataset, attack="poisoning", attack_mode="symmetric", intensity=args.poisoning_intensity)
    evaded = select(summary, scenario="s1", dataset=args.dataset, attack="evasion", attack_mode="fgsm", intensity=args.evasion_intensity)
    malicious = select(summary, scenario="s2", dataset=args.dataset, attack="byzantine", attack_mode="label_flip", partition=args.partition, intensity=args.federated_intensity)
    clean_fl = select(summary, scenario="s2", dataset=args.dataset, attack="byzantine", attack_mode="label_flip", partition=args.partition, intensity=0.0)

    lines = [
        f"# Reference systems — real numbers ({args.dataset}, 5 seeds, mean ± sd)",
        "",
        f"Poisoned ML = symmetric label flip at {args.poisoning_intensity:.0%} of training rows. "
        f"Malicious FL = {args.federated_intensity:.0%} label-flipping clients, {args.partition} partition. "
        f"Evaded ML = FGSM ε = {args.evasion_intensity} (added: the only system with a defined attack-success rate).",
        "",
        "| System | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |",
        "|---|---|---|---|---|---|",
        f"| Clean ML | {percent(clean, 'clean_accuracy', 100)} | {percent(clean, 'attacked_accuracy', 100)} | — | {percent(clean, 'accuracy_drop_pp', 1)} | |",
        f"| Poisoned ML | {percent(poisoned, 'clean_accuracy', 100)} | {percent(poisoned, 'attacked_accuracy', 100)} | {percent(poisoned, 'attack_success_rate', 100)} | {percent(poisoned, 'accuracy_drop_pp', 1)} | |",
        f"| Evaded ML (added) | {percent(evaded, 'clean_accuracy', 100)} | {percent(evaded, 'attacked_accuracy', 100)} | {percent(evaded, 'attack_success_rate', 100)} | {percent(evaded, 'accuracy_drop_pp', 1)} | |",
        f"| Malicious FL | {percent(clean_fl, 'clean_accuracy', 100)} | {percent(malicious, 'attacked_accuracy', 100)} | {percent(malicious, 'attack_success_rate', 100)} | {percent(malicious, 'accuracy_drop_pp', 1)} | |",
        "| Tampered QML | BLOCKED | BLOCKED | BLOCKED | BLOCKED | |",
        "",
        "Tampered QML is blocked until the variational circuit is supplied. "
        "Malicious FL clean accuracy is the federated clean baseline (FedAvg, 0 malicious clients), not the centralised one. "
        "Attack success rate is undefined for poisoning and federated attacks until it is defined (decision D10).",
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
