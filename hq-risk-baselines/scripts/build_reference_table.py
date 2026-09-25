"""Fill the project lead's reference-system table with real numbers from summary.csv, every attack at the strongest
point of its sweep. HQ-Risk column stays empty -- it is his.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (label, scenario, attack, attack_mode, what "attack success" means for this row)
SYSTEMS = [
    ("Clean ML", "s1", "poisoning", "symmetric", None),
    ("Poisoned ML", "s1", "poisoning", "symmetric", "undefined (D10)"),
    ("Targeted-poisoned ML", "s1", "poisoning", "targeted", "source rows → target class"),
    ("Evaded ML", "s1", "evasion", "fgsm", "correct → wrong"),
    ("Backdoored ML (ARF)", "s1", "backdoor", "trigger", "triggered rows → target class"),
    ("Stolen ML (ARF)", "s1", "extraction", "label_only", "fidelity of stolen copy"),
    ("Membership-inferred ML (ARF)", "s1", "membership", "confidence_threshold", "membership accuracy (50 = guess)"),
    ("Inverted ML (ARF)", "s1", "inversion", "zeroth_order", "cosine to class mean (×100)"),
    ("Clean FL", "s2", "byzantine", "label_flip", None),
    ("Malicious FL, label flip", "s2", "byzantine", "label_flip", "undefined (D10)"),
    ("Malicious FL, sign flip", "s2", "byzantine", "sign_flip", "undefined (D10)"),
    ("Malicious FL, targeted flip", "s2", "byzantine", "targeted_flip", "source rows → target class"),
    ("Backdoored FL (ARF)", "s2", "byzantine", "backdoor", "triggered rows → target class"),
    ("Clean QML", "s3", "circuit_tamper", "rx_half_pi", None),
    ("Tampered QML", "s3", "circuit_tamper", "rx_half_pi", "correct → wrong"),
    ("Shot-biased QML", "s3", "shot_bias", "forged_bitstring", "correct → wrong"),
    ("Noisy QML", "s3", "noise", "depolarizing_readout", "correct → wrong"),
    ("Transpiler drift QML (QTF · TSV)", "s3", "transpiler", "angle_drift", "correct → wrong"),
    ("Transpiler swap QML (QTF · TSV)", "s3", "transpiler", "qubit_swap", "correct → wrong"),
    ("Backdoored QML (ARF)", "s3", "backdoor", "trigger", "triggered rows → target class"),
    ("Stolen QML (ARF)", "s3", "extraction", "label_only", "fidelity of stolen copy"),
    ("Membership-inferred QML (ARF)", "s3", "membership", "confidence_threshold", "membership accuracy (50 = guess)"),
    ("Inverted QML (ARF)", "s3", "inversion", "zeroth_order", "cosine to class mean (×100)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--output", default="results/summary/reference_table.md")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--partition", default="iid")
    return parser.parse_args()


def sweep_rows(summary: pd.DataFrame, dataset: str, partition: str, scenario: str, attack: str, attack_mode: str) -> pd.DataFrame:
    rows = summary[(summary["scenario"] == scenario) & (summary["dataset"] == dataset) & (summary["attack"] == attack) & (summary["attack_mode"] == attack_mode)]
    if scenario == "s2":
        rows = rows[rows["partition"] == partition]
    if rows.empty:
        raise LookupError(f"no summary rows for {scenario}/{attack}/{attack_mode} on {dataset}")
    return rows.sort_values("intensity")


def percent(row: pd.Series, metric: str, scale: float) -> str:
    mean, sd = row[f"{metric}_mean"], row[f"{metric}_std"]
    if pd.isna(mean):
        return "—"
    return f"{mean * scale:.1f} ± {sd * scale:.1f}"


def line(label: str, clean: pd.Series, point: pd.Series, success: str | None) -> str:
    intensity = "none" if success is None else f"{point['intensity']:g}"
    value = percent(point, "attack_success_rate", 100)
    attack_success = "—" if success is None else (f"— {success}" if value == "—" else f"{value} ({success})")
    return (
        f"| {label} | {intensity} | {percent(clean, 'clean_accuracy', 100)} | {percent(point, 'attacked_accuracy', 100)} "
        f"| {attack_success} | {percent(point, 'accuracy_drop_pp', 1)} | |"
    )


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary["partition"] = summary["partition"].fillna("-")

    lines = [
        f"# Reference systems — real numbers ({args.dataset}, 5 seeds, mean ± sd)",
        "",
        f"Every attacked system is at the strongest point of its sweep. Federated rows use the {args.partition} partition. "
        "QML rows are evaluated on Qiskit Aer, 256 shots. ARF = adversarial-AI factor, QTF · TSV = quantum-threat factor, transpiler / supply chain.",
        "",
        "| System | Intensity | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, scenario, attack, attack_mode, success in SYSTEMS:
        rows = sweep_rows(summary, args.dataset, args.partition, scenario, attack, attack_mode)
        clean = rows.iloc[0]
        lines.append(line(label, clean, clean if success is None else rows.iloc[-1], success))
    lines += [
        "",
        "Clean QML is the project lead's QuantumNet circuit trained with Adam; QML drops are against it, not against logistic regression. "
        "FL clean accuracy is FedAvg with 0 malicious clients, not the centralised model. "
        "Stolen rows: attacked accuracy is the stolen copy's accuracy. Membership and inversion rows leave the model untouched (drop 0); "
        "their attack success is the attack's own score (METHODS D47, D48).",
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
