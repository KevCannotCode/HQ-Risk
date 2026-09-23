"""Accuracy vs attack intensity with sd error bars, one line per dataset, from summary.csv."""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERIES_COLOURS = {"breast_cancer": "#2a78d6", "digits": "#eb6834", "wine": "#1baf7a"}
SERIES_LABELS = {"breast_cancer": "Breast Cancer", "digits": "Digits", "wine": "Wine"}
PARTITION_STYLES = {"-": "-", "iid": "-", "non_iid": "--"}
SURFACE, INK, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

FIGURES = [
    ("s1_poisoning", "s1", "poisoning", "symmetric", "Poisoning rate (fraction of training rows)", "S1 — symmetric label-flip poisoning"),
    ("s1_poisoning_targeted", "s1", "poisoning", "targeted", "Poisoning rate (fraction of malignant training rows)", "S1 — targeted malignant→benign poisoning (Breast Cancer)"),
    ("s1_evasion", "s1", "evasion", "fgsm", "FGSM ε (standardised feature units)", "S1 — FGSM evasion on the test set"),
    ("s2_federated", "s2", "byzantine", "label_flip", "Malicious client fraction", "S2 — label-flipping clients, FedAvg (solid IID, dashed non-IID)"),
    ("s2_federated_signflip", "s2", "byzantine", "sign_flip", "Malicious client fraction", "S2 — sign-flipping clients, FedAvg (solid IID, dashed non-IID)"),
    ("s3_circuit_tamper", "s3", "circuit_tamper", "rx_half_pi", "Injected RX(π/2) gates", "S3 — circuit tampering, QuantumNet on Aer (256 shots)"),
    ("s3_shot_bias", "s3", "shot_bias", "forged_bitstring", "Fraction of shots forged", "S3 — shot manipulation, forged counts toward benign / class 0"),
    ("s3_noise", "s3", "noise", "depolarizing_readout", "Depolarizing + readout error probability p", "S3 — backend noise increase"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/summary/summary.csv")
    parser.add_argument("--output-dir", default="results/figures")
    return parser.parse_args()


def style_axes(axes: plt.Axes, x_label: str, title: str) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.grid(axis="y", color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=INK_MUTED, labelsize=10)
    axes.set_xlabel(x_label, color=INK, fontsize=11)
    axes.set_ylabel("Accuracy on clean test set (mean ± sd, 5 seeds)", color=INK, fontsize=11)
    axes.set_title(title, color=INK, fontsize=12, loc="left")
    axes.set_ylim(-0.02, 1.02)


def plot_series(axes: plt.Axes, rows: pd.DataFrame, dataset: str, partition: str) -> None:
    suffix = "" if partition == "-" else f" ({partition.replace('_', '-').upper()})"
    axes.errorbar(
        rows["intensity"],
        rows["attacked_accuracy_mean"],
        yerr=rows["attacked_accuracy_std"],
        color=SERIES_COLOURS[dataset],
        linestyle=PARTITION_STYLES[partition],
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        capsize=4,
        label=f"{SERIES_LABELS[dataset]}{suffix}",
    )


def make_figure(summary: pd.DataFrame, scenario: str, attack: str, attack_mode: str, x_label: str, title: str) -> plt.Figure | None:
    rows = summary[(summary["scenario"] == scenario) & (summary["attack"] == attack) & (summary["attack_mode"] == attack_mode)]
    if rows.empty:
        return None
    figure, axes = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style_axes(axes, x_label, title)
    for dataset in SERIES_COLOURS:
        for partition in PARTITION_STYLES:
            series = rows[(rows["dataset"] == dataset) & (rows["partition"] == partition)].sort_values("intensity")
            if not series.empty:
                plot_series(axes, series, dataset, partition)
    _, labels = axes.get_legend_handles_labels()
    if len(labels) > 1:
        axes.legend(frameon=False, fontsize=10, labelcolor=INK)
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary["partition"] = summary["partition"].fillna("-")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, scenario, attack, attack_mode, x_label, title in FIGURES:
        figure = make_figure(summary, scenario, attack, attack_mode, x_label, title)
        if figure is None:
            print(f"skip {name}: no rows in summary")
            continue
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(figure)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
