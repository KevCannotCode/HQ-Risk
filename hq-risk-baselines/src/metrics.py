"""Metric set for one run. Accuracy alone hides a targeted attack, so it never travels alone."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class RunMetrics:
    accuracy: float
    balanced_accuracy: float
    positive_recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)


class MetricsCalculator:
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, positive_label: int | None) -> RunMetrics:
        if positive_label is None:
            positive_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        else:
            positive_recall = recall_score(y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0)
            f1 = f1_score(y_true, y_pred, pos_label=positive_label, average="binary", zero_division=0)

        return RunMetrics(
            accuracy=float(np.mean(y_true == y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            positive_recall=float(positive_recall),
            f1=float(f1),
        )

    def accuracy_drop_pp(self, clean_accuracy: float, attacked_accuracy: float) -> float:
        return float((clean_accuracy - attacked_accuracy) * 100.0)

    def attack_success_rate(
        self, y_true: np.ndarray, y_pred_clean: np.ndarray, y_pred_attacked: np.ndarray
    ) -> float | None:
        was_correct = y_pred_clean == y_true
        if not was_correct.any():
            return None
        now_wrong = y_pred_attacked[was_correct] != y_true[was_correct]
        return float(np.mean(now_wrong))

    def targeted_success_rate(
        self, y_true: np.ndarray, y_pred: np.ndarray, target_label: int, source_label: int | None = None
    ) -> float | None:
        """Share of source-class rows (every non-target row when no source) predicted as the target class."""
        rows = y_true == source_label if source_label is not None else y_true != target_label
        if not rows.any():
            return None
        return float(np.mean(y_pred[rows] == target_label))
