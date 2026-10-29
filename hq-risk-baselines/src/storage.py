"""Append-only run log. One row per run, header written once, never overwritten."""

import csv
import fcntl
from pathlib import Path


class ResultWriter:
    COLUMNS = [
        "run_id",
        "timestamp",
        "scenario",
        "dataset",
        "model",
        "attack",
        "attack_mode",
        "intensity",
        "seed",
        "n_clients",
        "partition",
        "rounds",
        "clean_accuracy",
        "attacked_accuracy",
        "accuracy_drop_pp",
        "attack_success_rate",
        "balanced_accuracy",
        "positive_recall",
        "f1",
        "train_seconds",
        "git_commit",
        "config_hash",
    ]
    COMBINED_COLUMNS = COLUMNS + ["factors"]

    def __init__(self, path: str | Path, columns: list[str] | None = None):
        self.path = Path(path)
        self.columns = columns or self.COLUMNS
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise ValueError(f"row has columns outside the schema: {sorted(unknown)}")

        with self.path.open("a", newline="") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if handle.tell() == 0:
                csv.writer(handle).writerow(self.columns)
            csv.writer(handle).writerow([self._render(row.get(column)) for column in self.columns])
            fcntl.flock(handle, fcntl.LOCK_UN)

    def _render(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)
