"""Runs one configuration end to end and returns one schema row.

Every row carries its own clean baseline: the clean model is always trained on the same split,
so clean and attacked differ only by the attack (paired comparison).
"""

import hashlib
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from src.datasets import DatasetLoader
from src.metrics import MetricsCalculator
from src.run_config import RunConfig
from src.scenarios.classical import ClassicalScenario
from src.scenarios.federated import FederatedScenario
from src.scenarios.quantum import QuantumScenario

__all__ = ["ExperimentRunner", "RunConfig"]


class ExperimentRunner:
    def __init__(self):
        self.loader = DatasetLoader()
        self.metrics = MetricsCalculator()
        self.scenarios = {RunConfig.S1: ClassicalScenario(), RunConfig.S2: FederatedScenario(), RunConfig.S3: QuantumScenario()}

    def run(self, config: RunConfig) -> dict:
        started = time.perf_counter()
        if config.scenario not in self.scenarios:
            raise ValueError(f"unknown scenario {config.scenario!r}")
        data = self.loader.load(config.dataset, config.seed)
        test_fingerprint = self._fingerprint(data.x_test, data.y_test)

        outcome = self.scenarios[config.scenario].run(config, data)

        assert self._fingerprint(data.x_test, data.y_test) == test_fingerprint, "test set was modified during the run"
        clean = self.metrics.evaluate(data.y_test, outcome.clean_pred, data.positive_label)
        attacked = self.metrics.evaluate(data.y_test, outcome.attacked_pred, data.positive_label)
        return self._row(config, clean.accuracy, attacked, outcome.attack_success_rate, time.perf_counter() - started)

    def _row(self, config: RunConfig, clean_accuracy: float, attacked, attack_success_rate, seconds: float) -> dict:
        return {
            "run_id": config.run_id(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scenario": config.scenario,
            "dataset": config.dataset,
            "model": config.model,
            "attack": config.attack,
            "attack_mode": config.attack_mode,
            "intensity": config.intensity,
            "seed": config.seed,
            "n_clients": config.n_clients,
            "partition": config.partition,
            "rounds": config.rounds,
            "clean_accuracy": clean_accuracy,
            "attacked_accuracy": attacked.accuracy,
            "accuracy_drop_pp": self.metrics.accuracy_drop_pp(clean_accuracy, attacked.accuracy),
            "attack_success_rate": attack_success_rate,
            "balanced_accuracy": attacked.balanced_accuracy,
            "positive_recall": attacked.positive_recall,
            "f1": attacked.f1,
            "train_seconds": seconds,
            "git_commit": self._git_commit(),
            "config_hash": config.config_hash(),
        }

    def _fingerprint(self, x: np.ndarray, y: np.ndarray) -> str:
        return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()

    def _git_commit(self) -> str:
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "uncommitted"
