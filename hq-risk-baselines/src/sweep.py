"""Expands one YAML sweep file into the list of RunConfigs it describes. Adding an intensity is a YAML edit."""

import itertools
from pathlib import Path

import yaml

from src.experiment import RunConfig


class SweepConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open() as handle:
            self.raw = yaml.safe_load(handle)
        self.output = Path(self.raw.get("output", "results/raw/runs.csv"))

    def runs(self) -> list[RunConfig]:
        scenario = self.raw["scenario"]
        if scenario == RunConfig.S1:
            return self._classical_runs()
        if scenario == RunConfig.S2:
            return self._federated_runs()
        if scenario == RunConfig.S3:
            return self._quantum_runs()
        raise ValueError(f"unknown scenario {scenario!r} in {self.path}")

    def _classical_runs(self) -> list[RunConfig]:
        return [
            RunConfig(
                scenario=RunConfig.S1,
                dataset=dataset,
                model=self.raw["model"],
                attack=self.raw["attack"],
                attack_mode=self.raw["attack_mode"],
                intensity=float(intensity),
                seed=int(seed),
            )
            for dataset, intensity, seed in itertools.product(self.raw["datasets"], self.raw["intensities"], self.raw["seeds"])
        ]

    def _federated_runs(self) -> list[RunConfig]:
        federated = self.raw["federated"]
        return [
            RunConfig(
                scenario=RunConfig.S2,
                dataset=dataset,
                model=self.raw["model"],
                attack=self.raw["attack"],
                attack_mode=self.raw["attack_mode"],
                intensity=float(intensity),
                seed=int(seed),
                n_clients=int(federated["n_clients"]),
                partition=partition,
                rounds=int(federated["rounds"]),
                local_epochs=int(federated["local_epochs"]),
                learning_rate=float(federated["learning_rate"]),
                batch_size=int(federated["batch_size"]),
                dirichlet_alpha=float(federated["dirichlet_alpha"]),
            )
            for dataset, partition, intensity, seed in itertools.product(
                self.raw["datasets"], self.raw["partitions"], self.raw["intensities"], self.raw["seeds"]
            )
        ]

    def _quantum_runs(self) -> list[RunConfig]:
        quantum = self.raw["quantum"]
        return [
            RunConfig(
                scenario=RunConfig.S3,
                dataset=dataset,
                model=self.raw["model"],
                attack=self.raw["attack"],
                attack_mode=self.raw["attack_mode"],
                intensity=float(intensity),
                seed=int(seed),
                learning_rate=float(quantum["learning_rate"]),
                batch_size=int(quantum["batch_size"]),
                n_qubits=int(quantum["n_qubits"]),
                shots=int(quantum["shots"]),
                epochs=int(quantum["epochs"]),
            )
            for dataset, seed, intensity in itertools.product(self.raw["datasets"], self.raw["seeds"], self.raw["intensities"])
        ]
