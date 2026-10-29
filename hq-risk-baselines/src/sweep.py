"""Expands one YAML sweep file into the list of RunConfigs it describes. Adding an intensity is a YAML edit.

A single-attack sweep lists `intensities`. A combined sweep (attack: combined) lists `factors`, each with its own
`levels`; every combination of levels is one run and the `intensity` column repeats the first factor's level.
"""

import itertools
from pathlib import Path

import yaml

from src.run_config import RunConfig


class SweepConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open() as handle:
            self.raw = yaml.safe_load(handle)
        self.output = Path(self.raw.get("output", "results/raw/runs.csv"))

    @property
    def combined(self) -> bool:
        return self.raw["attack"] == RunConfig.COMBINED

    def runs(self) -> list[RunConfig]:
        scenario = self.raw["scenario"]
        if scenario not in (RunConfig.S1, RunConfig.S2, RunConfig.S3):
            raise ValueError(f"unknown scenario {scenario!r} in {self.path}")
        partitions = self.raw.get("partitions", [None]) if scenario == RunConfig.S2 else [None]
        return [
            RunConfig(
                scenario=scenario,
                dataset=dataset,
                model=self.raw["model"],
                attack=self.raw["attack"],
                attack_mode=self.raw["attack_mode"],
                intensity=intensity,
                seed=int(seed),
                partition=partition,
                factors=factors,
                **self._scenario_fields(scenario),
            )
            for dataset, partition, seed, (intensity, factors) in itertools.product(
                self.raw["datasets"], partitions, self.raw["seeds"], self._points()
            )
        ]

    def _points(self) -> list[tuple[float, tuple | None]]:
        if not self.combined:
            return [(float(v), None) for v in self.raw["intensities"]]
        factors = self.raw["factors"]
        points = []
        for levels in itertools.product(*[f["levels"] for f in factors]):
            combo = tuple((f["attack"], f["mode"], float(level)) for f, level in zip(factors, levels))
            points.append((float(levels[0]), combo))
        return points

    def _scenario_fields(self, scenario: str) -> dict:
        if scenario == RunConfig.S2:
            federated = self.raw["federated"]
            return dict(
                n_clients=int(federated["n_clients"]),
                rounds=int(federated["rounds"]),
                local_epochs=int(federated["local_epochs"]),
                learning_rate=float(federated["learning_rate"]),
                batch_size=int(federated["batch_size"]),
                dirichlet_alpha=float(federated["dirichlet_alpha"]),
            )
        if scenario == RunConfig.S3:
            quantum = self.raw["quantum"]
            return dict(
                learning_rate=float(quantum["learning_rate"]),
                batch_size=int(quantum["batch_size"]),
                n_qubits=int(quantum["n_qubits"]),
                shots=int(quantum["shots"]),
                epochs=int(quantum["epochs"]),
            )
        return {}
