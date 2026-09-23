"""Runs one configuration end to end and returns one schema row.

Every row carries its own clean baseline: the clean model is always trained on the same split,
so clean and attacked differ only by the attack (paired comparison).
"""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from src.attacks.backend_noise import BackendNoiseAttack
from src.attacks.byzantine import ByzantineBehaviour
from src.attacks.circuit_tamper import CircuitTamperAttack
from src.attacks.fgsm import FgsmAttack
from src.attacks.label_flip import LabelFlipAttack
from src.attacks.shot_bias import ShotBiasAttack
from src.datasets import DatasetLoader, DatasetSplit
from src.federated.client import FederatedClient
from src.federated.partition import DataPartitioner
from src.federated.server import FederatedServer
from src.metrics import MetricsCalculator
from src.models import ModelFactory
from src.quantum.circuit import QuantumNet
from src.quantum.encoder import PcaAngleEncoder
from src.quantum.executor import ShotExecutor
from src.quantum.trainer import QuantumTrainer


@dataclass(frozen=True)
class RunConfig:
    scenario: str
    dataset: str
    model: str
    attack: str
    attack_mode: str
    intensity: float
    seed: int
    n_clients: int | None = None
    partition: str | None = None
    rounds: int | None = None
    local_epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    dirichlet_alpha: float | None = None
    n_qubits: int | None = None
    shots: int | None = None
    epochs: int | None = None

    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    POISONING = "poisoning"
    EVASION = "evasion"
    BYZANTINE = "byzantine"
    CIRCUIT_TAMPER = "circuit_tamper"
    SHOT_BIAS = "shot_bias"
    NOISE = "noise"
    # Added for S3; left out of the hash when unset so every S1/S2 run_id already in runs.csv is unchanged.
    _S3_FIELDS = ("n_qubits", "shots", "epochs")

    def config_hash(self) -> str:
        fields = asdict(self)
        fields.pop("seed")
        return self._digest(fields)

    def run_id(self) -> str:
        return self._digest(asdict(self))

    def _digest(self, fields: dict) -> str:
        fields = {key: value for key, value in fields.items() if not (key in self._S3_FIELDS and value is None)}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]


class ExperimentRunner:
    def __init__(self):
        self.loader = DatasetLoader()
        self.factory = ModelFactory()
        self.metrics = MetricsCalculator()
        self._quantum_models = {}

    def run(self, config: RunConfig) -> dict:
        started = time.perf_counter()
        data = self.loader.load(config.dataset, config.seed)
        test_fingerprint = self._fingerprint(data.x_test, data.y_test)

        if config.scenario == RunConfig.S1:
            clean_pred, attacked_pred, attack_success_rate = self._run_classical(config, data)
        elif config.scenario == RunConfig.S2:
            clean_pred, attacked_pred, attack_success_rate = self._run_federated(config, data)
        elif config.scenario == RunConfig.S3:
            clean_pred, attacked_pred, attack_success_rate = self._run_quantum(config, data)
        else:
            raise ValueError(f"unknown scenario {config.scenario!r}")

        assert self._fingerprint(data.x_test, data.y_test) == test_fingerprint, "test set was modified during the run"

        clean = self.metrics.evaluate(data.y_test, clean_pred, data.positive_label)
        attacked = self.metrics.evaluate(data.y_test, attacked_pred, data.positive_label)
        return self._row(config, clean.accuracy, attacked, attack_success_rate, time.perf_counter() - started)

    def _run_classical(self, config: RunConfig, data: DatasetSplit):
        clean_model = self.factory.create(config.model, config.seed).fit(data.x_train, data.y_train)
        clean_pred = clean_model.predict(data.x_test)
        if config.intensity == 0:
            return clean_pred, clean_pred, None

        if config.attack == RunConfig.POISONING:
            poisoned_labels = self._poison(config, data)
            attacked_model = self.factory.create(config.model, config.seed).fit(data.x_train, poisoned_labels)
            return clean_pred, attacked_model.predict(data.x_test), None

        if config.attack == RunConfig.EVASION:
            adversarial = FgsmAttack().perturb(clean_model, data.x_test, data.y_test, config.intensity)
            attacked_pred = clean_model.predict(adversarial)
            return clean_pred, attacked_pred, self.metrics.attack_success_rate(data.y_test, clean_pred, attacked_pred)

        raise ValueError(f"unknown S1 attack {config.attack!r}")

    def _poison(self, config: RunConfig, data: DatasetSplit) -> np.ndarray:
        if config.attack_mode == LabelFlipAttack.TARGETED:
            if data.positive_label is None:
                raise ValueError(f"targeted poisoning needs a binary dataset, {data.name} is multiclass")
            attack = LabelFlipAttack(LabelFlipAttack.TARGETED, source_label=data.positive_label, target_label=1 - data.positive_label)
        else:
            attack = LabelFlipAttack(config.attack_mode)
        return attack.apply(data.y_train, config.intensity, data.n_classes, np.random.default_rng([config.seed, 0]))

    def _run_federated(self, config: RunConfig, data: DatasetSplit):
        if config.attack != RunConfig.BYZANTINE:
            raise ValueError(f"unknown S2 attack {config.attack!r}")
        partitioner = DataPartitioner(config.partition, config.dirichlet_alpha)
        shards = partitioner.split(data.y_train, config.n_clients, data.n_classes, np.random.default_rng([config.seed, 0]))

        clean_pred = self._train_federated(config, data, shards, n_malicious=0).predict(data.x_test)
        n_malicious = int(round(config.intensity * config.n_clients))
        if n_malicious == 0:
            return clean_pred, clean_pred, None
        attacked_pred = self._train_federated(config, data, shards, n_malicious).predict(data.x_test)
        return clean_pred, attacked_pred, None

    def _train_federated(self, config: RunConfig, data: DatasetSplit, shards: list[np.ndarray], n_malicious: int):
        clients = [
            FederatedClient(
                client_id=client_id,
                x=data.x_train[shard],
                y=data.y_train[shard],
                n_classes=data.n_classes,
                behaviour=ByzantineBehaviour(config.attack_mode if client_id < n_malicious else ByzantineBehaviour.HONEST),
                rng=np.random.default_rng([config.seed, 1 + client_id]),
            )
            for client_id, shard in enumerate(shards)
        ]
        model = self.factory.create_federated(config.model, data.x_train.shape[1], data.n_classes)
        return FederatedServer(model, clients).train(config.rounds, config.local_epochs, config.learning_rate, config.batch_size)

    def _run_quantum(self, config: RunConfig, data: DatasetSplit):
        net, x_test = self._trained_quantum_model(config, data)
        clean_executor = ShotExecutor(seed=config.seed)
        clean_pred = clean_executor.predict(net, clean_executor.counts(net, x_test, config.shots))
        if config.intensity == 0:
            return clean_pred, clean_pred, None

        attacked_counts = self._attacked_counts(config, data, net, x_test)
        attacked_pred = clean_executor.predict(net, attacked_counts)
        return clean_pred, attacked_pred, self.metrics.attack_success_rate(data.y_test, clean_pred, attacked_pred)

    def _trained_quantum_model(self, config: RunConfig, data: DatasetSplit):
        # Every S3 attack acts at inference, so one clean model per (dataset, seed, training knobs) serves the whole sweep.
        key = (config.dataset, config.seed, config.model, config.n_qubits, config.epochs, config.learning_rate, config.batch_size)
        if key not in self._quantum_models:
            if config.model != "quantum_net":
                raise ValueError(f"S3 supports only model 'quantum_net', got {config.model!r}")
            encoder = PcaAngleEncoder(config.n_qubits, config.seed).fit(data.x_train)
            net = QuantumNet(config.n_qubits, data.n_classes, config.seed)
            trainer = QuantumTrainer(config.epochs, config.learning_rate, config.batch_size, config.seed)
            trainer.fit(net, encoder.transform(data.x_train), data.y_train)
            self._quantum_models[key] = (net, encoder.transform(data.x_test))
        return self._quantum_models[key]

    def _attacked_counts(self, config: RunConfig, data: DatasetSplit, net: QuantumNet, x_test: np.ndarray):
        if config.attack == RunConfig.CIRCUIT_TAMPER:
            tampered = CircuitTamperAttack().apply(net.circuit, int(config.intensity), np.random.default_rng([config.seed, 0]))
            return ShotExecutor(seed=config.seed).counts(net, x_test, config.shots, circuit=tampered)

        if config.attack == RunConfig.SHOT_BIAS:
            attack = ShotBiasAttack()
            forged = attack.forged_shots(config.shots, config.intensity)
            honest = ShotExecutor(seed=config.seed).counts(net, x_test, config.shots - forged)
            target_label = 1 - data.positive_label if data.positive_label is not None else 0
            return attack.falsify(honest, attack.target_bitstring(net, target_label), forged)

        if config.attack == RunConfig.NOISE:
            noisy = ShotExecutor(seed=config.seed, noise_model=BackendNoiseAttack().noise_model(config.intensity))
            return noisy.counts(net, x_test, config.shots)

        raise ValueError(f"unknown S3 attack {config.attack!r}")

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
