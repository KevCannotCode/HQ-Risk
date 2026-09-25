"""S3: the project lead's QuantumNet, trained with Adam, evaluated on Aer. Execution attacks (tamper, shots, noise,
transpiler) act at inference on the clean model; backdoor retrains; query attacks treat it as a black box.
"""

import numpy as np

from src.attacks.backdoor import BackdoorAttack
from src.attacks.backend_noise import BackendNoiseAttack
from src.attacks.circuit_tamper import CircuitTamperAttack
from src.attacks.shot_bias import ShotBiasAttack
from src.attacks.transpiler import TranspilerAttack
from src.datasets import DatasetSplit
from src.metrics import MetricsCalculator
from src.quantum.circuit import QuantumNet
from src.quantum.classifier import QuantumClassifier
from src.quantum.encoder import PcaAngleEncoder
from src.quantum.executor import ShotExecutor
from src.quantum.trainer import QuantumTrainer
from src.run_config import RunConfig
from src.scenarios.outcome import Outcome
from src.scenarios.query_attacks import QueryAttacks


class QuantumScenario:
    MODEL = "quantum_net"

    def __init__(self):
        self.metrics = MetricsCalculator()
        self._clean_models = {}

    def run(self, config: RunConfig, data: DatasetSplit) -> Outcome:
        if config.model != self.MODEL:
            raise ValueError(f"S3 supports only model {self.MODEL!r}, got {config.model!r}")
        net, encoder = self._clean_model(config, data)
        x_test = encoder.transform(data.x_test)
        clean_executor = ShotExecutor(seed=config.seed)
        clean_pred = clean_executor.predict(net, clean_executor.counts(net, x_test, config.shots))

        if config.attack in RunConfig.QUERY_ATTACKS:
            return QueryAttacks().run(config, data, QuantumClassifier(net, encoder, config.shots, config.seed), clean_pred)
        if config.attack == RunConfig.BACKDOOR:
            return self._backdoor(config, data, clean_pred)
        if config.intensity == 0:
            return Outcome(clean_pred, clean_pred)

        attacked_pred = clean_executor.predict(net, self._attacked_counts(config, data, net, x_test))
        return Outcome(clean_pred, attacked_pred, self.metrics.attack_success_rate(data.y_test, clean_pred, attacked_pred))

    def _clean_model(self, config: RunConfig, data: DatasetSplit) -> tuple[QuantumNet, PcaAngleEncoder]:
        # Only the backdoor retrains, so one clean model per (dataset, seed, training knobs) serves the whole sweep.
        key = (config.dataset, config.seed, config.n_qubits, config.epochs, config.learning_rate, config.batch_size)
        if key not in self._clean_models:
            self._clean_models[key] = self._train(config, data.x_train, data.y_train, data.n_classes)
        return self._clean_models[key]

    def _train(self, config: RunConfig, x_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> tuple[QuantumNet, PcaAngleEncoder]:
        encoder = PcaAngleEncoder(config.n_qubits, config.seed).fit(x_train)
        net = QuantumNet(config.n_qubits, n_classes, config.seed)
        QuantumTrainer(config.epochs, config.learning_rate, config.batch_size, config.seed).fit(net, encoder.transform(x_train), y_train)
        return net, encoder

    def _backdoor(self, config: RunConfig, data: DatasetSplit, clean_pred: np.ndarray) -> Outcome:
        backdoor = BackdoorAttack(data.target_label)
        if config.intensity > 0:
            x_poisoned, y_poisoned = backdoor.poison(data.x_train, data.y_train, config.intensity, np.random.default_rng([config.seed, 0]))
            net, encoder = self._train(config, x_poisoned, y_poisoned, data.n_classes)
        else:
            net, encoder = self._clean_model(config, data)
        model = QuantumClassifier(net, encoder, config.shots, config.seed)
        triggered_pred = model.predict(backdoor.stamp(data.x_test))
        return Outcome(clean_pred, model.predict(data.x_test), self.metrics.targeted_success_rate(data.y_test, triggered_pred, data.target_label))

    def _attacked_counts(self, config: RunConfig, data: DatasetSplit, net: QuantumNet, x_test: np.ndarray):
        rng = np.random.default_rng([config.seed, 0])
        if config.attack == RunConfig.CIRCUIT_TAMPER:
            tampered = CircuitTamperAttack().apply(net.circuit, int(config.intensity), rng)
            return ShotExecutor(seed=config.seed).counts(net, x_test, config.shots, circuit=tampered)

        if config.attack == RunConfig.TRANSPILER:
            compiled = TranspilerAttack().compile(net.circuit, config.attack_mode, config.intensity, rng)
            return ShotExecutor(seed=config.seed).counts(net, x_test, config.shots, circuit=compiled)

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
