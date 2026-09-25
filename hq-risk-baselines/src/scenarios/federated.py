"""S2: FedAvg over simulated clients; clients 0..k-1 are malicious (D23)."""

import numpy as np

from src.attacks.backdoor import BackdoorAttack
from src.attacks.byzantine import ByzantineBehaviour
from src.datasets import DatasetSplit
from src.federated.client import FederatedClient
from src.federated.partition import DataPartitioner
from src.federated.server import FederatedServer
from src.metrics import MetricsCalculator
from src.models import ModelFactory
from src.run_config import RunConfig
from src.scenarios.outcome import Outcome


class FederatedScenario:
    def __init__(self):
        self.factory = ModelFactory()
        self.metrics = MetricsCalculator()

    def run(self, config: RunConfig, data: DatasetSplit) -> Outcome:
        if config.attack != RunConfig.BYZANTINE:
            raise ValueError(f"unknown S2 attack {config.attack!r}")
        partitioner = DataPartitioner(config.partition, config.dirichlet_alpha)
        shards = partitioner.split(data.y_train, config.n_clients, data.n_classes, np.random.default_rng([config.seed, 0]))

        clean_model = self._train(config, data, shards, n_malicious=0)
        clean_pred = clean_model.predict(data.x_test)
        n_malicious = int(round(config.intensity * config.n_clients))
        model = clean_model if n_malicious == 0 else self._train(config, data, shards, n_malicious)
        attacked_pred = model.predict(data.x_test)
        return Outcome(clean_pred, attacked_pred, self._success_rate(config, data, model, attacked_pred))

    def _success_rate(self, config: RunConfig, data: DatasetSplit, model, attacked_pred: np.ndarray) -> float | None:
        # Targeted modes report their natural rate at 0 malicious clients too (D43).
        if config.attack_mode == ByzantineBehaviour.TARGETED_FLIP:
            return self.metrics.targeted_success_rate(data.y_test, attacked_pred, data.target_label, data.source_label)
        if config.attack_mode == ByzantineBehaviour.BACKDOOR:
            triggered_pred = model.predict(BackdoorAttack(data.target_label).stamp(data.x_test))
            return self.metrics.targeted_success_rate(data.y_test, triggered_pred, data.target_label)
        return None

    def _train(self, config: RunConfig, data: DatasetSplit, shards: list[np.ndarray], n_malicious: int):
        clients = [
            FederatedClient(
                client_id=client_id,
                x=data.x_train[shard],
                y=data.y_train[shard],
                n_classes=data.n_classes,
                behaviour=ByzantineBehaviour(
                    config.attack_mode if client_id < n_malicious else ByzantineBehaviour.HONEST, data.source_label, data.target_label
                ),
                rng=np.random.default_rng([config.seed, 1 + client_id]),
            )
            for client_id, shard in enumerate(shards)
        ]
        model = self.factory.create_federated(config.model, data.x_train.shape[1], data.n_classes)
        return FederatedServer(model, clients).train(config.rounds, config.local_epochs, config.learning_rate, config.batch_size)
