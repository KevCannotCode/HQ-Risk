"""FedAvg server. Weighted mean of client parameters, no defence, every client every round."""

import numpy as np

from src.federated.client import FederatedClient
from src.models import SoftmaxRegression


class FederatedServer:
    def __init__(self, model: SoftmaxRegression, clients: list[FederatedClient]):
        self.model = model
        self.clients = clients

    def train(self, rounds: int, local_epochs: int, learning_rate: float, batch_size: int) -> SoftmaxRegression:
        for _ in range(rounds):
            global_parameters = self.model.get_parameters()
            updates = [
                (client.train(global_parameters, local_epochs, learning_rate, batch_size), client.n_samples)
                for client in self.clients
            ]
            self.model.set_parameters(self._federated_average(updates))
        return self.model

    def _federated_average(self, updates: list[tuple[tuple[np.ndarray, np.ndarray], int]]):
        total = sum(n for _, n in updates)
        weights = sum(parameters[0] * n for parameters, n in updates) / total
        bias = sum(parameters[1] * n for parameters, n in updates) / total
        return weights, bias
