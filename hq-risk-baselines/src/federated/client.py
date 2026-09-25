"""One federated client: trains the global model on its shard for a fixed number of local epochs."""

import numpy as np

from src.attacks.byzantine import ByzantineBehaviour
from src.models import SoftmaxRegression


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        behaviour: ByzantineBehaviour,
        rng: np.random.Generator,
    ):
        self.client_id = client_id
        self.n_classes = n_classes
        self.behaviour = behaviour
        self.rng = rng
        self.x, self.y = behaviour.corrupt_data(x, y, n_classes, rng)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def train(self, global_parameters, local_epochs: int, learning_rate: float, batch_size: int):
        model = SoftmaxRegression(self.x.shape[1], self.n_classes)
        model.set_parameters(global_parameters)
        for _ in range(local_epochs):
            model.fit_epoch(self.x, self.y, learning_rate, batch_size, self.rng)
        return self.behaviour.corrupt_update(global_parameters, model.get_parameters())
