"""Splits the training set across clients. IID is a random equal shard; non-IID is Dirichlet label skew."""

import numpy as np


class DataPartitioner:
    IID = "iid"
    NON_IID = "non_iid"
    # Wine has 142 training rows; a Dirichlet draw can leave a client empty. Redraw from the same stream (D45).
    MAX_DRAWS = 100

    def __init__(self, partition: str, dirichlet_alpha: float):
        if partition not in (self.IID, self.NON_IID):
            raise ValueError(f"unknown partition {partition!r}")
        self.partition = partition
        self.dirichlet_alpha = dirichlet_alpha

    def split(self, y: np.ndarray, n_clients: int, n_classes: int, rng: np.random.Generator) -> list[np.ndarray]:
        if self.partition == self.IID:
            shards = self._split_iid(len(y), n_clients, rng)
        else:
            for _ in range(self.MAX_DRAWS):
                shards = self._split_dirichlet(y, n_clients, n_classes, rng)
                if all(len(shard) for shard in shards):
                    break
        self._assert_every_client_has_data(shards)
        return shards

    def _split_iid(self, n_rows: int, n_clients: int, rng: np.random.Generator) -> list[np.ndarray]:
        order = rng.permutation(n_rows)
        return [np.sort(shard) for shard in np.array_split(order, n_clients)]

    def _split_dirichlet(self, y: np.ndarray, n_clients: int, n_classes: int, rng: np.random.Generator) -> list[np.ndarray]:
        shards = [[] for _ in range(n_clients)]
        for label in range(n_classes):
            rows = rng.permutation(np.flatnonzero(y == label))
            proportions = rng.dirichlet(np.full(n_clients, self.dirichlet_alpha))
            cut_points = (np.cumsum(proportions) * len(rows)).astype(int)[:-1]
            for client, piece in enumerate(np.split(rows, cut_points)):
                shards[client].extend(piece.tolist())
        return [np.sort(np.asarray(shard, dtype=np.int64)) for shard in shards]

    def _assert_every_client_has_data(self, shards: list[np.ndarray]) -> None:
        empty = [i for i, shard in enumerate(shards) if len(shard) == 0]
        if empty:
            raise RuntimeError(f"partition left clients {empty} with no rows; lower dirichlet_alpha or reduce n_clients")
