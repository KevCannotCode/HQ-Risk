"""Gradient-based training of QuantumNet: Adam on circuit weights and classical head together, exact gradients."""

import numpy as np
import torch
from torch.nn import functional as F

from src.quantum.circuit import QuantumNet


class QuantumTrainer:
    def __init__(self, epochs: int, learning_rate: float, batch_size: int, seed: int):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, net: QuantumNet, x: np.ndarray, y: np.ndarray) -> QuantumNet:
        features = torch.tensor(x, dtype=torch.float64)
        labels = torch.tensor(y, dtype=torch.long)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < 2:
                    continue  # BatchNorm needs two rows
                optimizer.zero_grad()
                F.cross_entropy(net(features[batch]), labels[batch]).backward()
                optimizer.step()
        net.eval()
        return net
