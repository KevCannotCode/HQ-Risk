"""The deployed QML model as an attacker sees it: standardised features in, shot-based answers out.
Same predict / predict_proba surface as a scikit-learn estimator, so the ARF attacks treat QML and classical alike.
"""

import numpy as np
import torch

from src.quantum.circuit import QuantumNet
from src.quantum.encoder import PcaAngleEncoder
from src.quantum.executor import ShotExecutor, z_expectations


class QuantumClassifier:
    def __init__(self, net: QuantumNet, encoder: PcaAngleEncoder, shots: int, seed: int):
        self.net = net
        self.encoder = encoder
        self.shots = shots
        self.executor = ShotExecutor(seed=seed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        counts = self.executor.counts(self.net, self.encoder.transform(x), self.shots)
        with torch.no_grad():
            logits = self.net.head(torch.tensor(z_expectations(counts, self.net.n_qubits), dtype=torch.float64))
        return torch.softmax(logits, dim=1).numpy()
