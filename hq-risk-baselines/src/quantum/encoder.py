"""Notebook preprocessing on top of DatasetLoader's standardised split: PCA to one feature per qubit,
then MinMax to [0, 1] so the encoding angle is pi * x. Both fitted on train only.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class PcaAngleEncoder:
    def __init__(self, n_qubits: int, seed: int):
        self.pca = PCA(n_components=n_qubits, random_state=seed)
        self.scaler = MinMaxScaler()

    def fit(self, x_train: np.ndarray) -> "PcaAngleEncoder":
        self.scaler.fit(self.pca.fit_transform(x_train))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.pca.transform(x))
