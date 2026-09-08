"""Training-set label poisoning. Only ever touches the rows it is handed -- the caller passes train, never test.

Symmetric fraction is of all rows; targeted fraction is of source-class rows, otherwise 40% cannot exist (D26).
"""

import numpy as np


class LabelFlipAttack:
    SYMMETRIC = "symmetric"
    TARGETED = "targeted"

    def __init__(self, mode: str, source_label: int | None = None, target_label: int | None = None):
        if mode not in (self.SYMMETRIC, self.TARGETED):
            raise ValueError(f"unknown poisoning mode {mode!r}")
        if mode == self.TARGETED and (source_label is None or target_label is None):
            raise ValueError("targeted poisoning needs source_label and target_label")
        self.mode = mode
        self.source_label = source_label
        self.target_label = target_label

    def apply(self, y_train: np.ndarray, fraction: float, n_classes: int, rng: np.random.Generator) -> np.ndarray:
        if fraction <= 0:
            return y_train.copy()
        if self.mode == self.SYMMETRIC:
            return self._flip_symmetric(y_train, int(round(fraction * len(y_train))), n_classes, rng)
        return self._flip_targeted(y_train, fraction, rng)

    def _flip_symmetric(self, y: np.ndarray, n_to_flip: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
        poisoned = y.copy()
        rows = rng.choice(len(y), size=n_to_flip, replace=False)
        offsets = rng.integers(1, n_classes, size=n_to_flip)
        poisoned[rows] = (y[rows] + offsets) % n_classes
        return poisoned

    def _flip_targeted(self, y: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
        poisoned = y.copy()
        candidates = np.flatnonzero(y == self.source_label)
        n_to_flip = int(round(fraction * len(candidates)))
        rows = rng.choice(candidates, size=n_to_flip, replace=False)
        poisoned[rows] = self.target_label
        return poisoned
