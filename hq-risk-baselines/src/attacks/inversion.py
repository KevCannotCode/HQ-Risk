"""Model inversion (infer): black-box reconstruction of a representative input per class. Starting from the
feature mean, the attacker climbs log p(class | x) with zeroth-order gradient estimates (antithetic Gaussian
directions) under an L2 penalty. Intensity is the query budget per class (D48).
"""

import numpy as np


class ModelInversionAttack:
    DIRECTIONS = 10
    SIGMA = 0.5
    STEP = 0.5
    L2_PENALTY = 0.05
    FLOOR = 1e-6

    def reconstruct(self, victim, n_classes: int, n_features: int, queries_per_class: int, rng: np.random.Generator) -> np.ndarray:
        steps = queries_per_class // (2 * self.DIRECTIONS)
        return np.stack([self._climb(victim, label, n_features, steps, rng) for label in range(n_classes)])

    def _climb(self, victim, label: int, n_features: int, steps: int, rng: np.random.Generator) -> np.ndarray:
        x = np.zeros(n_features)
        for _ in range(steps):
            directions = rng.standard_normal((self.DIRECTIONS, n_features))
            probes = np.r_[x + self.SIGMA * directions, x - self.SIGMA * directions]
            log_p = np.log(np.maximum(victim.predict_proba(probes)[:, label], self.FLOOR))
            difference = log_p[: self.DIRECTIONS] - log_p[self.DIRECTIONS :]
            gradient = difference @ directions / (2 * self.SIGMA * self.DIRECTIONS)
            x = x + self.STEP * (gradient - 2 * self.L2_PENALTY * x)
        return x

    def similarity(self, reconstructions: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """Mean cosine similarity between each class's reconstruction and that class's true training mean."""
        cosines = []
        for label, reconstruction in enumerate(reconstructions):
            mean = x_train[y_train == label].mean(axis=0)
            norm = np.linalg.norm(reconstruction) * np.linalg.norm(mean)
            cosines.append(0.0 if norm == 0 else float(reconstruction @ mean / norm))
        return float(np.mean(cosines))
