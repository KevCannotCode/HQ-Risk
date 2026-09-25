"""Backdoor (trojan) poisoning: stamp a fixed trigger on a fraction of training rows and relabel them to the target
class. Trigger = the last TRIGGER_SIZE features set to TRIGGER_VALUE standardised units (D42).
"""

import numpy as np


class BackdoorAttack:
    TRIGGER_SIZE = 3
    TRIGGER_VALUE = 3.0

    def __init__(self, target_label: int):
        self.target_label = target_label

    def stamp(self, x: np.ndarray) -> np.ndarray:
        stamped = x.copy()
        stamped[:, -self.TRIGGER_SIZE :] = self.TRIGGER_VALUE
        return stamped

    def poison(self, x: np.ndarray, y: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Fraction is of all rows it is handed (D8); only non-target rows are eligible."""
        candidates = np.flatnonzero(y != self.target_label)
        n_to_poison = min(int(round(fraction * len(y))), len(candidates))
        rows = rng.choice(candidates, size=n_to_poison, replace=False)
        x_poisoned, y_poisoned = x.copy(), y.copy()
        x_poisoned[rows] = self.stamp(x[rows])
        y_poisoned[rows] = self.target_label
        return x_poisoned, y_poisoned
