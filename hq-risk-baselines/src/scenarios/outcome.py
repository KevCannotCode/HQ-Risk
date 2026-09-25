"""What a scenario hands back for one run: clean and attacked test predictions, and the attack's own success score."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcome:
    clean_pred: np.ndarray
    attacked_pred: np.ndarray
    attack_success_rate: float | None = None
