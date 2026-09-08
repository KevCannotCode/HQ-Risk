"""What a malicious federated client does. Applied by FederatedClient, never by the server."""

import numpy as np

from src.attacks.label_flip import LabelFlipAttack


class ByzantineBehaviour:
    HONEST = "honest"
    LABEL_FLIP = "label_flip"
    SIGN_FLIP = "sign_flip"

    def __init__(self, mode: str):
        if mode not in (self.HONEST, self.LABEL_FLIP, self.SIGN_FLIP):
            raise ValueError(f"unknown byzantine mode {mode!r}")
        self.mode = mode

    def corrupt_labels(self, y: np.ndarray, n_classes: int, rng: np.random.Generator) -> np.ndarray:
        if self.mode != self.LABEL_FLIP:
            return y
        return LabelFlipAttack(LabelFlipAttack.SYMMETRIC).apply(y, fraction=1.0, n_classes=n_classes, rng=rng)

    def corrupt_update(self, global_parameters, local_parameters):
        if self.mode != self.SIGN_FLIP:
            return local_parameters
        return tuple(g - (l - g) for g, l in zip(global_parameters, local_parameters))
