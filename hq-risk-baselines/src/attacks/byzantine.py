"""What a malicious federated client does. Applied by FederatedClient, never by the server."""

import numpy as np

from src.attacks.backdoor import BackdoorAttack
from src.attacks.label_flip import LabelFlipAttack


class ByzantineBehaviour:
    HONEST = "honest"
    LABEL_FLIP = "label_flip"
    SIGN_FLIP = "sign_flip"
    TARGETED_FLIP = "targeted_flip"
    BACKDOOR = "backdoor"
    MODES = (HONEST, LABEL_FLIP, SIGN_FLIP, TARGETED_FLIP, BACKDOOR)
    # A backdoor client keeps half its rows clean so its update still looks like the honest ones (D44).
    BACKDOOR_LOCAL_FRACTION = 0.5

    def __init__(self, mode: str, source_label: int | None = None, target_label: int | None = None):
        if mode not in self.MODES:
            raise ValueError(f"unknown byzantine mode {mode!r}")
        self.mode = mode
        self.source_label = source_label
        self.target_label = target_label

    def corrupt_data(self, x: np.ndarray, y: np.ndarray, n_classes: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == self.LABEL_FLIP:
            return x, LabelFlipAttack(LabelFlipAttack.SYMMETRIC).apply(y, fraction=1.0, n_classes=n_classes, rng=rng)
        if self.mode == self.TARGETED_FLIP:
            attack = LabelFlipAttack(LabelFlipAttack.TARGETED, self.source_label, self.target_label)
            return x, attack.apply(y, fraction=1.0, n_classes=n_classes, rng=rng)
        if self.mode == self.BACKDOOR:
            return BackdoorAttack(self.target_label).poison(x, y, self.BACKDOOR_LOCAL_FRACTION, rng)
        return x, y

    def corrupt_update(self, global_parameters, local_parameters):
        if self.mode != self.SIGN_FLIP:
            return local_parameters
        return tuple(g - (l - g) for g, l in zip(global_parameters, local_parameters))
