"""S1: centralised scikit-learn classifier. Training-time attacks retrain on corrupted data; evasion and the
query attacks act on the clean model.
"""

import numpy as np

from src.attacks.backdoor import BackdoorAttack
from src.attacks.fgsm import FgsmAttack
from src.attacks.label_flip import LabelFlipAttack
from src.datasets import DatasetSplit
from src.metrics import MetricsCalculator
from src.models import ModelFactory
from src.run_config import RunConfig
from src.scenarios.outcome import Outcome
from src.scenarios.query_attacks import QueryAttacks


class ClassicalScenario:
    def __init__(self):
        self.factory = ModelFactory()
        self.metrics = MetricsCalculator()

    def run(self, config: RunConfig, data: DatasetSplit) -> Outcome:
        clean_model = self.factory.create(config.model, config.seed).fit(data.x_train, data.y_train)
        clean_pred = clean_model.predict(data.x_test)

        if config.attack in RunConfig.QUERY_ATTACKS:
            return QueryAttacks().run(config, data, clean_model, clean_pred)
        if config.attack == RunConfig.BACKDOOR:
            return self._backdoor(config, data, clean_model, clean_pred)
        if config.attack == RunConfig.POISONING:
            return self._poisoning(config, data, clean_pred)
        if config.attack == RunConfig.EVASION:
            if config.intensity == 0:
                return Outcome(clean_pred, clean_pred)
            adversarial = FgsmAttack().perturb(clean_model, data.x_test, data.y_test, config.intensity)
            attacked_pred = clean_model.predict(adversarial)
            return Outcome(clean_pred, attacked_pred, self.metrics.attack_success_rate(data.y_test, clean_pred, attacked_pred))
        raise ValueError(f"unknown S1 attack {config.attack!r}")

    def _poisoning(self, config: RunConfig, data: DatasetSplit, clean_pred: np.ndarray) -> Outcome:
        targeted = config.attack_mode == LabelFlipAttack.TARGETED
        attacked_pred = clean_pred
        if config.intensity > 0:
            attack = LabelFlipAttack(config.attack_mode, data.source_label, data.target_label) if targeted else LabelFlipAttack(config.attack_mode)
            poisoned_labels = attack.apply(data.y_train, config.intensity, data.n_classes, np.random.default_rng([config.seed, 0]))
            attacked_pred = self.factory.create(config.model, config.seed).fit(data.x_train, poisoned_labels).predict(data.x_test)
        if not targeted:
            return Outcome(clean_pred, attacked_pred)
        # Targeted success is a rate even without an attack (the natural source -> target error), so intensity 0 has one (D43).
        return Outcome(clean_pred, attacked_pred, self.metrics.targeted_success_rate(data.y_test, attacked_pred, data.target_label, data.source_label))

    def _backdoor(self, config: RunConfig, data: DatasetSplit, clean_model, clean_pred: np.ndarray) -> Outcome:
        backdoor = BackdoorAttack(data.target_label)
        model = clean_model
        if config.intensity > 0:
            x_poisoned, y_poisoned = backdoor.poison(data.x_train, data.y_train, config.intensity, np.random.default_rng([config.seed, 0]))
            model = self.factory.create(config.model, config.seed).fit(x_poisoned, y_poisoned)
        triggered_pred = model.predict(backdoor.stamp(data.x_test))
        return Outcome(clean_pred, model.predict(data.x_test), self.metrics.targeted_success_rate(data.y_test, triggered_pred, data.target_label))
