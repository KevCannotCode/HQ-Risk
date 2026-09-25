"""ARF steal / infer attacks against any deployed model with predict / predict_proba (scikit-learn or QML).
The victim is untouched, so attacked predictions are the clean ones -- except extraction, where they are the stolen model's.
"""

import numpy as np

from src.attacks.extraction import ModelExtractionAttack
from src.attacks.inversion import ModelInversionAttack
from src.attacks.membership import MembershipInferenceAttack
from src.datasets import DatasetSplit
from src.run_config import RunConfig
from src.scenarios.outcome import Outcome


class QueryAttacks:
    def run(self, config: RunConfig, data: DatasetSplit, victim, clean_pred: np.ndarray) -> Outcome:
        if config.intensity == 0:
            return Outcome(clean_pred, clean_pred)
        rng = np.random.default_rng([config.seed, 0])
        n_features = data.x_train.shape[1]

        if config.attack == RunConfig.EXTRACTION:
            stolen = ModelExtractionAttack().steal(victim, int(config.intensity), n_features, rng)
            stolen_pred = stolen.predict(data.x_test)
            return Outcome(clean_pred, stolen_pred, float(np.mean(stolen_pred == clean_pred)))

        if config.attack == RunConfig.MEMBERSHIP:
            attack = MembershipInferenceAttack()
            members = attack.scores(victim, data.x_train, data.y_train)
            outsiders = attack.scores(victim, data.x_test, data.y_test)
            return Outcome(clean_pred, clean_pred, attack.accuracy(members, outsiders, int(config.intensity), rng))

        if config.attack == RunConfig.INVERSION:
            attack = ModelInversionAttack()
            reconstructions = attack.reconstruct(victim, data.n_classes, n_features, int(config.intensity), rng)
            return Outcome(clean_pred, clean_pred, attack.similarity(reconstructions, data.x_train, data.y_train))

        raise ValueError(f"{config.attack!r} is not a query attack")
