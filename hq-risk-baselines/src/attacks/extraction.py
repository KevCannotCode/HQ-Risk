"""Model extraction (steal): a label-only API is queried with n synthetic inputs and a logistic regression is fitted
to the answers. Queries are N(0, I) in standardised feature space -- the attacker knows the feature scaling (D46).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class ModelExtractionAttack:
    def queries(self, n_queries: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal((n_queries, n_features))

    def steal(self, victim, n_queries: int, n_features: int, rng: np.random.Generator):
        queries = self.queries(n_queries, n_features, rng)
        answers = victim.predict(queries)
        if np.unique(answers).size == 1:
            return ConstantModel(answers[0])
        return LogisticRegression(max_iter=5000).fit(queries, answers)


class ConstantModel:
    """What the attacker can build when every query came back with the same label."""

    def __init__(self, label: int):
        self.label = label

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.label)
