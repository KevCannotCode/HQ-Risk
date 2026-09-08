"""Classifiers used by both scenarios.

S1 uses scikit-learn estimators. S2 needs to start local training from arbitrary global weights,
which scikit-learn's solvers do not expose, so federated runs use SoftmaxRegression -- the same
logistic-regression hypothesis class, trained by mini-batch gradient descent.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class SoftmaxRegression:
    def __init__(self, n_features: int, n_classes: int):
        self.weights = np.zeros((n_features, n_classes), dtype=np.float64)
        self.bias = np.zeros(n_classes, dtype=np.float64)

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weights.copy(), self.bias.copy()

    def set_parameters(self, parameters: tuple[np.ndarray, np.ndarray]) -> None:
        weights, bias = parameters
        self.weights = np.asarray(weights, dtype=np.float64).copy()
        self.bias = np.asarray(bias, dtype=np.float64).copy()

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(x @ self.weights + self.bias, axis=1)

    def fit_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        order = rng.permutation(len(x))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            self._gradient_step(x[batch], y[batch], learning_rate)

    def _gradient_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        probabilities = self.probabilities(x)
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(y)), y] = 1.0
        error = (probabilities - targets) / len(y)
        self.weights -= learning_rate * (x.T @ error)
        self.bias -= learning_rate * error.sum(axis=0)


class ModelFactory:
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"

    def available(self) -> list[str]:
        return [self.LOGISTIC_REGRESSION, self.MLP]

    def create(self, name: str, seed: int):
        if name == self.LOGISTIC_REGRESSION:
            return LogisticRegression(max_iter=5000, random_state=seed)
        if name == self.MLP:
            return MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=seed)
        raise ValueError(f"unknown model {name!r}, expected one of {self.available()}")

    def create_federated(self, name: str, n_features: int, n_classes: int) -> SoftmaxRegression:
        if name != self.LOGISTIC_REGRESSION:
            raise ValueError(f"federated runs support only {self.LOGISTIC_REGRESSION!r}, got {name!r}")
        return SoftmaxRegression(n_features, n_classes)
