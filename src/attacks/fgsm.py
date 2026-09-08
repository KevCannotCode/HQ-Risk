"""White-box FGSM on the test set. Epsilon is in standardised feature units (see DatasetLoader)."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class FgsmAttack:
    def perturb(self, model, x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
        if epsilon <= 0:
            return x.copy()
        gradient = self.loss_gradient(model, x, y)
        return x + epsilon * np.sign(gradient)

    def loss_gradient(self, model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(model, LogisticRegression):
            return self._logistic_regression_gradient(model, x, y)
        if isinstance(model, MLPClassifier):
            return self._mlp_gradient(model, x, y)
        raise TypeError(f"no gradient available for {type(model).__name__}")

    def _logistic_regression_gradient(self, model: LogisticRegression, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        residual = self._output_residual(model.predict_proba(x), y, model.classes_)
        return residual @ model.coef_

    def _mlp_gradient(self, model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(model.coefs_) != 2 or model.activation != "relu":
            raise ValueError("MLP gradient implemented for one relu hidden layer only")
        hidden_weights, output_weights = model.coefs_
        hidden_bias, _ = model.intercepts_

        pre_activation = x @ hidden_weights + hidden_bias
        residual = self._output_residual(model.predict_proba(x), y, model.classes_)
        hidden_gradient = (residual @ output_weights.T) * (pre_activation > 0)
        return hidden_gradient @ hidden_weights.T

    def _output_residual(self, probabilities: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        class_index = np.searchsorted(classes, y)
        if probabilities.shape[1] == 2:
            positive = probabilities[:, 1]
            return (positive - (class_index == 1))[:, None]
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(y)), class_index] = 1.0
        return probabilities - targets
