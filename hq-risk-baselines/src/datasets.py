"""Benchmark datasets with a stratified split and a scaler fitted on train only.

Scaling is what gives the FGSM epsilon a meaning, so it lives here rather than in the attack.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_classes: int
    positive_label: int | None
    source_label: int
    target_label: int


class DatasetLoader:
    TEST_FRACTION = 0.2

    _LOADERS = {
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
        "wine": load_wine,
    }

    _POSITIVE_LABEL = {
        "breast_cancer": 1,
        "digits": None,
        "wine": None,
    }

    # Targeted attacks move source-class rows to the target class. Multiclass: lowest two labels, no cherry-picking (D41).
    _TARGETED_PAIR = {
        "breast_cancer": (1, 0),
        "digits": (0, 1),
        "wine": (0, 1),
    }

    def available(self) -> list[str]:
        return sorted(self._LOADERS)

    def load(self, name: str, seed: int) -> DatasetSplit:
        if name not in self._LOADERS:
            raise ValueError(f"unknown dataset {name!r}, expected one of {self.available()}")

        bundle = self._LOADERS[name]()
        features = np.asarray(bundle.data, dtype=np.float64)
        labels = np.asarray(bundle.target, dtype=np.int64)

        if name == "breast_cancer":
            labels = 1 - labels

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.TEST_FRACTION,
            random_state=seed,
            stratify=labels,
        )

        scaler = StandardScaler().fit(x_train)
        assert scaler.n_samples_seen_ == len(x_train), "scaler saw rows outside the training set"

        return DatasetSplit(
            name=name,
            x_train=scaler.transform(x_train),
            y_train=y_train,
            x_test=scaler.transform(x_test),
            y_test=y_test,
            n_classes=int(np.unique(labels).size),
            positive_label=self._POSITIVE_LABEL[name],
            source_label=self._TARGETED_PAIR[name][0],
            target_label=self._TARGETED_PAIR[name][1],
        )
