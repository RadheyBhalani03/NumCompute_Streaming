"""Tree-based ensemble models implemented with NumPy only."""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier
from .utils import check_2d_array, check_X_y


class RandomForestClassifier:
    """Random forest classifier using bootstrap samples and majority voting.

    Each tree is trained on a bootstrap sample and a random subset of features.
    Streaming partial_fit() accumulates chunks and rebuilds the forest from all
    data seen so far.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        sample_ratio: float = 1.0,
        max_features="sqrt",
        random_state=None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if not 0 < sample_ratio <= 1:
            raise ValueError("sample_ratio must be in (0, 1].")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_ratio = sample_ratio
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.feature_indices_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self._X_buffer = None
        self._y_buffer = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """Fit the forest on one full batch."""
        X, y = check_X_y(X, y, dtype=float)
        self._X_buffer = X.copy()
        self._y_buffer = y.copy()
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._fit_forest(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Update forest using a streaming chunk."""
        X, y = check_X_y(X, y, dtype=float)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self._X_buffer = X.copy()
            self._y_buffer = y.copy()
            self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError("X has a different number of features from previous chunks.")
            self._X_buffer = np.vstack((self._X_buffer, X))
            self._y_buffer = np.concatenate((self._y_buffer, y))
            if classes is None:
                self.classes_ = np.unique(np.concatenate((self.classes_, np.unique(y))))
        self._fit_forest(self._X_buffer, self._y_buffer)
        return self

    def predict(self, X):
        """Predict by majority vote across trees."""
        if not self.trees_:
            raise ValueError("RandomForestClassifier must be fitted before predict().")
        X = check_2d_array(X, "X", dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data.")

        predictions = []
        for tree, features in zip(self.trees_, self.feature_indices_):
            predictions.append(tree.predict(X[:, features]))
        predictions = np.asarray(predictions)  # shape (n_estimators, n_samples)
        return np.array([self._majority_vote(predictions[:, i]) for i in range(X.shape[0])])

    def _fit_forest(self, X, y):
        self.trees_ = []
        self.feature_indices_ = []
        n_samples, n_features = X.shape
        sample_size = max(1, int(round(self.sample_ratio * n_samples)))

        for i in range(self.n_estimators):
            sample_idx = self._rng.choice(n_samples, size=sample_size, replace=True)
            features = self._choose_features(n_features)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=None,
                random_state=None if self.random_state is None else self.random_state + i + 1,
            )
            tree.fit(X[sample_idx][:, features], y[sample_idx])
            self.trees_.append(tree)
            self.feature_indices_.append(features)

    def _choose_features(self, n_features):
        if self.max_features is None:
            k = n_features
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(n_features, max(1, self.max_features))
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("float max_features must be in (0, 1].")
            k = max(1, int(np.ceil(self.max_features * n_features)))
        else:
            raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'.")
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    def _majority_vote(self, values):
        labels, counts = np.unique(values, return_counts=True)
        max_count = np.max(counts)
        tied = labels[counts == max_count]
        return np.sort(tied)[0]


class BaggingClassifier(RandomForestClassifier):
    """Bagging classifier using all features for every tree."""

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        sample_ratio: float = 1.0,
        random_state=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            sample_ratio=sample_ratio,
            max_features=None,
            random_state=random_state,
        )

# Friendly aliases used in examples and video explanation.
StreamForestClassifier = RandomForestClassifier
ChunkForestClassifier = RandomForestClassifier
