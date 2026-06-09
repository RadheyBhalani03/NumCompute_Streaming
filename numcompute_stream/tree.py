"""Decision tree classifier implemented from scratch using NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import check_2d_array, check_X_y


@dataclass
class Node:
    """A single node in the decision tree."""

    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None
    n_samples: int = 0
    n_left: int = 0
    n_right: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """A NumPy-only classification decision tree.

    Streaming support is provided through partial_fit(). Each incoming chunk is
    added to an internal buffer and the tree is rebuilt from all data seen so far.
    This gives a clear chunk-wise learning interface while keeping the algorithm
    transparent and testable for a NumPy-only assignment.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features=None,
        random_state=None,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.root_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self._X_buffer = None
        self._y_buffer = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        """Fit the decision tree on a complete batch."""
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._X_buffer = X.copy()
        self._y_buffer = y.copy()
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def partial_fit(self, X, y, classes=None):
        """Update the tree using a streaming chunk.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        classes : optional
            Accepted for API consistency. If supplied on first call, it defines
            the full class set.
        """
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
        self.root_ = self._build_tree(self._X_buffer, self._y_buffer, depth=0)
        return self

    def predict(self, X):
        """Predict class labels for X."""
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier must be fitted before predict().")
        X = check_2d_array(X, "X", dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data.")
        return np.array([self._predict_one(row, self.root_) for row in X])

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        max_count = np.max(counts)
        tied = values[counts == max_count]
        return np.sort(tied)[0]

    def _gini(self, y):
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        return float(1.0 - np.sum(probs ** 2))

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
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
        return self._rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y) -> Tuple[Optional[int], Optional[float], float]:
        parent_impurity = self._gini(y)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape

        for feature in self._feature_indices(n_features):
            col = X[:, feature]
            valid_col = col[~np.isnan(col)]
            if valid_col.size == 0:
                continue
            thresholds = np.unique(valid_col)
            if thresholds.size > 32:
                thresholds = np.quantile(thresholds, np.linspace(0.05, 0.95, 32))
                thresholds = np.unique(thresholds)
            for threshold in thresholds:
                left_mask = col <= threshold
                right_mask = col > threshold
                # Send NaNs to the larger child candidate deterministically.
                nan_mask = np.isnan(col)
                if np.any(nan_mask):
                    if np.sum(left_mask) >= np.sum(right_mask):
                        left_mask = left_mask | nan_mask
                    else:
                        right_mask = right_mask | nan_mask
                n_left = int(np.sum(left_mask))
                n_right = int(np.sum(right_mask))
                if n_left == 0 or n_right == 0:
                    continue
                weighted = (n_left / n_samples) * self._gini(y[left_mask]) + (n_right / n_samples) * self._gini(y[right_mask])
                gain = parent_impurity - weighted
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        return best_feature, best_threshold, float(best_gain if best_gain != -np.inf else 0.0)

    def _build_tree(self, X, y, depth):
        node = Node(n_samples=int(y.size))
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            node.value = self._majority_class(y)
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= self.min_impurity_decrease:
            node.value = self._majority_class(y)
            return node

        col = X[:, feature]
        left_mask = col <= threshold
        right_mask = col > threshold
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            if np.sum(left_mask) >= np.sum(right_mask):
                left_mask = left_mask | nan_mask
            else:
                right_mask = right_mask | nan_mask

        node.feature = feature
        node.threshold = threshold
        node.n_left = int(np.sum(left_mask))
        node.n_right = int(np.sum(right_mask))
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _predict_one(self, row, node):
        while not node.is_leaf:
            value = row[node.feature]
            if np.isnan(value):
                node = node.left if node.n_left >= node.n_right else node.right
            elif value <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

# Friendly alias used in the demo notebook.
ChunkTreeClassifier = DecisionTreeClassifier
