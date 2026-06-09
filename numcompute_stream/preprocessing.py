"""Streaming preprocessing tools.

The classes in this file follow a scikit-learn-like API but are implemented
from scratch using only NumPy.
"""

from __future__ import annotations

import numpy as np

from .utils import check_2d_array


class StreamingImputer:
    """Impute missing numeric values using running column means.

    The imputer supports chunk-wise updates through partial_fit(). Missing values
    are represented using np.nan. Columns with no observed values are filled with
    fill_value.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value
        self.count_ = None
        self.sum_ = None
        self.statistics_ = None
        self.n_features_in_ = None

    def partial_fit(self, X, y=None):
        """Update running column means using a new chunk."""
        del y
        X = check_2d_array(X, "X", dtype=float)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=float)
            self.sum_ = np.zeros(X.shape[1], dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks.")

        mask = ~np.isnan(X)
        self.count_ += np.sum(mask, axis=0)
        self.sum_ += np.nansum(X, axis=0)
        self.statistics_ = np.divide(
            self.sum_,
            self.count_,
            out=np.full_like(self.sum_, self.fill_value, dtype=float),
            where=self.count_ != 0,
        )
        return self

    def fit(self, X, y=None):
        """Fit imputer on one full batch."""
        self.count_ = None
        self.sum_ = None
        self.statistics_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y)

    def transform(self, X):
        """Replace np.nan values using learned running means."""
        if self.statistics_ is None:
            raise ValueError("StreamingImputer must be fitted before transform().")
        X = check_2d_array(X, "X", dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data.")
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)


class StreamingStandardScaler:
    """Standardize numeric features using running means and variances.

    Standard deviation values of zero are replaced with 1.0 to avoid division by
    zero. NaNs are ignored during fitting and remain NaN during transform unless
    an imputer is used before the scaler.
    """

    def __init__(self):
        self.count_ = None
        self.sum_ = None
        self.sum_sq_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def partial_fit(self, X, y=None):
        """Update scaler state using a new chunk."""
        del y
        X = check_2d_array(X, "X", dtype=float)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=float)
            self.sum_ = np.zeros(X.shape[1], dtype=float)
            self.sum_sq_ = np.zeros(X.shape[1], dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks.")

        mask = ~np.isnan(X)
        safe_X = np.where(mask, X, 0.0)
        self.count_ += np.sum(mask, axis=0)
        self.sum_ += np.sum(safe_X, axis=0)
        self.sum_sq_ += np.sum(safe_X ** 2, axis=0)

        self.mean_ = np.divide(
            self.sum_,
            self.count_,
            out=np.zeros_like(self.sum_, dtype=float),
            where=self.count_ != 0,
        )
        second_moment = np.divide(
            self.sum_sq_,
            self.count_,
            out=np.zeros_like(self.sum_sq_, dtype=float),
            where=self.count_ != 0,
        )
        self.var_ = np.maximum(second_moment - self.mean_ ** 2, 0.0)
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def fit(self, X, y=None):
        """Fit scaler on one full batch."""
        self.count_ = None
        self.sum_ = None
        self.sum_sq_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_features_in_ = None
        return self.partial_fit(X, y)

    def transform(self, X):
        """Standardize X using the current running statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StreamingStandardScaler must be fitted before transform().")
        X = check_2d_array(X, "X", dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from fitted data.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)


# Backward-compatible aliases for older NumCompute notebooks.
Imputer = StreamingImputer
StandardScaler = StreamingStandardScaler
