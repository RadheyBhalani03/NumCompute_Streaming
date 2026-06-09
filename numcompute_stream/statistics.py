"""Statistical functions and streaming statistics for NumCompute Stream."""

from __future__ import annotations

import numpy as np


def mean(X, axis=None):
    """Compute mean while ignoring NaNs."""
    return np.nanmean(np.asarray(X, dtype=float), axis=axis)


def mean_loop(X):
    """Loop-based mean for benchmarking against the vectorised implementation."""
    X = np.ravel(np.asarray(X, dtype=float))
    if X.size == 0:
        raise ValueError("mean of empty array is undefined.")
    total = 0.0
    count = 0
    for value in X:
        if not np.isnan(value):
            total += value
            count += 1
    return np.nan if count == 0 else total / count


def std(X, axis=None):
    """Compute standard deviation while ignoring NaNs."""
    return np.nanstd(np.asarray(X, dtype=float), axis=axis)


def std_loop(X):
    """Loop-based standard deviation for benchmarking."""
    X = np.ravel(np.asarray(X, dtype=float))
    mu = mean_loop(X)
    if np.isnan(mu):
        return np.nan
    total = 0.0
    count = 0
    for value in X:
        if not np.isnan(value):
            total += (value - mu) ** 2
            count += 1
    return np.nan if count == 0 else float(np.sqrt(total / count))


def min(X, axis=None):  # pylint: disable=redefined-builtin
    """Compute minimum while ignoring NaNs."""
    return np.nanmin(np.asarray(X, dtype=float), axis=axis)


def max(X, axis=None):  # pylint: disable=redefined-builtin
    """Compute maximum while ignoring NaNs."""
    return np.nanmax(np.asarray(X, dtype=float), axis=axis)


def histogram(X, bins=10):
    """Compute histogram after removing NaN values."""
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X)]
    if X.size == 0:
        raise ValueError("histogram requires at least one non-NaN value.")
    return np.histogram(X, bins=bins)


def quantile(X, q):
    """Compute quantile while ignoring NaNs."""
    X = np.asarray(X, dtype=float)
    if np.any((np.asarray(q) < 0) | (np.asarray(q) > 1)):
        raise ValueError("q must be between 0 and 1.")
    return np.nanquantile(X, q)


class RunningMean:
    """Incrementally compute column-wise means for streaming data.

    NaNs are ignored. The state stores only counts and sums, not all previous data.
    """

    def __init__(self):
        self.count_ = None
        self.sum_ = None
        self.n_features_in_ = None

    def update(self, X):
        """Update running mean using a new chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.count_ = np.zeros(X.shape[1], dtype=float)
            self.sum_ = np.zeros(X.shape[1], dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from previous chunks.")

        mask = ~np.isnan(X)
        self.count_ += np.sum(mask, axis=0)
        self.sum_ += np.nansum(X, axis=0)
        return self

    def compute(self):
        """Return current column-wise means."""
        if self.count_ is None:
            raise ValueError("RunningMean has not received any data yet.")
        return np.divide(
            self.sum_,
            self.count_,
            out=np.full_like(self.sum_, np.nan, dtype=float),
            where=self.count_ != 0,
        )


class RunningVariance:
    """Incrementally compute column-wise variance using sums and squared sums."""

    def __init__(self):
        self.count_ = None
        self.sum_ = None
        self.sum_sq_ = None
        self.n_features_in_ = None

    def update(self, X):
        """Update running variance using a new chunk."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")
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
        return self

    def mean(self):
        """Return current column-wise means."""
        if self.count_ is None:
            raise ValueError("RunningVariance has not received any data yet.")
        return np.divide(
            self.sum_,
            self.count_,
            out=np.full_like(self.sum_, np.nan, dtype=float),
            where=self.count_ != 0,
        )

    def compute(self):
        """Return current column-wise population variances."""
        if self.count_ is None:
            raise ValueError("RunningVariance has not received any data yet.")
        mu = self.mean()
        var = np.divide(
            self.sum_sq_,
            self.count_,
            out=np.full_like(self.sum_sq_, np.nan, dtype=float),
            where=self.count_ != 0,
        ) - mu ** 2
        return np.maximum(var, 0.0)

    def std(self):
        """Return current column-wise standard deviations."""
        return np.sqrt(self.compute())
