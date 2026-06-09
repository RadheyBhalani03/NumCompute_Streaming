"""General utility functions used across the package."""

from __future__ import annotations

import numpy as np


def check_2d_array(X, name: str = "X", dtype=float) -> np.ndarray:
    """Convert input to a 2D NumPy array and validate shape."""
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (n_samples, n_features).")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def check_1d_array(y, name: str = "y", dtype=None) -> np.ndarray:
    """Convert input to a 1D NumPy array and validate shape."""
    arr = np.asarray(y, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array with shape (n_samples,).")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    return arr


def check_X_y(X, y, dtype=float):
    """Validate feature matrix X and target vector y."""
    X_arr = check_2d_array(X, "X", dtype=dtype)
    y_arr = check_1d_array(y, "y")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    return X_arr, y_arr


def sigmoid(x):
    """Compute the sigmoid function with simple overflow protection."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    """Compute a numerically stable softmax along an axis."""
    x = np.asarray(x, dtype=float)
    x_max = np.nanmax(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    denom = np.nansum(exp_x, axis=axis, keepdims=True)
    return exp_x / denom


def logsumexp(x, axis=None):
    """Compute log(sum(exp(x))) stably."""
    x = np.asarray(x, dtype=float)
    x_max = np.nanmax(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.nansum(np.exp(x - x_max), axis=axis, keepdims=True))
    return np.squeeze(result)


def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.sqrt(np.nansum((a - b) ** 2)))


def batch_iterator(X, y=None, batch_size: int = 32, shuffle: bool = False, random_state=None):
    """Yield streaming mini-batches.

    If y is provided, yields (X_batch, y_batch). Otherwise yields X_batch.
    """
    X = np.asarray(X)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    y_arr = None if y is None else np.asarray(y)
    if y_arr is not None and len(y_arr) != n:
        raise ValueError("X and y must have the same number of samples.")
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        if y_arr is None:
            yield X[idx]
        else:
            yield X[idx], y_arr[idx]
