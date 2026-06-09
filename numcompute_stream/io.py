"""Input/output helpers for NumCompute Stream.

Only NumPy and the Python standard library are used. The main loader returns
feature and target arrays so it can be used directly in the streaming demo.
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np


_MISSING_TOKENS = {"", "NA", "N/A", "nan", "NaN", "null", "None", "?"}


def _normalise_cell(value):
    """Convert common missing-value tokens to np.nan."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() in _MISSING_TOKENS:
        return np.nan
    return value


def load_csv(
    path: str,
    target_column: Optional[int] = -1,
    delimiter: str = ",",
    skip_header: bool | int = True,
    dtype=float,
    return_header: bool = False,
):
    """Load a CSV file using NumPy and optionally split features/target.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    target_column : int or None, default=-1
        Column index to use as target. If None, the full data array is returned.
    delimiter : str, default="," 
        CSV delimiter.
    skip_header : bool or int, default=True
        True skips one header row. False skips none. An integer skips that many rows.
    dtype : data-type, default=float
        Desired output dtype. For ML models, float is recommended.
    return_header : bool, default=False
        If True, returns the header row as a list when available.

    Returns
    -------
    X, y : np.ndarray, np.ndarray
        If target_column is not None, returns feature matrix and target vector.
    data : np.ndarray
        If target_column is None, returns the full data matrix.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If data is empty, malformed, or target_column is invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    if isinstance(skip_header, bool):
        skip_rows = 1 if skip_header else 0
    elif isinstance(skip_header, int) and skip_header >= 0:
        skip_rows = skip_header
    else:
        raise ValueError("skip_header must be a bool or non-negative integer.")

    header = None
    if skip_rows > 0:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            header = first_line.split(delimiter) if first_line else None

    raw = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_rows,
        dtype=object,
        encoding="utf-8",
        autostrip=True,
    )

    if raw.size == 0:
        raise ValueError("Loaded CSV data is empty.")

    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    normalised = np.vectorize(_normalise_cell, otypes=[object])(raw)

    try:
        data = normalised.astype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "CSV contains non-numeric values. For this assignment, use numeric CSV "
            "data or encode categorical values before model training."
        ) from exc

    if data.ndim != 2:
        raise ValueError("Loaded data must be a 2D array.")

    if target_column is None:
        return (data, header) if return_header else data

    n_features_total = data.shape[1]
    if not -n_features_total <= target_column < n_features_total:
        raise ValueError(
            f"target_column {target_column} is out of range for {n_features_total} columns."
        )

    target_idx = target_column % n_features_total
    X = np.delete(data, target_idx, axis=1)
    y = data[:, target_idx]

    if X.shape[1] == 0:
        raise ValueError("CSV must contain at least one feature column.")

    return (X, y, header) if return_header else (X, y)


def train_test_split(
    X,
    y,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
):
    """Split arrays into train and test subsets using NumPy only."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("y must be 1D with shape (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    n_test = max(1, int(round(n * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_stream_chunks(X, y=None, chunk_size: int = 32, shuffle: bool = False, random_state=None):
    """Yield chunks of X, or paired chunks of (X, y), for streaming demos."""
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must contain at least one sample.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    y_array = None if y is None else np.asarray(y)
    if y_array is not None and len(y_array) != n:
        raise ValueError("X and y must contain the same number of samples.")

    for start in range(0, n, chunk_size):
        batch_idx = indices[start : start + chunk_size]
        if y_array is None:
            yield X[batch_idx]
        else:
            yield X[batch_idx], y_array[batch_idx]
