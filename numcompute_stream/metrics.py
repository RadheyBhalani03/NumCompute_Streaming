"""Classification and regression metrics with streaming support."""

from __future__ import annotations

import numpy as np


def validate_inputs(y_true, y_pred):
    """Validate metric input vectors."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("Input arrays cannot be empty.")
    return y_true, y_pred


def accuracy_score(y_true, y_pred):
    """Return the fraction of correct predictions."""
    y_true, y_pred = validate_inputs(y_true, y_pred)
    valid = ~(np.isnan(y_true.astype(float, copy=False)) | np.isnan(y_pred.astype(float, copy=False))) if np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number) else np.ones_like(y_true, dtype=bool)
    if np.sum(valid) == 0:
        return 0.0
    return float(np.mean(y_true[valid] == y_pred[valid]))


def accuracy(y_true, y_pred):
    """Backward-compatible alias for accuracy_score."""
    return accuracy_score(y_true, y_pred)


def error_rate(y_true, y_pred):
    """Return 1 - accuracy."""
    return float(1.0 - accuracy_score(y_true, y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute a confusion matrix.

    Rows represent true classes and columns represent predicted classes.
    """
    y_true, y_pred = validate_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        true_mask = y_true == true_label
        for j, pred_label in enumerate(labels):
            matrix[i, j] = int(np.sum(true_mask & (y_pred == pred_label)))
    return matrix


def precision(y_true, y_pred, labels=None):
    """Compute macro-averaged precision."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    return float(np.mean(np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)))


def recall(y_true, y_pred, labels=None):
    """Compute macro-averaged recall."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = np.sum(cm, axis=1) - tp
    return float(np.mean(np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)))


def f1_score(y_true, y_pred, labels=None):
    """Compute macro F1 score from macro precision and recall."""
    p = precision(y_true, y_pred, labels=labels)
    r = recall(y_true, y_pred, labels=labels)
    return float(0.0 if (p + r) == 0 else 2 * p * r / (p + r))


def mse(y_true, y_pred):
    """Compute mean squared error."""
    y_true, y_pred = validate_inputs(y_true, y_pred)
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    return float(np.nanmean((y_true - y_pred) ** 2))


class StreamingAccuracy:
    """Incrementally track accuracy over streaming chunks."""

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.history_ = []

    def update(self, y_true, y_pred):
        """Update metric state using a new chunk."""
        y_true, y_pred = validate_inputs(y_true, y_pred)
        self.correct += int(np.sum(y_true == y_pred))
        self.total += int(y_true.size)
        self.history_.append(self.compute())
        return self

    def compute(self):
        """Return cumulative accuracy."""
        return 0.0 if self.total == 0 else float(self.correct / self.total)

    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
        self.history_ = []
        return self


class StreamingConfusionMatrix:
    """Incrementally update a confusion matrix over chunks."""

    def __init__(self, labels=None):
        self.labels = None if labels is None else np.asarray(labels)
        self.matrix_ = None

    def update(self, y_true, y_pred):
        """Update confusion matrix with a chunk."""
        y_true, y_pred = validate_inputs(y_true, y_pred)
        if self.labels is None:
            new_labels = np.unique(np.concatenate((y_true, y_pred)))
            if self.matrix_ is None:
                self.labels = new_labels
                self.matrix_ = np.zeros((len(self.labels), len(self.labels)), dtype=int)
            else:
                all_labels = np.unique(np.concatenate((self.labels, new_labels)))
                expanded = np.zeros((len(all_labels), len(all_labels)), dtype=int)
                for i, old_i in enumerate(self.labels):
                    for j, old_j in enumerate(self.labels):
                        new_i = np.where(all_labels == old_i)[0][0]
                        new_j = np.where(all_labels == old_j)[0][0]
                        expanded[new_i, new_j] = self.matrix_[i, j]
                self.labels = all_labels
                self.matrix_ = expanded
        elif self.matrix_ is None:
            self.matrix_ = np.zeros((len(self.labels), len(self.labels)), dtype=int)

        self.matrix_ += confusion_matrix(y_true, y_pred, labels=self.labels)
        return self

    def compute(self):
        """Return cumulative confusion matrix."""
        if self.matrix_ is None:
            if self.labels is None:
                return np.zeros((0, 0), dtype=int)
            return np.zeros((len(self.labels), len(self.labels)), dtype=int)
        return self.matrix_.copy()
