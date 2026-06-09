"""Matplotlib visualisation helpers for streaming ML experiments."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_history(values, title="Metric over time", ylabel="Metric"):
    """Plot one metric value per streaming chunk.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be a 1D sequence.")
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_model_comparison(single_tree_scores, ensemble_scores, labels=("Decision Tree", "Random Forest")):
    """Plot streaming metric histories for two models."""
    single_tree_scores = np.asarray(single_tree_scores, dtype=float)
    ensemble_scores = np.asarray(ensemble_scores, dtype=float)
    if single_tree_scores.ndim != 1 or ensemble_scores.ndim != 1:
        raise ValueError("score histories must be 1D sequences.")
    if len(single_tree_scores) != len(ensemble_scores):
        raise ValueError("score histories must have the same length.")
    chunks = np.arange(1, len(single_tree_scores) + 1)
    fig, ax = plt.subplots()
    ax.plot(chunks, single_tree_scores, marker="o", label=labels[0])
    ax.plot(chunks, ensemble_scores, marker="s", label=labels[1])
    ax.set_title("Streaming model comparison")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_confusion_matrix(cm, class_labels=None, title="Confusion Matrix"):
    """Display a confusion matrix using matplotlib."""
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square 2D matrix.")
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]
    if len(class_labels) != cm.shape[0]:
        raise ValueError("class_labels length must match cm dimensions.")

    fig, ax = plt.subplots()
    image = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(cm.shape[0]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    return fig, ax
