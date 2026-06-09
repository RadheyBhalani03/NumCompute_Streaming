"""
NumCompute Stream
=================

A NumPy-only streaming machine learning framework for Assignment 2.2.

Main features
-------------
- CSV loading with missing-value handling
- Streaming preprocessing with partial_fit()
- Incremental statistics and metrics with update()
- Decision tree classifier from scratch
- Random forest classifier from scratch
- Streaming pipeline
- Matplotlib visualisation helpers
"""

from .io import load_csv, train_test_split, make_stream_chunks
from .metrics import accuracy_score, error_rate, confusion_matrix, StreamingAccuracy
from .preprocessing import StreamingImputer, StreamingStandardScaler
from .tree import DecisionTreeClassifier, ChunkTreeClassifier
from .ensemble import RandomForestClassifier, BaggingClassifier, StreamForestClassifier, ChunkForestClassifier
from .pipeline import StreamingPipeline

__version__ = "0.1.0"

__all__ = [
    "load_csv",
    "train_test_split",
    "make_stream_chunks",
    "accuracy_score",
    "error_rate",
    "confusion_matrix",
    "StreamingAccuracy",
    "StreamingImputer",
    "StreamingStandardScaler",
    "DecisionTreeClassifier",
    "ChunkTreeClassifier",
    "RandomForestClassifier",
    "BaggingClassifier",
    "StreamForestClassifier",
    "ChunkForestClassifier",
    "StreamingPipeline",
]
