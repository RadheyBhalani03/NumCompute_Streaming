"""Benchmark streaming decision tree vs random forest.

Run from the project root:
    python benchmark/benchmark_streaming.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numcompute_stream import DecisionTreeClassifier, RandomForestClassifier
from numcompute_stream.io import make_stream_chunks, train_test_split
from numcompute_stream.metrics import accuracy_score
from numcompute_stream.statistics import mean, mean_loop


def time_function(func, *args, repeat=5):
    """Measure average execution time in seconds."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))


def make_dataset(n_samples=300, random_state=42):
    """Create a simple numeric binary classification dataset."""
    rng = np.random.default_rng(random_state)
    X0 = rng.normal(loc=-1.0, scale=0.8, size=(n_samples // 2, 4))
    X1 = rng.normal(loc=1.0, scale=0.8, size=(n_samples - n_samples // 2, 4))
    X = np.vstack((X0, X1))
    y = np.array([0] * len(X0) + [1] * len(X1))
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def run_streaming_model(model, X_train, y_train, X_test, y_test, chunk_size=30):
    """Train model incrementally and return final accuracy."""
    scores = []
    for X_chunk, y_chunk in make_stream_chunks(X_train, y_train, chunk_size=chunk_size):
        model.partial_fit(X_chunk, y_chunk)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
    return float(scores[-1]), scores


def benchmark_models():
    """Compare single tree and ensemble under streaming learning."""
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    tree = DecisionTreeClassifier(max_depth=5, random_state=1)
    forest = RandomForestClassifier(n_estimators=7, max_depth=5, random_state=1)

    start = time.perf_counter()
    tree_acc, tree_scores = run_streaming_model(tree, X_train, y_train, X_test, y_test)
    tree_time = time.perf_counter() - start

    start = time.perf_counter()
    forest_acc, forest_scores = run_streaming_model(forest, X_train, y_train, X_test, y_test)
    forest_time = time.perf_counter() - start

    print("\nStreaming Model Benchmark")
    print("-" * 35)
    print(f"Decision Tree final accuracy: {tree_acc:.3f}")
    print(f"Decision Tree runtime:        {tree_time:.6f} sec")
    print(f"Random Forest final accuracy: {forest_acc:.3f}")
    print(f"Random Forest runtime:        {forest_time:.6f} sec")

    return {
        "tree_accuracy": tree_acc,
        "tree_runtime": tree_time,
        "forest_accuracy": forest_acc,
        "forest_runtime": forest_time,
        "tree_scores": tree_scores,
        "forest_scores": forest_scores,
    }


def benchmark_vectorised_vs_loop():
    """Compare vectorised and loop-based mean implementations."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=10000)
    t_vec = time_function(mean, data, repeat=20)
    t_loop = time_function(mean_loop, data, repeat=20)
    speedup = t_loop / t_vec if t_vec > 0 else np.inf

    print("\nVectorised vs Loop Benchmark")
    print("-" * 35)
    print(f"Vectorised mean: {t_vec:.8f} sec")
    print(f"Loop mean:       {t_loop:.8f} sec")
    print(f"Speedup:         {speedup:.2f}x")

    return {"vectorised_mean": t_vec, "loop_mean": t_loop, "speedup": speedup}


if __name__ == "__main__":
    results = benchmark_models()
    vec_results = benchmark_vectorised_vs_loop()

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Streaming Benchmark Results\n\n")
        f.write("| Model | Final Accuracy | Runtime (seconds) | Notes |\n")
        f.write("|---|---:|---:|---|\n")
        f.write(f"| Decision Tree | {results['tree_accuracy']:.3f} | {results['tree_runtime']:.6f} | Faster single model |\n")
        f.write(f"| Random Forest | {results['forest_accuracy']:.3f} | {results['forest_runtime']:.6f} | Ensemble with majority voting |\n\n")
        f.write("## Accuracy by stream chunk\n\n")
        f.write(f"Decision Tree: {results['tree_scores']}\n\n")
        f.write(f"Random Forest: {results['forest_scores']}\n\n")
        f.write("## Vectorised vs loop mean\n\n")
        f.write("| Method | Average Time (seconds) |\n")
        f.write("|---|---:|\n")
        f.write(f"| Vectorised mean | {vec_results['vectorised_mean']:.8f} |\n")
        f.write(f"| Loop mean | {vec_results['loop_mean']:.8f} |\n")
        f.write(f"\nSpeedup: {vec_results['speedup']:.2f}x\n")
    print(f"\nSaved results to {output_path}")
