# NumCompute Streaming Trees

This repository contains my Assignment 2.2 extension of NumCompute. It adds a small NumPy-only streaming machine-learning framework with a decision tree classifier, a tree-based ensemble, incremental preprocessing, streaming metrics, custom CSV loading, benchmarking, and matplotlib visualisation.

The project is written using only plain Python, NumPy, and matplotlib for the main package. It does not use scikit-learn, pandas, PyTorch, TensorFlow, or any external machine-learning library.

## Main features

- Chunk-based learning through `.partial_fit()`
- Decision tree classifier from scratch using Gini impurity
- Random forest style ensemble using bootstrapped trees and majority voting
- Streaming imputer and standard scaler
- Streaming accuracy and confusion-matrix style metrics
- Custom CSV loading and train/test splitting without pandas
- Built-in plotting functions in `visualise.py`
- Demo notebook showing streaming training and visualisation
- Benchmark comparing single tree, ensemble, and vectorised vs loop computation
- 40+ unit tests covering normal behaviour and edge cases

## Project structure

```text
numcompute_stream/
  __init__.py
  io.py
  utils.py
  statistics.py
  metrics.py
  preprocessing.py
  tree.py
  ensemble.py
  pipeline.py
  visualise.py

tests/
  test_io.py
  test_metrics.py
  test_statistics.py
  test_preprocessing.py
  test_tree.py
  test_ensemble.py
  test_pipeline.py
  test_visualise.py

demo/
  sample_data.csv
  stream_demo.ipynb
  make_sample_data.py

benchmark/
  benchmark_streaming.py
  benchmark_results.md

report.pdf
requirements.txt
pyproject.toml
```

## Installation

From the root folder:

```bash
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers CSV loading, chunk creation, streaming preprocessing, incremental statistics, metrics, decision tree fitting and prediction, random forest fitting and prediction, pipeline behaviour, visualisation functions, and edge cases such as NaNs, tied labels, invalid shapes, zero-variance chunks, and transform-before-fit errors.

## Run the benchmark

```bash
python benchmark/benchmark_streaming.py
```

The benchmark writes results to:

```text
benchmark/benchmark_results.md
```

## Run the demo notebook

Open and run:

```text
demo/stream_demo.ipynb
```

The notebook:

1. Loads `sample_data.csv` using the custom CSV loader.
2. Splits the data using NumPy.
3. Breaks the training set into stream chunks.
4. Trains the decision tree and random forest incrementally using `.partial_fit()`.
5. Logs accuracy after each chunk.
6. Uses `visualise.py` to plot model performance over time.