# Streaming Benchmark Results

| Model | Final Accuracy | Runtime (seconds) | Notes |
|---|---:|---:|---|
| Decision Tree | 0.920 | 0.300218 | Faster single model |
| Random Forest | 0.947 | 1.038911 | Ensemble with majority voting |

## Accuracy by stream chunk

Decision Tree: [0.9066666666666666, 0.88, 0.8933333333333333, 0.9066666666666666, 0.92, 0.96, 0.9733333333333334, 0.92]

Random Forest: [0.9466666666666667, 0.9333333333333333, 0.9066666666666666, 0.9466666666666667, 0.9733333333333334, 0.9333333333333333, 0.9733333333333334, 0.9466666666666667]

## Vectorised vs loop mean

| Method | Average Time (seconds) |
|---|---:|
| Vectorised mean | 0.00011098 |
| Loop mean | 0.01269329 |

Speedup: 114.37x
