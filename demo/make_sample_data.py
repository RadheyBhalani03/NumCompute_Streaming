"""Generate sample CSV data for the streaming demo."""

import numpy as np

rng = np.random.default_rng(42)
X0 = rng.normal(loc=-1.0, scale=0.8, size=(75, 4))
X1 = rng.normal(loc=1.0, scale=0.8, size=(75, 4))
X = np.vstack((X0, X1))
y = np.array([0] * 75 + [1] * 75)
idx = rng.permutation(len(y))
X = X[idx]
y = y[idx]
# Add a few missing values to show imputer behaviour.
X[0, 1] = np.nan
X[10, 2] = np.nan

data = np.column_stack((X, y))
np.savetxt(
    "demo/sample_data.csv",
    data,
    delimiter=",",
    header="f1,f2,f3,f4,label",
    comments="",
    fmt="%.6f",
)
print("Saved demo/sample_data.csv")
