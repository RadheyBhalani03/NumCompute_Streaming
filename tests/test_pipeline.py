import unittest
import numpy as np

from numcompute_stream.pipeline import StreamingPipeline
from numcompute_stream.preprocessing import StreamingImputer, StreamingStandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


class TestPipeline(unittest.TestCase):
    def test_pipeline_partial_fit_predict(self):
        X = np.array([[0.0, np.nan], [1.0, 2.0], [3.0, 4.0], [4.0, 5.0]])
        y = np.array([0, 0, 1, 1])
        pipe = StreamingPipeline([
            ("imputer", StreamingImputer()),
            ("scaler", StreamingStandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=2)),
        ])
        pipe.partial_fit(X[:2], y[:2])
        pipe.partial_fit(X[2:], y[2:])
        preds = pipe.predict(X)
        self.assertEqual(preds.shape, (4,))

    def test_named_steps(self):
        pipe = StreamingPipeline([("model", DecisionTreeClassifier())])
        self.assertIn("model", pipe.named_steps)

    def test_invalid_steps(self):
        with self.assertRaises(ValueError):
            StreamingPipeline([])


if __name__ == "__main__":
    unittest.main()
