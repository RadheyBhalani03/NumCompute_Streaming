import unittest
import numpy as np

from numcompute_stream.metrics import (
    accuracy_score,
    error_rate,
    confusion_matrix,
    precision,
    recall,
    f1_score,
    mse,
    StreamingAccuracy,
    StreamingConfusionMatrix,
)


class TestMetrics(unittest.TestCase):
    def test_accuracy_score(self):
        self.assertAlmostEqual(accuracy_score([1, 0, 1], [1, 1, 1]), 2 / 3)

    def test_error_rate(self):
        self.assertAlmostEqual(error_rate([1, 0, 1], [1, 1, 1]), 1 / 3)

    def test_confusion_matrix(self):
        cm = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), labels=[0, 1])
        np.testing.assert_array_equal(cm, np.array([[1, 1], [0, 1]]))

    def test_precision_recall_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        self.assertGreaterEqual(precision(y_true, y_pred), 0)
        self.assertGreaterEqual(recall(y_true, y_pred), 0)
        self.assertGreaterEqual(f1_score(y_true, y_pred), 0)

    def test_mse(self):
        self.assertAlmostEqual(mse([1, 2], [1, 4]), 2.0)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            accuracy_score([1, 2], [1])

    def test_streaming_accuracy(self):
        metric = StreamingAccuracy()
        metric.update(np.array([1, 0]), np.array([1, 1]))
        metric.update(np.array([1, 1]), np.array([1, 1]))
        self.assertAlmostEqual(metric.compute(), 0.75)
        self.assertEqual(len(metric.history_), 2)

    def test_streaming_confusion_matrix(self):
        metric = StreamingConfusionMatrix(labels=[0, 1])
        metric.update(np.array([0, 1]), np.array([0, 1]))
        np.testing.assert_array_equal(metric.compute(), np.array([[1, 0], [0, 1]]))


if __name__ == "__main__":
    unittest.main()
