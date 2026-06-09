import unittest
import numpy as np

from numcompute_stream.statistics import mean, std, histogram, quantile, RunningMean, RunningVariance


class TestStatistics(unittest.TestCase):
    def test_mean_ignores_nan(self):
        self.assertAlmostEqual(mean([1, np.nan, 3]), 2.0)

    def test_std_zero_variance(self):
        self.assertAlmostEqual(std([2, 2, 2]), 0.0)

    def test_histogram(self):
        counts, edges = histogram([1, 2, 3], bins=2)
        self.assertEqual(counts.sum(), 3)
        self.assertEqual(len(edges), 3)

    def test_quantile(self):
        self.assertAlmostEqual(quantile([1, 2, 3], 0.5), 2.0)

    def test_running_mean_multiple_chunks(self):
        rm = RunningMean()
        rm.update([[1, 2], [3, np.nan]])
        rm.update([[5, 6]])
        np.testing.assert_allclose(rm.compute(), np.array([3.0, 4.0]))

    def test_running_variance_zero_variance(self):
        rv = RunningVariance()
        rv.update([[2, 2], [2, 2]])
        np.testing.assert_allclose(rv.compute(), np.array([0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
