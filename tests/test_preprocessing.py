import unittest
import numpy as np

from numcompute_stream.preprocessing import StreamingImputer, StreamingStandardScaler


class TestPreprocessing(unittest.TestCase):
    def test_imputer_replaces_nan(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imputer = StreamingImputer().fit(X)
        Xt = imputer.transform(X)
        self.assertFalse(np.isnan(Xt).any())
        self.assertAlmostEqual(Xt[0, 1], 4.0)

    def test_imputer_across_chunks(self):
        imputer = StreamingImputer()
        imputer.partial_fit(np.array([[1.0], [3.0]]))
        imputer.partial_fit(np.array([[5.0]]))
        self.assertAlmostEqual(imputer.statistics_[0], 3.0)

    def test_imputer_transform_before_fit(self):
        with self.assertRaises(ValueError):
            StreamingImputer().transform([[1]])

    def test_scaler_zero_mean(self):
        X = np.array([[1.0], [2.0], [3.0]])
        scaler = StreamingStandardScaler().fit(X)
        Xt = scaler.transform(X)
        self.assertAlmostEqual(float(np.mean(Xt)), 0.0)

    def test_scaler_zero_variance(self):
        X = np.array([[2.0], [2.0]])
        scaler = StreamingStandardScaler().fit(X)
        Xt = scaler.transform(X)
        np.testing.assert_allclose(Xt, np.zeros_like(X))

    def test_scaler_wrong_feature_count(self):
        scaler = StreamingStandardScaler().fit(np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            scaler.transform(np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
