import unittest
import numpy as np

from numcompute_stream.ensemble import RandomForestClassifier, BaggingClassifier


class TestEnsemble(unittest.TestCase):
    def test_random_forest_fits(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1).fit(X, y)
        self.assertEqual(len(rf.trees_), 3)

    def test_random_forest_predict_shape(self):
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1).fit(X, y)
        self.assertEqual(rf.predict(X).shape, (6,))

    def test_partial_fit(self):
        rf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1)
        rf.partial_fit(np.array([[0], [1]], dtype=float), np.array([0, 0]))
        rf.partial_fit(np.array([[4], [5]], dtype=float), np.array([1, 1]))
        self.assertEqual(len(rf.trees_), 3)
        self.assertEqual(rf.predict(np.array([[0], [5]], dtype=float)).shape, (2,))

    def test_feature_subsets_exist(self):
        X = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
        y = np.array([0, 0, 1, 1])
        rf = RandomForestClassifier(n_estimators=4, max_depth=2, max_features="sqrt", random_state=2).fit(X, y)
        self.assertEqual(len(rf.feature_indices_), 4)

    def test_bagging_uses_all_features(self):
        X = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=float)
        y = np.array([0, 0, 1, 1])
        model = BaggingClassifier(n_estimators=2, random_state=1).fit(X, y)
        self.assertEqual(model.feature_indices_[0].size, 2)


if __name__ == "__main__":
    unittest.main()
