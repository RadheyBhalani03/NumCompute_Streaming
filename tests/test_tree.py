import unittest
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTree(unittest.TestCase):
    def test_tree_fits_simple_dataset(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        preds = clf.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_predict_shape(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        self.assertEqual(clf.predict(X).shape, (4,))

    def test_pure_class(self):
        X = np.array([[0], [1], [2]], dtype=float)
        y = np.array([1, 1, 1])
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
        np.testing.assert_array_equal(clf.predict(X), y)

    def test_max_depth_zero_majority(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
        preds = clf.predict(X)
        self.assertTrue(np.all(preds == 0))

    def test_tie_resolution_chooses_smaller_label(self):
        X = np.array([[0], [1]], dtype=float)
        y = np.array([1, 0])
        clf = DecisionTreeClassifier(max_depth=0).fit(X, y)
        self.assertEqual(clf.predict(np.array([[5.0]]))[0], 0)

    def test_partial_fit_across_chunks(self):
        clf = DecisionTreeClassifier(max_depth=2)
        clf.partial_fit(np.array([[0], [1]], dtype=float), np.array([0, 0]))
        clf.partial_fit(np.array([[2], [3]], dtype=float), np.array([1, 1]))
        preds = clf.predict(np.array([[0], [3]], dtype=float))
        np.testing.assert_array_equal(preds, np.array([0, 1]))

    def test_nan_prediction_does_not_crash(self):
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        pred = clf.predict(np.array([[np.nan]], dtype=float))
        self.assertEqual(pred.shape, (1,))


if __name__ == "__main__":
    unittest.main()
