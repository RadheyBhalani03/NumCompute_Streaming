import os
import tempfile
import unittest
import numpy as np

from numcompute_stream.io import load_csv, train_test_split, make_stream_chunks


class TestIO(unittest.TestCase):
    def test_load_csv_splits_target(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("a,b,label\n1,2,0\n3,4,1\n")
            path = f.name
        try:
            X, y = load_csv(path)
            self.assertEqual(X.shape, (2, 2))
            np.testing.assert_array_equal(y, np.array([0.0, 1.0]))
        finally:
            os.remove(path)

    def test_load_csv_missing_value(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("a,b,label\n1,,0\n3,4,1\n")
            path = f.name
        try:
            X, _ = load_csv(path)
            self.assertTrue(np.isnan(X[0, 1]))
        finally:
            os.remove(path)

    def test_load_csv_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("missing_file.csv")

    def test_train_test_split_shapes(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 10)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 10)

    def test_make_stream_chunks(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        chunks = list(make_stream_chunks(X, y, chunk_size=4))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
