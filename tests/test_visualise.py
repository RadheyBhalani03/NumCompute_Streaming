import unittest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from numcompute_stream.visualise import plot_metric_history, plot_model_comparison, plot_confusion_matrix


class TestVisualise(unittest.TestCase):
    def test_plot_metric_history_returns_fig_ax(self):
        fig, ax = plot_metric_history([0.5, 0.7, 0.8])
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_model_comparison_returns_fig_ax(self):
        fig, ax = plot_model_comparison([0.5, 0.7], [0.6, 0.8])
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_confusion_matrix_returns_fig_ax(self):
        fig, ax = plot_confusion_matrix(np.array([[1, 0], [0, 1]]), class_labels=[0, 1])
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_plot_model_comparison_length_mismatch(self):
        with self.assertRaises(ValueError):
            plot_model_comparison([0.5], [0.6, 0.7])


if __name__ == "__main__":
    unittest.main()
