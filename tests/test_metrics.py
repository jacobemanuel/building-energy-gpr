import unittest

import numpy as np

from src.metrics import interval_coverage, regression_metrics, standardized_residuals


class MetricsTests(unittest.TestCase):
    def test_regression_metrics_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.5, 2.5, 4.5])

        out = regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(out["mae"], 0.375, places=6)
        self.assertAlmostEqual(out["rmse"], np.sqrt(0.1875), places=6)
        self.assertTrue(0.8 < out["r2"] <= 1.0)

    def test_interval_coverage_and_standardized_residuals(self) -> None:
        y_true = np.array([10.0, 11.0, 9.5, 10.5])
        y_mean = np.array([10.1, 10.9, 9.6, 10.4])
        y_std = np.array([0.5, 0.5, 0.4, 0.6])

        cov = interval_coverage(y_true, y_mean, y_std, z=1.96)
        z = standardized_residuals(y_true, y_mean, y_std)

        self.assertGreaterEqual(cov, 0.75)
        self.assertEqual(z.shape, y_true.shape)
        self.assertTrue(np.isfinite(z).all())


if __name__ == "__main__":
    unittest.main()
