import unittest

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.data import coerce_dtypes, load_raw_data, split_features_targets
from src.train import build_models, build_preprocessor


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        df = coerce_dtypes(load_raw_data())
        x, y = split_features_targets(df)
        cls.x_train = x.iloc[:120].copy()
        cls.y_train = y.iloc[:120].copy()
        cls.x_test = x.iloc[120:140].copy()

    def test_ridge_pipeline_fit_predict(self) -> None:
        pre = build_preprocessor(categorical_as_numeric=False)
        ridge = build_models(seed=42)["ridge"]
        pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(ridge))])
        pipe.fit(self.x_train, self.y_train["Y1"])
        pred = pipe.predict(self.x_test)
        self.assertEqual(pred.shape[0], len(self.x_test))
        self.assertTrue(np.isfinite(pred).all())

    def test_gpr_pipeline_returns_std(self) -> None:
        pre = build_preprocessor(categorical_as_numeric=False)
        gpr = build_models(seed=42)["gpr"]
        pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(gpr))])
        pipe.fit(self.x_train, self.y_train["Y1"])

        x_t = pipe.named_steps["preprocess"].transform(self.x_test)
        y_mean, y_std = pipe.named_steps["model"].predict(x_t, return_std=True)
        self.assertEqual(y_mean.shape[0], len(self.x_test))
        self.assertEqual(y_std.shape[0], len(self.x_test))
        self.assertTrue((y_std >= 0).all())


if __name__ == "__main__":
    unittest.main()
