import unittest

import numpy as np

from src.config import FEATURES, TARGETS
from src.data import (
    coerce_dtypes,
    load_raw_data,
    split_features_targets,
    validate_data,
    validate_enb2012_or_raise,
)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = coerce_dtypes(load_raw_data())

    def test_required_columns_present(self) -> None:
        expected = set(FEATURES + TARGETS)
        self.assertTrue(expected.issubset(set(self.df.columns)))

    def test_split_features_targets_shapes(self) -> None:
        x, y = split_features_targets(self.df)
        self.assertEqual(list(x.columns), FEATURES)
        self.assertEqual(list(y.columns), TARGETS)
        self.assertEqual(len(x), len(y))
        self.assertGreater(len(x), 700)

    def test_data_validation_basic(self) -> None:
        info = validate_data(self.df)
        self.assertEqual(info["missing_total"], 0)
        self.assertEqual(info["duplicate_rows"], 0)
        self.assertEqual(info["shape"][1], 10)

    def test_validation_gate_rejects_missing_values(self) -> None:
        broken = self.df.copy()
        broken.loc[0, "X1"] = np.nan
        with self.assertRaises(ValueError):
            validate_enb2012_or_raise(broken)

    def test_validation_gate_rejects_invalid_categorical_levels(self) -> None:
        broken = self.df.copy()
        broken.loc[0, "X6"] = 99
        with self.assertRaises(ValueError):
            validate_enb2012_or_raise(broken)


if __name__ == "__main__":
    unittest.main()
