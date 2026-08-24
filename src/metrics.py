from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def interval_coverage(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, z: float) -> float:
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def standardized_residuals(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return (y_true - y_mean) / (y_std + 1e-8)
