from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from .config import (
    DEFAULT_SEED,
    OUTPUTS_DIR,
    TARGETS,
)
from .data import coerce_dtypes, load_raw_data, split_features_targets, validate_enb2012_or_raise
from .metrics import interval_coverage, regression_metrics, standardized_residuals
from .plots import plot_pred_vs_true, plot_reliability, plot_residuals, plot_standardized_residuals
from .utils import ensure_dir, save_json


def _load_split_indices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path)
        return data["train_idx"], data["test_idx"]
    raise FileNotFoundError(
        f"Missing split indices at {path}. Run training first to create a deterministic split."
    )


def _predict_with_std(model, X):
    if hasattr(model, "named_steps") and isinstance(model.named_steps.get("model"), GaussianProcessRegressor):
        preprocess = model.named_steps["preprocess"]
        gpr = model.named_steps["model"]
        X_t = preprocess.transform(X)
        y_mean, y_std = gpr.predict(X_t, return_std=True)
        return y_mean, y_std

    try:
        y_mean, y_std = model.predict(X, return_std=True)
        return y_mean, y_std
    except TypeError:
        y_mean = model.predict(X)
        return y_mean, None


def _reliability_curve(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict:
    nominal = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    empirical = []
    for coverage in nominal:
        z = norm.ppf(0.5 + coverage / 2.0)
        empirical.append(interval_coverage(y_true, y_pred, y_std, z))
    return {"nominal": nominal.tolist(), "empirical": empirical}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=None)
    args = parser.parse_args()

    figures_dir = args.output_dir / "figures"
    ensure_dir(args.output_dir / "metrics")
    ensure_dir(args.output_dir / "predictions")
    ensure_dir(figures_dir)

    df = load_raw_data(args.data_path)
    df = coerce_dtypes(df)
    validate_enb2012_or_raise(df)
    X, y = split_features_targets(df)

    _, test_idx = _load_split_indices(args.output_dir / "metrics" / "split_indices.npz")
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    model_dir = args.output_dir / "models"
    model_paths = sorted(model_dir.glob("*.joblib"))
    targets = set(args.targets or TARGETS)
    model_paths = [p for p in model_paths if p.stem.split("_", 1)[0] in targets]
    results = []
    gpr_plot_data = {"y_true": {}, "y_pred": {}, "z": {}, "reliability": {}}

    for path in model_paths:
        name = path.stem
        if "_" not in name:
            continue
        target, model_name = name.split("_", 1)
        try:
            model = joblib.load(path)
        except Exception as exc:
            print(f"[WARNING] Could not load model {path}: {exc}. Skipping.")
            continue

        y_pred, y_std = _predict_with_std(model, X_test)
        y_true = y_test[target].to_numpy()

        metrics = regression_metrics(y_true, y_pred)
        row = {
            "target": target,
            "model": model_name,
            **metrics,
        }

        if y_std is not None:
            cov90 = interval_coverage(y_true, y_pred, y_std, 1.645)
            cov95 = interval_coverage(y_true, y_pred, y_std, 1.96)
            row["coverage_90"] = cov90
            row["coverage_95"] = cov95
            row["pred_std_mean"] = float(np.mean(y_std))
        else:
            row["coverage_90"] = None
            row["coverage_95"] = None
            row["pred_std_mean"] = None

        results.append(row)

        pred_df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_std": y_std if y_std is not None else np.nan,
            }
        )
        pred_df.to_csv(args.output_dir / "predictions" / f"{target}_{model_name}.csv", index=False)

        if model_name == "gpr":
            gpr_plot_data["y_true"][target] = y_true
            gpr_plot_data["y_pred"][target] = y_pred
            if y_std is not None:
                gpr_plot_data["z"][target] = standardized_residuals(y_true, y_pred, y_std)
                gpr_plot_data["reliability"][target] = _reliability_curve(y_true, y_pred, y_std)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "metrics" / "results_table.csv", index=False)
    results_df.to_latex(args.output_dir / "metrics" / "results_table.tex", index=False, float_format="%.3f")

    save_json({"results": results}, args.output_dir / "metrics" / "results_table.json")

    legacy_design_path = args.output_dir / "metrics" / "design_insight.json"
    if legacy_design_path.exists():
        legacy_design_path.unlink()

    if gpr_plot_data["y_true"]:
        plot_pred_vs_true(gpr_plot_data["y_true"], gpr_plot_data["y_pred"], figures_dir / "gpr_pred_vs_true.png")
        plot_residuals(gpr_plot_data["y_true"], gpr_plot_data["y_pred"], figures_dir / "gpr_residuals.png")
        if gpr_plot_data["z"]:
            plot_standardized_residuals(gpr_plot_data["z"], figures_dir / "gpr_std_residuals.png")
        if gpr_plot_data["reliability"]:
            plot_reliability(gpr_plot_data["reliability"], figures_dir / "gpr_reliability.png")


if __name__ == "__main__":
    main()
