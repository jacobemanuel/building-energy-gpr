from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_SEED,
    FEATURES,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    TARGETS,
    TEST_SIZE,
)
from .data import coerce_dtypes, load_raw_data, split_features_targets, validate_enb2012_or_raise
from .utils import ensure_dir, save_json, seed_everything


def build_preprocessor(categorical_as_numeric: bool) -> ColumnTransformer:
    if categorical_as_numeric:
        return ColumnTransformer(
            transformers=[("num", StandardScaler(), FEATURES)],
            remainder="drop",
        )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models(seed: int):
    gpr_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    models = {
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "gpr": GaussianProcessRegressor(
            kernel=gpr_kernel,
            normalize_y=True,
            n_restarts_optimizer=2,  # keep restarts low so exact GPR stays reasonable on CPU
            random_state=seed,
        ),
    }
    return models


def train_single_target(
    X_train,
    y_train,
    model_name: str,
    estimator,
    preprocessor,
    output_dir: Path,
    target: str,
    preprocess_label: str,
) -> dict:
    pipeline = Pipeline(
        steps=[("preprocess", clone(preprocessor)), ("model", clone(estimator))],
    )
    start = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start

    model_path = output_dir / f"{target}_{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    model_step = pipeline.named_steps.get("model")
    model_info: dict[str, object] = {}
    if isinstance(model_step, RidgeCV):
        model_info["alpha"] = float(model_step.alpha_)
    if isinstance(model_step, RandomForestRegressor):
        model_info["n_estimators"] = int(model_step.n_estimators)
        model_info["max_depth"] = model_step.max_depth
    if isinstance(model_step, GaussianProcessRegressor):
        model_info["kernel"] = str(model_step.kernel_)
        model_info["log_marginal_likelihood"] = float(model_step.log_marginal_likelihood_value_)

    return {
        "target": target,
        "model": model_name,
        "model_path": str(model_path),
        "fit_seconds": float(duration),
        "preprocess": preprocess_label,
        "model_info": model_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models for Energy Efficiency dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir / "models")
    ensure_dir(args.output_dir / "metrics")

    df = load_raw_data(args.data_path)
    df = coerce_dtypes(df)
    validate_enb2012_or_raise(df)
    X, y = split_features_targets(df)

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=args.seed,
        shuffle=True,
    )
    np.savez(args.output_dir / "metrics" / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    preprocessor_cat = build_preprocessor(categorical_as_numeric=False)
    preprocessor_num = build_preprocessor(categorical_as_numeric=True)
    models = build_models(args.seed)

    train_log = []
    targets = args.targets or TARGETS
    for target in targets:
        for model_name, estimator in models.items():
            info = train_single_target(
                X_train,
                y_train[target],
                model_name,
                estimator,
                preprocessor_cat,
                args.output_dir / "models",
                target,
                "categorical",
            )
            train_log.append(info)

        # simple ablation: let GPR treat the categories as numeric
        info = train_single_target(
            X_train,
            y_train[target],
            "gpr_numeric",
            build_models(args.seed)["gpr"],
            preprocessor_num,
            args.output_dir / "models",
            target,
            "numeric",
        )
        train_log.append(info)

    save_json(
        {
            "seed": args.seed,
            "test_size": TEST_SIZE,
            "features": FEATURES,
            "targets": targets,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        args.output_dir / "metrics" / "train_config.json",
    )
    save_json({"runs": train_log}, args.output_dir / "metrics" / "train_log.json")


if __name__ == "__main__":
    main()
