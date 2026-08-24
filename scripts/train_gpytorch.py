from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.backends.gpytorch_backend import GPyTorchConfig, predict, train_exact_gp
from src.config import CATEGORICAL_FEATURES, DEFAULT_SEED, NUMERIC_FEATURES, OUTPUTS_DIR, TARGETS, TEST_SIZE
from src.data import coerce_dtypes, load_raw_data, split_features_targets
from src.metrics import regression_metrics
from src.utils import ensure_dir, save_json, seed_everything


def _build_preprocessor() -> ColumnTransformer:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _load_split_indices(path: Path, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        data = np.load(path)
        return data["train_idx"], data["test_idx"]

    # Use the same split method as src/train.py for consistency.
    indices = np.arange(n_samples)
    return train_test_split(
        indices, test_size=TEST_SIZE, random_state=seed, shuffle=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPyTorch backend (optional, large-scale).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target", choices=TARGETS, default="Y1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir / "metrics")
    ensure_dir(args.output_dir / "predictions")

    df = load_raw_data(args.data_path)
    df = coerce_dtypes(df)
    X, y = split_features_targets(df)

    split_path = args.output_dir / "metrics" / "split_indices.npz"
    train_idx, test_idx = _load_split_indices(split_path, len(df), args.seed)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx][args.target]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx][args.target]

    preprocessor = _build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    config = GPyTorchConfig(epochs=args.epochs, lr=args.lr, use_cuda=args.use_cuda)
    model, likelihood, device = train_exact_gp(X_train_t, y_train.to_numpy(), config)
    y_pred, y_std = predict(model, likelihood, X_test_t, device)

    metrics = regression_metrics(y_test.to_numpy(), y_pred)
    save_json(
        {
            "target": args.target,
            "metrics": metrics,
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "use_cuda": args.use_cuda,
            },
        },
        args.output_dir / "metrics" / f"gpytorch_{args.target}.json",
    )

    pred_path = args.output_dir / "predictions" / f"{args.target}_gpytorch.csv"
    np.savetxt(
        pred_path,
        np.column_stack([y_test.to_numpy(), y_pred, y_std]),
        delimiter=",",
        header="y_true,y_pred,y_std",
        comments="",
    )

    print(f"Saved metrics to {args.output_dir / 'metrics' / f'gpytorch_{args.target}.json'}")
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
