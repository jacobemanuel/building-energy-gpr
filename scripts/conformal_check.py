from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# lets this run as "python3 scripts/conformal_check.py" from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SEED, OUTPUTS_DIR, TARGETS
from src.data import coerce_dtypes, load_raw_data, split_features_targets, validate_enb2012_or_raise
from src.train import build_models, build_preprocessor
from src.utils import ensure_dir, save_json, seed_everything


def _split_conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal quantile for two-sided intervals."""
    n = len(abs_residuals)
    if n == 0:
        raise ValueError("Calibration residuals are empty.")
    level = math.ceil((n + 1) * coverage) / n
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(abs_residuals, level, method="higher"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal calibration check for GPR.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--cal-size", type=float, default=0.2)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.output_dir / "metrics")
    ensure_dir(args.output_dir / "tables")

    df = load_raw_data(args.data_path)
    df = coerce_dtypes(df)
    validate_enb2012_or_raise(df)
    X, y = split_features_targets(df)

    split_path = args.output_dir / "metrics" / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing split indices at {split_path}. Run training first."
        )
    _split = np.load(split_path)
    train_idx, test_idx = _split["train_idx"], _split["test_idx"]
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    train_sub_idx, cal_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=args.cal_size,
        random_state=args.seed,
        shuffle=True,
    )

    X_sub = X_train.iloc[train_sub_idx]
    y_sub = y_train.iloc[train_sub_idx]
    X_cal = X_train.iloc[cal_idx]
    y_cal = y_train.iloc[cal_idx]

    preprocessor = build_preprocessor(categorical_as_numeric=False)
    gpr = build_models(args.seed)["gpr"]

    rows = []
    for target in TARGETS:
        model = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", clone(gpr)),
            ]
        )
        model.fit(X_sub, y_sub[target])

        cal_pred = model.predict(X_cal)
        cal_res = np.abs(y_cal[target].to_numpy() - cal_pred)

        q90 = _split_conformal_quantile(cal_res, 0.90)
        q95 = _split_conformal_quantile(cal_res, 0.95)

        test_pred = model.predict(X_test)
        y_true = y_test[target].to_numpy()

        cov90 = float(np.mean((y_true >= test_pred - q90) & (y_true <= test_pred + q90)))
        cov95 = float(np.mean((y_true >= test_pred - q95) & (y_true <= test_pred + q95)))

        rows.append(
            {
                "target": target,
                "q90": q90,
                "q95": q95,
                "cov90": cov90,
                "cov95": cov95,
            }
        )

    out_json = args.output_dir / "metrics" / "conformal_check.json"
    save_json({"seed": args.seed, "cal_size": args.cal_size, "rows": rows}, out_json)

    df_out = pd.DataFrame(rows)
    tex_path = args.output_dir / "tables" / "conformal_coverage.tex"
    lines = []
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Target} & \\textbf{$q_{0.90}$} & \\textbf{$q_{0.95}$} & \\textbf{Cov90} & \\textbf{Cov95} & \\textbf{Gap90} & \\textbf{Gap95} \\\\")
    lines.append("\\midrule")
    for _, row in df_out.iterrows():
        gap90 = row["cov90"] - 0.90
        gap95 = row["cov95"] - 0.95
        lines.append(
            f"{row['target']} & {row['q90']:.3f} & {row['q95']:.3f} & {row['cov90']:.3f} & {row['cov95']:.3f} & {gap90:+.3f} & {gap95:+.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
