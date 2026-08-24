from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, TABLES_DIR


def format_value(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "--"
    return f"{val:.3f}"


def _fmt_cell(val: float | None, *, bold: bool = False) -> str:
    text = format_value(val)
    if text == "--":
        return text
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _best_masks(sub: pd.DataFrame) -> dict[str, set[int]]:
    masks: dict[str, set[int]] = {}
    n_rows = len(sub)
    if n_rows == 0:
        return masks

    min_metrics = ["mae", "rmse"]
    max_metrics = ["r2"]

    for metric in min_metrics:
        vals = pd.to_numeric(sub[metric], errors="coerce")
        if vals.notna().any():
            best = vals.min()
            masks[metric] = set(vals.index[np.isclose(vals, best, atol=1e-12)])
        else:
            masks[metric] = set()

    for metric in max_metrics:
        vals = pd.to_numeric(sub[metric], errors="coerce")
        if vals.notna().any():
            best = vals.max()
            masks[metric] = set(vals.index[np.isclose(vals, best, atol=1e-12)])
        else:
            masks[metric] = set()

    coverage_targets = {"coverage_90": 0.90, "coverage_95": 0.95}
    for metric, target in coverage_targets.items():
        vals = pd.to_numeric(sub[metric], errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            masks[metric] = set()
            continue
        diffs = (valid - target).abs()
        best_diff = diffs.min()
        masks[metric] = set(diffs.index[np.isclose(diffs, best_diff, atol=1e-12)])

    return masks


def build_table(df: pd.DataFrame, target: str) -> str:
    sub = df[df["target"] == target].copy()
    order = ["ridge", "rf", "gpr", "gpr_numeric"]
    sub["model"] = pd.Categorical(sub["model"], categories=order, ordered=True)
    sub = sub.sort_values("model").reset_index(drop=True)
    best_masks = _best_masks(sub)
    rows = []
    for idx, row in sub.iterrows():
        rows.append(
            [
                str(row["model"]).replace("_", "\\_"),
                _fmt_cell(row["mae"], bold=idx in best_masks.get("mae", set())),
                _fmt_cell(row["rmse"], bold=idx in best_masks.get("rmse", set())),
                _fmt_cell(row["r2"], bold=idx in best_masks.get("r2", set())),
                _fmt_cell(
                    row["coverage_90"],
                    bold=idx in best_masks.get("coverage_90", set()),
                ),
                _fmt_cell(
                    row["coverage_95"],
                    bold=idx in best_masks.get("coverage_95", set()),
                ),
            ]
        )

    header = "\\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{$R^2$} & \\textbf{Cov90} & \\textbf{Cov95} \\\\"
    lines = ["\\begin{tabular}{lccccc}", "\\toprule", header, "\\midrule"]
    for model, mae, rmse, r2, cov90, cov95 in rows:
        lines.append(f"{model} & {mae} & {rmse} & {r2} & {cov90} & {cov95} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None or pd.isna(mean):
        return "--"
    if std is None or pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def build_design_insight_table(data: dict) -> str:
    rows = data.get("rows", [])
    header = "\\textbf{Target} & \\textbf{X7 min} & \\textbf{X7 max} & \\textbf{$\\hat{y}$(min)} & \\textbf{$\\hat{y}$(max)} & \\textbf{$\\Delta$} \\\\"
    lines = ["\\begin{tabular}{lccccc}", "\\toprule", header, "\\midrule"]
    for row in rows:
        y_min = _format_mean_std(row.get("y_min"), row.get("y_min_std"))
        y_max = _format_mean_std(row.get("y_max"), row.get("y_max_std"))
        delta = format_value(row.get("delta"))
        lines.append(
            f"{row.get('target')} & {row.get('x7_min'):.2f} & {row.get('x7_max'):.2f} & {y_min} & {y_max} & {delta} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    results_path = OUTPUTS_DIR / "metrics" / "results_table.csv"
    out_dir = TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)

    targets = [str(t) for t in df["target"].unique()]
    protected_tables = {"results_repeated"}
    for existing in out_dir.glob("results_*.tex"):
        stem = existing.stem
        if stem in protected_tables:
            continue
        existing_target = stem.replace("results_", "")
        if existing_target not in targets:
            existing.unlink()

    for target in targets:
        table_tex = build_table(df, target)
        out_path = out_dir / f"results_{target}.tex"
        out_path.write_text(table_tex)

    design_path = OUTPUTS_DIR / "metrics" / "design_insight.json"
    design_table_path = out_dir / "design_insight.tex"
    if design_path.exists():
        data = json.loads(design_path.read_text())
        design_table_path.write_text(build_design_insight_table(data))
    elif design_table_path.exists():
        design_table_path.unlink()


if __name__ == "__main__":
    main()
