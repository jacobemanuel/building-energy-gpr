from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, TABLES_DIR, TARGETS

MODEL_ORDER = ["ridge", "rf", "gpr", "gpr_numeric"]
TARGET_ORDER = TARGETS


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:02d}"


def _format_pm(mean: float | None, std: float | None, *, bold: bool = False) -> str:
    if mean is None or std is None:
        return "--"
    text = f"{mean:.3f} $\\pm$ {std:.3f}"
    if bold:
        return f"\\textbf{{{text}}}"
    return text


def _best_masks(grouped: pd.DataFrame) -> dict[tuple[str, str], bool]:
    metric_rules = {
        "mae": ("min", None),
        "rmse": ("min", None),
        "r2": ("max", None),
        "coverage_90": ("closest", 0.90),
        "coverage_95": ("closest", 0.95),
    }
    best_lookup: dict[tuple[str, str], bool] = {}

    for target in grouped["target"].unique():
        target_df = grouped[grouped["target"] == target]
        for metric, (rule, ref) in metric_rules.items():
            means = pd.to_numeric(target_df[(metric, "mean")], errors="coerce")
            valid = means.dropna()
            if valid.empty:
                continue
            if rule == "min":
                best_val = valid.min()
                best_rows = valid.index[valid.eq(best_val)]
            elif rule == "max":
                best_val = valid.max()
                best_rows = valid.index[valid.eq(best_val)]
            else:
                diffs = (valid - float(ref)).abs()
                best_diff = diffs.min()
                best_rows = diffs.index[diffs.eq(best_diff)]

            for row_idx in best_rows:
                best_lookup[(str(row_idx), metric)] = True

    return best_lookup


def aggregate_results(repeats_dir: Path, out_tex: Path, seed_dirs: list[Path] | None = None) -> None:
    rows = []
    if seed_dirs is None:
        seed_dirs = sorted(repeats_dir.glob("seed_*"))
    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "metrics" / "results_table.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        df["seed_dir"] = seed_dir.name
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No results_table.csv found under repeated split directories.")

    full = pd.concat(rows, ignore_index=True)

    metrics = ["mae", "rmse", "r2", "coverage_90", "coverage_95"]
    grouped = (
        full.groupby(["target", "model"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["target"] = pd.Categorical(grouped["target"], categories=TARGET_ORDER, ordered=True)
    grouped["model"] = pd.Categorical(grouped["model"], categories=MODEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["target", "model"]).reset_index(drop=True)
    grouped["row_id"] = grouped.index.astype(str)
    best_lookup = _best_masks(grouped)

    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Target} & \\textbf{Model} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{$R^2$} & \\textbf{Cov90} & \\textbf{Cov95} \\\\")
    lines.append("\\midrule")

    for _, row in grouped.iterrows():
        target = row[("target", "")]
        model = row[("model", "")]
        row_id = row[("row_id", "")]
        model = model.replace("_", "\\_")
        cells = []
        for metric in metrics:
            mean = row[(metric, "mean")]
            std = row[(metric, "std")]
            if pd.isna(mean) or pd.isna(std):
                cells.append("--")
            else:
                is_best = best_lookup.get((str(row_id), metric), False)
                cells.append(_format_pm(float(mean), float(std), bold=is_best))
        line = f"{target} & {model} & " + " & ".join(cells) + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated train/eval splits and aggregate results.")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR / "repeated")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    repeats_dir = args.output_root
    if args.clean and repeats_dir.exists():
        shutil.rmtree(repeats_dir)
    repeats_dir.mkdir(parents=True, exist_ok=True)
    run_seed_dirs: list[Path] = []

    for i in range(args.repeats):
        seed = args.start_seed + i
        out_dir = _seed_dir(repeats_dir, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_seed_dirs.append(out_dir)

        train_cmd = [
            "python3",
            "-m",
            "src.train",
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
        ]
        if args.data_path:
            train_cmd += ["--data-path", str(args.data_path)]
        _run(train_cmd)

        eval_cmd = [
            "python3",
            "-m",
            "src.evaluate",
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
        ]
        if args.data_path:
            eval_cmd += ["--data-path", str(args.data_path)]
        _run(eval_cmd)

    aggregate_results(
        repeats_dir,
        TABLES_DIR / "results_repeated.tex",
        seed_dirs=run_seed_dirs,
    )


if __name__ == "__main__":
    main()
