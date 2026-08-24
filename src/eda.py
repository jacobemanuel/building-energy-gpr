from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    FIGURES_DIR,
    TARGETS,
    X6_LEVELS,
    X8_LEVELS,
)
from .data import coerce_dtypes, load_raw_data, validate_data
from .data import validate_enb2012_or_raise
from .utils import ensure_dir, save_json


def _set_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        font="serif",
        rc={
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        },
    )


def _plot_numeric_histograms(df: pd.DataFrame, path: Path) -> None:
    _set_plot_style()
    numeric_cols = [col for col in NUMERIC_FEATURES + TARGETS if col in df.columns]
    discrete_cols = []
    discrete_data = {}
    for col in numeric_cols:
        values = df[col].dropna()
        if values.nunique() <= 15:
            counts = values.value_counts().sort_index()
            shares = counts / counts.sum()
            labels = [f"{val:g}" for val in counts.index]
            discrete_cols.append(col)
            discrete_data[col] = (labels, shares.values)
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 2.8 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        values = df[col].dropna()
        if col in discrete_cols:
            labels, shares = discrete_data[col]
            x = list(range(len(labels)))
            ax.bar(x, shares, color="#4c78a8", edgecolor="white", linewidth=0.5)
            if len(labels) > 8:
                tick_idx = x[::2]
                ax.set_xticks(tick_idx)
                ax.set_xticklabels([labels[i] for i in tick_idx], rotation=45)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
            ax.margins(x=0.02)
        else:
            bins = min(20, max(2, values.nunique()))
            sns.histplot(
                values,
                bins=bins,
                stat="probability",
                common_norm=False,
                color="#4c78a8",
                edgecolor="white",
                ax=ax,
            )
            if values.nunique() > 10:
                ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("share")
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(numeric_cols) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_categorical_counts(df: pd.DataFrame, path: Path) -> None:
    _set_plot_style()
    n_cols = 2
    n_rows = (len(CATEGORICAL_FEATURES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 4.2 * n_rows))
    axes = axes.flatten()
    max_share = 0.0

    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        if col == "X6":
            full_index = X6_LEVELS
        elif col == "X8":
            full_index = X8_LEVELS
        else:
            min_cat = int(df[col].min())
            max_cat = int(df[col].max())
            full_index = list(range(min_cat, max_cat + 1))
        counts = df[col].value_counts().sort_index().reindex(full_index, fill_value=0)
        shares = counts / counts.sum()
        max_share = max(max_share, shares.max())
        sns.barplot(x=counts.index.astype(str), y=shares.values * 100.0, color="#f58518", ax=ax)
        ax.set_title(col)
        ax.set_ylabel("share (%)")
        ax.set_xlabel("category")
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")
        sns.despine(ax=ax, top=True, right=True)

    for ax in axes[len(CATEGORICAL_FEATURES) :]:
        ax.axis("off")

    for ax in axes[: len(CATEGORICAL_FEATURES)]:
        ax.set_ylim(0, max_share * 100.0 * 1.1)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    _set_plot_style()
    cols = [col for col in NUMERIC_FEATURES + TARGETS if col in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_by_feature(
    df: pd.DataFrame,
    path: Path,
    hue_col: str,
    palette: str,
    title: str,
    legend_title: str,
) -> None:
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.scatterplot(
        data=df,
        x="Y1",
        y="Y2",
        hue=hue_col,
        palette=palette,
        s=35,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Y1 (Heating Load)")
    ax.set_ylabel("Y2 (Cooling Load)")
    ax.legend(title=legend_title, loc="lower right")
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_eda(data_path: Path | None = None) -> None:
    ensure_dir(OUTPUTS_DIR / "metrics")
    ensure_dir(FIGURES_DIR)

    df = load_raw_data(data_path)
    df = coerce_dtypes(df)

    validation = validate_data(df)
    save_json(validation, OUTPUTS_DIR / "metrics" / "data_validation.json")
    validate_enb2012_or_raise(df)

    describe = df.describe(include="all").T
    describe.to_csv(OUTPUTS_DIR / "metrics" / "data_describe.csv")

    _plot_numeric_histograms(df, FIGURES_DIR / "feature_target_histograms.png")
    _plot_categorical_counts(df, FIGURES_DIR / "categorical_counts.png")
    _plot_correlation_heatmap(df, FIGURES_DIR / "correlation_heatmap.png")
    _plot_joint_by_feature(
        df,
        FIGURES_DIR / "target_joint_by_x6.png",
        hue_col="X6",
        palette="viridis",
        title="Y1 vs Y2 by X6 (Orientation)",
        legend_title="X6",
    )
    _plot_joint_by_feature(
        df,
        FIGURES_DIR / "target_joint_by_x8.png",
        hue_col="X8",
        palette="magma",
        title="Y1 vs Y2 by X8 (Glazing Dist.)",
        legend_title="X8",
    )


if __name__ == "__main__":
    run_eda()
