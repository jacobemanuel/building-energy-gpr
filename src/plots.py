from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "font.size": 11,
        }
    )


def _make_axes(n_targets: int, height: float = 4.0):
    if n_targets <= 0:
        return None, []
    if n_targets == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, height))
        return fig, [ax]
    fig, axes = plt.subplots(1, n_targets, figsize=(5.2 * n_targets, height))
    return fig, list(axes)


def plot_pred_vs_true(y_true_dict: dict, y_pred_dict: dict, path: Path) -> None:
    _apply_plot_style()
    targets = sorted(y_true_dict.keys())
    if len(targets) == 0:
        return
    if len(targets) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.6))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.6))
        axes = list(np.atleast_1d(axes))
    min_val = None
    max_val = None
    for target in targets:
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        cur_min = min(y_true.min(), y_pred.min())
        cur_max = max(y_true.max(), y_pred.max())
        min_val = cur_min if min_val is None else min(min_val, cur_min)
        max_val = cur_max if max_val is None else max(max_val, cur_max)
    if min_val is None or max_val is None:
        return
    span = max_val - min_val
    pad = span * 0.05 if span > 0 else 1.0
    min_val -= pad
    max_val += pad
    for ax, target in zip(axes, targets):
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        ax.scatter(
            y_true,
            y_pred,
            s=22,
            alpha=0.8,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.plot([min_val, max_val], [min_val, max_val], color="#f58518", linewidth=2)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{target}: pred vs true")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true_dict: dict, y_pred_dict: dict, path: Path) -> None:
    _apply_plot_style()
    targets = sorted(y_true_dict.keys())
    fig, axes = _make_axes(len(targets))
    if fig is None:
        return
    all_residuals = []
    for target in targets:
        all_residuals.append(y_true_dict[target] - y_pred_dict[target])
    if all_residuals:
        residuals_all = np.concatenate(all_residuals)
        max_abs = float(np.max(np.abs(residuals_all)))
        y_min = -max_abs
        y_max = max_abs
    for ax, target in zip(axes, targets):
        y_true = y_true_dict[target]
        y_pred = y_pred_dict[target]
        residuals = y_true - y_pred
        ax.scatter(
            y_pred,
            residuals,
            s=22,
            alpha=0.75,
            color="#54a24b",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(0.0, color="#f58518", linewidth=1.6)
        if all_residuals:
            ax.set_ylim(y_min, y_max)
        ax.set_title(f"{target}: residuals")
        ax.set_xlabel("pred")
        ax.set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_standardized_residuals(z_dict: dict, path: Path) -> None:
    _apply_plot_style()
    targets = sorted(z_dict.keys())
    fig, axes = _make_axes(len(targets))
    if fig is None:
        return
    max_density = 0.0
    hist_data = {}
    for target in targets:
        z = z_dict[target]
        counts, _ = np.histogram(z, bins=20, density=True)
        max_density = max(max_density, counts.max() if len(counts) else 0.0)
        hist_data[target] = z
    for ax, target in zip(axes, targets):
        z = hist_data[target]
        ax.hist(z, bins=20, density=True, color="#9d755d", edgecolor="white", alpha=0.85)
        x = np.linspace(-4, 4, 200)
        y = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        ax.plot(x, y, color="#4c78a8", linewidth=2)
        ax.axvline(0.0, color="#f58518", linewidth=1.2, alpha=0.8)
        ax.set_xlim(-4, 4)
        if max_density > 0:
            ax.set_ylim(0, max_density * 1.1)
        ax.set_title(f"{target}: standardized residuals")
        ax.set_xlabel("z")
        ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(curve_dict: dict, path: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for target, payload in curve_dict.items():
        nominal = np.asarray(payload["nominal"])
        empirical = np.asarray(payload["empirical"])
        ax.plot(nominal, empirical, marker="o", linewidth=2, label=target)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Uncertainty reliability curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
