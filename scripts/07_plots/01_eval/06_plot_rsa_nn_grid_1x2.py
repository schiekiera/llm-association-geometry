#!/usr/bin/env python3
"""
1x2 grid: RSA mean correlations (left) + NN mean overlap (right).
Shared legend across both subplots.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(
    "data/eval"
)
OUTPUT_DIR = Path(
    "output"
)

RSA_ROOT = INPUT_DIR / "01_RSA"
RSA_LAYERWISE_PATTERN = "layerwise_correlations_pairs500000_*.csv"
RSA_PROMPTS = ["averaged", "template", "forced_choice", "free_association"]

NN_ROOT = INPUT_DIR / "02_NN"
NN_PATTERN = "layerwise_neighbors_k*_*.csv"
NN_K_TICKS = [5, 10, 20, 50, 100, 200]

# ICML Style Constants
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_XLABEL = 12
FONT_SIZE_YLABEL = 12
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12
LEGEND_LINEWIDTH = 2.5
DPI = 300

RSA_Y_LIM = (0, 0.8)
NN_Y_LIM = (0.1, 0.6)

FIG_W = 7.0
FIG_H = 4.45
PLT_LAYOUT = [0, 0.02, 1, 0.9]
BBOX_LEGEND = (0.5, 0.96)
GRID_TITLE_PAD = 0.01


def set_icml_style():
    sns.set_style("whitegrid", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5})
    sns.set_context(
        "paper",
        rc={"font.size": FONT_SIZE_LABEL, "axes.titlesize": FONT_SIZE_TITLE, "axes.labelsize": FONT_SIZE_LABEL},
    )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{times}",
                    r"\usepackage{amsmath}",
                    r"\usepackage{amssymb}",
                ]
            ),
        }
    )
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times"
    plt.rcParams["mathtext.it"] = "Times:italic"
    plt.rcParams["mathtext.bf"] = "Times:bold"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return sns.color_palette("colorblind", n_colors=5)


# =============================================================================
# RSA helpers
# =============================================================================

def _latest_eval_file(prompt_dir: Path) -> Path | None:
    candidates = sorted(prompt_dir.glob(RSA_LAYERWISE_PATTERN))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_fc_col(columns: list[str]) -> str | None:
    for col in columns:
        if col.startswith("pearson_forced_choice_"):
            return col
    for col in columns:
        if col.startswith("pearson_fc_"):
            return col
    return None


def _get_fa_col(columns: list[str]) -> str | None:
    for col in columns:
        if col.startswith("pearson_fa_"):
            return col
    return None


def _load_rsa_layerwise() -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in RSA_ROOT.iterdir() if p.is_dir()]):
        if model_dir.name == "99_old":
            continue
        for prompt in RSA_PROMPTS:
            prompt_dir = model_dir / prompt
            if not prompt_dir.exists():
                continue
            latest = _latest_eval_file(prompt_dir)
            if latest is None:
                continue
            df = pd.read_csv(latest)
            df["model"] = model_dir.name
            df["prompt"] = prompt
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No RSA layerwise files found under {RSA_ROOT}.")
    return pd.concat(rows, ignore_index=True)


def _standardize_rsa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fc_col = _get_fc_col(list(df.columns))
    fa_col = _get_fa_col(list(df.columns))
    if fc_col:
        df["pearson_fc"] = df[fc_col]
    if fa_col:
        df["pearson_fa"] = df[fa_col]
    if "pearson_fasttext" not in df.columns:
        df["pearson_fasttext"] = np.nan
    if "pearson_bert" not in df.columns:
        df["pearson_bert"] = np.nan
    if "pearson_crossmodel" not in df.columns:
        df["pearson_crossmodel"] = np.nan
    return df


def _rsa_avg_long(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_rsa(df)
    metric_cols = [
        "pearson_fc",
        "pearson_fa",
        "pearson_fasttext",
        "pearson_bert",
        "pearson_crossmodel",
    ]
    avg_df = (
        df.groupby("layer", dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    avg_long = avg_df.melt(
        id_vars=["layer"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    return avg_long


# =============================================================================
# NN helpers
# =============================================================================

def _load_nn_inputs() -> pd.DataFrame:
    frames = []
    for path in sorted(NN_ROOT.rglob(NN_PATTERN)):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        if "model" not in df.columns or "prompt" not in df.columns:
            try:
                prompt_name = path.parent.name
                model_name = path.parent.parent.name
                df["model"] = df.get("model", model_name)
                df["prompt"] = df.get("prompt", prompt_name)
            except Exception:
                pass
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No NN layerwise files found under {NN_ROOT}.")
    return pd.concat(frames, ignore_index=True)


def _nn_long(df: pd.DataFrame) -> pd.DataFrame:
    required = {"layer", "k", "model", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    metrics = [
        "nn_forced_choice_PPMI",
        "nn_fa_PPMI",
        "nn_fasttext",
        "nn_bert",
        "nn_crossmodel",
    ]
    available = [m for m in metrics if m in df.columns]
    if not available:
        raise ValueError("No NN metric columns found in inputs.")
    df_long = df.melt(
        id_vars=["layer", "k", "model", "prompt"],
        value_vars=available,
        var_name="metric",
        value_name="overlap",
    )
    return df_long


# =============================================================================
# Plotting
# =============================================================================

METRIC_LABELS = {
    "pearson_fc": r"$\mathbf{S}^{\mathrm{FC}}$",
    "pearson_fa": r"$\mathbf{S}^{\mathrm{FA}}$",
    "pearson_fasttext": r"$\mathbf{S}^{\mathrm{FT}}$",
    "pearson_bert": r"$\mathbf{S}^{\mathrm{BERT}}$",
    "pearson_crossmodel": r"$\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$",
    "nn_forced_choice_PPMI": r"$\mathbf{S}^{\mathrm{FC}}$",
    "nn_fa_PPMI": r"$\mathbf{S}^{\mathrm{FA}}$",
    "nn_fasttext": r"$\mathbf{S}^{\mathrm{FT}}$",
    "nn_bert": r"$\mathbf{S}^{\mathrm{BERT}}$",
    "nn_crossmodel": r"$\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$",
}

LABEL_ORDER = [
    r"$\mathbf{S}^{\mathrm{FC}}$",
    r"$\mathbf{S}^{\mathrm{FA}}$",
    r"$\mathbf{S}^{\mathrm{FT}}$",
    r"$\mathbf{S}^{\mathrm{BERT}}$",
    r"$\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$",
]


def _legend_handles(palette: list, labels: list[str]) -> list[Line2D]:
    color_map = {
        r"$\mathbf{S}^{\mathrm{FC}}$": palette[0],
        r"$\mathbf{S}^{\mathrm{FA}}$": palette[3],
        r"$\mathbf{S}^{\mathrm{FT}}$": palette[1],
        r"$\mathbf{S}^{\mathrm{BERT}}$": palette[2],
        r"$\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$": palette[4],
    }
    handles = []
    for label in labels:
        handles.append(Line2D([0], [0], color=color_map[label], lw=LEGEND_LINEWIDTH, label=label))
    return handles


def _plot_rsa(ax: plt.Axes, df_long: pd.DataFrame, palette: list) -> list[str]:
    df_long["metric_label"] = df_long["metric"].map(METRIC_LABELS)
    metric_labels = [l for l in LABEL_ORDER if l in set(df_long["metric_label"].dropna())]
    sns.lineplot(
        data=df_long,
        x="layer",
        y="value",
        hue="metric_label",
        hue_order=metric_labels,
        linewidth=2.0,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Layer", fontsize=FONT_SIZE_XLABEL)
    ax.set_ylabel("Mean correlation", fontsize=FONT_SIZE_YLABEL)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlim(0, 42)
    ax.set_ylim(RSA_Y_LIM)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    if ax.legend_:
        ax.legend_.remove()
    return metric_labels


def _plot_nn(ax: plt.Axes, df_long: pd.DataFrame, palette: list) -> list[str]:
    avg_df = (
        df_long.groupby(["k", "metric"], dropna=False)["overlap"]
        .mean()
        .reset_index()
    )
    avg_df["metric_label"] = avg_df["metric"].map(METRIC_LABELS)
    metric_labels = [l for l in LABEL_ORDER if l in set(avg_df["metric_label"].dropna())]
    sns.lineplot(
        data=avg_df,
        x="k",
        y="overlap",
        hue="metric_label",
        hue_order=metric_labels,
        marker="o",
        markersize=6,
        linewidth=2.0,
        palette=palette,
        ax=ax,
    )
    ax.set_xscale("log")
    present_ticks = [k for k in NN_K_TICKS if k in set(avg_df["k"].astype(int))]
    ax.set_xticks(present_ticks)
    ax.set_xticklabels([str(k) for k in present_ticks])
    ax.set_xlabel("k (log scale)", fontsize=FONT_SIZE_XLABEL)
    ax.set_ylabel("Mean overlap", fontsize=FONT_SIZE_YLABEL)
    ax.set_ylim(NN_Y_LIM)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
    if ax.legend_:
        ax.legend_.remove()
    return metric_labels


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    palette = set_icml_style()

    rsa_df = _load_rsa_layerwise()
    rsa_long = _rsa_avg_long(rsa_df)
    nn_df = _load_nn_inputs()
    nn_long = _nn_long(nn_df)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    rsa_labels = _plot_rsa(axes[0], rsa_long, palette)
    nn_labels = _plot_nn(axes[1], nn_long, palette)

    for ax, title in zip(axes, [r"\textbf{RSA}", r"\textbf{NN@k}"]):
        ax.text(
            0.5,
            1.0 + GRID_TITLE_PAD,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_TITLE,
            fontweight="bold",
        )

    shared_labels = [l for l in LABEL_ORDER if l in set(rsa_labels + nn_labels)]
    handles = _legend_handles(palette, shared_labels)

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=BBOX_LEGEND,
        ncol=len(shared_labels),
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=4.0,
    )

    plt.tight_layout(rect=PLT_LAYOUT)
    out_base = OUTPUT_DIR / "rsa_nn_grid_1x2"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", dpi=DPI)
    print(f"Plot saved to {out_base}.pdf/png")


if __name__ == "__main__":
    main()
