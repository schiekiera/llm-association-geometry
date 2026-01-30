#!/usr/bin/env python3
"""
NN@k line-plot grid: rows=models, columns=prompts.
Each subplot shows 5 geometry lines averaged across layers.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, NullFormatter

# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = "output"
INPUT_DIR = "data/eval/02_NN"
FILE_PATTERN = "layerwise_neighbors_k*_*.csv"
PROMPT_NAMES = ["averaged", "template", "forced_choice", "free_association"]
K_ORDER = [5, 10, 20, 50, 100, 200]

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct",
}

# ICML Style Constants
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_XLABEL = 12
FONT_SIZE_YLABEL = 12
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 13
LEGEND_LINEWIDTH = 2.5
LINE_WIDTH = 2.0
DPI = 300
Y_LIM_DEFAULT = (0.0, 1.0)
PLT_LAYOUT = [0, 0.03, 1, 0.9]
BBOX_LEGEND = (0.5, 0.93)



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


def _metric_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = list(df.columns)
    fc_col = next((c for c in cols if c.startswith("nn_forced_choice_")), None)
    fa_col = next((c for c in cols if c.startswith("nn_fa_")), None)
    fasttext_col = "nn_fasttext" if "nn_fasttext" in cols else None
    bert_col = "nn_bert" if "nn_bert" in cols else None
    cross_col = "nn_crossmodel" if "nn_crossmodel" in cols else None
    return {
        "fc": fc_col,
        "fa": fa_col,
        "fasttext": fasttext_col,
        "bert": bert_col,
        "cross": cross_col,
    }


def _collect_prompt_data(prompt_dir: Path) -> pd.DataFrame | None:
    rows = []
    for path in sorted(prompt_dir.glob(FILE_PATTERN)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        k_val = int(df["k"].iloc[0]) if "k" in df.columns else None
        if k_val is None:
            continue
        cols = _metric_columns(df)
        if not cols["fc"] or not cols["fa"]:
            continue
        row = {
            "k": k_val,
            "nn_fc": float(df[cols["fc"]].mean()),
            "nn_fa": float(df[cols["fa"]].mean()),
            "nn_fasttext": float(df[cols["fasttext"]].mean()) if cols["fasttext"] else np.nan,
            "nn_bert": float(df[cols["bert"]].mean()) if cols["bert"] else np.nan,
            "nn_cross": float(df[cols["cross"]].mean()) if cols["cross"] else np.nan,
        }
        rows.append(row)
    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values("k")
    return out


def _collect_all() -> pd.DataFrame:
    rows = []
    for model_dir in sorted([p for p in Path(INPUT_DIR).iterdir() if p.is_dir()]):
        for prompt in PROMPT_NAMES:
            prompt_dir = model_dir / prompt
            if not prompt_dir.exists():
                continue
            df = _collect_prompt_data(prompt_dir)
            if df is None:
                continue
            df["model"] = model_dir.name
            df["prompt"] = prompt
            rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No NN layerwise files found under {INPUT_DIR}.")
    out = pd.concat(rows, ignore_index=True)
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out


def _compute_y_limits(df_all: pd.DataFrame) -> tuple[float, float]:
    value_cols = ["nn_fc", "nn_fa", "nn_fasttext", "nn_bert", "nn_cross"]
    values = df_all[value_cols].to_numpy().ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Y_LIM_DEFAULT
    return float(values.min()), float(values.max())


def plot_grid(df_all: pd.DataFrame, palette: list) -> plt.Figure:
    models = sorted(df_all["model"].unique())
    n_rows = len(models)
    n_cols = len(PROMPT_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), sharex=True, sharey=True)
    y_min, y_max = _compute_y_limits(df_all)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    title_map = {
        "averaged": r"\textbf{Averaged}",
        "template": r"\textbf{Meaning}",
        "forced_choice": r"\textbf{Task (FC)}",
        "free_association": r"\textbf{Task (FA)}",
    }

    for i, model in enumerate(models):
        for j, prompt in enumerate(PROMPT_NAMES):
            ax = axes[i, j]
            df = df_all[(df_all["model"] == model) & (df_all["prompt"] == prompt)]
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax.set_axis_off()
                continue

            df = df.sort_values("k")
            ax.plot(
                df["k"],
                df["nn_fc"],
                label="FC",
                color=palette[0],
                lw=LINE_WIDTH,
            )
            ax.plot(
                df["k"],
                df["nn_fa"],
                label="FA",
                color=palette[3],
                lw=LINE_WIDTH,
                alpha=0.95,
            )
            ax.plot(
                df["k"],
                df["nn_fasttext"],
                label="FastText",
                color=palette[1],
                lw=LINE_WIDTH,
            )
            ax.plot(
                df["k"],
                df["nn_bert"],
                label="BERT",
                color=palette[2],
                lw=LINE_WIDTH,
            )
            if df["nn_cross"].notna().any():
                ax.plot(
                    df["k"],
                    df["nn_cross"],
                    label="Cross-model",
                    color=palette[4],
                    lw=LINE_WIDTH,
                )

            if i == 0:
                ax.set_title(title_map.get(prompt, prompt.replace("_", " ").title()), fontweight="bold")
            if j == 0:
                ax.set_ylabel(model[:25] + "..." if len(model) > 25 else model, fontweight="bold", fontsize=FONT_SIZE_YLABEL)
            if i == n_rows - 1:
                ax.set_xlabel("k (log scale)", fontsize=FONT_SIZE_XLABEL)
            present_ticks = [k for k in K_ORDER if k in set(df["k"].astype(int))]

            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(present_ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in present_ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            if i != n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)
            sns.despine(ax=ax, trim=False)

    legend_handles = [
        Line2D([0], [0], color=palette[0], lw=LEGEND_LINEWIDTH, label="Forced Choice ($\mathbf{S}^{\mathrm{FC}}$)"),
        Line2D([0], [0], color=palette[3], lw=LEGEND_LINEWIDTH, label="Free Association ($\mathbf{S}^{\mathrm{FA}}$)"),
        Line2D([0], [0], color=palette[1], lw=LEGEND_LINEWIDTH, label="FastText ($\mathbf{S}^{\\mathrm{FT}}$)"),
        Line2D([0], [0], color=palette[2], lw=LEGEND_LINEWIDTH, label="BERT ($\mathbf{S}^{\\mathrm{BERT}}$)"),
        Line2D([0], [0], color=palette[4], lw=LEGEND_LINEWIDTH, label="Cross-model ($\mathbf{S}^{\\mathrm{X}}$)"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=BBOX_LEGEND,
        ncol=5,
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=4.0,
    )
    plt.tight_layout(rect=PLT_LAYOUT)
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    palette = set_icml_style()
    df_all = _collect_all()
    fig = plot_grid(df_all, palette)
    out_base = os.path.join(OUTPUT_DIR, "nn_line_plot_8x4_grid")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", dpi=DPI)
    print(f"Plot saved to {out_base}.pdf/png")


if __name__ == "__main__":
    main()
