#!/usr/bin/env python3
"""
Standalone plotting script: per-model heatmap grid for prediction performance.

Variant: only Min / Max / Mean columns.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42



plt.rcParams["font.family"] = "serif"
def apply_icml_text_style() -> None:
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


apply_icml_text_style()


INPUT_DIR = Path(
    "data/prediction/01_held_out_words"
)
OUTPUT_DIR = Path(
    "output"
)

PROMPT_ORDER = ["averaged", "template", "forced_choice", "free_association"]
PROMPT_LABEL = {
    "averaged": "Averaged",
    "template": "Meaning",
    "forced_choice": "Task (FC)",
    "free_association": "Task (FA)",
}

METRICS = ["min", "max", "mean"]
METRIC_LABEL = {
    "min": "Min",
    "max": "Max",
    "mean": "Mean",
}

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct-2407",
}


def _apply_model_renames(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    out = df.copy()
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out

# =============================================================================
# Visual settings (tweak here)
# =============================================================================

FONT_SCALE = 1.2
TITLE_FONTSIZE = 12
TITLE_PAD = 8
X_TICK_FONTSIZE = 12
Y_TICK_FONTSIZE = 12
FULL_VALUE_FONTSIZE = 13
BASE_VALUE_FONTSIZE = 11
COLORBAR_LABEL_FONTSIZE = 10

FIG_WIDTH_PER_COL = 3.0
FIG_HEIGHT_PER_ROW = 3.0
WSPACE = 0.10
HSPACE = 0.30
BOTTOM_PAD = 0.12
COLORBAR_FRACTION = 0.020
COLORBAR_PAD = 0.02


def _collect_latest_per_prompt_results(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        if model_dir.name == "01_summary":
            continue
        for prompt_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            candidates = sorted(
                prompt_dir.glob("ridge_predict_held_out_words_cosine_centered_*.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                continue
            newest = candidates[-1]
            try:
                df = pd.read_csv(newest)
            except Exception:
                continue
            required = {"model", "prompt", "layer", "r2_full", "r2_cross_add"}
            if not required.issubset(set(df.columns)):
                continue
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            "No per-layer prediction CSVs found. Expected files like "
            f"`{results_root}/<MODEL>/<PROMPT>/ridge_predict_held_out_words_cosine_centered_*.csv`."
        )

    return pd.concat(rows, ignore_index=True)


def _prompt_columns_present(df: pd.DataFrame) -> list[str]:
    present = list(df["prompt"].astype(str).unique())
    return [p for p in PROMPT_ORDER if p in present]


@dataclass(frozen=True)
class SummaryCell:
    full: float
    base: float


def _summarize_curve(full_vals: np.ndarray, base_vals: np.ndarray) -> dict[str, SummaryCell]:
    if full_vals.size == 0:
        return {m: SummaryCell(np.nan, np.nan) for m in METRICS}

    idx_max = int(np.argmax(full_vals))
    idx_min = int(np.argmin(full_vals))

    return {
        "max": SummaryCell(float(full_vals[idx_max]), float(base_vals[idx_max])),
        "min": SummaryCell(float(full_vals[idx_min]), float(base_vals[idx_min])),
        "mean": SummaryCell(float(np.mean(full_vals)), float(np.mean(base_vals))),
    }


def _fmt_no_leading_zero(val: float, decimals: int = 2) -> str:
    if not np.isfinite(val):
        return ""
    s = f"{val:.{decimals}f}"
    if s.startswith("-0."):
        return s.replace("-0.", "-.", 1)
    if s.startswith("0."):
        return s.replace("0.", ".", 1)
    return s


def plot_model_performance_grid(full_df: pd.DataFrame, out_pdf: Path) -> None:
    required = {"model", "prompt", "layer", "r2_full", "r2_cross_add"}
    missing = required - set(full_df.columns)
    if missing:
        if {"peak_r2_full", "mean_r2_cross"}.issubset(set(full_df.columns)):
            raise ValueError(
                "You passed a summary CSV (e.g. `summary_prediction_analysis_*.csv`). "
                "This plot needs per-layer results with columns: "
                f"{sorted(required)}."
            )
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    full_df = _apply_model_renames(full_df)

    sns.set_theme(style="whitegrid", context="paper", font_scale=FONT_SCALE)
    plt.rcParams["font.family"] = "serif"

    apply_icml_text_style()

    models = sorted(full_df["model"].astype(str).unique())
    prompts = _prompt_columns_present(full_df)
    if not prompts:
        raise ValueError("No known prompts found in input dataframe.")

    prompt_labels = [PROMPT_LABEL[p] for p in prompts]

    model_to_full_grid: dict[str, pd.DataFrame] = {}
    model_to_base_grid: dict[str, pd.DataFrame] = {}

    global_min = np.inf
    global_max = -np.inf

    for model in models:
        rows_full = []
        rows_base = []
        for prompt in prompts:
            subset = (
                full_df[(full_df["model"] == model) & (full_df["prompt"] == prompt)]
                .sort_values("layer")
            )
            if subset.empty:
                summary = {m: SummaryCell(np.nan, np.nan) for m in METRICS}
            else:
                full_vals = subset["r2_full"].to_numpy(dtype=float)
                base_vals = subset["r2_cross_add"].to_numpy(dtype=float)
                summary = _summarize_curve(full_vals, base_vals)

            row_full = {m: summary[m].full for m in METRICS}
            row_base = {m: summary[m].base for m in METRICS}

            vals = np.array(list(row_full.values()), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                global_min = min(global_min, float(vals.min()))
                global_max = max(global_max, float(vals.max()))

            rows_full.append(row_full)
            rows_base.append(row_base)

        df_full = pd.DataFrame(rows_full, index=prompt_labels)[METRICS]
        df_base = pd.DataFrame(rows_base, index=prompt_labels)[METRICS]
        model_to_full_grid[model] = df_full
        model_to_base_grid[model] = df_base

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 0.0, 1.0

    n_cols = 4
    n_rows = int(np.ceil(len(models) / n_cols))

    fig_w = FIG_WIDTH_PER_COL * n_cols
    fig_h = FIG_HEIGHT_PER_ROW * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE, bottom=BOTTOM_PAD)

    x_ticklabels = [METRIC_LABEL[m] for m in METRICS]

    for i, model in enumerate(models):
        ax = axes[i]
        df_full = model_to_full_grid[model]
        df_base = model_to_base_grid[model]

        sns.heatmap(
            df_full,
            ax=ax,
            annot=False,
            cmap="viridis",
            vmin=global_min,
            vmax=global_max,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
        )

        for y in range(df_full.shape[0]):
            for x in range(df_full.shape[1]):
                val_full = float(df_full.iloc[y, x])
                if not np.isfinite(val_full):
                    continue
                val_base = float(df_base.iloc[y, x])

                norm_val = (val_full - global_min) / (global_max - global_min) if global_max > global_min else 0.5
                text_color = "white" if norm_val < 0.60 else "black"

                ax.text(
                    x + 0.5,
                    y + 0.40,
                    r"\textbf{" + _fmt_no_leading_zero(val_full, decimals=2) + r"}",
                    color=text_color,
                    ha="center",
                    va="center",
                    fontsize=FULL_VALUE_FONTSIZE,
                )
                ax.text(
                    x + 0.5,
                    y + 0.75,
                    f"({_fmt_no_leading_zero(val_base, decimals=2)})"
                    if np.isfinite(val_base)
                    else "",
                    color=text_color,
                    ha="center",
                    va="center",
                    fontsize=BASE_VALUE_FONTSIZE,
                )

        ax.set_title(r"\textbf{" + model + r"}", fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(x_ticklabels, rotation=0, fontsize=X_TICK_FONTSIZE)
        ax.tick_params(axis="x", pad=2)

        if i % n_cols != 0:
            ax.set_yticks([])
        else:
            ax.set_yticklabels(prompt_labels, rotation=0, fontsize=Y_TICK_FONTSIZE)

    for j in range(len(models), len(axes)):
        axes[j].axis("off")

    norm = plt.Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.tolist(),
        orientation="vertical",
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
    )
    cbar.set_label("R² Score (Full Model)", fontsize=COLORBAR_LABEL_FONTSIZE)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help=(
            "Optional: path to a CSV that already contains per-layer results "
            "(must include: model,prompt,layer,r2_full,r2_cross_add). "
            "If omitted, the script will collect the newest per-layer outputs from --results-root."
        ),
    )
    ap.add_argument(
        "--results-root",
        type=str,
        default=str(INPUT_DIR),
        help="Root directory containing per-model/prompt per-layer outputs.",
    )
    ap.add_argument(
        "--out-pdf",
        type=str,
        default=None,
        help="Output PDF path. Defaults to output/05_plots/02_prediction/01_held_out_words/rr_model_performance_grid_2x4.pdf",
    )
    args = ap.parse_args()

    if args.out_pdf is None:
        out_pdf = OUTPUT_DIR / "rr_model_performance_grid_2x4.pdf"
    else:
        out_pdf = Path(args.out_pdf)

    if args.input_csv is None:
        df = _collect_latest_per_prompt_results(Path(args.results_root))
    else:
        df = pd.read_csv(Path(args.input_csv))
    plot_model_performance_grid(df, out_pdf)
    print(f"Saved model performance grid (min/max/mean) to: {out_pdf}")


if __name__ == "__main__":
    main()
