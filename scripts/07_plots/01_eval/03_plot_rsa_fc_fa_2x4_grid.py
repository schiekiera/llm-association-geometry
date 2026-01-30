#!/usr/bin/env python3
"""
Standalone RSA plot: 2x4 grid of model panels, each with FC/FA sub-heatmaps.
Uses latest summary_all_results_pairs500000_*.csv by mtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================================================
# Configuration
# =============================================================================

# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


TIMES_FONT_PATH = Path("/System/Library/Fonts/Times.ttc")
if TIMES_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(TIMES_FONT_PATH))
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
    "data/eval/01_RSA"
)
SUMMARY_PATTERN = "summary_all_results_pairs500000_*.csv"
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
METRIC_ORDER = ["Min", "Max", "Mean"]

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct-2407",
}


def _apply_model_renames(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    out = df.copy()
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out

# Layout & style
N_COLS = 4
PANEL_WSPACE = 0.35
PANEL_HSPACE = 0.35
SUBPLOT_WSPACE = 0.15
FIG_W = 4.9 * N_COLS
FIG_H = 3.3 * 2
CBAR_FRACTION = 0.015
CBAR_PAD = 0.02

MAIN_TITLE_SIZE = 18
SUB_TITLE_SIZE = 20
FONT_SIZE_XTICK = 13
FONT_SIZE_YTICK = 13
FONT_SIZE_ANNOT = 15
FONT_SIZE_CBAR = 18
FONT_SIZE_CBAR_TICK = 13
TITLE_Y_OFFSET = 0.07

# Color scale defaults (used only if data-derived bounds are not finite)
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 0.6


@dataclass(frozen=True)
class SummaryCell:
    min: float
    max: float
    mean: float


def _latest_summary_csv() -> Path:
    candidates = sorted(INPUT_DIR.glob(SUMMARY_PATTERN))
    if not candidates:
        raise FileNotFoundError(f"No files found matching {SUMMARY_PATTERN} in {INPUT_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_prefixed_col(group: pd.DataFrame, prefixes: list[str]) -> str | None:
    candidates = [c for c in group.columns if any(c.startswith(p) for p in prefixes)]
    if not candidates:
        return None
    non_nan_counts = [(c, int(group[c].notna().sum())) for c in candidates]
    non_nan_counts.sort(key=lambda x: x[1], reverse=True)
    best_col, best_n = non_nan_counts[0]
    return best_col if best_n > 0 else None


def _fmt_no_leading_zero(val: float, decimals: int = 2) -> str:
    if not np.isfinite(val):
        return ""
    s = f"{val:.{decimals}f}"
    if s.startswith("-0."):
        return s.replace("-0.", "-.", 1)
    if s.startswith("0."):
        return s.replace("0.", ".", 1)
    return s


def _summarize_group(group: pd.DataFrame, col: str) -> SummaryCell:
    vals = group[col].astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return SummaryCell(np.nan, np.nan, np.nan)
    return SummaryCell(float(np.min(vals)), float(np.max(vals)), float(np.mean(vals)))


def _compute_tables(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    df = df[df["layer"].astype(int) > 0].copy()
    models = sorted(df["model"].astype(str).unique())
    tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    rows: List[dict] = []

    for model in models:
        tables[model] = {}
        for target, prefixes in [
            ("FC", ["pearson_forced_choice_", "pearson_fc_"]),
            ("FA", ["pearson_fa_"]),
        ]:
            data = []
            for prompt in PROMPT_ORDER:
                group = df[(df["model"] == model) & (df["prompt"] == prompt)]
                if group.empty:
                    cell = SummaryCell(np.nan, np.nan, np.nan)
                else:
                    col = _resolve_prefixed_col(group, prefixes)
                    if col is None:
                        cell = SummaryCell(np.nan, np.nan, np.nan)
                    else:
                        cell = _summarize_group(group, col)
                data.append([cell.min, cell.max, cell.mean])

                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "prompt": prompt,
                        "metric": "min",
                        "value": cell.min,
                    }
                )
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "prompt": prompt,
                        "metric": "max",
                        "value": cell.max,
                    }
                )
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "prompt": prompt,
                        "metric": "mean",
                        "value": cell.mean,
                    }
                )

            df_table = pd.DataFrame(data, index=[PROMPT_LABEL[p] for p in PROMPT_ORDER], columns=METRIC_ORDER)
            tables[model][target] = df_table

    csv_df = pd.DataFrame(rows)
    return tables, csv_df


def _global_vmin_vmax(tables: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[float, float]:
    vals = []
    for model in tables:
        for target in tables[model]:
            arr = tables[model][target].to_numpy(dtype=float).ravel()
            vals.append(arr)
    if not vals:
        return DEFAULT_VMIN, DEFAULT_VMAX
    all_vals = np.concatenate(vals)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return DEFAULT_VMIN, DEFAULT_VMAX
    return float(np.min(all_vals)), float(np.max(all_vals))


def plot_grid() -> Path:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    apply_icml_text_style()
    df = _apply_model_renames(pd.read_csv(_latest_summary_csv()))
    tables, csv_df = _compute_tables(df)
    vmin, vmax = _global_vmin_vmax(tables)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = DEFAULT_VMIN, DEFAULT_VMAX

    models = sorted(tables.keys())
    n_rows = int(np.ceil(len(models) / N_COLS))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(n_rows, N_COLS, wspace=PANEL_WSPACE, hspace=PANEL_HSPACE)

    axes_all = []
    model_axes: List[Tuple[str, plt.Axes, plt.Axes]] = []

    for idx, model in enumerate(models):
        row = idx // N_COLS
        col = idx % N_COLS
        sub = outer[row, col].subgridspec(1, 2, wspace=SUBPLOT_WSPACE)
        ax_left = fig.add_subplot(sub[0, 0])
        ax_right = fig.add_subplot(sub[0, 1])

        for ax, target in [(ax_left, "FC"), (ax_right, "FA")]:
            table = tables[model][target]
            annot_df = table.map(lambda v: _fmt_no_leading_zero(float(v), decimals=2))
            sns.heatmap(
                table,
                ax=ax,
                annot=annot_df,
                fmt="",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                square=True,
                linewidths=0.5,
                linecolor="white",
                annot_kws={"fontsize": FONT_SIZE_ANNOT},
            )
            ax.set_title(
                "$\mathbf{S}^{\mathrm{FC}}$" if target == "FC" else "$\mathbf{S}^{\mathrm{FA}}$",
                fontsize=SUB_TITLE_SIZE,
                fontweight="bold",
                pad=8,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels(METRIC_ORDER, rotation=0, fontsize=FONT_SIZE_XTICK)
            ax.tick_params(axis="x", pad=2)
            if ax is ax_left and col == 0:
                ax.set_yticklabels(table.index, rotation=0, fontsize=FONT_SIZE_YTICK)
            else:
                ax.set_yticklabels([])

        # Panel title centered above the two subplots
        model_axes.append((model, ax_left, ax_right))
        axes_all.extend([ax_left, ax_right])

    # Shared colorbar
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes_all,
        orientation="vertical",
        fraction=CBAR_FRACTION,
        pad=CBAR_PAD,
    )
    cbar.set_label("Pearson $r$", fontsize=FONT_SIZE_CBAR)
    cbar.ax.tick_params(labelsize=FONT_SIZE_CBAR_TICK)

    # Panel titles centered above each FC/FA pair (use final axis positions)
    for model, ax_left, ax_right in model_axes:
        pos_left = ax_left.get_position()
        pos_right = ax_right.get_position()
        center_x = (pos_left.x0 + pos_right.x1) / 2
        top_y = max(pos_left.y1, pos_right.y1)
        fig.text(
            center_x,
            top_y + TITLE_Y_OFFSET,
            r"\textbf{" + model + r"}",
            ha="center",
            va="bottom",
            fontsize=MAIN_TITLE_SIZE,
            fontweight="bold",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUTPUT_DIR / "rsa_fc_fa_2x4_grid.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return out_pdf


def main() -> None:
    out_pdf = plot_grid()
    print(f"Saved RSA grid to: {out_pdf}")


if __name__ == "__main__":
    main()
