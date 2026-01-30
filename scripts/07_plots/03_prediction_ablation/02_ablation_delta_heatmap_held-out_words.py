#!/usr/bin/env python3
"""
Standalone script: Delta ($\Delta R^2$) summary heatmap.

Creates a heatmap with:
- rows: models
- columns: paradigms (FC, FA, FC+FA)
- square cells
- 3-decimal annotations

Input: `summary_prediction_analysis_*.csv` (model, prompt, delta_fc,
delta_fa, delta_full).
We average deltas across prompts per model and order models by delta_full descending.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm


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
    "data/prediction/02_held_out_words_ablation/01_summary"
)
OUTPUT_DIR = Path(
    "output"
)

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct-2407",
}


def _apply_model_renames(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    out = df.copy()
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out


def _find_latest_summary_csv(summary_dir: Path) -> Path | None:
    candidates = sorted(summary_dir.rglob("summary_prediction_analysis_*.csv"))
    return candidates[-1] if candidates else None


def _fmt_no_leading_zero(val: float, decimals: int = 3) -> str:
    if not np.isfinite(val):
        return ""
    s = f"{val:.{decimals}f}"
    if s.startswith("-0."):
        return s.replace("-0.", "-.", 1)
    if s.startswith("0."):
        return s.replace("0.", ".", 1)
    return s


def plot_delta_heatmap(summary_df: pd.DataFrame, out_pdf: Path) -> None:
    required = {
        "model",
        "prompt",
        "delta_fc",
        "delta_fa",
        "delta_full",
    }
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {sorted(missing)}")

    summary_df = _apply_model_renames(summary_df)

    # Mean deltas per model across prompts
    mean_deltas = (
        summary_df
        .groupby(
            "model",
            as_index=False,
        )[
            [
                "delta_fc",
                "delta_fa",
                "delta_full",
            ]
        ]
        .mean()
    )
    mean_deltas = mean_deltas[~mean_deltas["model"].str.upper().eq("MEAN")]

    # Order models by delta_full (desc)
    mean_deltas = mean_deltas.sort_values(
        "delta_full", ascending=False
    ).reset_index(drop=True)
    models = mean_deltas["model"].tolist()

    hm_df = pd.DataFrame(
        {
            "FC+FA": mean_deltas["delta_full"].to_numpy(dtype=float),
            "FC": mean_deltas["delta_fc"].to_numpy(dtype=float),
            "FA": mean_deltas["delta_fa"].to_numpy(dtype=float),
        },
        index=models,
    )

    # Symmetric scale around 0, with extremes at ±abs_max
    delta_vals = hm_df.to_numpy(dtype=float).ravel()
    delta_vals = delta_vals[np.isfinite(delta_vals)]
    abs_max = float(np.nanmax(np.abs(delta_vals))) if delta_vals.size else 0.0
    abs_max = abs_max if abs_max > 0 else 1e-6

    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap = plt.get_cmap("PuOr_r")  # reversed to make orange positive, purple negative

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    apply_icml_text_style()

    # height scales with number of models; width fixed (3 cols)
    fig_h = max(6.0, 0.6 * len(models))
    fig, ax = plt.subplots(1, 1, figsize=(6.0, fig_h))

    annot_df = hm_df.applymap(lambda v: _fmt_no_leading_zero(float(v), decimals=3))
    sns.heatmap(
        hm_df,
        ax=ax,
        cmap=cmap,
        norm=norm,
        cbar=True,
        annot=annot_df,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": r"$\Delta R^2$"},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to summary_prediction_analysis_*.csv. If omitted, uses latest under the summary directory.",
    )
    ap.add_argument(
        "--out-pdf",
        type=str,
        default=None,
        help="Output PDF path. Default: ablation_rr_delta_heatmap.pdf in the plots directory.",
    )
    args = ap.parse_args()

    if args.summary_csv is None:
        latest = _find_latest_summary_csv(INPUT_DIR)
        if latest is None:
            raise FileNotFoundError(f"No summary_prediction_analysis_*.csv found under: {INPUT_DIR}")
        summary_csv = latest
    else:
        summary_csv = Path(args.summary_csv)

    if args.out_pdf is None:
        out_pdf = OUTPUT_DIR / "ablation_rr_delta_heatmap.pdf"
    else:
        out_pdf = Path(args.out_pdf)

    df = pd.read_csv(summary_csv)
    plot_delta_heatmap(df, out_pdf)
    print(f"Saved delta heatmap to: {out_pdf}")


if __name__ == "__main__":
    main()

