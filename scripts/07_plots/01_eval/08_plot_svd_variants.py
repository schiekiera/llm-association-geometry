#!/usr/bin/env python3
"""
Plot mean RSA correlations across SVD variants for FC/FA.

Two lines per panel (Counts vs PPMI), averaged across models, prompts, layers.
"""

from __future__ import annotations

import argparse
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

ROOT = Path("projects/icml_project")
INPUT_DIR = Path(
    "data/eval/03_SVD_variants"
)
OUTPUT_DIR = Path(
    "output"
)

TASK_ORDER = ["ForcedChoice", "FreeAssociation"]
TASK_LABELS = {
    "ForcedChoice": r"\textbf{Forced Choice} ($\mathbf{S}^{\mathrm{FC}}$)",
    "FreeAssociation": r"\textbf{Free Association} ($\mathbf{S}^{\mathrm{FA}}$)",
}

LABEL_PPMI = "$\mathbf{S}^p_{\mathrm{PPMI}}$ from PPMI-weighted $\mathbf{B}^p$"
LABEL_COUNTS = "$\mathbf{S}^p_{\mathrm{Counts}}$ from Raw Counts $\mathbf{B}^p$"

VARIANT_ORDER = [
    ("svd100_counts", 100, "Counts"),
    ("svd300_counts", 300, "Counts"),
    ("svd600_counts", 600, "Counts"),
    ("counts", 1000, "Counts"),
    ("svd100_ppmi", 100, "PPMI"),
    ("svd300_ppmi", 300, "PPMI"),
    ("svd600_ppmi", 600, "PPMI"),
    ("ppmi", 1000, "PPMI"),
]

X_TICKS = [100, 300, 600, 1000]
X_LABELS = ["100", "300", "600", "None"]

# ICML Style Constants
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 12
FONT_SIZE_XLABEL = 12
FONT_SIZE_YLABEL = 12
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12
LEGEND_LINEWIDTH = 2.5
LINE_WIDTH = 2.0
MARKER_STYLE = "o"
MARKER_SIZE = 6
DPI = 300
Y_LIM = (0.1, 0.45)
PLT_LAYOUT = [0, 0.03, 1, 0.9]
BBOX_LEGEND = (0.5, 1.01)
TITLE_PAD = 12


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
    return sns.color_palette("colorblind")


def _find_inputs(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(input_dir.rglob(pattern))


def _load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    required = {"task", "variant", "pearson"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    var_info = pd.DataFrame(VARIANT_ORDER, columns=["variant", "svd_dim", "kind"])
    df = df.merge(var_info, on="variant", how="inner")
    if df.empty:
        raise ValueError("No matching variants found in inputs.")

    grouped = (
        df.groupby(["task", "variant", "svd_dim", "kind"], dropna=False)["pearson"]
        .mean()
        .reset_index()
    )
    return grouped


def plot_svd_variants(df: pd.DataFrame, out_path: Path) -> None:
    palette = set_icml_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)

    if len(TASK_ORDER) == 1:
        axes = [axes]

    color_map = {"Counts": palette[0], "PPMI": palette[3]}
    legend_label_map = {"Counts": LABEL_COUNTS, "PPMI": LABEL_PPMI}
    for ax, task in zip(axes, TASK_ORDER):
        subset = df[df["task"] == task]
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_axis_off()
            continue

        for kind in ["Counts", "PPMI"]:
            line_df = subset[subset["kind"] == kind].copy()
            line_df = line_df.sort_values("svd_dim")
            ax.plot(
                line_df["svd_dim"],
                line_df["pearson"],
                label=legend_label_map.get(kind, kind),
                color=color_map[kind],
                lw=LINE_WIDTH,
                marker=MARKER_STYLE,
                markersize=MARKER_SIZE,
            )

        ax.set_title(TASK_LABELS.get(task, task), fontweight="bold", pad=TITLE_PAD)
        ax.set_xlabel("SVD regime of $\mathbf{B}^p$", fontsize=FONT_SIZE_XLABEL)
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels(X_LABELS)
        ax.set_ylim(Y_LIM)
        ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK)

    axes[0].set_ylabel("Mean Pearson r", fontsize=FONT_SIZE_YLABEL)

    if axes[0].legend_:
        axes[0].legend_.remove()
    if axes[1].legend_:
        axes[1].legend_.remove()

    legend_handles = [
        Line2D([0], [0], color=color_map["Counts"], lw=LEGEND_LINEWIDTH, label=LABEL_COUNTS),
        Line2D([0], [0], color=color_map["PPMI"], lw=LEGEND_LINEWIDTH, label=LABEL_PPMI),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=BBOX_LEGEND,
        ncol=2,
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=3.5,
    )

    plt.tight_layout(rect=PLT_LAYOUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SVD variant comparisons for FC/FA.")
    parser.add_argument(
        "--input-dir",
        default=str(INPUT_DIR),
        help="Directory containing svd_variants_results_* CSVs.",
    )
    parser.add_argument(
        "--pattern",
        default="svd_variants_results_*_pairs*.csv",
        help="Glob pattern for input CSVs within --input-dir (searched recursively).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional explicit list of input CSV files (overrides --input-dir/--pattern).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Default: svd_variants.pdf",
    )
    args = parser.parse_args()

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = _find_inputs(Path(args.input_dir), args.pattern)

    if not input_paths:
        raise FileNotFoundError("No SVD variant CSVs found. Check --input-dir/--pattern.")

    df = _load_inputs(input_paths)
    if df.empty:
        raise ValueError("Input files were empty or unreadable.")

    agg = _aggregate(df)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / "svd_variants.pdf"

    plot_svd_variants(agg, out_path)


if __name__ == "__main__":
    main()
