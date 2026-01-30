#!/usr/bin/env python3
"""
Plot a 1x2 grid of RSA layerwise correlations for FC and FA.
Each panel shows 4 prompt lines (mean across models).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42



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


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = Path(
    "data/eval/01_RSA"
)
RSA_LAYERWISE_PATTERN = "layerwise_correlations_pairs500000_*.csv"
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

GEOMETRIES = [
    ("fc", r"\textbf{Forced Choice} ($\mathbf{S}^{\mathrm{FC}}$)"),
    ("fa", r"\textbf{Free Association} ($\mathbf{S}^{\mathrm{FA}}$)"),
]

RSA_YLIM = (0.0, 0.6)
FIG_W = 7.6
FIG_H = 3.6
WSPACE = 0.25
HSPACE = 0.3

FONT_SIZE_TITLE = 14
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 12

LEGEND_BBOX_TO_ANCHOR = (0.5, 1.09)

@dataclass(frozen=True)
class RsaCurve:
    layer: np.ndarray
    value: np.ndarray


def _latest_file(root: Path, pattern: str) -> Path | None:
    candidates = sorted(root.rglob(pattern))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _resolve_prefixed_col(columns: List[str], prefixes: List[str]) -> str | None:
    candidates = [c for c in columns if any(c.startswith(p) for p in prefixes)]
    if not candidates:
        return None
    return candidates[0]


def _load_rsa_layerwise() -> pd.DataFrame | None:
    rows = []
    for model_dir in sorted([p for p in INPUT_DIR.iterdir() if p.is_dir()]):
        if model_dir.name == "99_old":
            continue
        for prompt in PROMPT_ORDER:
            prompt_dir = model_dir / prompt
            if not prompt_dir.exists():
                continue
            latest = _latest_file(prompt_dir, RSA_LAYERWISE_PATTERN)
            if latest is None:
                continue
            df = pd.read_csv(latest)
            df["model"] = model_dir.name
            df["prompt"] = prompt
            rows.append(df)
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _rsa_metric_column(df: pd.DataFrame, geom: str) -> str | None:
    cols = list(df.columns)
    if geom == "fc":
        return _resolve_prefixed_col(cols, ["pearson_forced_choice_", "pearson_fc_"])
    if geom == "fa":
        return _resolve_prefixed_col(cols, ["pearson_fa_"])
    return None


def _rsa_curves(df: pd.DataFrame) -> Dict[str, Dict[str, RsaCurve]]:
    df = df.copy()
    df = df[df["layer"].astype(int) > 0]
    curves: Dict[str, Dict[str, RsaCurve]] = {g: {} for g, _ in GEOMETRIES}

    for geom, _ in GEOMETRIES:
        col = _rsa_metric_column(df, geom)
        if col is None:
            continue
        for prompt in PROMPT_ORDER:
            subset = df[df["prompt"] == prompt]
            if subset.empty:
                continue
            per_model = subset.groupby(["model", "layer"], as_index=False)[col].mean()
            per_layer = per_model.groupby("layer", as_index=False)[col].mean()
            curves[geom][prompt] = RsaCurve(
                layer=per_layer["layer"].to_numpy(dtype=int),
                value=per_layer[col].to_numpy(dtype=float),
            )
    return curves


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    apply_icml_text_style()
    palette = sns.color_palette("colorblind", n_colors=len(PROMPT_ORDER))

    rsa_df = _load_rsa_layerwise()
    if rsa_df is None:
        raise FileNotFoundError("No RSA layerwise CSVs found.")

    rsa_curves = _rsa_curves(rsa_df)

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H), gridspec_kw={"wspace": WSPACE, "hspace": HSPACE}
    )

    for col_idx, (geom, title) in enumerate(GEOMETRIES):
        ax = axes[col_idx]
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=8)
        if geom not in rsa_curves or not rsa_curves[geom]:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=FONT_SIZE_LABEL)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            continue
        for i, prompt in enumerate(PROMPT_ORDER):
            if prompt not in rsa_curves[geom]:
                continue
            curve = rsa_curves[geom][prompt]
            ax.plot(curve.layer, curve.value, color=palette[i], lw=1.6, label=PROMPT_LABEL[prompt])
        ax.set_ylim(RSA_YLIM)
        ax.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_xlim(0, 42)
        ax.tick_params(axis="x", labelsize=FONT_SIZE_XTICK)
        if col_idx == 0:
            ax.set_ylabel("Pearson r", fontsize=FONT_SIZE_LABEL)
            ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICK)
        else:
            ax.set_yticklabels([])

    handles = [
        plt.Line2D([0], [0], color=palette[i], lw=2.0, label=PROMPT_LABEL[p])
        for i, p in enumerate(PROMPT_ORDER)
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        ncol=4,
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=3.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rsa_line_plot_1x2_grid_fc_fa.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
