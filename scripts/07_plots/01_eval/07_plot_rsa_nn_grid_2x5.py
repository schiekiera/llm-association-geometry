#!/usr/bin/env python3
"""
Standalone 2x5 grid: RSA layer profiles (row 1) + NN@k profiles (row 2).
Each column is a reference geometry; each subplot has 4 prompt lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    "data/eval"
)
RSA_ROOT = INPUT_DIR / "01_RSA"
RSA_SUMMARY_PATTERN = "summary_all_results_pairs500000_*.csv"
RSA_LAYERWISE_PATTERN = "layerwise_correlations_pairs500000_*.csv"

NN_ROOT = INPUT_DIR / "02_NN"
NN_SUMMARY_PATTERN = "summary_nn_per_k_*.csv"
NN_FALLBACK_PATTERN = "summary_all_neighbors_k*_*.csv"
NN_K_ORDER = [5, 10, 20, 50, 100, 200]
NN_LOG_TICKS = [5, 10, 20, 50, 100, 200]

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
    ("fasttext", r"\textbf{FastText} ($\mathbf{S}^{\mathrm{FT}}$)"),
    ("bert", r"\textbf{BERT} ($\mathbf{S}^{\mathrm{BERT}}$)"),
    ("crossmodel", r"\textbf{Cross-Model} ($\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$)"),
]

RSA_YLIM = (0, 0.9)
NN_YLIM = (0.1, 0.62)
RSA_YTICKS = np.arange(0.0, 1.0, 0.1)
NN_YTICKS = np.arange(0.1, 0.62, 0.1)

FIG_W = 15
FIG_H = 6.8
WSPACE = 0.25
HSPACE = 0.35

FONT_SIZE_TITLE = 16
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 14


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


def _load_rsa_summary() -> pd.DataFrame | None:
    latest = _latest_file(RSA_ROOT, RSA_SUMMARY_PATTERN)
    if latest is None:
        return None
    return pd.read_csv(latest)


def _load_rsa_layerwise() -> pd.DataFrame | None:
    rows = []
    for model_dir in sorted([p for p in RSA_ROOT.iterdir() if p.is_dir()]):
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
    if geom == "fasttext":
        return "pearson_fasttext" if "pearson_fasttext" in cols else None
    if geom == "bert":
        return "pearson_bert" if "pearson_bert" in cols else None
    if geom == "crossmodel":
        return "pearson_crossmodel" if "pearson_crossmodel" in cols else None
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
            # mean per model per layer, then average across models
            per_model = (
                subset.groupby(["model", "layer"], as_index=False)[col].mean()
            )
            per_layer = per_model.groupby("layer", as_index=False)[col].mean()
            curves[geom][prompt] = RsaCurve(
                layer=per_layer["layer"].to_numpy(dtype=int),
                value=per_layer[col].to_numpy(dtype=float),
            )
    return curves


def _load_nn_summary() -> pd.DataFrame | None:
    latest = _latest_file(NN_ROOT, NN_SUMMARY_PATTERN)
    if latest is not None:
        return pd.read_csv(latest)

    # Fallback: aggregate per-layer summaries into per-k means
    fallback = sorted(NN_ROOT.rglob(NN_FALLBACK_PATTERN))
    if not fallback:
        return None
    frames = []
    for path in fallback:
        df = pd.read_csv(path)
        if "model" not in df.columns or "prompt" not in df.columns:
            df["prompt"] = df.get("prompt", path.parent.name)
            df["model"] = df.get("model", path.parent.parent.name)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    return df_all


def _nn_metric_column(df: pd.DataFrame, geom: str) -> str | None:
    cols = list(df.columns)
    if geom == "fc":
        return _resolve_prefixed_col(cols, ["nn_forced_choice_"])
    if geom == "fa":
        return _resolve_prefixed_col(cols, ["nn_fa_"])
    if geom == "fasttext":
        if "nn_fasttext" in cols:
            return "nn_fasttext"
        return "nn_fasttext_mean" if "nn_fasttext_mean" in cols else None
    if geom == "bert":
        if "nn_bert" in cols:
            return "nn_bert"
        return "nn_bert_mean" if "nn_bert_mean" in cols else None
    if geom == "crossmodel":
        if "nn_crossmodel" in cols:
            return "nn_crossmodel"
        return "nn_crossmodel_mean" if "nn_crossmodel_mean" in cols else None
    return None


def _nn_curves(df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {g: {} for g, _ in GEOMETRIES}

    for geom, _ in GEOMETRIES:
        col = _nn_metric_column(df, geom)
        if col is None:
            continue
        for prompt in PROMPT_ORDER:
            subset = df[df["prompt"] == prompt]
            if subset.empty:
                continue
            per_k = subset.groupby("k", as_index=False)[col].mean()
            curves[geom][prompt] = (
                per_k["k"].to_numpy(dtype=int),
                per_k[col].to_numpy(dtype=float),
            )
    return curves


def _plot_unavailable(ax: plt.Axes, message: str = "Not available") -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=FONT_SIZE_LABEL)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    apply_icml_text_style()
    palette = sns.color_palette("colorblind", n_colors=len(PROMPT_ORDER))

    rsa_df = _load_rsa_summary()
    if rsa_df is None:
        rsa_df = _load_rsa_layerwise()
    if rsa_df is None:
        raise FileNotFoundError("No RSA summary or layerwise CSVs found.")

    nn_df = _load_nn_summary()
    if nn_df is None:
        raise FileNotFoundError("No NN summary_nn_per_k_*.csv found.")

    rsa_curves = _rsa_curves(rsa_df)
    nn_curves = _nn_curves(nn_df)

    fig, axes = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), gridspec_kw={"wspace": WSPACE, "hspace": HSPACE})

    # RSA row
    max_layer = 0
    for geom, _ in GEOMETRIES:
        for prompt in rsa_curves.get(geom, {}):
            max_layer = max(max_layer, int(np.max(rsa_curves[geom][prompt].layer)))
    tick_step = 10 if max_layer > 20 else 5

    for col_idx, (geom, title) in enumerate(GEOMETRIES):
        ax = axes[0, col_idx]
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight="bold", pad=8)
        if geom not in rsa_curves or not rsa_curves[geom]:
            _plot_unavailable(ax)
            continue
        for i, prompt in enumerate(PROMPT_ORDER):
            if prompt not in rsa_curves[geom]:
                continue
            curve = rsa_curves[geom][prompt]
            ax.plot(curve.layer, curve.value, color=palette[i], lw=1.6, label=PROMPT_LABEL[prompt])
        ax.set_ylim(RSA_YLIM)
        ax.set_yticks(RSA_YTICKS)
        ax.set_xlabel("Layer", fontsize=FONT_SIZE_LABEL)
        ax.set_xticks(np.arange(1, max_layer + 1, tick_step))
        ax.tick_params(axis="x", labelsize=FONT_SIZE_XTICK)
        if col_idx == 0:
            ax.set_ylabel("Pearson r", fontsize=FONT_SIZE_LABEL)
            ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICK)
        else:
            ax.set_yticklabels([])

    # NN row
    for col_idx, (geom, title) in enumerate(GEOMETRIES):
        ax = axes[1, col_idx]
        if geom not in nn_curves or not nn_curves[geom]:
            _plot_unavailable(ax)
            continue
        ks_all = set()
        for prompt in nn_curves[geom]:
            ks_all.update(nn_curves[geom][prompt][0].tolist())
        ks_sorted = [k for k in NN_K_ORDER if k in ks_all] or sorted(ks_all)

        for i, prompt in enumerate(PROMPT_ORDER):
            if prompt not in nn_curves[geom]:
                continue
            ks, vals = nn_curves[geom][prompt]
            ax.plot(ks, vals, color=palette[i], lw=1.6, label=PROMPT_LABEL[prompt])
        ax.set_ylim(NN_YLIM)
        ax.set_yticks(NN_YTICKS)
        ax.set_xscale("log")
        ax.set_xlabel("k (log scale)", fontsize=FONT_SIZE_LABEL)
        ax.set_xticks(NN_LOG_TICKS)
        ax.set_xticklabels([str(k) for k in NN_LOG_TICKS])
        ax.tick_params(axis="x", labelsize=FONT_SIZE_XTICK)
        if col_idx == 0:
            ax.set_ylabel("NN@k overlap", fontsize=FONT_SIZE_LABEL)
            ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICK)
        else:
            ax.set_yticklabels([])

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color=palette[i], lw=2.0, label=PROMPT_LABEL[p])
        for i, p in enumerate(PROMPT_ORDER)
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        handlelength=3.5,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rsa_nn_grid_2x5.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
