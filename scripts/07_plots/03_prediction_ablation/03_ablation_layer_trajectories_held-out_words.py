#!/usr/bin/env python3
"""
Standalone script: layer trajectories for held-out words.
Plots R²_full vs layer, one subplot per prompt, colored by model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    "data/prediction/02_held_out_words_ablation"
)
OUTPUT_DIR = Path(
    "output"
)

PROMPT_ORDER = ["averaged", "template", "forced_choice", "free_association"]
PROMPT_LABEL = {
    "averaged": r"\textbf{Averaged}",
    "template": r"\textbf{Meaning}",
    "forced_choice": r"\textbf{Task (Forced Choice)}",
    "free_association": r"\textbf{Task (Free Association)}",
}

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct-2407",
}

# Style parameters
TITLE_FONT_SIZE = 33
LABEL_FONT_SIZE = 30
TICK_FONT_SIZE = 30
LEGEND_FONT_SIZE = 30
FIG_W_PER_PROMPT = 6
FIG_H = 8
LINE_WIDTH = 2
TITLE_PAD = 20


def _apply_model_renames(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    out = df.copy()
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out


def _collect_latest_per_prompt_results(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        if model_dir.name == "01_summary":
            continue
        for prompt_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            candidates = sorted(
                prompt_dir.glob("ridge_predict_held_out_words_cosine_*.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if not candidates:
                continue
            newest = candidates[-1]
            try:
                df = pd.read_csv(newest)
            except Exception:
                continue
            required = {"model", "prompt", "layer", "r2_full"}
            if not required.issubset(set(df.columns)):
                continue
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            "No per-layer prediction CSVs found. Expected files like "
            f"`{results_root}/<MODEL>/<PROMPT>/ridge_predict_held_out_words_cosine_centered_*.csv`."
        )
    combined = pd.concat(rows, ignore_index=True)
    return _apply_model_renames(combined)


def _prompt_columns_present(df: pd.DataFrame) -> list[str]:
    present = list(df["prompt"].astype(str).unique())
    return [p for p in PROMPT_ORDER if p in present]


def plot_layer_trajectories(full_df: pd.DataFrame, out_dir: Path) -> None:
    prompts = _prompt_columns_present(full_df)
    if not prompts:
        raise ValueError("No known prompts found in input dataframe.")

    # Color-blind friendly categorical colors (same as barplot)
    model_to_color = {
        "Falcon3-10B-Instruct": "#CC79A7",
        "Llama-3.1-8B-Instruct": "#E69F00",
        "Mistral-7B-Instruct-v0.2": "#56B4E9",
        "Mistral-Nemo-Instruct-2407": "#009E73",
        "Qwen2.5-7B-Instruct": "#F0E442",
        "gemma-2-9b-it": "#0072B2",
        "phi-4": "#D55E00",
        "rnj-1-instruct": "#999999",
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    apply_icml_text_style()

    fig, axes = plt.subplots(
        1,
        len(prompts),
        figsize=(FIG_W_PER_PROMPT * len(prompts), FIG_H),
        sharey=True,
    )
    if len(prompts) == 1:
        axes = [axes]

    for ax, prompt in zip(axes, prompts):
        subset = full_df[full_df["prompt"] == prompt]
        sns.lineplot(
            data=subset,
            x="layer",
            y="r2_full",
            hue="model",
            style="model",
            markers=False,  # No points, lines only
            dashes=False,
            ax=ax,
            linewidth=LINE_WIDTH,
            palette=model_to_color,
        )
        ax.set_title(
            f"{PROMPT_LABEL.get(prompt, prompt)}",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            pad=TITLE_PAD,
        )
        ax.set_ylabel("R² (Full Model)", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel("Layer Index", fontsize=LABEL_FONT_SIZE)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
        ax.legend().set_visible(False)  # Hide individual legends

    # Create single legend for the whole plot with larger text (factor 3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_rr_layer_trajectories.pdf"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=str,
        default=str(INPUT_DIR),
        help="Root directory containing per-model/prompt per-layer outputs.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for PDFs.",
    )
    args = ap.parse_args()

    df = _collect_latest_per_prompt_results(Path(args.results_root))
    plot_layer_trajectories(df, Path(args.out_dir))


if __name__ == "__main__":
    main()
