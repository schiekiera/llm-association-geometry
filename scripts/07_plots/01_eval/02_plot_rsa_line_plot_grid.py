import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.lines import Line2D

# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = "data/eval/01_RSA"
OUTPUT_DIR = "output"
EVAL_PATTERN = "layerwise_correlations_pairs500000_*.csv"
PROMPT_NAMES = ["averaged", "template", "forced_choice", "free_association"]

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
DPI = 300
Y_LIM = (-0.2, 1.0)
PLT_LAYOUT = [0, 0.03, 1, 0.9]
BBOX_LEGEND = (0.5, 0.93)

# =============================================================================
# Plotting Helpers
# =============================================================================

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


def set_icml_style():
    
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
    sns.set_context("paper", rc={
        "font.size": FONT_SIZE_LABEL,
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL
    })
    apply_icml_text_style()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    return sns.color_palette("colorblind")

def infer_sample_tag(df: pd.DataFrame) -> str:
    if "pair_mode" in df.columns and "n_pairs" in df.columns:
        modes = df["pair_mode"].dropna().unique().tolist()
        pairs = df["n_pairs"].dropna().unique().tolist()
        if len(modes) == 1 and len(pairs) == 1:
            if modes[0] == "sampled": return f"pairs{int(pairs[0])}"
            return f"pairsAll_{int(pairs[0])}"
    return "pairsUnknown"

def get_forced_choice_col(columns):
    for col in columns:
        if col.startswith("pearson_forced_choice_"):
            return col
    for col in columns:
        if col.startswith("pearson_fc_"):
            return col
    return None

def get_fa_col(columns):
    for col in columns:
        if col.startswith("pearson_fa_"): return col
    return None

def plot_combined_grid(df_summary, models, prompts, max_layers, palette):
    n_rows = len(models)
    n_cols = len(prompts)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.5), sharex=True, sharey=True)

    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :]
    elif n_cols == 1: axes = axes[:, np.newaxis]

    title_map = {
        "averaged": r"\textbf{Averaged}",
        "template": r"\textbf{Meaning}",
        "forced_choice": r"\textbf{Task (FC)}",
        "free_association": r"\textbf{Task (FA)}",
    }

    for i, model in enumerate(models):
        for j, prompt in enumerate(prompts):
            ax = axes[i, j]
            df = df_summary[(df_summary["model"] == model) & (df_summary["prompt"] == prompt)]
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_axis_off()
                continue

            tourn_col = get_forced_choice_col(df.columns)
            fa_col = get_fa_col(df.columns)

            if tourn_col:
                ax.plot(df["layer"], df[tourn_col], label="FC", color=palette[0], lw=1.6)
            if fa_col:
                ax.plot(df["layer"], df[fa_col], label="FA", color=palette[3], lw=1.6, alpha=0.95)
            if "pearson_fasttext" in df.columns:
                ax.plot(df["layer"], df["pearson_fasttext"], label="FastText", ls="-", color=palette[1], lw=1.2)
            if "pearson_bert" in df.columns:
                ax.plot(df["layer"], df["pearson_bert"], label="BERT", ls="-", color=palette[2], lw=1.2)
            if "pearson_crossmodel" in df.columns:
                ax.plot(
                    df["layer"],
                    df["pearson_crossmodel"],
                    label="Cross-model consensus",
                    ls="-",
                    color=palette[4],
                    lw=1.4,
                )

            if i == 0: ax.set_title(title_map.get(prompt, prompt.replace("_", " ").title()), fontweight='bold')
            if j == 0: ax.set_ylabel(model[:25] + "..." if len(model) > 25 else model, fontweight='bold', fontsize=FONT_SIZE_YLABEL)
            if i == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=FONT_SIZE_XLABEL)
                ax.set_xticks(np.arange(0, max_layers + 1, 10 if max_layers > 20 else 5))

            ax.set_ylim(Y_LIM)
            ax.set_yticks(np.arange(Y_LIM[0], Y_LIM[1] + 0.001, 0.2))
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

def _latest_eval_file(prompt_dir: str) -> str | None:
    candidates = sorted(
        [os.path.join(prompt_dir, f) for f in os.listdir(prompt_dir) if f.startswith("layerwise_correlations_pairs500000_")]
    )
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _collect_eval_files():
    rows = []
    for model_name in sorted(os.listdir(INPUT_DIR)):
        model_dir = os.path.join(INPUT_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        for prompt in PROMPT_NAMES:
            prompt_dir = os.path.join(model_dir, prompt)
            if not os.path.isdir(prompt_dir):
                continue
            csv_path = _latest_eval_file(prompt_dir)
            if not csv_path or not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df["model"] = model_name
            df["prompt"] = prompt
            rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No evaluation files found under {INPUT_DIR} for {EVAL_PATTERN}"
        )
    out = pd.concat(rows, ignore_index=True)
    out["model"] = out["model"].replace(MODEL_RENAMES)
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    palette = set_icml_style()

    df = _collect_eval_files()
    sample_tag = infer_sample_tag(df)

    model_info = [{"model": m, "n_layers": df[df["model"] == m]["layer"].max() + 1} for m in df["model"].unique()]
    models = pd.DataFrame(model_info).sort_values(["n_layers", "model"])["model"].tolist()

    fig = plot_combined_grid(df, models, PROMPT_NAMES, df["layer"].max(), palette)
    out_base = os.path.join(OUTPUT_DIR, "rsa_line_plot_8x4_grid")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    print(f"Plot saved to {out_base}.pdf")

if __name__ == "__main__":
    main()
