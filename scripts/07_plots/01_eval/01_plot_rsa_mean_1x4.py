#!/usr/bin/env python3
"""
Standalone script to create a mean layer heatmap from RSA results.
Creates a 2x2 grid showing mean correlations across layers for each target type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Use Type-1 fonts in PDF/PS outputs
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Times Font for ICML (via LaTeX)
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

# Configuration
INPUT_DIR = Path("data/eval/01_RSA")
INPUT_PATTERN = "summary_all_results_pairs500000_*.csv"
OUTPUT_DIR = Path("output")

MODEL_RENAMES = {
    "Mistral-Nemo-Instruct-v1": "Mistral-Nemo-Instruct-2407",
}

# ICML Style Constants
vmin, vmax = 0, 0.9
FONT_SIZE_TITLE = 13
FONT_SIZE_XTICK = 12
FONT_SIZE_YTICK = 12
FONT_SIZE_ANNOT = 12
FONT_SIZE_COLORBAR = 12
FONT_SIZE_COLORBAR_TICK = 12
FIG_H = 5.0
FIG_ASPECT = 19.6 / 6.6  # Match 03_plot_rsa_fc_fa_minmaxmean_grid.py ratio



def _format_label_with_subscript(label: str) -> str:
    """
    Render underscores as subscripts in matplotlib tick labels.
    Example: "Template_FC" -> "Template$_{FC}$"
    If multiple underscores exist, everything after the first underscore becomes the subscript.
    """
    s = str(label)
    if "_" not in s:
        return s
    base, sub = s.split("_", 1)
    # Escape underscores that might still appear in the subscript text
    sub = sub.replace("_", r"\_")
    return f"{base}$_{{{sub}}}$"

def load_and_prepare_data(csv_path):
    """Load RSA results and prepare for plotting."""
    df = pd.read_csv(csv_path)
    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_RENAMES)
    
    # Get unique models and prompts
    models = sorted(df['model'].unique())
    # Enforce a consistent prompt order (used across all heatmaps)
    desired_prompt_order = ["averaged", "template", "forced_choice", "free_association"]
    prompts_present = list(df["prompt"].unique())
    prompts = [p for p in desired_prompt_order if p in prompts_present]
    
    return df, models, prompts

def compute_mean_correlations(df, models, prompts, target_col):
    """Compute mean correlation across layers for each model/prompt combination."""
    data = []
    
    for model in models:
        for prompt in prompts:
            # Get all layers for this model/prompt
            subset = df[(df['model'] == model) & (df['prompt'] == prompt)]
            
            if not subset.empty:
                # Compute mean across layers
                mean_corr = subset[target_col].mean()
            else:
                mean_corr = np.nan
                
            data.append({
                'model': model,
                'prompt': prompt, 
                'mean_correlation': mean_corr
            })
    
    # Create pivot table
    df_long = pd.DataFrame(data)
    pivot = df_long.pivot(index='model', columns='prompt', values='mean_correlation')
    
    
    return pivot

def _latest_summary_csv() -> Path:
    candidates = sorted(INPUT_DIR.glob(INPUT_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"No files found matching {INPUT_PATTERN} in {INPUT_DIR}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def create_mean_heatmap():
    """Create the main heatmap plot."""
    # Load data
    input_path = _latest_summary_csv()
    df, models, prompts = load_and_prepare_data(input_path)
    
    # Target columns and their display names (include crossmodel if present)
    targets = [
        ("pearson_fc_PPMI", r"\textbf{Forced Choice} ($\mathbf{S}^{\mathrm{FC}}$)"),
        ("pearson_fa_PPMI", r"\textbf{Free Association} ($\mathbf{S}^{\mathrm{FA}}$)"),
        ("pearson_fasttext", r"\textbf{FastText} ($\mathbf{S}^{\mathrm{FT}}$)"),
        ("pearson_bert", r"\textbf{BERT} ($\mathbf{S}^{\mathrm{BERT}}$)"),
    ]
    if "pearson_crossmodel" in df.columns:
        targets.append(("pearson_crossmodel", r"\textbf{Cross-model} ($\mathbf{S}^{\mathrm{X}}_{\mathrm{m}}$)"))
    
    # Desired column order + display labels
    # averaged -> Averaged
    # template -> Meaning
    # forced_choice -> Task (FC)
    # free_association -> Task (FA)
    prompt_label_map = {
        "averaged": "Averaged",
        "template": "Meaning",
        "forced_choice": "Task (FC)",
        "free_association": "Task (FA)",
    }
    display_prompts = [prompt_label_map[p] for p in prompts]
    
    # Compute pivot tables for each target
    pivots = []
    all_values = []
    
    for target_col, _ in targets:
        pivot = compute_mean_correlations(df, models, prompts, target_col)
        
        # Rename columns to display labels (and keep the enforced order)
        pivot.columns = display_prompts
        
        pivots.append(pivot)
        
        # Collect values for global color scale
        vals = pivot.to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            all_values.extend(vals)
    

    
    n_targets = len(targets)
    # Create 1xN subplot grid (single row)
    # Note: with square cells and 8x4 matrices, each heatmap is inherently narrow.
    # A smaller figure width reduces the large apparent gaps between panels.
    fig_w = FIG_H * FIG_ASPECT
    fig, axes = plt.subplots(
        1,
        n_targets,
        figsize=(fig_w, FIG_H),
        gridspec_kw={"wspace": 0.02},
    )
    if n_targets == 1:
        axes = [axes]
    
    # Plot each heatmap
    def _format_cell(val: float) -> str:
        if not np.isfinite(val):
            return ""
        s = f"{val:.2f}"
        if s.startswith("-0."):
            return s.replace("-0.", "-.", 1)
        if s.startswith("0."):
            return s.replace("0.", ".", 1)
        return s

    for i, ((target_col, title), pivot) in enumerate(zip(targets, pivots)):
        ax = axes[i]
        annot_df = pivot.applymap(_format_cell)
        
        # Create heatmap with viridis colormap
        sns.heatmap(
            pivot,
            ax=ax,
            annot=annot_df,
            fmt="",
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,  # Force square cells
            linewidths=0.5,
            annot_kws={"fontsize": FONT_SIZE_ANNOT}
        )
        
        # Customize appearance
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Fix x-axis tick label aesthetics (clean rotation/alignment/padding)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_XTICK, pad=2)
        ax.tick_params(axis='y', rotation=0, labelsize=FONT_SIZE_YTICK)
        xt = [_format_label_with_subscript(t.get_text()) for t in ax.get_xticklabels()]
        ax.set_xticklabels(
            xt,
            rotation=35,
            ha='right',
            rotation_mode='anchor',
        )
        
        # Only show y-labels on leftmost subplot
        if i != 0:  # Not the first subplot
            ax.set_yticklabels([])
        
        # Show x-labels on all subplots (since it's a single row)
        # Keep x-labels for all
    
    # Enough bottom margin for rotated ticks; wspace handled via gridspec_kw above
    plt.subplots_adjust(bottom=0.22)
    
    # Add shared colorbar
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    # Position colorbar on the right side
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', 
                       fraction=0.02, pad=0.04, aspect=30)
    cbar.set_label('Mean Pearson Correlation', fontsize=FONT_SIZE_COLORBAR, fontweight='bold')
    cbar.ax.tick_params(labelsize=FONT_SIZE_COLORBAR_TICK)
    
    # Save plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "rsa_mean_1x4_grid.pdf"
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved heatmap to: {output_path}")
    




def main():
    """Main function."""
    print("Creating RSA mean layer heatmap...")
    input_path = _latest_summary_csv()
    print(f"Input file: {input_path}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Check if input file exists
    try:
        create_mean_heatmap()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return
    print("Done!")

if __name__ == "__main__":
    main()