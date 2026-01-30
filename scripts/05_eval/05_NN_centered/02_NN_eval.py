#!/usr/bin/env python3
"""
Summarize nearest-neighbors (NN@k) results across layers.

Reads one or more `summary_all_neighbors_k*_*.csv` files and produces:
1) A per-k summary (mean/max per metric with best layer).
2) A cross-k summary (best k per metric).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path("projects/icml_project")
DEFAULT_INPUT_DIR = ROOT / "output/eval/02_NN"
DEFAULT_OUTPUT_DIR = ROOT / "output/eval/02_NN"
DEFAULT_KS = [5, 10, 20, 50, 100, 200]

METRICS = [
    "nn_forced_choice_PPMI",
    "nn_fa_PPMI",
    "nn_fasttext",
    "nn_bert",
    "nn_crossmodel",
]

GROUP_COLS = [
    "model",
    "prompt",
    "beh_type",
    "fa_type",
    "k",
    "k_fasttext",
    "n_words",
    "n_words_fasttext",
]


@dataclass(frozen=True)
class SummaryCell:
    mean: float
    max: float
    best_layer: int


def _find_input_csvs(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(input_dir.rglob(pattern))


def _summarize_group(df: pd.DataFrame, metrics: list[str]) -> dict:
    summary = {}
    for metric in metrics:
        vals = df[metric].astype(float)
        mean_val = vals.mean()
        max_val = vals.max()
        best_layer = int(df.loc[vals.idxmax(), "layer"])
        summary[f"{metric}_mean"] = mean_val
        summary[f"{metric}_max"] = max_val
        summary[f"{metric}_best_layer"] = best_layer
    summary["mean_all_metrics"] = df[metrics].astype(float).mean(axis=1).mean()
    summary["max_all_metrics"] = df[metrics].astype(float).max(axis=1).max()
    return summary


def summarize_per_k(df_all: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, df_group in df_all.groupby(GROUP_COLS, dropna=False):
        row = dict(zip(GROUP_COLS, keys))
        row.update(_summarize_group(df_group, metrics))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_best_k(summary_k: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    base_cols = [c for c in GROUP_COLS if c not in ("k", "k_fasttext")]
    for metric in metrics:
        mean_col = f"{metric}_mean"
        max_col = f"{metric}_max"
        for keys, df_group in summary_k.groupby(base_cols, dropna=False):
            df_group = df_group.sort_values(by=mean_col, ascending=False)
            best_mean = df_group.iloc[0]
            df_group_max = df_group.sort_values(by=max_col, ascending=False).iloc[0]
            row = dict(zip(base_cols, keys))
            row.update(
                {
                    "metric": metric,
                    "best_k_mean": int(best_mean["k"]),
                    "best_k_fasttext_mean": int(best_mean["k_fasttext"]),
                    "best_mean": float(best_mean[mean_col]),
                    "best_mean_layer": int(best_mean[f"{metric}_best_layer"]),
                    "best_k_max": int(df_group_max["k"]),
                    "best_k_fasttext_max": int(df_group_max["k_fasttext"]),
                    "best_max": float(df_group_max[max_col]),
                    "best_max_layer": int(df_group_max[f"{metric}_best_layer"]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _fmt_metric_row(values: dict[str, float], ndigits: int = 3) -> str:
    parts = [f"{m}={values[m]:.{ndigits}f}" for m in values]
    return "{ " + ", ".join(parts) + " }"


def print_summary(summary_k: pd.DataFrame, metrics: list[str], ndigits: int = 3) -> None:
    metric_mean_cols = {m: f"{m}_mean" for m in metrics}
    metric_best_layer_cols = {m: f"{m}_best_layer" for m in metrics}

    prompts = sorted(summary_k["prompt"].astype(str).unique())
    models = sorted(summary_k["model"].astype(str).unique())
    ks = sorted(summary_k["k"].astype(int).unique())

    print(f"Models: {len(models)} | Prompts: {len(prompts)} | Ks: {ks}")

    print("\nMean across models/prompts (mean over layers) by k:")
    for k in ks:
        row = {}
        subset = summary_k[summary_k["k"].astype(int) == k]
        for metric, col in metric_mean_cols.items():
            row[metric] = float(subset[col].mean())
        print(f"  k={k:3d}  {_fmt_metric_row(row, ndigits=ndigits)}")

    print("\nBest k per metric (global mean over models/prompts):")
    best_k_by_metric = {}
    for metric, col in metric_mean_cols.items():
        means = summary_k.groupby("k")[col].mean().sort_values(ascending=False)
        best_k = int(means.index[0])
        best_k_by_metric[metric] = best_k
        print(f"  {metric:18s} best k={best_k:3d}  mean={means.iloc[0]:.{ndigits}f}")

    print("\nPrompt ranking per metric at best k:")
    for metric, best_k in best_k_by_metric.items():
        col = metric_mean_cols[metric]
        subset = summary_k[summary_k["k"].astype(int) == best_k]
        prompt_means = subset.groupby("prompt")[col].mean().sort_values(ascending=False)
        print(f"  {metric} (k={best_k}):")
        for prompt, val in prompt_means.items():
            print(f"    {str(prompt):15s} {val:.{ndigits}f}")

    print("\nPeak layer per prompt (avg of per-model best layers at best k):")
    for metric, best_k in best_k_by_metric.items():
        layer_col = metric_best_layer_cols[metric]
        subset = summary_k[summary_k["k"].astype(int) == best_k]
        print(f"  {metric} (k={best_k}):")
        for prompt in prompts:
            prompt_subset = subset[subset["prompt"] == prompt]
            if prompt_subset.empty:
                continue
            mean_layer = float(prompt_subset[layer_col].mean())
            print(f"    {prompt:15s} layer~{mean_layer:.1f}")

    print("\nBest/worst (model, prompt, k) by mean over layers:")
    for metric, col in metric_mean_cols.items():
        best_row = summary_k.loc[summary_k[col].idxmax()]
        worst_row = summary_k.loc[summary_k[col].idxmin()]
        print(f"  {metric}:")
        print(
            f"    best  {best_row['model']} | {best_row['prompt']} | k={int(best_row['k'])} "
            f"mean={best_row[col]:.{ndigits}f}"
        )
        print(
            f"    worst {worst_row['model']} | {worst_row['prompt']} | k={int(worst_row['k'])} "
            f"mean={worst_row[col]:.{ndigits}f}"
        )


def load_inputs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "model" not in df.columns or "prompt" not in df.columns:
            try:
                df["prompt"] = df.get("prompt", path.parent.name)
                df["model"] = df.get("model", path.parent.parent.name)
            except Exception:
                pass
        # Infer beh_type/fa_type when missing (from column names)
        if "beh_type" not in df.columns:
            beh_cols = [c for c in df.columns if c.startswith("nn_forced_choice_")]
            df["beh_type"] = beh_cols[0].replace("nn_forced_choice_", "") if beh_cols else ""
        if "fa_type" not in df.columns:
            fa_cols = [c for c in df.columns if c.startswith("nn_fa_")]
            df["fa_type"] = fa_cols[0].replace("nn_fa_", "") if fa_cols else ""
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _layerwise_patfc_for_k(input_dir: Path, k: int) -> list[Path]:
    pattern = f"layerwise_neighbors_k{k}_*.csv"
    return _find_input_csvs(input_dir, pattern)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize nearest-neighbors NN@k results across layers and k."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing per-layer NN results (searched recursively).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Optional glob pattern for input CSVs within --input-dir. "
            "If omitted, uses layerwise_neighbors_k{K}_*.csv for each K."
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional explicit list of input CSV files (overrides --input-dir/--pattern).",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=DEFAULT_KS,
        help="List of k values to summarize when reading layerwise files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write summary CSVs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d")

    per_k_frames = []
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
        df_all = load_inputs(input_paths)
        if df_all.empty:
            raise ValueError("Input files were empty or unreadable.")
        per_k_frames.append(df_all)
    else:
        for k in args.ks:
            if args.pattern:
                pattern = args.pattern
                paths = _find_input_csvs(input_dir, pattern)
            else:
                paths = _layerwise_patfc_for_k(input_dir, k)
            if not paths:
                continue
            df_k = load_inputs(paths)
            if df_k.empty:
                continue
            per_k_frames.append(df_k)

            # Write per-k summary_all_neighbors_k{k}_<date>.csv
            summary_path = out_dir / f"summary_all_neighbors_k{k}_{tag}.csv"
            df_k.to_csv(summary_path, index=False)

    if not per_k_frames:
        raise FileNotFoundError("No NN layerwise CSVs found. Check --input-dir/--pattern/--ks.")

    df_all = pd.concat(per_k_frames, ignore_index=True)

    available_metrics = [m for m in METRICS if m in df_all.columns]
    missing = set(["layer"] + GROUP_COLS) - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not available_metrics:
        raise ValueError("No NN metric columns found in inputs.")

    summary_k = summarize_per_k(df_all, available_metrics)
    summary_best_k = summarize_best_k(summary_k, available_metrics)
    per_k_path = out_dir / f"summary_nn_per_k_{tag}.csv"
    best_k_path = out_dir / f"summary_nn_best_k_{tag}.csv"

    summary_k.to_csv(per_k_path, index=False)
    summary_best_k.to_csv(best_k_path, index=False)

    print(f"Wrote per-k summary to: {per_k_path}")
    print(f"Wrote best-k summary to: {best_k_path}")
    print_summary(summary_k, available_metrics)


if __name__ == "__main__":
    main()
