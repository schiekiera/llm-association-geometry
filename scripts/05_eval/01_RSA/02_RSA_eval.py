#!/usr/bin/env python3
"""
Summarize RSA evaluation results for the paper.

This script computes the summary statistics used in the RSA Results paragraph:
- mean RSA per prompt (mean over layers, then mean over models)
- prompt rankings per reference geometry
- peak layer per prompt (mean over models)
- overall peak layer (mean over all model/prompt rows)
- best/worst (model, prompt) cells by mean-over-layers

Input format: `context/data/rsa/eval_rsa2.csv`
Expected columns:
  layer, pearson_fc_PPMI, pearson_fa_PPMI, pearson_fasttext, pearson_bert,
  model, prompt
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


METRICS: List[Tuple[str, str]] = [
    ("FC", "pearson_fc_PPMI"),
    ("FA", "pearson_fa_PPMI"),
    ("FastText", "pearson_fasttext"),
    ("BERT", "pearson_bert"),
    ("Cross-model consensus", "pearson_crossmodel"),
]

MODEL_ALIAS = ["Falcon3-10B-Instruct", "gemma-2-9b-it", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.2", "Mistral-Nemo-Instruct-v1", "phi-4", "Qwen2.5-7B-Instruct", "rnj-1-instruct"]
PROMPT_NAMES = ["template", "averaged", "forced_choice", "free_association"]

DEFAULT_SUMMARY_DIR = Path("data/eval/01_RSA")
DEFAULT_SUMMARY_PATTERN = "summary_all_results_pairs500000_*.csv"
LAYERWISE_PATTERN = "layerwise_correlations_pairs500000_*.csv"


def _read_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["layer", "model", "prompt"] + [c for _, c in METRICS] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
        for row in reader:
            yield row


def _find_latest_layerwise(model: str, prompt: str) -> Path | None:
    target_dir = DEFAULT_SUMMARY_DIR / model / prompt
    if not target_dir.exists():
        return None
    candidates = sorted(target_dir.glob(LAYERWISE_PATTERN))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _normalize_layerwise_rows(csv_path: Path, model: str, prompt: str) -> Iterable[dict]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {csv_path}")
        fc_cols = [c for c in reader.fieldnames if c.startswith("pearson_fc_")]
        fa_cols = [c for c in reader.fieldnames if c.startswith("pearson_fa_")]
        if not fc_cols or not fa_cols:
            raise ValueError(
                f"Missing FC/FA columns in {csv_path} "
                f"(fc={fc_cols}, fa={fa_cols})"
            )
        fc_col = fc_cols[0]
        fa_col = fa_cols[0]
        for row in reader:
            yield {
                "layer": row["layer"],
                "pearson_fc_PPMI": row[fc_col],
                "pearson_fa_PPMI": row[fa_col],
                "pearson_fasttext": row["pearson_fasttext"],
                "pearson_bert": row["pearson_bert"],
                "pearson_crossmodel": row.get("pearson_crossmodel", ""),
                "model": model,
                "prompt": prompt,
            }


def build_summary_from_layerwise(summary_dir: Path) -> Path:
    missing = []
    rows: List[dict] = []
    for model in MODEL_ALIAS:
        for prompt in PROMPT_NAMES:
            latest = _find_latest_layerwise(model, prompt)
            if latest is None:
                missing.append((model, prompt))
                continue
            rows.extend(list(_normalize_layerwise_rows(latest, model, prompt)))

    if missing:
        missing_str = ", ".join([f"{m}/{p}" for m, p in missing])
        raise FileNotFoundError(
            f"Missing layerwise CSVs for: {missing_str} (pattern {LAYERWISE_PATTERN})"
        )

    out_path = summary_dir / f"summary_all_results_pairs500000_{_date_tag()}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "pearson_fc_PPMI",
                "pearson_fa_PPMI",
                "pearson_fasttext",
                "pearson_bert",
                "pearson_crossmodel",
                "model",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _fmt_row(d: Dict[str, float], ndigits: int = 3) -> str:
    parts = [f"{name}={d[name]:.{ndigits}f}" for name, _ in METRICS]
    return "{ " + ", ".join(parts) + " }"


def summarize(csv_path: Path, ndigits: int = 3) -> None:
    # (model,prompt) -> sums per metric; count layers
    group_sums: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0] * len(METRICS))
    group_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    # (prompt, layer) -> sums per metric; count models
    prompt_layer_sums: Dict[Tuple[str, int], List[float]] = defaultdict(lambda: [0.0] * len(METRICS))
    prompt_layer_counts: Dict[Tuple[str, int], int] = defaultdict(int)

    # layer -> sums per metric; count rows
    layer_sums: Dict[int, List[float]] = defaultdict(lambda: [0.0] * len(METRICS))
    layer_counts: Dict[int, int] = defaultdict(int)

    models = set()
    prompts = set()
    layers = set()

    for row in _read_rows(csv_path):
        model = row["model"]
        prompt = row["prompt"]
        layer = int(row["layer"])

        models.add(model)
        prompts.add(prompt)
        layers.add(layer)

        key = (model, prompt)
        for i, (_, col) in enumerate(METRICS):
            v = float(row[col])
            group_sums[key][i] += v
            prompt_layer_sums[(prompt, layer)][i] += v
            layer_sums[layer][i] += v
        group_counts[key] += 1
        prompt_layer_counts[(prompt, layer)] += 1
        layer_counts[layer] += 1

    # Mean over layers per (model,prompt)
    group_means: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, sums in group_sums.items():
        n = group_counts[key]
        group_means[key] = {name: sums[i] / n for i, (name, _) in enumerate(METRICS)}

    # Mean across models per prompt (mean over layers first, then models)
    prompt_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: {name: 0.0 for name, _ in METRICS})
    prompt_counts: Dict[str, int] = defaultdict(int)
    for (model, prompt), vals in group_means.items():
        for name, _ in METRICS:
            prompt_sums[prompt][name] += vals[name]
        prompt_counts[prompt] += 1

    prompt_means: Dict[str, Dict[str, float]] = {}
    for prompt in prompt_sums:
        prompt_means[prompt] = {name: prompt_sums[prompt][name] / prompt_counts[prompt] for name, _ in METRICS}

    print(f"Loaded: {csv_path}")
    print(f"Models: {len(models)} | Prompts: {len(prompts)} | Layers: {len(layers)}")
    print(f"Prompt set: {sorted(prompts)}")

    print("\nMean across models (mean over layers, then models):")
    for prompt in sorted(prompt_means):
        print(f"  {prompt:15s} {_fmt_row(prompt_means[prompt], ndigits=ndigits)}")

    print("\nPrompt ranking per reference geometry:")
    for name, _ in METRICS:
        items = sorted(((p, prompt_means[p][name]) for p in prompt_means), key=lambda x: x[1], reverse=True)
        print(f"  {name}:")
        for p, v in items:
            print(f"    {p:15s} {v:.{ndigits}f}")

    # Peak layer per prompt (mean across models at each layer)
    print("\nPeak layer per prompt (mean across models):")
    for prompt in sorted(prompts):
        # collect per-layer means for this prompt
        layer_means: List[Tuple[int, Dict[str, float]]] = []
        for layer in sorted(layers):
            key = (prompt, layer)
            if key not in prompt_layer_sums:
                continue
            n = prompt_layer_counts[key]
            vals = {name: prompt_layer_sums[key][i] / n for i, (name, _) in enumerate(METRICS)}
            layer_means.append((layer, vals))
        if not layer_means:
            continue
        print(f"  {prompt}:")
        for name, _ in METRICS:
            best_layer, best_vals = max(layer_means, key=lambda lv: lv[1][name])
            print(f"    {name:8s} layer {best_layer:2d}  mean r={best_vals[name]:.{ndigits}f}")

    # Overall peak layer across all models/prompts
    print("\nOverall peak layer (all models + prompts):")
    for name, _ in METRICS:
        best_layer = None
        best_val = -1e9
        for layer in sorted(layers):
            n = layer_counts[layer]
            v = layer_sums[layer][[n for n, _ in METRICS].index(name)] / n
            if v > best_val:
                best_val = v
                best_layer = layer
        print(f"  {name:8s} layer {best_layer:2d}  mean r={best_val:.{ndigits}f}")

    # Extremes: best/worst (model,prompt) by mean-over-layers
    print("\nBest/worst (model, prompt) by mean over layers:")
    for name, _ in METRICS:
        best_key, best_vals = max(group_means.items(), key=lambda kv: kv[1][name])
        worst_key, worst_vals = min(group_means.items(), key=lambda kv: kv[1][name])
        print(f"  {name}:")
        print(f"    best  {best_key[0]} | {best_key[1]}  r={best_vals[name]:.{ndigits}f}")
        print(f"    worst {worst_key[0]} | {worst_key[1]}  r={worst_vals[name]:.{ndigits}f}")


def _latest_summary_csv(summary_dir: Path) -> Path:
    candidates = sorted(summary_dir.glob(DEFAULT_SUMMARY_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"No files found matching {DEFAULT_SUMMARY_PATTERN} in {summary_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _date_tag() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help=(
            "Path to RSA evaluation CSV. If omitted or 'latest', uses the newest "
            f"{DEFAULT_SUMMARY_PATTERN} in {DEFAULT_SUMMARY_DIR}."
        ),
    )
    ap.add_argument(
        "--ndigits",
        type=int,
        default=3,
        help="Decimal precision for printed values.",
    )
    args = ap.parse_args()

    if not args.csv or args.csv == "latest":
        csv_path = build_summary_from_layerwise(DEFAULT_SUMMARY_DIR)
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    summarize(csv_path=csv_path, ndigits=args.ndigits)


if __name__ == "__main__":
    main()
