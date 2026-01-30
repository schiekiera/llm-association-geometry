#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path(
    "data/prediction/03_held_out_words_cosine"
)
SUMMARY_DIR = RESULTS_ROOT / "01_summary"

PROMPT_ORDER = ["averaged", "template", "forced_choice", "free_association"]


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
            required = {
                "model",
                "prompt",
                "layer",
                "r2_baseline",
                "r2_cross_add",
                "r2_cross_fc",
                "r2_cross_fa",
                "r2_full",
                "delta_cross",
                "delta_cross_fc",
                "delta_cross_fa",
                "delta_full",
            }
            if not required.issubset(set(df.columns)):
                continue
            rows.append(df)

    if not rows:
        raise FileNotFoundError(
            "No per-layer held-out-words cosine CSVs found. Expected files like "
            f"`{results_root}/<MODEL>/<PROMPT>/ridge_predict_held_out_words_cosine_*.csv`."
        )
    return pd.concat(rows, ignore_index=True)


def analyze_predictions(results_root: Path) -> Path:
    full_df = _collect_latest_per_prompt_results(results_root)

    all_prompts = sorted(full_df["prompt"].unique())
    filtered_prompts = [p for p in all_prompts if p in PROMPT_ORDER]
    full_df = full_df[full_df["prompt"].isin(PROMPT_ORDER)]

    print(f"Found prompts: {all_prompts}")
    print(f"Using prompts: {filtered_prompts}")

    mean_df = (
        full_df.groupby(["model", "prompt"], as_index=False)[
            ["r2_baseline", "r2_cross_add", "r2_cross_fc", "r2_cross_fa", "r2_full"]
        ]
        .mean()
        .rename(
            columns={
                "r2_baseline": "mean_r2_baseline",
                "r2_cross_add": "mean_r2_cross",
                "r2_cross_fc": "mean_r2_cross_fc",
                "r2_cross_fa": "mean_r2_cross_fa",
                "r2_full": "mean_r2_full",
            }
        )
    )

    # Peak per model/prompt (by r2_full)
    idx = full_df.groupby(["model", "prompt"])["r2_full"].idxmax()
    peak_df = full_df.loc[idx].copy()
    peak_df["best_layer"] = peak_df["layer"]
    peak_df["peak_r2_baseline"] = peak_df["r2_baseline"]
    peak_df["peak_r2_cross"] = peak_df["r2_cross_add"]
    peak_df["peak_r2_cross_fc"] = peak_df["r2_cross_fc"]
    peak_df["peak_r2_cross_fa"] = peak_df["r2_cross_fa"]
    peak_df["peak_r2_full"] = peak_df["r2_full"]
    peak_df["delta_cross_minus_baseline"] = peak_df["delta_cross"]
    peak_df["delta_cross_fc_minus_baseline"] = peak_df["delta_cross_fc"]
    peak_df["delta_cross_fa_minus_baseline"] = peak_df["delta_cross_fa"]
    peak_df["delta_full_minus_baseline"] = peak_df["delta_full"]
    peak_df["delta_fc"] = peak_df["delta_cross_fc"] - peak_df["delta_cross"]
    peak_df["delta_fa"] = peak_df["delta_cross_fa"] - peak_df["delta_cross"]
    peak_df["delta_full"] = peak_df["delta_full"] - peak_df["delta_cross"]

    summary_df = (
        peak_df.merge(mean_df, on=["model", "prompt"], how="left")
        .loc[
            :,
            [
                "model",
                "prompt",
                "best_layer",
                "mean_r2_cross",
                "mean_r2_cross_fc",
                "mean_r2_cross_fa",
                "mean_r2_full",
                "peak_r2_full",
                "delta_fc",
                "delta_fa",
                "delta_full",
            ],
        ]
        .sort_values(["prompt", "peak_r2_full"], ascending=[True, False])
    )

    print("\n=== Summary Predictive Performance ===")
    numeric_cols = summary_df.select_dtypes(include="number").columns
    mean_row = summary_df[numeric_cols].mean(numeric_only=True).round(3)
    mean_dict = {col: mean_row.get(col, "") for col in summary_df.columns}
    mean_dict["model"] = "MEAN"
    mean_dict["prompt"] = "MEAN"
    summary_display = pd.concat(
        [summary_df, pd.DataFrame([mean_dict])],
        ignore_index=True,
    )
    print(summary_display.to_string(index=False, float_format="%.3f"))

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUMMARY_DIR / f"summary_prediction_analysis_{timestamp}.csv"
    summary_display.to_csv(out_path, index=False)
    print(f"\nSaved summary CSV to: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Root directory containing held-out-words cosine prediction CSVs.",
    )
    args = ap.parse_args()
    analyze_predictions(Path(args.results_root))


if __name__ == "__main__":
    main()
