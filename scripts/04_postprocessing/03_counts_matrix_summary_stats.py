#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("projects/icml_project")
FC_ROOT_DEFAULT = PROJECT_ROOT / "output/02_behavioral_associations/01_forced_choice/03_postprocessed"
FA_ROOT_DEFAULT = PROJECT_ROOT / "output/02_behavioral_associations/02_free_associations/03_postprocessed"
OUT_DIR_DEFAULT = PROJECT_ROOT / "output/02_behavioral_associations"


def _load_csr_row_stats(npz_path: Path) -> tuple[int, float, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        fmt = data.get("format", None)
        if fmt is not None:
            fmt = str(fmt.tolist(), "utf-8") if hasattr(fmt, "tolist") else str(fmt)
        if fmt != "csr":
            raise ValueError(f"Unsupported sparse format in {npz_path}: {fmt}")
        indptr = data["indptr"]
        data_vals = data["data"]
    n_rows = int(len(indptr) - 1)
    if n_rows <= 0:
        return 0, float("nan"), float("nan")
    # Sum each row, then take the mean of row sums.
    # Using reduceat avoids densifying the matrix.
    row_sums = np.add.reduceat(data_vals, indptr[:-1])
    mean_row_sum = float(np.mean(row_sums)) if row_sums.size else float("nan")
    total_sum = float(np.sum(data_vals)) if data_vals.size else float("nan")
    return n_rows, mean_row_sum, total_sum


def _collect_paradigm_stats(root: Path, paradigm: str) -> list[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    rows: list[dict] = []
    total_rows = 0
    total_row_sum = 0.0
    total_sum_all = 0.0

    for model_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        npz_path = model_dir / "counts_matrix.npz"
        if not npz_path.exists():
            continue
        n_rows, mean_row_sum, total_sum = _load_csr_row_stats(npz_path)
        total_rows += n_rows
        total_row_sum += mean_row_sum * n_rows
        total_sum_all += total_sum
        rows.append(
            {
                "paradigm": paradigm,
                "model": model_dir.name,
                "n_rows": n_rows,
                "mean_row_sum": mean_row_sum,
                "total_sum": total_sum,
            }
        )

    mean_row_sum_all = float(total_row_sum / total_rows) if total_rows > 0 else float("nan")
    rows.append(
        {
            "paradigm": paradigm,
            "model": "ALL",
            "n_rows": total_rows,
            "mean_row_sum": mean_row_sum_all,
            "total_sum": total_sum_all,
        }
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fc-root", type=str, default=str(FC_ROOT_DEFAULT))
    ap.add_argument("--fa-root", type=str, default=str(FA_ROOT_DEFAULT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    rows: list[dict] = []
    rows.extend(_collect_paradigm_stats(Path(args.fc_root), "forced_choice"))
    rows.extend(_collect_paradigm_stats(Path(args.fa_root), "free_association"))

    df = pd.DataFrame(rows)
    df["mean_row_sum"] = df["mean_row_sum"].round(1)
    df["total_sum"] = df["total_sum"].round(1)
    df = df.sort_values(["paradigm", "model"], ascending=[True, True]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"counts_matrix_summary_stats_{ts}.csv"

    print(df.to_string(index=False))
    df.to_csv(out_path, index=False)
    print(f"\nSaved summary CSV to: {out_path}")


if __name__ == "__main__":
    main()
