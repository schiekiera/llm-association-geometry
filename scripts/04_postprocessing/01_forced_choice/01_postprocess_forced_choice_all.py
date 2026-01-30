#!/usr/bin/env python3
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# =============================================================================
# Constants & Path Setup
# =============================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

DEFAULT_BASE_DIR = _repo_root() / "output" / "02_behavioral_associations" / "01_forced_choice"
INPUT_DIR = DEFAULT_BASE_DIR / "02_processed"
OUTPUT_BASE_DIR = DEFAULT_BASE_DIR / "03_postprocessed"

# Pattern to extract: {MODEL_ALIAS}_forced_choice_round{ROUND}_processed_{TIMESTAMP}.csv
FILE_PATTERN = re.compile(r"^(.*)_forced_choice_round(\d+)_processed_(.*)\.csv$")

# =============================================================================
# Core Logic (Adapted from Single Script)
# =============================================================================

def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    norm = lambda c: re.sub(r"[\s_]+", "", str(c).strip().lower())
    norm_to_col = {norm(c): c for c in df.columns}
    cols = (norm_to_col.get("input"), norm_to_col.get("pick1"), norm_to_col.get("pick2"))
    if not all(cols):
        raise ValueError(f"Missing columns in {list(df.columns)}")
    return cols

def _clean_series(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.lower()
    out = out.replace({"": pd.NA, "na": pd.NA, "nan": pd.NA})
    has_alnum = out.str.contains(r"[^\W_]", regex=True, na=False)
    return out.where(has_alnum, pd.NA)

def process_model_group(model_alias: str, csv_paths: List[Path], out_dir: Path, no_cosine: bool = False):
    print(f"\n>>> Processing Model: {model_alias} ({len(csv_paths)} files)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Concat
    frames = []
    for p in csv_paths:
        header = pd.read_csv(p, nrows=0)
        ic, p1c, p2c = _resolve_columns(header)
        df = pd.read_csv(p, usecols=[ic, p1c, p2c], dtype="string")
        frames.append(df)
    full_df = pd.concat(frames, ignore_index=True)
    ic, p1c, p2c = _resolve_columns(full_df)

    # 2. Build Count Matrix
    inp, p1, p2 = _clean_series(full_df[ic]), _clean_series(full_df[p1c]), _clean_series(full_df[p2c])
    row_vocab = pd.Index(sorted(inp.dropna().unique().tolist()), dtype="string")
    col_vocab = pd.Index(sorted(pd.concat([p1, p2]).dropna().unique().tolist()), dtype="string")
    
    inp2, picks = pd.concat([inp, inp]), pd.concat([p1, p2])
    valid = inp2.notna() & picks.notna() & (picks != inp2)
    row_ids = row_vocab.get_indexer(inp2[valid])
    col_ids = col_vocab.get_indexer(picks[valid])
    
    data = np.ones(len(row_ids), dtype=np.int32)
    counts = sp.coo_matrix((data, (row_ids, col_ids)), shape=(len(row_vocab), len(col_vocab))).tocsr()
    N = int(counts.sum())

    # 3. Compute PPMI
    row_sums = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    col_sums = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    coo = counts.tocoo()
    denom = row_sums[coo.row] * col_sums[coo.col]
    pmi = np.log((coo.data.astype(np.float64) * float(N)) / denom)
    ppmi_data = np.maximum(pmi, 0.0)
    ppmi = sp.coo_matrix((ppmi_data.astype(np.float32), (coo.row, coo.col)), shape=counts.shape).tocsr()

    # 4. Save Matrices & Vocabs
    sp.save_npz(out_dir / "counts_matrix.npz", counts)
    sp.save_npz(out_dir / "ppmi_matrix.npz", ppmi)
    
    # Cosine Sim function
    def save_cosine(mat, name):
        if sp.issparse(mat):
            A = mat.astype(np.float32)
            row_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        else:
            A = mat.astype(np.float32)
            row_norm = np.linalg.norm(A, axis=1)
            
        inv = np.zeros_like(row_norm); nz = row_norm > 0; inv[nz] = 1.0 / row_norm[nz]
        if sp.issparse(A):
            A_norm = sp.diags(inv) @ A
            sim = (A_norm @ A_norm.T).toarray()
        else:
            A_norm = A * inv[:, np.newaxis]
            sim = A_norm @ A_norm.T
        np.save(out_dir / f"cosine_sim_{name}.npy", sim.astype(np.float32))

    if not no_cosine:
        # 1. Raw Matrices
        save_cosine(counts, "counts")
        save_cosine(ppmi, "ppmi")

        # 2. SVD Variants
        svd_dims = [100, 300, 600]
        matrices = {"counts": counts, "ppmi": ppmi}
        
        for m_name, matrix in matrices.items():
            for dim in svd_dims:
                # TruncatedSVD
                if matrix.shape[1] > dim:
                    print(f"   Computing SVD-{dim} for {m_name}...")
                    svd = TruncatedSVD(n_components=dim, random_state=42)
                    reduced = svd.fit_transform(matrix)
                    
                    # Save Embedding
                    np.save(out_dir / f"svd{dim}_{m_name}_embedding.npy", reduced.astype(np.float32))
                    
                    # Compute & Save Cosine Sim
                    save_cosine(reduced, f"svd{dim}_{m_name}")
                else:
                    print(f"   Skipping SVD-{dim} for {m_name} (dim {matrix.shape[1]} <= {dim})")

    pd.DataFrame({"input_word": row_vocab}).to_csv(out_dir / "row_vocab.csv", index=False)
    pd.DataFrame({"output_word": col_vocab}).to_csv(out_dir / "col_vocab.csv", index=False)
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"n_rows": int(len(row_vocab)), "total_counts": N}, f, indent=2)
    print(f"Done. Results in {out_dir}")

# =============================================================================
# Main Discovery Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cosine", action="store_true")
    args = parser.parse_args()

    # 1. Discover all files
    all_files = list(INPUT_DIR.glob("*.csv"))
    groups: Dict[str, Dict[str, List[Path]]] = {}

    for f in all_files:
        match = FILE_PATTERN.match(f.name)
        if match:
            alias, round_num, ts = match.groups()
            if alias not in groups: groups[alias] = {}
            if ts not in groups[alias]: groups[alias][ts] = []
            groups[alias][ts].append(f)

    # 2. Process each model (latest timestamp)
    for alias, ts_dict in groups.items():
        latest_ts = sorted(ts_dict.keys())[-1]
        files = ts_dict[latest_ts]
        
        # We expect multiple rounds (usually 3-4 for a finished run)
        if len(files) < 1: continue
        
        out_dir = OUTPUT_BASE_DIR / alias
        process_model_group(alias, files, out_dir, args.no_cosine)

if __name__ == "__main__":
    main()





