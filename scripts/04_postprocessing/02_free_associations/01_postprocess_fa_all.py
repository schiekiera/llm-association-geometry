import os
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
from glob import glob
from tqdm import tqdm
import json
import argparse
import shutil
from sklearn.decomposition import TruncatedSVD

# =============================================================================
# Configuration
# =============================================================================

RAW_DIR = "data/02_behavioral_associations/02_free_associations/01_raw"
PROCESSED_BASE_DIR = "data/02_behavioral_associations/02_free_associations/02_processed"
POSTPROCESSED_BASE_DIR = "data/02_behavioral_associations/02_free_associations/03_postprocessed"

# =============================================================================
# Extraction & Quality Logic
# =============================================================================

def _clean_series(s: pd.Series) -> pd.Series:
    """
    Clean text fields: lowercase, strip, and set empty/non-alphanumeric to NA.
    Mirrors the FC postprocess cleaning logic.
    """
    out = s.astype("string").str.strip().str.lower()
    out = out.replace({"": pd.NA, "na": pd.NA, "nan": pd.NA})
    has_alnum = out.str.contains(r"[^\W_]", regex=True, na=False)
    return out.where(has_alnum, pd.NA)

def extract_associations(text: str) -> list:
    """
    Robustly extract up to 5 associations from the model response.
    Handles 'output:', 'Output:', and comma-separated lists.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # 1. Look for the "output:" prefix
    m = re.search(r"output\s*:(.*?)(?:\.|\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        content = m.group(1).strip()
    else:
        # Fallback: take the first non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        content = lines[0] if lines else ""

    # 2. Split by commas and clean
    # Remove trailing period if it exists
    content = content.rstrip('.')
    raw_words = [w.strip().lower() for w in content.split(',')]
    
    # 3. Clean each word (remove non-alphanumeric if leading/trailing)
    # Note: We keep internal hyphens for compound words
    cleaned_words = []
    for w in raw_words:
        w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if w_clean:
            cleaned_words.append(w_clean)
            
    return cleaned_words[:5]

# =============================================================================
# Matrix Generation Logic (Counts + PPMI) from 02_processed/*/long_df.csv
# =============================================================================

def _build_vocabs_from_long_df(long_df_path: str, chunksize: int = 500_000):
    row_set = set()
    col_set = set()

    for chunk in pd.read_csv(long_df_path, usecols=["cue_word", "association"], dtype="string", chunksize=chunksize):
        cue = _clean_series(chunk["cue_word"])
        assoc = _clean_series(chunk["association"])
        valid = cue.notna() & assoc.notna()
        cue = cue[valid]
        assoc = assoc[valid]
        # Keep cue repetitions in vocab, but they can be filtered when building counts
        row_set.update(cue.unique().tolist())
        col_set.update(assoc.unique().tolist())

    row_vocab = pd.Index(sorted(row_set), dtype="string")
    col_vocab = pd.Index(sorted(col_set), dtype="string")
    return row_vocab, col_vocab

def generate_matrices_from_processed(model_alias: str, no_cosine: bool = False, chunksize: int = 500_000):
    """
    Build and save cue x association matrices for a model using its processed long_df.csv.
    Output scheme mirrors FC postprocessing:
      - counts_matrix.npz
      - ppmi_matrix.npz
      - cosine_sim_counts.npy / cosine_sim_ppmi.npy (optional)
      - row_vocab.csv (input_word)
      - col_vocab.csv (output_word)
      - summary.json
    """
    in_dir = os.path.join(PROCESSED_BASE_DIR, model_alias)
    long_df_path = os.path.join(in_dir, "long_df.csv")
    if not os.path.exists(long_df_path):
        print(f"Skipping matrix gen for {model_alias}: missing {long_df_path}")
        return None

    out_dir = os.path.join(POSTPROCESSED_BASE_DIR, model_alias)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n>>> Generating matrices for {model_alias} from {long_df_path}")

    # Pass 1: vocab discovery (chunked)
    row_vocab, col_vocab = _build_vocabs_from_long_df(long_df_path, chunksize=chunksize)
    row_to_id = {w: i for i, w in enumerate(row_vocab.tolist())}
    col_to_id = {w: i for i, w in enumerate(col_vocab.tolist())}

    # Pass 2: build sparse counts (chunked)
    counts = sp.csr_matrix((len(row_vocab), len(col_vocab)), dtype=np.int32)
    for chunk in pd.read_csv(long_df_path, usecols=["cue_word", "association"], dtype="string", chunksize=chunksize):
        cue = _clean_series(chunk["cue_word"])
        assoc = _clean_series(chunk["association"])
        valid = cue.notna() & assoc.notna() & (assoc != cue)
        cue = cue[valid]
        assoc = assoc[valid]
        if cue.empty:
            continue

        r = cue.map(row_to_id).astype(np.int32)
        c = assoc.map(col_to_id).astype(np.int32)

        # Aggregate within chunk to reduce duplicates before building sparse matrix
        tmp = pd.DataFrame({"r": r.values, "c": c.values})
        vc = tmp.value_counts(sort=False).reset_index(name="v")
        coo = sp.coo_matrix(
            (vc["v"].astype(np.int32).to_numpy(), (vc["r"].to_numpy(), vc["c"].to_numpy())),
            shape=counts.shape
        ).tocsr()
        counts = counts + coo

    N = int(counts.sum())

    # Compute PPMI (same formula as FC)
    row_sums = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    col_sums = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    coo = counts.tocoo()
    denom = row_sums[coo.row] * col_sums[coo.col]
    pmi = np.log((coo.data.astype(np.float64) * float(N)) / denom)
    ppmi_data = np.maximum(pmi, 0.0)
    ppmi = sp.coo_matrix((ppmi_data.astype(np.float32), (coo.row, coo.col)), shape=counts.shape).tocsr()

    # Save matrices
    sp.save_npz(os.path.join(out_dir, "counts_matrix.npz"), counts)
    sp.save_npz(os.path.join(out_dir, "ppmi_matrix.npz"), ppmi)

    # Helper for cosine sim
    def save_cosine(mat, name):
        if sp.issparse(mat):
            A = mat.astype(np.float32)
            row_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
        else:
            A = mat.astype(np.float32)
            row_norm = np.linalg.norm(A, axis=1)
            
        inv = np.zeros_like(row_norm)
        nz = row_norm > 0
        inv[nz] = 1.0 / row_norm[nz]
        
        if sp.issparse(mat):
            A_norm = sp.diags(inv) @ A
            sim = (A_norm @ A_norm.T).toarray()
        else:
            A_norm = A * inv[:, np.newaxis]
            sim = A_norm @ A_norm.T
            
        np.save(os.path.join(out_dir, f"cosine_sim_{name}.npy"), sim.astype(np.float32))

    # Optional cosine similarities (cue-cue)
    if not no_cosine:
        # 1. Raw Matrices
        save_cosine(counts, "counts")
        save_cosine(ppmi, "ppmi")

        # 2. SVD Variants
        svd_dims = [100, 300, 600]
        matrices = {"counts": counts, "ppmi": ppmi}
        
        for m_name, matrix in matrices.items():
            for dim in svd_dims:
                # Check if matrix is large enough
                if matrix.shape[1] > dim:
                    print(f"   Computing SVD-{dim} for {m_name}...")
                    svd = TruncatedSVD(n_components=dim, random_state=42)
                    reduced = svd.fit_transform(matrix)
                    
                    # Save Embedding
                    np.save(os.path.join(out_dir, f"svd{dim}_{m_name}_embedding.npy"), reduced.astype(np.float32))
                    
                    # Compute & Save Cosine Sim
                    save_cosine(reduced, f"svd{dim}_{m_name}")
                else:
                    print(f"   Skipping SVD-{dim} for {m_name} (dim {matrix.shape[1]} <= {dim})")

    # Save vocabs
    pd.DataFrame({"input_word": row_vocab}).to_csv(os.path.join(out_dir, "row_vocab.csv"), index=False)
    pd.DataFrame({"output_word": col_vocab}).to_csv(os.path.join(out_dir, "col_vocab.csv"), index=False)

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "model": model_alias,
                "n_rows": int(len(row_vocab)),
                "n_cols": int(len(col_vocab)),
                "total_counts": int(N),
            },
            f,
            indent=2,
        )

    print(f"Done. Results in {out_dir} (rows={len(row_vocab)}, cols={len(col_vocab)}, total_counts={N})")
    return {"model": model_alias, "out_dir": out_dir, "n_rows": int(len(row_vocab)), "n_cols": int(len(col_vocab)), "total_counts": int(N)}

def process_model(model_alias, file_list):
    print(f"\nProcessing Model: {model_alias} ({len(file_list)} runs)")
    
    all_trials = []
    for f in file_list:
        df = pd.read_csv(f)
        # Some older files might have 'cue_word', some might have 'word'
        cue_col = 'cue_word' if 'cue_word' in df.columns else 'word'
        
        for _, row in df.iterrows():
            raw_resp = row.get('response', '')
            cue = str(row[cue_col]).lower()
            assocs = extract_associations(raw_resp)
            
            trial_data = {
                "cue_word": cue,
                "run_file": os.path.basename(f),
                "n_assocs": len(assocs),
                "repeats_cue": cue in assocs,
                "assocs": assocs
            }
            all_trials.append(trial_data)

    df_processed = pd.DataFrame(all_trials)
    
    # 1. Quality Metrics
    quality = {
        "model": model_alias,
        "total_trials": len(df_processed),
        "mean_assocs": df_processed["n_assocs"].mean(),
        "pct_full_compliance": (df_processed["n_assocs"] == 5).mean() * 100,
        "pct_zero_assocs": (df_processed["n_assocs"] == 0).mean() * 100,
        "pct_cue_repetition": df_processed["repeats_cue"].mean() * 100,
    }
    
    # 2. Diversity Metrics
    all_produced_words = [item for sublist in df_processed["assocs"] for item in sublist]
    unique_words_total = len(set(all_produced_words))
    
    # Unique associates per cue
    diversity_per_cue = df_processed.groupby("cue_word")["assocs"].apply(
        lambda x: len(set([item for sublist in x for item in sublist]))
    )
    
    quality["unique_words_total"] = unique_words_total
    quality["mean_unique_per_cue"] = diversity_per_cue.mean()
    
    # 3. Save Processed Data
    model_out_dir = os.path.join(PROCESSED_BASE_DIR, model_alias)
    os.makedirs(model_out_dir, exist_ok=True)
    
    # Write long_df.csv incrementally to save memory.
    final_long_df_path = os.path.join(model_out_dir, "long_df.csv")
    tmp_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
    if tmp_root and os.path.isdir(tmp_root):
        tmp_long_df_path = os.path.join(tmp_root, f"{model_alias}_long_df_{os.getpid()}.csv")
        tmp_hint = f"(temp: {tmp_long_df_path})"
    else:
        tmp_long_df_path = final_long_df_path + ".tmp"
        tmp_hint = f"(temp: {tmp_long_df_path})"
    print(f"Writing {final_long_df_path} {tmp_hint} ...")
    
    first_chunk = True
    # Process trials in chunks to avoid massive memory allocation
    chunk_size = 20000
    try:
        for i in range(0, len(all_trials), chunk_size):
            chunk = all_trials[i:i + chunk_size]
            rows = []
            for trial in chunk:
                for pos, val in enumerate(trial["assocs"]):
                    rows.append({
                        "cue_word": trial["cue_word"],
                        "run_file": trial["run_file"],
                        "position": pos + 1,
                        "association": val
                    })
            
            # Use pandas' internal chunked writer too, to reduce large write bursts.
            pd.DataFrame(rows).to_csv(
                tmp_long_df_path,
                mode='a' if not first_chunk else 'w',
                header=first_chunk,
                index=False,
                chunksize=50_000
            )
            first_chunk = False

        # Move finished file into place (cross-filesystem safe)
        if os.path.abspath(tmp_long_df_path) != os.path.abspath(final_long_df_path):
            shutil.move(tmp_long_df_path, final_long_df_path)
        else:
            # same path (shouldn't happen), but keep for completeness
            pass
    finally:
        # Best-effort cleanup if a temp file remains
        if os.path.exists(tmp_long_df_path) and os.path.abspath(tmp_long_df_path) != os.path.abspath(final_long_df_path):
            # If move succeeded, tmp won't exist. If it failed, remove partial temp.
            try:
                os.remove(tmp_long_df_path)
            except OSError:
                pass
    
    # We return an empty long_df to main because we've already written it to disk
    # and we only need the pairs for Jaccard overlap
    all_pairs = set()
    for t in all_trials:
        for a in t["assocs"]:
            all_pairs.add((t["cue_word"], a))

    return quality, all_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cosine", action="store_true", help="Do not compute cosine similarities (saves time/memory).")
    parser.add_argument("--matrix-chunksize", type=int, default=500_000, help="CSV chunk size for building matrices.")
    args = parser.parse_args()

    # 1. Group files by model
    all_csvs = glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    model_files = {}
    
    if not all_csvs:
        print(f"Warning: No CSV files found in {RAW_DIR}")
        return
    
    for f in all_csvs:
        basename = os.path.basename(f)
        if "_FA_run" in basename:
            alias = basename.split("_FA_run")[0]
        elif "_associations_run" in basename:
            alias = basename.split("_associations_run")[0]
        else:
            continue
            
        if alias not in model_files:
            model_files[alias] = []
        model_files[alias].append(f)

    # 2. Process each model
    all_quality_reports = []
    all_model_assocs = {} # alias -> set of (cue, assoc) pairs

    for alias, files in model_files.items():
        q_report, pairs = process_model(alias, files)
        all_quality_reports.append(q_report)
        all_model_assocs[alias] = pairs

    # 3. Summarize Quality
    df_quality = pd.DataFrame(all_quality_reports)
    if df_quality.empty:
        print("No quality reports generated. Check file naming conventions.")
        return

    quality_path = os.path.join(PROCESSED_BASE_DIR, "overall_quality_report.csv")
    df_quality.to_csv(quality_path, index=False)
    
    print("\n--- Overall Quality Report ---")
    print(df_quality.sort_values("pct_full_compliance", ascending=False).to_string(index=False))

    # 4. Cross-Model Consistency (Jaccard Similarity of Association Pairs)
    if len(all_model_assocs) > 1:
        print("\n--- Cross-Model Association Overlap (Jaccard) ---")
        models = sorted(all_model_assocs.keys())
        overlap_matrix = np.zeros((len(models), len(models)))
        
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                s1 = all_model_assocs[m1]
                s2 = all_model_assocs[m2]
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
        
        df_overlap = pd.DataFrame(overlap_matrix, index=models, columns=models)
        overlap_path = os.path.join(PROCESSED_BASE_DIR, "cross_model_overlap.csv")
        df_overlap.to_csv(overlap_path)
        print(df_overlap)

    # 5. Build matrices from processed long_df.csv for each model
    print("\n--- Building FA Count + PPMI Matrices (from 02_processed) ---")
    matrix_summaries = []
    for alias in sorted(model_files.keys()):
        summary = generate_matrices_from_processed(
            alias,
            no_cosine=args.no_cosine,
            chunksize=args.matrix_chunksize,
        )
        if summary is not None:
            matrix_summaries.append(summary)

    if matrix_summaries:
        df_mat = pd.DataFrame(matrix_summaries).sort_values(["n_rows", "model"])
        print("\n--- Matrix Build Summary ---")
        print(df_mat.to_string(index=False))
        df_mat.to_csv(os.path.join(POSTPROCESSED_BASE_DIR, "matrix_build_summary.csv"), index=False)
        print(f"\nSaved matrix build summary to: {os.path.join(POSTPROCESSED_BASE_DIR, 'matrix_build_summary.csv')}")

if __name__ == "__main__":
    main()

