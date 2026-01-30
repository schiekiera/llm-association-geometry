#!/usr/bin/env python3
import os
import re
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("projects/icml_project")
FC_POSTPROCESSED_ROOT = PROJECT_ROOT / "output/02_behavioral_associations/01_forced_choice/03_postprocessed"
FA_POSTPROCESSED_ROOT = PROJECT_ROOT / "output/02_behavioral_associations/02_free_associations/03_postprocessed"
HIDDEN_ROOT = PROJECT_ROOT / "output/01_hidden_state_embeddings/final_hidden_states"
OUTPUT_DIR_BASE = PROJECT_ROOT / "output/eval/03_SVD_variants"

# The 8 variants to compare
VARIANTS = [
    ("counts", "cosine_sim_counts.npy"),
    ("ppmi", "cosine_sim_ppmi.npy"),
    ("svd100_counts", "cosine_sim_svd100_counts.npy"),
    ("svd300_counts", "cosine_sim_svd300_counts.npy"),
    ("svd600_counts", "cosine_sim_svd600_counts.npy"),
    ("svd100_ppmi", "cosine_sim_svd100_ppmi.npy"),
    ("svd300_ppmi", "cosine_sim_svd300_ppmi.npy"),
    ("svd600_ppmi", "cosine_sim_svd600_ppmi.npy"),
]

PROMPT_NAMES = ["averaged", "template", "force_choice", "free_association"]

# =============================================================================
# Logging
# =============================================================================

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"eval_variants_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================

def _torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def center_and_normalize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.maximum(norms, 1e-12)

def _tri_offsets(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return (i * (2 * n - i - 1)) // 2

def _tri_k_to_ij(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = _tri_offsets(n)
    i = np.searchsorted(offsets, k, side="right") - 1
    j = i + 1 + (k - offsets[i])
    return i.astype(np.int64), j.astype(np.int64)

def _sample_pairs(n: int, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    m = n * (n - 1) // 2
    if sample_size >= m:
        upper = np.triu_indices(n, k=1)
        return upper[0].astype(np.int64), upper[1].astype(np.int64)
    rng = np.random.default_rng(seed)
    k = rng.choice(m, size=sample_size, replace=False).astype(np.int64)
    return _tri_k_to_ij(k, n)

def get_clean_alias(alias):
    alias = re.sub(r"_\d{8}_\d{6}$", "", alias)
    alias = alias.replace("_small", "").replace("_large", "").replace("_old", "")
    return alias

def evaluate_variant(model_alias, prompt, hidden_states, hidden_words,
                     beh_sim, beh_vocab, variant_name, task_type, sample_size, seed, hidden_cosine=None):
    
    hidden_cosine = None  # Always recompute from centered hidden states.
    # Align Vocabs
    h_map = {w: i for i, w in enumerate(hidden_words)}
    b_map = {w: i for i, w in enumerate(beh_vocab)}
    
    # Common words
    common = sorted([w for w in hidden_words if w in b_map])
    if len(common) < 50:
        return None
        
    idx_h = [h_map[w] for w in common]
    idx_b = [b_map[w] for w in common]
    
    # Subset behavioral matrix
    curr_beh = beh_sim[np.ix_(idx_b, idx_b)]
    
    if hidden_cosine is not None:
        # hidden_cosine is (L, N_orig, N_orig)
        curr_hid_cosine = hidden_cosine[:, idx_h][:, :, idx_h]
        L = curr_hid_cosine.shape[0]
        N = len(common)
    else:
        curr_hid = hidden_states[idx_h]
        N, L, H = curr_hid.shape
    
    # Sample pairs
    ii, jj = _sample_pairs(N, sample_size, seed)
    v_beh = curr_beh[ii, jj]
    logger.info(
        f"    Variant={variant_name} Prompt={prompt} Task={task_type} | "
        f"N_common={N} pairs={len(ii)} layers={L}"
    )
    
    results = []
    # Layerwise Correlation
    for l in range(L):
        if hidden_cosine is not None:
            layer_matrix = curr_hid_cosine[l].numpy() if isinstance(curr_hid_cosine, torch.Tensor) else curr_hid_cosine[l]
            v_hid = layer_matrix[ii, jj]
        else:
            layer_tensor = curr_hid[:, l, :].to(torch.float32)
            normed = center_and_normalize(layer_tensor.numpy())
            v_hid = np.einsum("ij,ij->i", normed[ii], normed[jj])
        
        r, _ = pearsonr(v_hid, v_beh)
        
        results.append({
            "task": task_type,
            "model": model_alias,
            "prompt": prompt,
            "variant": variant_name,
            "layer": l,
            "pearson": r,
            "n_common_words": N,
            "n_pairs": len(ii)
        })
        
    return results

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None, help="Optional single model alias to run.")
    parser.add_argument("--job-index", type=int, default=None, help="Zero-based job index for array runs.")
    parser.add_argument("--job-count", type=int, default=None, help="Total number of array jobs.")
    args = parser.parse_args()

    setup_logging(OUTPUT_DIR_BASE)
    
    # Define tasks to process
    tasks = [
        ("ForcedChoice", FC_POSTPROCESSED_ROOT),
        ("FreeAssociation", FA_POSTPROCESSED_ROOT)
    ]
    
    all_results = []
    models_seen = set()
    
    for task_name, task_root in tasks:
        logger.info(f"\n=== Processing Task: {task_name} ===")
        
        if not task_root.exists():
            logger.warning(f"Root directory for {task_name} not found: {task_root}")
            continue
        
        # Discover available models in postprocessed directory
        model_dirs = sorted([d for d in task_root.iterdir() if d.is_dir()])
        if args.model:
            model_dirs = [d for d in model_dirs if d.name == args.model]
        if args.job_index is not None or args.job_count is not None:
            if args.job_index is None or args.job_count is None:
                raise ValueError("--job-index and --job-count must be provided together.")
            model_dirs = [d for i, d in enumerate(model_dirs) if i % args.job_count == args.job_index]
            logger.info(f"Array selection: job {args.job_index + 1}/{args.job_count}")
        logger.info(f"Found {len(model_dirs)} models to evaluate for {task_name}.")
        
        for model_idx, model_dir in enumerate(model_dirs, start=1):
            model_alias = model_dir.name
            logger.info(f"[{model_idx}/{len(model_dirs)}] Processing Model: {model_alias}")
            models_seen.add(model_alias)
            
            # Load Vocab for this model
            vocab_path = model_dir / "row_vocab.csv"
            if not vocab_path.exists():
                logger.warning(f"  Missing row_vocab.csv for {model_alias}, skipping.")
                continue
                
            vocab_df = pd.read_csv(vocab_path)
            # Assuming column is 'input_word' based on postprocessing script
            beh_vocab = vocab_df["input_word"].astype(str).str.strip().str.lower().tolist()
    
            # Load Hidden States (Once per model, for all prompts)
            model_clean = get_clean_alias(model_alias)
            hidden_data_cache = {} 
            
            for prompt in PROMPT_NAMES:
                hpath = HIDDEN_ROOT / model_clean / prompt / "hidden_states.pt"
                if not hpath.exists():
                    # Try fallback to full alias if clean didn't work (or vice versa logic from other scripts)
                    hpath = HIDDEN_ROOT / model_alias / prompt / "hidden_states.pt"
                
                if hpath.exists():
                    try:
                        data = _torch_load_compat(hpath)
                        words = [str(w).strip().lower() for w in data["cue_words"]]
                        hid_cos = None
                        hidden_data_cache[prompt] = (data["hidden_states"], words, hid_cos)
                        logger.info(f"  Loaded hidden states for prompt={prompt} (layers={data['hidden_states'].shape[1]})")
                    except Exception as e:
                        logger.error(f"  Error loading hidden states {hpath}: {e}")
                else:
                    logger.warning(f"  Hidden states not found for {model_alias}/{prompt}")
    
            if not hidden_data_cache:
                logger.warning(f"  Skipping {model_alias} (no hidden states found)")
                continue
    
            # Iterate Variants
            for var_idx, (var_name, filename) in enumerate(VARIANTS, start=1):
                mat_path = model_dir / filename
                if not mat_path.exists():
                    # Optional warning or debug
                    # logger.debug(f"  Missing variant {var_name} ({filename}) for {model_alias}")
                    continue
                    
                logger.info(f"  [{var_idx}/{len(VARIANTS)}] Evaluating variant: {var_name}")
                try:
                    beh_sim = np.load(mat_path)
                except Exception as e:
                    logger.error(f"  Failed to load {mat_path}: {e}")
                    continue
                
                # Evaluate against all prompts
                for prompt_idx, (prompt, (h_states, h_words, h_cos)) in enumerate(hidden_data_cache.items(), start=1):
                    logger.info(f"    [{prompt_idx}/{len(hidden_data_cache)}] Prompt: {prompt}")
                    res = evaluate_variant(
                        model_alias, prompt, h_states, h_words, 
                        beh_sim, beh_vocab, var_name, task_name,
                        args.pair_sample_size, args.seed, 
                        hidden_cosine=h_cos
                    )
                    if res:
                        all_results.extend(res)
    
    # Save Results
    if all_results:
        df_res = pd.DataFrame(all_results)
        OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        for model_alias in sorted(models_seen):
            df_model = df_res[df_res["model"] == model_alias]
            if df_model.empty:
                continue
            safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_alias)
            out_csv = OUTPUT_DIR_BASE / (
                f"svd_variants_results_{safe_model}_pairs{args.pair_sample_size}_{date_str}.csv"
            )
            df_model.to_csv(out_csv, index=False)
            logger.info(f"Model results saved to: {out_csv}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
