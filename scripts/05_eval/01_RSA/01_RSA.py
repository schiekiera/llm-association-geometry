import os
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse
import gzip
import logging
import time
from datetime import datetime
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"
HIDDEN_ROOT = "data/01_hidden_state_embeddings/final_hidden_states"
BEHAVIORAL_ROOT = "data/02_behavioral_associations/01_forced_choice/03_postprocessed"
FA_BEHAVIORAL_ROOT = "data/02_behavioral_associations/02_free_associations/03_postprocessed"
FASTTEXT_PATH = "data/further_embeddings/fasttext/cc.en.300.vec.gz"
FASTTEXT_BENCH_DIR = "data/further_embeddings/fasttext"
BERT_BENCH_DIR = "data/further_embeddings/bert"
OUTPUT_DIR_BASE = "data/eval/01_RSA"

BERT_MODEL = "bert-base-uncased"
BASE_PROMPT = "This is a "

# List of prompt names used in extraction
PROMPT_NAMES = ["template", "averaged", "forced_choice", "free_association"]

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_{timestamp}.log")
    
    # Configure logging
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
# Pair sampling utilities
# =============================================================================

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

# =============================================================================
# Torch load helper
# =============================================================================

def _torch_load_compat(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


_HIDDEN_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}


def _load_hidden_bundle(model_alias: str, prompt_name: str, *, use_cache: bool = True) -> dict[str, Any] | None:
    cache_key = (model_alias, prompt_name)
    if use_cache and cache_key in _HIDDEN_CACHE:
        return _HIDDEN_CACHE[cache_key]

    clean_alias = get_clean_alias(model_alias)
    hid_path = os.path.join(HIDDEN_ROOT, clean_alias, prompt_name, "hidden_states.pt")
    if not os.path.exists(hid_path):
        hid_path = os.path.join(HIDDEN_ROOT, model_alias, prompt_name, "hidden_states.pt")
        if not os.path.exists(hid_path):
            if use_cache:
                _HIDDEN_CACHE[cache_key] = None
            return None

    try:
        hid_data = _torch_load_compat(hid_path)
        bundle = {
            "hidden_states": hid_data.get("hidden_states"),
            "cue_words": hid_data.get("cue_words"),
            "cosine_similarity_matrices": hid_data.get("cosine_similarity_matrices"),
        }
        if use_cache:
            _HIDDEN_CACHE[cache_key] = bundle
        return bundle
    except Exception as e:
        logger.error(f"  Failed to load hidden states for {model_alias} {prompt_name}: {e}")
        if use_cache:
            _HIDDEN_CACHE[cache_key] = None
        return None

# =============================================================================
# Helper Functions
# =============================================================================

def _load_benchmark_bundle(bench_dir: str, words: list[str], kind: str):
    words_path = os.path.join(bench_dir, "benchmark_words.csv")
    sim_path = os.path.join(bench_dir, "benchmark_cosine_sim.npy")
    if not os.path.exists(words_path) or not os.path.exists(sim_path):
        raise FileNotFoundError(f"Missing precomputed {kind} benchmarks in {bench_dir}. Run: python scripts/01_preprocessing/process_benchmarks.py --only {kind}")

    bench_words = pd.read_csv(words_path)["word"].astype(str).str.strip().str.lower().tolist()
    words_norm = [str(w).strip().lower() for w in words]
    pos = {w: i for i, w in enumerate(bench_words)}
    idx = np.array([pos[w] for w in words_norm], dtype=int)
    sim = np.load(sim_path)
    sim = sim[np.ix_(idx, idx)]
    out = {"sim": sim, "bench_words": bench_words}
    if kind == "fasttext":
        valid = np.load(os.path.join(bench_dir, "benchmark_valid_mask.npy")).astype(bool)[idx]
        out["valid_mask"] = valid
    return out

def _load_cosine_sim_and_vocab(root_dir: str, model_alias: str):
    model_dir = os.path.join(root_dir, model_alias)
    ppmi_path = os.path.join(model_dir, "cosine_sim_ppmi.npy")
    counts_path = os.path.join(model_dir, "cosine_sim_counts.npy")
    vocab_path = os.path.join(model_dir, "row_vocab.csv")
    if not os.path.exists(vocab_path): return None, None, None
    vocab_df = pd.read_csv(vocab_path)
    row_vocab = vocab_df["input_word"].astype(str).str.strip().str.lower().tolist()
    if os.path.exists(ppmi_path): return np.load(ppmi_path), "PPMI", row_vocab
    if os.path.exists(counts_path): return np.load(counts_path), "Counts", row_vocab
    return None, None, row_vocab

def get_clean_alias(alias):
    alias = re.sub(r"_\d{8}_\d{6}$", "", alias)
    return alias.replace("_small", "").replace("_large", "").replace("_old", "")

def center_and_normalize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.maximum(norms, 1e-12)

def build_crossmodel_consensus_pairs(
    m_target: str,
    prompt_name: str,
    common_words: list[str],
    ii: np.ndarray,
    jj: np.ndarray,
    model_aliases: list[str],
    *,
    show_progress: bool = True,
) -> tuple[np.ndarray, dict]:
    v_sum = np.zeros(len(ii), dtype=np.float32)
    denom = 0
    n_models = 0
    layer_count = 0

    start_time = time.time()
    pbar = tqdm(
        total=0,
        desc=f"Consensus {m_target}|{prompt_name}",
        unit="layer",
        disable=not show_progress,
    )

    for alias in model_aliases:
        if alias == m_target:
            continue

        bundle = _load_hidden_bundle(alias, prompt_name, use_cache=False)
        if bundle is None:
            continue

        cue_words = bundle.get("cue_words") or []
        if not cue_words:
            continue

        cue_words_norm = [str(w).strip().lower() for w in cue_words]
        h_map = {w: i for i, w in enumerate(cue_words_norm)}
        if not all(w in h_map for w in common_words):
            continue

        idx_src = np.array([h_map[w] for w in common_words], dtype=np.int64)
        ii_src = idx_src[ii]
        jj_src = idx_src[jj]

        # Use hidden states to avoid materializing full cosine matrices.
        hid_states = bundle.get("hidden_states")
        if hid_states is None:
            continue

        n_models += 1
        if isinstance(hid_states, torch.Tensor):
            hid_states = hid_states.cpu()
        L = hid_states.shape[1]
        pbar.total += max(0, L - 1)
        pbar.refresh()
        for l in range(1, L):
            layer_hid = hid_states[idx_src, l, :]
            if isinstance(layer_hid, torch.Tensor):
                layer_hid = layer_hid.to(torch.float32).numpy()
            layer_hid = center_and_normalize(layer_hid)
            v_sum += np.einsum("ij,ij->i", layer_hid[ii_src], layer_hid[jj_src]).astype(
                np.float32, copy=False
            )
            denom += 1
            layer_count += 1
            pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time
    logger.info(
        f"  Consensus build time: {elapsed:.1f}s | models={n_models} layers={layer_count}"
    )

    meta = {"cross_n_models": n_models, "cross_n_layers": layer_count}
    if denom == 0:
        v_cons = np.full(len(ii), np.nan, dtype=np.float32)
    else:
        v_cons = v_sum / float(denom)
    return v_cons, meta

def evaluate_model_prompt(
    model_alias,
    prompt_name,
    ft_sim,
    ft_valid_words,
    bert_sim,
    words,
    pair_sample_size: int,
    seed: int,
    model_aliases: list[str],
    *,
    show_progress: bool = False,
):
    logger.info(f"Evaluating: {model_alias} | Prompt: {prompt_name}")
    fc_sim, fc_type, fc_vocab = _load_cosine_sim_and_vocab(BEHAVIORAL_ROOT, model_alias)
    if fc_sim is None: return None
    fa_sim, fa_type, fa_vocab = _load_cosine_sim_and_vocab(FA_BEHAVIORAL_ROOT, model_alias)
    if fa_sim is None: return None
    
    bundle = _load_hidden_bundle(model_alias, prompt_name)
    if bundle is None:
        return None
    hid_states = bundle.get("hidden_states")
    hid_words = bundle.get("cue_words")
    hid_cosine = None  # Always recompute from centered hidden states.
    
    hid_words_norm = [str(w).strip().lower() for w in hid_words]
    h_map, fc_map, fa_map = {w: i for i, w in enumerate(hid_words_norm)}, {w: i for i, w in enumerate(fc_vocab)}, {w: i for i, w in enumerate(fa_vocab)}
    words_norm = [str(w).strip().lower() for w in words]
    common_words = [w for w in words_norm if (w in h_map) and (w in fc_map) and (w in fa_map)]
    if len(common_words) < 100: return None

    idx_h, idx_hs, idx_fa = np.array([h_map[w] for w in common_words]), np.array([fc_map[w] for w in common_words]), np.array([fa_map[w] for w in common_words])
    
    # Slice precomputed hidden cosine if available
    if hid_cosine is not None:
        # hid_cosine is (L, N_orig, N_orig)
        # We need (L, N_common, N_common)
        # Use np.ix_ logic for the last two dims
        hid_cosine = hid_cosine[:, idx_h][:, :, idx_h] # Slicing tensor
    else:
        hid_states = hid_states[idx_h] # Only slice states if we need to compute

    fc_sim, fa_sim = fc_sim[np.ix_(idx_hs, idx_hs)], fa_sim[np.ix_(idx_fa, idx_fa)]
    wpos = {w: i for i, w in enumerate(words_norm)}
    idx_global = np.array([wpos[w] for w in common_words])
    ft_valid_words, ft_sim, bert_sim = ft_valid_words[idx_global], ft_sim[np.ix_(idx_global, idx_global)], bert_sim[np.ix_(idx_global, idx_global)]
    
    # N is common words count
    N = len(common_words)
    # L is number of layers
    L = hid_cosine.shape[0] if hid_cosine is not None else hid_states.shape[1]
    # Exclude layer 0 (embedding layer; not a hidden state from a transformer block)
    if L <= 1:
        logger.warning(f"  Not enough layers after excluding layer 0 (L={L}). Skipping.")
        return None

    if pair_sample_size > 0:
        ii, jj = _sample_pairs(N, pair_sample_size, seed=seed); pair_mode = "sampled"
    else:
        upper = np.triu_indices(N, k=1); ii, jj = upper[0].astype(np.int64), upper[1].astype(np.int64); pair_mode = "full"
    
    n_pairs = int(len(ii))
    logger.info(
        f"  Common words={N} | layers={L - 1} | pairs={n_pairs} | pair_mode={pair_mode}"
    )
    date_tag = datetime.now().strftime("%Y%m%d")
    sample_tag = f"pairs{n_pairs}" if pair_mode == "sampled" else f"pairsAll_{n_pairs}"
    
    valid_pairs_ft = ft_valid_words[ii] & ft_valid_words[jj]
    v_hs, v_fa, v_ft, v_bert = fc_sim[ii, jj], fa_sim[ii, jj], ft_sim[ii, jj][valid_pairs_ft], bert_sim[ii, jj]
    
    v_consensus, cross_meta = build_crossmodel_consensus_pairs(
        model_alias,
        prompt_name,
        common_words,
        ii,
        jj,
        model_aliases,
        show_progress=show_progress,
    )
    logger.info(
        f"  Cross-model consensus: models={cross_meta['cross_n_models']} "
        f"layers={cross_meta['cross_n_layers']}"
    )
    results = []
    target_out_dir = os.path.join(OUTPUT_DIR_BASE, model_alias, prompt_name)
    os.makedirs(target_out_dir, exist_ok=True)

    # Start at layer 1 to exclude embedding layer 0
    layer_iter = range(1, L)
    if show_progress:
        layer_iter = tqdm(layer_iter, desc=f"Layers RSA {model_alias}|{prompt_name}", unit="layer")
    for l in layer_iter:
        if hid_cosine is not None:
            # Use precomputed matrix
            # hid_cosine is torch tensor, convert to numpy
            layer_matrix = hid_cosine[l].numpy() if isinstance(hid_cosine, torch.Tensor) else hid_cosine[l]
            v_hid = layer_matrix[ii, jj]
        else:
            # Compute on the fly
            layer_hid = center_and_normalize(hid_states[:, l, :].to(torch.float32).numpy())
            v_hid = np.einsum("ij,ij->i", layer_hid[ii], layer_hid[jj]) if pair_mode == "sampled" else cosine_similarity(layer_hid)[ii, jj]
            
        results.append({
            "layer": l,
            f"pearson_forced_choice_{fc_type}": pearsonr(v_hid, v_hs)[0],
            f"pearson_fa_{fa_type}": pearsonr(v_hid, v_fa)[0],
            "pearson_fasttext": pearsonr(v_hid[valid_pairs_ft], v_ft)[0],
            "pearson_bert": pearsonr(v_hid, v_bert)[0],
            "pearson_crossmodel": pearsonr(v_hid, v_consensus)[0] if np.isfinite(v_consensus).all() else np.nan,
            "pair_mode": pair_mode, "n_pairs": n_pairs, "n_pairs_fasttext": int(valid_pairs_ft.sum()), "seed": seed,
            "cross_n_models": cross_meta["cross_n_models"], "cross_n_layers": cross_meta["cross_n_layers"],
        })
    
    df_res = pd.DataFrame(results)
    output_csv = os.path.join(target_out_dir, f"layerwise_correlations_{sample_tag}_{date_tag}.csv")
    df_res.to_csv(output_csv, index=False)
    
    return df_res, fc_type, fa_type

def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def main():
    setup_logging(OUTPUT_DIR_BASE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-sample-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=None, help="Restrict to a single model alias.")
    ap.add_argument("--prompt", type=str, default=None, help="Restrict to a single prompt name.")
    ap.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Job index for array execution (maps to model/prompt).",
    )
    ap.add_argument(
        "--job-count",
        type=int,
        default=None,
        help="Total number of array jobs (for sanity check).",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary/plots (useful in array mode).",
    )
    args = ap.parse_args()

    vocab_df = pd.read_csv(VOCAB_PATH)
    words = sorted(vocab_df["word"].astype(str).unique().tolist())
    ft_bundle = _load_benchmark_bundle(FASTTEXT_BENCH_DIR, words, kind="fasttext")
    bert_bundle = _load_benchmark_bundle(BERT_BENCH_DIR, words, kind="bert")
    
    model_aliases = sorted(
        [d for d in os.listdir(BEHAVIORAL_ROOT) if os.path.isdir(os.path.join(BEHAVIORAL_ROOT, d))]
    )
    targets = [(m, p) for m in model_aliases for p in PROMPT_NAMES]

    if args.model or args.prompt:
        model_filter = args.model
        prompt_filter = args.prompt
        targets = [
            (m, p)
            for (m, p) in targets
            if (model_filter is None or m == model_filter)
            and (prompt_filter is None or p == prompt_filter)
        ]
    if args.job_index is not None:
        if args.job_count is not None and args.job_count != len(targets):
            logger.warning(f"Job count {args.job_count} != {len(targets)} targets.")
        if args.job_index < 0 or args.job_index >= len(targets):
            raise ValueError(f"job-index {args.job_index} out of range (0..{len(targets)-1})")
        targets = [targets[args.job_index]]

    write_summary = (
        (not args.no_summary)
        and args.job_index is None
        and args.model is None
        and args.prompt is None
    )

    all_summaries, all_layerwise = [], []

    show_progress = len(targets) == 1
    logger.info(f"Targets to process: {len(targets)}")

    for alias, prompt_name in targets:
            res_tuple = evaluate_model_prompt(
                alias,
                prompt_name,
                ft_bundle["sim"],
                ft_bundle["valid_mask"],
                bert_bundle["sim"],
                words,
                args.pair_sample_size,
                args.seed,
                model_aliases,
                show_progress=show_progress,
            )
            if res_tuple:
                res, fc_t, fa_t = res_tuple
                res["model"], res["prompt"] = alias, prompt_name
                all_layerwise.append(res)
                if write_summary:
                    all_summaries.append(
                        {
                            "model": alias,
                            "prompt": prompt_name,
                            "beh_type": fc_t,
                            "peak_forced_choice": res[f"pearson_forced_choice_{fc_t}"].max(),
                            "peak_layer": res.loc[res[f"pearson_forced_choice_{fc_t}"].idxmax(), "layer"],
                            "fa_type": fa_t,
                            "peak_fa": res[f"pearson_fa_{fa_t}"].max(),
                            "peak_layer_fa": res.loc[res[f"pearson_fa_{fa_t}"].idxmax(), "layer"],
                        }
                    )
    
    if all_layerwise and write_summary:
        date_tag = datetime.now().strftime("%Y%m%d")
        sample_tag = f"pairs{args.pair_sample_size}" if args.pair_sample_size > 0 else "pairsAll"
        
        # Save summary CSV
        df_all = pd.concat(all_layerwise, ignore_index=True)
        summary_path = os.path.join(OUTPUT_DIR_BASE, f"summary_all_results_{sample_tag}_{date_tag}.csv")
        df_all.to_csv(summary_path, index=False)
        logger.info(f"Full results saved to {summary_path}")
        
if __name__ == "__main__":
    main()
