#!/usr/bin/env python3
"""
Nearest-neighbor consistency evaluation.

For each input word i, retrieve its k nearest neighbors in both spaces and
compute the overlap proportion:
    NN@k(i) = |N^B_k(i) ∩ N^G_k(i)| / k
We report the mean NN@k across words per layer.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"
HIDDEN_ROOT = "data/01_hidden_state_embeddings/final_hidden_states"
BEHAVIORAL_ROOT = "data/02_behavioral_associations/01_forced_choice/03_postprocessed"
FA_BEHAVIORAL_ROOT = "data/02_behavioral_associations/02_free_associations/03_postprocessed"
FASTTEXT_BENCH_DIR = "data/further_embeddings/fasttext"
BERT_BENCH_DIR = "data/further_embeddings/bert"
OUTPUT_DIR_BASE = "data/eval/02_NN"

PROMPT_NAMES = ["template", "averaged", "forced_choice", "free_association"]

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)

# =============================================================================
# IO helpers
# =============================================================================

def _torch_load_compat(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_benchmark_bundle(bench_dir: str, words: list[str], kind: str):
    words_path = os.path.join(bench_dir, "benchmark_words.csv")
    sim_path = os.path.join(bench_dir, "benchmark_cosine_sim.npy")
    if not os.path.exists(words_path) or not os.path.exists(sim_path):
        raise FileNotFoundError(
            f"Missing precomputed {kind} benchmarks in {bench_dir}. "
            f"Run: python scripts/01_preprocessing/process_benchmarks.py --only {kind}"
        )

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
    if not os.path.exists(vocab_path):
        return None, None, None
    vocab_df = pd.read_csv(vocab_path)
    row_vocab = vocab_df["input_word"].astype(str).str.strip().str.lower().tolist()
    if os.path.exists(ppmi_path):
        return np.load(ppmi_path), "PPMI", row_vocab
    if os.path.exists(counts_path):
        return np.load(counts_path), "Counts", row_vocab
    return None, None, row_vocab


def get_clean_alias(alias: str) -> str:
    alias = re.sub(r"_\d{8}_\d{6}$", "", alias)
    return alias.replace("_small", "").replace("_large", "").replace("_old", "")


# Cache hidden bundles and consensus sums within a process
_HIDDEN_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}
_CONSENSUS_CACHE: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}


def _load_hidden_bundle(
    model_alias: str,
    prompt_name: str,
    *,
    use_cache: bool = True,
) -> dict[str, Any] | None:
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
# Nearest neighbors computation
# =============================================================================

def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)

def center_and_normalize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    return Xc / np.maximum(norms, 1e-12)

def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

def _topk_from_embeddings(
    emb: np.ndarray,
    k: int,
    *,
    device: str,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    n = emb.shape[0]
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.empty((n, 0), dtype=int), 0

    if device == "cpu":
        sim = emb @ emb.T
        return _topk_indices(sim, k_eff)

    emb_t = torch.from_numpy(emb).to(device)
    topk_idx = np.empty((n, k_eff), dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sim = emb_t[start:end] @ emb_t.T
        row_idx = torch.arange(start, end, device=device)
        sim[torch.arange(end - start, device=device), row_idx] = float("-inf")
        _, idx = torch.topk(sim, k_eff, dim=1)
        topk_idx[start:end] = idx.cpu().numpy()
        del sim, idx
    return topk_idx, k_eff

def _accumulate_cosine_sum(
    s_sum: np.ndarray,
    emb: np.ndarray,
    *,
    device: str,
    chunk_size: int,
) -> None:
    if device == "cpu":
        s_sum += (emb @ emb.T).astype(np.float32, copy=False)
        return

    emb_t = torch.from_numpy(emb).to(device)
    n = emb_t.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        sim = emb_t[start:end] @ emb_t.T
        s_sum[start:end] += sim.float().cpu().numpy()
        del sim

def _topk_indices(sim: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    n = sim.shape[0]
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.empty((n, 0), dtype=int), 0
    sim = np.nan_to_num(sim, nan=-np.inf)
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
    return idx, k_eff


def _mean_overlap(topk_a: np.ndarray, topk_b: np.ndarray, k: int) -> float:
    if k <= 0 or topk_a.size == 0 or topk_b.size == 0:
        return float("nan")
    overlaps = np.empty(topk_a.shape[0], dtype=float)
    for i in range(topk_a.shape[0]):
        overlaps[i] = np.intersect1d(topk_a[i], topk_b[i]).size / k
    return float(np.mean(overlaps))


def build_crossmodel_consensus_topk(
    m_target: str,
    prompt_name: str,
    common_words: list[str],
    k: int,
    model_aliases: list[str],
    *,
    show_progress: bool = True,
    device: str = "cpu",
    chunk_size: int = 1024,
) -> tuple[np.ndarray | None, int, dict]:
    key = (m_target, prompt_name, tuple(common_words))
    if key in _CONSENSUS_CACHE:
        cached = _CONSENSUS_CACHE[key]
        s_sum = cached["s_sum"]
        denom = cached["denom"]
        meta = cached["meta"]
    else:
        n_words = len(common_words)
        s_sum = np.zeros((n_words, n_words), dtype=np.float32)
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
                _accumulate_cosine_sum(
                    s_sum,
                    layer_hid,
                    device=device,
                    chunk_size=chunk_size,
                )
                denom += 1
                layer_count += 1
                pbar.update(1)

        pbar.close()
        elapsed = time.time() - start_time
        logger.info(
            f"  Consensus build time: {elapsed:.1f}s | models={n_models} layers={layer_count}"
        )

        meta = {"cross_n_models": n_models, "cross_n_layers": layer_count, "cross_denom": denom}
        _CONSENSUS_CACHE[key] = {"s_sum": s_sum, "denom": denom, "meta": meta}

    k_eff = min(k, s_sum.shape[0] - 1) if s_sum is not None else 0
    if s_sum is None or denom == 0 or k_eff <= 0:
        return None, k_eff, meta

    s_cons = s_sum / float(denom)
    consensus_topk, _ = _topk_indices(s_cons, k_eff)
    return consensus_topk, k_eff, meta


def evaluate_model_prompt(
    model_alias: str,
    prompt_name: str,
    ft_sim: np.ndarray,
    ft_valid_words: np.ndarray,
    bert_sim: np.ndarray,
    words: list[str],
    k: int,
    model_aliases: list[str],
    *,
    show_progress: bool = False,
    device: str = "cpu",
    chunk_size: int = 1024,
):
    logger.info(f"Evaluating NN@{k}: {model_alias} | Prompt: {prompt_name}")
    fc_sim, fc_type, fc_vocab = _load_cosine_sim_and_vocab(BEHAVIORAL_ROOT, model_alias)
    if fc_sim is None:
        return None
    fa_sim, fa_type, fa_vocab = _load_cosine_sim_and_vocab(FA_BEHAVIORAL_ROOT, model_alias)
    if fa_sim is None:
        return None

    bundle = _load_hidden_bundle(model_alias, prompt_name, use_cache=False)
    if bundle is None:
        return None
    hid_states = bundle.get("hidden_states")
    hid_words = bundle.get("cue_words")
    hid_cosine = None  # Always recompute from centered hidden states.

    hid_words_norm = [str(w).strip().lower() for w in hid_words]
    h_map = {w: i for i, w in enumerate(hid_words_norm)}
    fc_map = {w: i for i, w in enumerate(fc_vocab)}
    fa_map = {w: i for i, w in enumerate(fa_vocab)}
    words_norm = [str(w).strip().lower() for w in words]

    common_words = [w for w in words_norm if (w in h_map) and (w in fc_map) and (w in fa_map)]
    if len(common_words) < 100:
        return None

    idx_h = np.array([h_map[w] for w in common_words])
    idx_hs = np.array([fc_map[w] for w in common_words])
    idx_fa = np.array([fa_map[w] for w in common_words])

    # Slice precomputed hidden cosine if available
    if hid_cosine is not None:
        hid_cosine = hid_cosine[:, idx_h][:, :, idx_h]
    else:
        if isinstance(hid_states, torch.Tensor):
            hid_states = hid_states.cpu()
        hid_states = hid_states[idx_h]

    fc_sim = fc_sim[np.ix_(idx_hs, idx_hs)]
    fa_sim = fa_sim[np.ix_(idx_fa, idx_fa)]
    wpos = {w: i for i, w in enumerate(words_norm)}
    idx_global = np.array([wpos[w] for w in common_words])
    ft_valid_words = ft_valid_words[idx_global]
    ft_sim = ft_sim[np.ix_(idx_global, idx_global)]
    bert_sim = bert_sim[np.ix_(idx_global, idx_global)]

    n_words = len(common_words)
    L = hid_cosine.shape[0] if hid_cosine is not None else hid_states.shape[1]
    if L <= 1:
        logger.warning(f"  Not enough layers after excluding layer 0 (L={L}). Skipping.")
        return None

    # Precompute top-k for targets (full word set)
    fc_topk, k_eff = _topk_indices(fc_sim, k)
    fa_topk, _ = _topk_indices(fa_sim, k_eff)
    bert_topk, _ = _topk_indices(bert_sim, k_eff)

    consensus_topk, k_eff_cons, cross_meta = build_crossmodel_consensus_topk(
        model_alias,
        prompt_name,
        common_words,
        k_eff,
        model_aliases,
        show_progress=show_progress,
        device=device,
        chunk_size=chunk_size,
    )
    logger.info(
        f"  Cross-model consensus: models={cross_meta.get('cross_n_models', 0)} "
        f"layers={cross_meta.get('cross_n_layers', 0)}"
    )

    # FastText uses only valid words
    valid_idx = np.where(ft_valid_words)[0]
    if valid_idx.size >= 2:
        ft_sim_valid = ft_sim[np.ix_(valid_idx, valid_idx)]
        ft_topk, k_eff_ft = _topk_indices(ft_sim_valid, k_eff)
    else:
        ft_topk, k_eff_ft = np.empty((0, 0), dtype=int), 0

    results = []
    target_out_dir = os.path.join(OUTPUT_DIR_BASE, model_alias, prompt_name)
    os.makedirs(target_out_dir, exist_ok=True)

    # Start at layer 1 to exclude embedding layer 0
    layer_iter = range(1, L)
    if show_progress:
        layer_iter = tqdm(layer_iter, desc=f"Layers NN@{k_eff}", unit="layer")
    for l in layer_iter:
        if hid_cosine is not None:
            layer_matrix = hid_cosine[l]
            if isinstance(layer_matrix, torch.Tensor):
                layer_matrix = layer_matrix.numpy()
            hid_topk, _ = _topk_indices(layer_matrix, k_eff)
        else:
            layer_hid = hid_states[:, l, :]
            if isinstance(layer_hid, torch.Tensor):
                layer_hid = layer_hid.to(torch.float32).numpy()
            layer_hid = center_and_normalize(layer_hid)
            hid_topk, _ = _topk_from_embeddings(
                layer_hid,
                k_eff,
                device=device,
                chunk_size=chunk_size,
            )
        nn_hs = _mean_overlap(hid_topk, fc_topk, k_eff)
        nn_fa = _mean_overlap(hid_topk, fa_topk, k_eff)
        nn_bert = _mean_overlap(hid_topk, bert_topk, k_eff)
        nn_cross = float("nan")
        if consensus_topk is not None and k_eff_cons > 0:
            nn_cross = _mean_overlap(hid_topk, consensus_topk, k_eff_cons)

        nn_ft = float("nan")
        if valid_idx.size >= 2 and k_eff_ft > 0:
            if hid_cosine is not None:
                layer_valid = layer_matrix[np.ix_(valid_idx, valid_idx)]
                hid_topk_ft, _ = _topk_indices(layer_valid, k_eff_ft)
            else:
                layer_valid = layer_hid[valid_idx]
                hid_topk_ft, _ = _topk_from_embeddings(
                    layer_valid,
                    k_eff_ft,
                    device=device,
                    chunk_size=chunk_size,
                )
            nn_ft = _mean_overlap(hid_topk_ft, ft_topk, k_eff_ft)

        results.append(
            {
                "layer": l,
                f"nn_forced_choice_{fc_type}": nn_hs,
                f"nn_fa_{fa_type}": nn_fa,
                "nn_fasttext": nn_ft,
                "nn_bert": nn_bert,
                "nn_crossmodel": nn_cross,
                "k": k_eff,
                "k_fasttext": k_eff_ft,
                "n_words": n_words,
                "n_words_fasttext": int(valid_idx.size),
                "cross_n_models": cross_meta.get("cross_n_models", 0),
                "cross_n_layers": cross_meta.get("cross_n_layers", 0),
                "cross_denom": cross_meta.get("cross_denom", 0),
            }
        )

    df_res = pd.DataFrame(results)
    date_tag = datetime.now().strftime("%Y%m%d")
    output_csv = os.path.join(target_out_dir, f"layerwise_neighbors_k{k_eff}_{date_tag}.csv")
    df_res.to_csv(output_csv, index=False)

    return df_res, fc_type, fa_type


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[10],
        help="One or more k values for NN@k (e.g., --k 10 20 50).",
    )
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
        help="Skip summary_all_neighbors CSVs (useful in array mode).",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for cosine/top-k (auto uses CUDA if available).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Row chunk size for GPU matmul/top-k.",
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)

    vocab_df = pd.read_csv(VOCAB_PATH)
    words = sorted(vocab_df["word"].astype(str).unique().tolist())
    ft_bundle = _load_benchmark_bundle(FASTTEXT_BENCH_DIR, words, kind="fasttext")
    bert_bundle = _load_benchmark_bundle(BERT_BENCH_DIR, words, kind="bert")

    model_aliases = sorted(
        [d for d in os.listdir(BEHAVIORAL_ROOT) if os.path.isdir(os.path.join(BEHAVIORAL_ROOT, d))]
    )
    targets = model_aliases

    if args.model or args.prompt:
        model_filter = args.model
        targets = [
            m
            for m in targets
            if (model_filter is None or m == model_filter)
        ]
    if args.job_index is not None:
        if args.job_count is not None and args.job_count != len(targets):
            logger.warning(f"Job count {args.job_count} != {len(targets)} targets.")
        if args.job_index < 0 or args.job_index >= len(targets):
            raise ValueError(f"job-index {args.job_index} out of range (0..{len(targets)-1})")
        targets = [targets[args.job_index]]

    prompt_filter = args.prompt
    prompts = PROMPT_NAMES if prompt_filter is None else [prompt_filter]
    show_progress = len(targets) == 1 and len(prompts) == 1
    write_summary = (not args.no_summary) and args.job_index is None and args.model is None and args.prompt is None

    for k in args.k:
        logger.info(f"Processing NN@{k}...")
        all_layerwise = []

        for alias in targets:
            for prompt_name in prompts:
                res_tuple = evaluate_model_prompt(
                    alias,
                    prompt_name,
                    ft_bundle["sim"],
                    ft_bundle["valid_mask"],
                    bert_bundle["sim"],
                    words,
                    k,
                    model_aliases,
                    show_progress=show_progress,
                    device=device,
                    chunk_size=args.chunk_size,
                )
                if res_tuple:
                    res, fc_t, fa_t = res_tuple
                    res["model"] = alias
                    res["prompt"] = prompt_name
                    res["beh_type"] = fc_t
                    res["fa_type"] = fa_t
                    all_layerwise.append(res)

        if all_layerwise and write_summary:
            date_tag = datetime.now().strftime("%Y%m%d")
            df_all = pd.concat(all_layerwise, ignore_index=True)
            summary_path = os.path.join(OUTPUT_DIR_BASE, f"summary_all_neighbors_k{k}_{date_tag}.csv")
            df_all.to_csv(summary_path, index=False)
            logger.info(f"Full NN results saved to {summary_path}")
        elif not all_layerwise:
            logger.warning(f"No results for NN@{k}")


if __name__ == "__main__":
    main()
