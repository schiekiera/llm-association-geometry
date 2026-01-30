#!/usr/bin/env python3
"""
Held-out words prediction using cosine features (no layer-meta features).

Predict target hidden cosine similarities using:
- FastText cosine
- BERT cosine
- Cross-model cosine features (all other models, all layers except layer 0)
- Behavioral cosine (FC, FA)

Hidden-state cosine similarities are computed after mean-centering per layer
using training words only.

Train on sampled pairs from train words; evaluate on all pairs from test words.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path("projects/icml_project")

HIDDEN_ROOT = PROJECT_ROOT / "output/01_hidden_state_embeddings/final_hidden_states"
FC_ROOT = PROJECT_ROOT / "output/02_behavioral_associations/01_forced_choice/03_postprocessed"
FA_ROOT = PROJECT_ROOT / "output/02_behavioral_associations/02_free_associations/03_postprocessed"
FASTTEXT_DIR = PROJECT_ROOT / "input/further_embeddings/fasttext"
BERT_DIR = PROJECT_ROOT / "input/further_embeddings/bert"

OUTPUT_DIR_BASE = PROJECT_ROOT / "output/prediction/01_held_out_words"

PROMPT_NAMES = ["template", "averaged", "forced_choice", "free_association"]

# =============================================================================
# Logging
# =============================================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
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


def get_clean_alias(alias):
    alias = re.sub(r"_\d{8}_\d{6}$", "", alias)
    alias = alias.replace("_small", "").replace("_large", "").replace("_old", "")
    return alias


def load_vocab(csv_path: Path, col_name: str):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return df[col_name].astype(str).str.strip().str.lower().tolist()


def _sample_pairs(n: int, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    m = n * (n - 1) // 2

    if sample_size >= m:
        upper = np.triu_indices(n, k=1)
        return upper[0].astype(np.int64), upper[1].astype(np.int64)

    rng = np.random.default_rng(seed)

    def _tri_offsets(n):
        i = np.arange(n, dtype=np.int64)
        return (i * (2 * n - i - 1)) // 2

    def _tri_k_to_ij(k, n):
        offsets = _tri_offsets(n)
        i = np.searchsorted(offsets, k, side="right") - 1
        j = i + 1 + (k - offsets[i])
        return i.astype(np.int64), j.astype(np.int64)

    k = rng.choice(m, size=sample_size, replace=False).astype(np.int64)
    return _tri_k_to_ij(k, n)


def _r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def _load_benchmark_cosine(bench_dir: Path, name: str):
    sim_path = bench_dir / "benchmark_cosine_sim.npy"
    vocab_path = bench_dir / "benchmark_words.csv"
    if not sim_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"Missing benchmark cosine or vocab for {name} in {bench_dir}")
    sim = np.load(sim_path).astype(np.float32)
    vocab = load_vocab(vocab_path, "word" if "word" in pd.read_csv(vocab_path).columns else "input_word")
    return sim, vocab


def _load_behavior_cosine(root_dir: Path, model_alias: str):
    sim_path = root_dir / model_alias / "cosine_sim_counts.npy"
    vocab_path = root_dir / model_alias / "row_vocab.csv"
    if not sim_path.exists() or not vocab_path.exists():
        return None, None
    sim = np.load(sim_path).astype(np.float32)
    vocab = load_vocab(vocab_path, "input_word")
    return sim, vocab


def _compute_centered_cosine_matrices_from_train(
    hid_states: np.ndarray, train_idx: np.ndarray
) -> np.ndarray:
    if hid_states.dtype == np.float64:
        hid_states = hid_states.astype(np.float32)
    n_words, n_layers, _ = hid_states.shape
    mats = []
    for l in range(n_layers):
        X = hid_states[:, l, :]
        X_train = X[train_idx]
        train_mean = X_train.mean(axis=0, keepdims=True)
        Xc = X - train_mean
        norms = np.linalg.norm(Xc, axis=1, keepdims=True)
        Xn = Xc / np.maximum(norms, 1e-12)
        sim = (Xn @ Xn.T).astype(np.float32, copy=False)
        mats.append(sim)
    return np.stack(mats, axis=0)


def _load_hidden_states(model_alias: str, prompt: str):
    clean_alias = get_clean_alias(model_alias)
    hid_path = HIDDEN_ROOT / clean_alias / prompt / "hidden_states.pt"
    if not hid_path.exists():
        hid_path = HIDDEN_ROOT / model_alias / prompt / "hidden_states.pt"
        if not hid_path.exists():
            return None, None
    hid_data = _torch_load_compat(hid_path)
    hid_cues = [w.strip().lower() for w in hid_data["cue_words"]]
    hid_states = hid_data.get("hidden_states")
    if hid_states is None:
        return None, None
    if isinstance(hid_states, torch.Tensor):
        hid_states = hid_states.to(torch.float32).cpu().numpy()
    return hid_states, hid_cues


def _slice_to_common(sim_matrix: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return sim_matrix[np.ix_(idx, idx)]


def _pair_sims(sim_matrix: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    return sim_matrix[ii, jj].astype(np.float32)


def _build_crossmodel_features(source_hid_mats: list[np.ndarray], ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    # source_hid_mats are mean-centered cosine matrices (train-mean)
    values = []
    for mats in source_hid_mats:
        L_src = mats.shape[0]
        for l in range(1, L_src):
            values.append(mats[l][ii, jj])
    if not values:
        return np.zeros((len(ii), 0), dtype=np.float32)
    # One consensus scalar: average across all non-target models' layers.
    avg = np.mean(np.column_stack(values), axis=1, dtype=np.float32)
    return avg[:, None].astype(np.float32)


def _expected_cross_dim(source_hid_mats: list[np.ndarray]) -> int:
    return 1 if source_hid_mats else 0


# =============================================================================
# Core fit
# =============================================================================

def fit_single_layer_batched(
    l: int,
    y_train: np.ndarray,
    hid_target_mats: np.ndarray,
    ii_te: np.ndarray,
    jj_te: np.ndarray,
    ft_sim: np.ndarray,
    bert_sim: np.ndarray,
    fc_sim: np.ndarray,
    fa_sim: np.ndarray,
    source_hid_mats: list[np.ndarray],
    ii_tr: np.ndarray,
    jj_tr: np.ndarray,
    batch_size: int,
):
    # Prepare training static features
    ft_tr = _pair_sims(ft_sim, ii_tr, jj_tr)[:, None]
    bert_tr = _pair_sims(bert_sim, ii_tr, jj_tr)[:, None]
    fc_tr = _pair_sims(fc_sim, ii_tr, jj_tr)[:, None]
    fa_tr = _pair_sims(fa_sim, ii_tr, jj_tr)[:, None]
    X_cross_tr = _build_crossmodel_features(source_hid_mats, ii_tr, jj_tr)

    X_base_tr = np.hstack([ft_tr, bert_tr])

    def fit_model(X_tr, name):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        model = RidgeCV(alphas=np.logspace(-2, 6, 15), cv=5)
        model.fit(X_tr_sc, y_train)
        r2_train = float(model.score(X_tr_sc, y_train))
        logger.info(f"      Layer {l} - {name}: alpha_={model.alpha_}")
        return scaler, model, r2_train

    scaler_base, model_base, r2_train_base = fit_model(X_base_tr, "Baseline")
    scaler_cross, model_cross, r2_train_cross = fit_model(
        np.hstack([X_base_tr, X_cross_tr]), "Baseline+CrossModel"
    )
    scaler_cross_fc, model_cross_fc, r2_train_cross_fc = fit_model(
        np.hstack([X_base_tr, X_cross_tr, fc_tr]),
        "Baseline+CrossModel+FC",
    )
    scaler_cross_fa, model_cross_fa, r2_train_cross_fa = fit_model(
        np.hstack([X_base_tr, X_cross_tr, fa_tr]),
        "Baseline+CrossModel+FA",
    )
    scaler_full, model_full, r2_train_full = fit_model(
        np.hstack([X_base_tr, X_cross_tr, fc_tr, fa_tr]),
        "Full",
    )

    # Test (batched)
    y_test = hid_target_mats[l][ii_te, jj_te]
    n_test = len(ii_te)

    preds_base = []
    preds_cross = []
    preds_cross_fc = []
    preds_cross_fa = []
    preds_full = []

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        ii_b = ii_te[start:end]
        jj_b = jj_te[start:end]

        ft_b = _pair_sims(ft_sim, ii_b, jj_b)[:, None]
        bert_b = _pair_sims(bert_sim, ii_b, jj_b)[:, None]
        fc_b = _pair_sims(fc_sim, ii_b, jj_b)[:, None]
        fa_b = _pair_sims(fa_sim, ii_b, jj_b)[:, None]
        X_cross_b = _build_crossmodel_features(source_hid_mats, ii_b, jj_b)

        X_base_b = np.hstack([ft_b, bert_b])
        preds_base.extend(model_base.predict(scaler_base.transform(X_base_b)))
        preds_cross.extend(
            model_cross.predict(scaler_cross.transform(np.hstack([X_base_b, X_cross_b])))
        )
        preds_cross_fc.extend(
            model_cross_fc.predict(scaler_cross_fc.transform(np.hstack([X_base_b, X_cross_b, fc_b])))
        )
        preds_cross_fa.extend(
            model_cross_fa.predict(scaler_cross_fa.transform(np.hstack([X_base_b, X_cross_b, fa_b])))
        )
        preds_full.extend(
            model_full.predict(scaler_full.transform(np.hstack([X_base_b, X_cross_b, fc_b, fa_b])))
        )

    r2_base = _r2_score(y_test, np.array(preds_base))
    r2_cross = _r2_score(y_test, np.array(preds_cross))
    r2_cross_fc = _r2_score(y_test, np.array(preds_cross_fc))
    r2_cross_fa = _r2_score(y_test, np.array(preds_cross_fa))
    r2_full = _r2_score(y_test, np.array(preds_full))

    return {
        "layer": l,
        "r2_baseline": r2_base,
        "r2_cross_add": r2_cross,
        "r2_cross_fc": r2_cross_fc,
        "r2_cross_fa": r2_cross_fa,
        "r2_full": r2_full,
        "r2_train_baseline": r2_train_base,
        "r2_train_cross_add": r2_train_cross,
        "r2_train_cross_fc": r2_train_cross_fc,
        "r2_train_cross_fa": r2_train_cross_fa,
        "r2_train_full": r2_train_full,
        "delta_cross": r2_cross - r2_base,
        "delta_cross_fc": r2_cross_fc - r2_base,
        "delta_cross_fa": r2_cross_fa - r2_base,
        "delta_full": r2_full - r2_base,
    }


# =============================================================================
# Main processing
# =============================================================================

def process_target_prompt(
    target_model: str,
    prompt: str,
    args,
    ft_bundle,
    bert_bundle,
    all_models: list[str],
):
    logger.info(f"Processing target={target_model} | prompt={prompt}")

    hid_target_states, hid_target_vocab = _load_hidden_states(target_model, prompt)
    if hid_target_states is None:
        logger.warning(f"  Missing target hidden states for {target_model}/{prompt}")
        return []

    fc_sim, fc_vocab = _load_behavior_cosine(FC_ROOT, target_model)
    fa_sim, fa_vocab = _load_behavior_cosine(FA_ROOT, target_model)
    if fc_sim is None or fa_sim is None:
        logger.warning(f"  Missing FC/FA cosine for {target_model}")
        return []

    ft_sim, ft_vocab = ft_bundle
    bert_sim, bert_vocab = bert_bundle

    vocab_sets = [
        set(hid_target_vocab),
        set(fc_vocab),
        set(fa_vocab),
        set(ft_vocab),
        set(bert_vocab),
    ]
    common_words = sorted(list(set.intersection(*vocab_sets)))
    if len(common_words) < 100:
        logger.warning(f"  Only {len(common_words)} common words for {target_model}/{prompt}")
        return []

    source_models = [m for m in all_models if m != target_model]
    source_hid_mats = []
    source_vocab_maps = []

    for src in source_models:
        states, vocab = _load_hidden_states(src, prompt)
        if states is None:
            logger.warning(f"  Missing hidden states for source {src}/{prompt}, skipping")
            continue
        source_hid_mats.append(states)
        source_vocab_maps.append({w: i for i, w in enumerate(vocab)})

    if not source_hid_mats:
        logger.warning("  No source models with hidden states available.")
        return []

    for w2i in source_vocab_maps:
        common_words = [w for w in common_words if w in w2i]
    if len(common_words) < 100:
        logger.warning(f"  Only {len(common_words)} common words after source alignment.")
        return []

    logger.info(f"  Common vocabulary: {len(common_words)} words")

    w2i_target = {w: i for i, w in enumerate(hid_target_vocab)}
    w2i_fc = {w: i for i, w in enumerate(fc_vocab)}
    w2i_fa = {w: i for i, w in enumerate(fa_vocab)}
    w2i_ft = {w: i for i, w in enumerate(ft_vocab)}
    w2i_bert = {w: i for i, w in enumerate(bert_vocab)}
    w2i_common = {w: i for i, w in enumerate(common_words)}

    idx_target = np.array([w2i_target[w] for w in common_words])
    idx_fc = np.array([w2i_fc[w] for w in common_words])
    idx_fa = np.array([w2i_fa[w] for w in common_words])
    idx_ft = np.array([w2i_ft[w] for w in common_words])
    idx_bert = np.array([w2i_bert[w] for w in common_words])

    hid_target_states = hid_target_states[idx_target]
    fc_sim = _slice_to_common(fc_sim, idx_fc)
    fa_sim = _slice_to_common(fa_sim, idx_fa)
    ft_sim = _slice_to_common(ft_sim, idx_ft)
    bert_sim = _slice_to_common(bert_sim, idx_bert)

    source_hid_states_sliced = []
    for states, w2i in zip(source_hid_mats, source_vocab_maps):
        idx_src = np.array([w2i[w] for w in common_words])
        states = states[idx_src]
        source_hid_states_sliced.append(states)

    L_target = hid_target_states.shape[1]
    if L_target <= 1:
        logger.warning("  Not enough layers after excluding layer 0.")
        return []

    expected_cross_dim = _expected_cross_dim(source_hid_states_sliced)
    logger.info(f"  Cross-model feature dim (sum L_src-1): {expected_cross_dim}")

    all_seed_results: list[pd.DataFrame] = []

    for seed_idx, seed in enumerate(args.seeds):
        logger.info(f"  Seed {seed} ({seed_idx+1}/{len(args.seeds)}): Starting...")

        train_words, test_words = train_test_split(common_words, test_size=0.2, random_state=seed)
        if set(train_words) & set(test_words):
            raise RuntimeError("Train/test words overlap (leakage detected).")

        idx_common_train = np.array([w2i_common[w] for w in train_words])
        idx_common_test = np.array([w2i_common[w] for w in test_words])

        ii_tr, jj_tr = _sample_pairs(len(train_words), args.pair_sample_size, seed)
        ii_te, jj_te = np.triu_indices(len(test_words), k=1)

        ii_tr = idx_common_train[ii_tr]
        jj_tr = idx_common_train[jj_tr]
        ii_te = idx_common_test[ii_te]
        jj_te = idx_common_test[jj_te]

        logger.info(f"    Train pairs: {len(ii_tr):,} | Test pairs (all): {len(ii_te):,}")

        hid_target_mats = _compute_centered_cosine_matrices_from_train(
            hid_target_states, idx_common_train
        )
        source_hid_mats_sliced = [
            _compute_centered_cosine_matrices_from_train(states, idx_common_train)
            for states in source_hid_states_sliced
        ]

        X_cross_tr = _build_crossmodel_features(source_hid_mats_sliced, ii_tr, jj_tr)
        if X_cross_tr.shape[1] != expected_cross_dim:
            raise RuntimeError(
                f"Cross-model feature dim mismatch: got {X_cross_tr.shape[1]}, expected {expected_cross_dim}"
            )

        layer_results = []
        total_layers = L_target - 1
        for layer_idx, l in enumerate(range(1, L_target)):
            logger.info(f"    Layer {l} ({layer_idx+1}/{total_layers})")
            y_train = hid_target_mats[l][ii_tr, jj_tr]
            result = fit_single_layer_batched(
                l=l,
                y_train=y_train,
                hid_target_mats=hid_target_mats,
                ii_te=ii_te,
                jj_te=jj_te,
                ft_sim=ft_sim,
                bert_sim=bert_sim,
                fc_sim=fc_sim,
                fa_sim=fa_sim,
                source_hid_mats=source_hid_mats_sliced,
                ii_tr=ii_tr,
                jj_tr=jj_tr,
                batch_size=args.batch_size,
            )
            layer_results.append(result)

        df_seed = pd.DataFrame(layer_results)
        df_seed["seed"] = seed
        all_seed_results.append(df_seed)

    if not all_seed_results:
        return []

    df_all = pd.concat(all_seed_results, ignore_index=True)

    metrics = [
        "r2_baseline",
        "r2_cross_add",
        "r2_cross_fc",
        "r2_cross_fa",
        "r2_full",
        "r2_train_baseline",
        "r2_train_cross_add",
        "r2_train_cross_fc",
        "r2_train_cross_fa",
        "r2_train_full",
        "delta_cross",
        "delta_cross_fc",
        "delta_cross_fa",
        "delta_full",
    ]

    agg = df_all.groupby("layer")[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["layer"] + [f"{m}" if s == "mean" else f"{m}_sd" for (m, s) in agg.columns.tolist()[1:]]
    agg["model"] = target_model
    agg["prompt"] = prompt
    agg["n_seeds"] = len(args.seeds)
    agg["n_train_pairs"] = int(args.pair_sample_size)
    agg["n_test_pairs"] = int(len(ii_te))
    agg["cross_dim"] = int(expected_cross_dim)
    return agg


def process_and_save(target_model: str, prompt: str, args, ft_bundle, bert_bundle, all_models: list[str]) -> bool:
    out_dir = OUTPUT_DIR_BASE / target_model / prompt
    out_dir.mkdir(parents=True, exist_ok=True)
    datetime_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"ridge_predict_held_out_words_cosine_centered_{datetime_tag}.csv"

    if csv_path.exists() and not args.overwrite:
        logger.info(f"  [SKIP] {target_model} | {prompt} -> {csv_path} (exists; use --overwrite to recompute)")
        return True

    df = process_target_prompt(target_model, prompt, args, ft_bundle, bert_bundle, all_models)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(csv_path, index=False)
        logger.info(f"  [SAVED] {target_model} | {prompt} -> {csv_path}")
        return True
    return False


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-sample-size", type=int, default=10000, help="Train pairs sampled from train words")
    ap.add_argument("--batch-size", type=int, default=50000, help="Test batch size")
    ap.add_argument("--seed", type=int, default=42, help="Legacy: single seed. Prefer --seeds.")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--model", type=str, default=None, help="Optional: run only this target model alias (or comma-separated list)")
    ap.add_argument("--prompt", type=str, default=None, help="Optional: run only this prompt (or comma-separated list)")
    ap.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing output CSVs")
    args = ap.parse_args()

    if args.seeds is None:
        args.seeds = [args.seed]
    else:
        args.seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    ft_sim, ft_vocab = _load_benchmark_cosine(FASTTEXT_DIR, "fasttext")
    bert_sim, bert_vocab = _load_benchmark_cosine(BERT_DIR, "bert")
    ft_bundle = (ft_sim, ft_vocab)
    bert_bundle = (bert_sim, bert_vocab)

    all_models = sorted([d.name for d in FC_ROOT.iterdir() if d.is_dir()])

    if args.model:
        targets = [m.strip() for m in args.model.split(",") if m.strip()]
    else:
        targets = all_models

    if args.prompt:
        prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    else:
        prompts = PROMPT_NAMES

    for model_alias in targets:
        for prompt in prompts:
            process_and_save(model_alias, prompt, args, ft_bundle, bert_bundle, all_models)


if __name__ == "__main__":
    main()
