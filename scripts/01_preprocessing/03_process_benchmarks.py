#!/usr/bin/env python3
import os
import argparse
import time
from datetime import datetime
import gzip

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


DEFAULT_VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"

# Source FastText vectors (large .vec.gz file)
DEFAULT_FASTTEXT_VEC_PATH = "data/further_embeddings/fasttext/cc.en.300.vec.gz"

# Output dirs
DEFAULT_FASTTEXT_OUT_DIR = "data/further_embeddings/fasttext"
DEFAULT_BERT_OUT_DIR = "data/further_embeddings/bert"

DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_BASE_PROMPT = "This is a "


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def load_fasttext_vectors(vec_gz_path: str, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load FastText vectors case-insensitively and return matrix + valid mask (same as eval script)."""
    word_set = set(w.lower() for w in words)
    vectors: dict[str, np.ndarray] = {}

    start_time = time.time()
    with gzip.open(vec_gz_path, "rt", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0].lower()
            if word in word_set:
                vectors[word] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vectors) == len(word_set):
                    break

    elapsed = time.time() - start_time
    print(f"[FastText] Loaded {len(vectors)}/{len(word_set)} vectors in {elapsed:.2f}s from {vec_gz_path}")

    matrix = np.zeros((len(words), 300), dtype=np.float32)
    valid_mask = np.zeros(len(words), dtype=bool)
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in vectors:
            matrix[i] = vectors[lw]
            valid_mask[i] = True
    return matrix, valid_mask


def load_bert_embeddings(words: list[str], base_prompt: str, bert_model: str, device: str, batch_size: int) -> np.ndarray:
    """Extract BERT embeddings isolating only the target word tokens (same logic as eval script, but batched)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
    model = AutoModel.from_pretrained(bert_model).to(device)
    model.eval()

    all_vecs: list[np.ndarray] = []

    def _embed_batch(batch_words: list[str]) -> list[np.ndarray]:
        texts = [base_prompt + w for w in batch_words]
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").cpu().numpy()  # [B, T, 2]
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            hs = out.last_hidden_state  # [B, T, H]

        # Special tokens mask (per batch)
        special_ids = torch.tensor(tok.all_special_ids, device=device)
        special_mask = torch.isin(enc["input_ids"], special_ids)  # [B, T]

        vecs: list[np.ndarray] = []
        for bi, w in enumerate(batch_words):
            w_start = len(base_prompt)
            w_end = w_start + len(w)
            off = offsets[bi]
            mask = (off[:, 1] > w_start) & (off[:, 0] < w_end)
            mask = mask & (~special_mask[bi].detach().cpu().numpy())
            idx = np.where(mask)[0]
            if idx.size == 0:
                idx = np.where(~special_mask[bi].detach().cpu().numpy())[0]
            v = hs[bi, idx, :].mean(dim=0).detach().cpu().numpy()
            vecs.append(v)
        return vecs

    for i in tqdm(range(0, len(words), batch_size), desc="Extracting BERT baseline"):
        batch_words = words[i : i + batch_size]
        all_vecs.extend(_embed_batch(batch_words))

    return np.stack(all_vecs).astype(np.float32)


def apply_svd_and_save(matrix: np.ndarray, n_components: int, out_dir: str, prefix: str):
    """
    Applies TruncatedSVD to the matrix, normalizes, computes cosine similarity, and saves results.
    """
    print(f"[{prefix}] Computing SVD-{n_components}...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(matrix)
    
    # Cosine Similarity
    sim = cosine_similarity(_normalize(reduced))
    
    # Save
    emb_path = os.path.join(out_dir, f"benchmark_embeddings_svd{n_components}.npy")
    sim_path = os.path.join(out_dir, f"benchmark_cosine_sim_svd{n_components}.npy")
    
    np.save(emb_path, reduced.astype(np.float32))
    np.save(sim_path, sim.astype(np.float32))
    
    print(f"[{prefix}] Saved SVD-{n_components}:")
    print(f"  - {emb_path}")
    print(f"  - {sim_path}")


def save_fasttext(out_dir: str, words: list[str], ft_matrix: np.ndarray, valid_mask: np.ndarray, ft_sim: np.ndarray, overwrite: bool):
    os.makedirs(out_dir, exist_ok=True)

    words_path = os.path.join(out_dir, "benchmark_words.csv")
    vec_path = os.path.join(out_dir, "benchmark_vectors.npy")
    valid_path = os.path.join(out_dir, "benchmark_valid_mask.npy")
    sim_path = os.path.join(out_dir, "benchmark_cosine_sim.npy")

    if (not overwrite) and any(os.path.exists(p) for p in [words_path, vec_path, valid_path, sim_path]):
        raise FileExistsError(f"[FastText] Outputs already exist in {out_dir}. Use --overwrite to replace.")

    pd.DataFrame({"word": words}).to_csv(words_path, index=False)
    np.save(vec_path, ft_matrix.astype(np.float32))
    np.save(valid_path, valid_mask.astype(bool))
    np.save(sim_path, ft_sim.astype(np.float32))

    print(f"[FastText] Saved to {out_dir}:")
    print(f"  - {words_path}")
    print(f"  - {vec_path}")
    print(f"  - {valid_path}")
    print(f"  - {sim_path}")
    
    # SVD 100 for FastText
    apply_svd_and_save(ft_matrix, 100, out_dir, "FastText")


def save_bert(out_dir: str, words: list[str], bert_matrix: np.ndarray, bert_sim: np.ndarray, overwrite: bool):
    os.makedirs(out_dir, exist_ok=True)

    words_path = os.path.join(out_dir, "benchmark_words.csv")
    emb_path = os.path.join(out_dir, "benchmark_embeddings.npy")
    sim_path = os.path.join(out_dir, "benchmark_cosine_sim.npy")

    if (not overwrite) and any(os.path.exists(p) for p in [words_path, emb_path, sim_path]):
        raise FileExistsError(f"[BERT] Outputs already exist in {out_dir}. Use --overwrite to replace.")

    pd.DataFrame({"word": words}).to_csv(words_path, index=False)
    np.save(emb_path, bert_matrix.astype(np.float32))
    np.save(sim_path, bert_sim.astype(np.float32))

    print(f"[BERT] Saved to {out_dir}:")
    print(f"  - {words_path}")
    print(f"  - {emb_path}")
    print(f"  - {sim_path}")
    
    # SVD 100 and 300 for BERT
    apply_svd_and_save(bert_matrix, 300, out_dir, "BERT")
    apply_svd_and_save(bert_matrix, 100, out_dir, "BERT")


def main():
    ap = argparse.ArgumentParser(description="Precompute benchmark semantic spaces + cosine sims for FastText and BERT.")
    ap.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB_PATH)
    ap.add_argument("--only", choices=["all", "fasttext", "bert"], default="all")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--fasttext-vec-path", type=str, default=DEFAULT_FASTTEXT_VEC_PATH)
    ap.add_argument("--fasttext-out-dir", type=str, default=DEFAULT_FASTTEXT_OUT_DIR)

    ap.add_argument("--bert-out-dir", type=str, default=DEFAULT_BERT_OUT_DIR)
    ap.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    ap.add_argument("--base-prompt", type=str, default=DEFAULT_BASE_PROMPT)
    ap.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu. Default: auto.")
    ap.add_argument("--bert-batch-size", type=int, default=32)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting benchmark preprocessing at {ts}")

    vocab_df = pd.read_csv(args.vocab_path)
    words = sorted(vocab_df["word"].astype(str).unique().tolist())
    print(f"Loaded vocab: {len(words)} words from {args.vocab_path}")

    if args.only in ("all", "fasttext"):
        ft_matrix, ft_valid_mask = load_fasttext_vectors(args.fasttext_vec_path, words)
        ft_sim = cosine_similarity(_normalize(ft_matrix))
        save_fasttext(args.fasttext_out_dir, words, ft_matrix, ft_valid_mask, ft_sim, overwrite=args.overwrite)

    if args.only in ("all", "bert"):
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BERT] Using device: {device}")
        bert_matrix = load_bert_embeddings(
            words=words,
            base_prompt=args.base_prompt,
            bert_model=args.bert_model,
            device=device,
            batch_size=args.bert_batch_size,
        )
        bert_sim = cosine_similarity(_normalize(bert_matrix))
        save_bert(args.bert_out_dir, words, bert_matrix, bert_sim, overwrite=args.overwrite)

    print("Benchmark preprocessing complete.")


if __name__ == "__main__":
    main()

