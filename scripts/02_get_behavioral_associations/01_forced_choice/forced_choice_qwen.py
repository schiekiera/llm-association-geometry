import os
import re
import random
import hashlib
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

import math

# =============================================================================
# Configuration
# =============================================================================

VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"
HF_TOKEN_PATH = "data/token/hf_token.txt"
HF_CACHE_ROOT = "cache/huggingface"

RAW_DIR = "data/02_behavioral_associations/01_forced_choice/01_raw"
PROCESSED_DIR = "data/02_behavioral_associations/01_forced_choice/02_processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ALIAS = "Qwen2.5-7B-Instruct"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# forced_choice hyperparameters
N_CANDIDATES = 16
N_PICKS = 2
MAX_ROUNDS = 1            # hard stop
STOP_POOL_SIZE = 2         # stop per-input when pool size <= 2

# Performance hyperparameters
BATCH_SIZE = 128
MAX_NEW_TOKENS = 10

# Retry / compliance hyperparameters
MAX_RETRIES = 5
RETRY_TEMPERATURE = 0.5
RETRY_TOP_P = 0.9
RETRY_BATCH_SIZE = 64

BASE_SEED = 12345
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TERMINATORS = None  # will be set in main()


# =============================================================================
# Prompt template (supports variable candidate count)
# =============================================================================

BASE_PROMPT_TEMPLATE = (
    "You will be given one input word and a list of candidate words.\n"
    "Your task is to select exactly {n_picks} words from the list that are most\n"
    "similar or closely related to the input word.\n\n"
    "Rules:\n"
    "- Select exactly {n_picks} words.\n"
    "- Both selected words must come from the provided candidate list.\n"
    "- Do not select the input word.\n"
    "- Output must contain only the {n_picks} chosen words.\n"
    "- Use the format: output: word1, word2\n"
    "- Do not add any explanation, reasoning, commentary, or extra text.\n"
    "- Do not change spelling or number of words.\n\n"
    "Example:\n"
    "input word: dog\n"
    "candidates: [banana, violin, therapy, beer, tango, paper, cat, kiwi, jeans, car, vacation, note, leash, bath, ceiling, ivy]\n"
    "output: cat, leash\n\n"
    "Now follow the same format.\n\n"
    "input word: {input_word}\n"
    "candidates: [{candidate_list}]\n"
    "output: "
)


# =============================================================================
# Hugging Face cache
# =============================================================================

os.makedirs(HF_CACHE_ROOT, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_ROOT

# =============================================================================
# Postprocessing helpers
# =============================================================================

def extract_picks(output_text: object, input_word: str) -> Tuple[object, object]:
    """
    Stage 1: Extract two tokens/words from the raw `output` field.
    Also strips model format drift like 'table: platform, desk' (input word + colon).
    """
    if not isinstance(output_text, str) or not output_text.strip():
        return pd.NA, pd.NA

    text = output_text.strip()

    # --- NEW: remove a leading "<input_word>:" prefix (case-insensitive) ---
    if isinstance(input_word, str) and input_word.strip():
        iw = re.escape(input_word.strip())
        # Examples matched: "table: ...", "Table : ...", "table:\n..."
        text = re.sub(rf"^\s*{iw}\s*:\s*", "output: ", text, flags=re.IGNORECASE)

    # Prefer explicit patterns: "output: word1, word2"
    matches = re.findall(r"output:\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)", text, flags=re.IGNORECASE)
    if matches:
        pick1, pick2 = matches[-1]
        return pick1, pick2

    # Fallback: scan lines from bottom for comma-separated pair without colon
    lines: List[str] = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        # IMPORTANT: after substitution above, the bad "input:" case is gone
        if "," in line and ":" not in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

    return pd.NA, pd.NA


def split_candidates(cand_text: object) -> List[str]:
    """Split the candidates string into a list (variable length)."""
    if isinstance(cand_text, str) and cand_text.strip():
        return [w.strip() for w in cand_text.split(",") if w.strip()]
    return []


def validate_picks(input_word: str, candidates: List[str], p1: object, p2: object) -> Tuple[object, object]:
    """
    Stage 2: Validate extracted picks:
      - must be a candidate
      - must not be the input word
    Invalid -> pd.NA
    """
    cand = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    eligible = set(c for c in cand if c != input_word)

    def clean(p: object) -> object:
        if isinstance(p, str):
            p = p.strip()
            if p in eligible:
                return p
        return pd.NA

    vp1 = clean(p1)
    vp2 = clean(p2)

    # Optional sanity: if both valid but identical, blank out second pick
    if isinstance(vp1, str) and isinstance(vp2, str) and vp1 == vp2:
        vp2 = pd.NA

    return vp1, vp2


def _stable_int_seed(*parts: str) -> int:
    """Deterministic seed from content."""
    s = "||".join(parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def pick_winners(
    input_word: str,
    candidates: List[str],
    pick1: object,
    pick2: object,
    round_id: int,
    trial_id: int,
) -> Tuple[object, object]:
    """
    Stage 3: Ensure exactly 2 winners per trial (when possible), always drawn from candidates
    and never the input word. Invalid picks are replaced with deterministic random candidates
    to keep forced_choice stable and reproducible.

    Rules:
    - If both picks invalid -> sample 2 random candidates (or 1 duplicated if only one eligible)
    - If one valid -> keep it and fill second with random remaining candidate
    - If both valid -> keep both (enforce distinct if possible)
    """
    cand = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    eligible = [c for c in cand if c != input_word]

    rng = random.Random(_stable_int_seed(str(round_id), str(trial_id), input_word))

    def valid(p: object) -> bool:
        return isinstance(p, str) and p in eligible and p != input_word

    w1 = pick1 if valid(pick1) else None
    w2 = pick2 if valid(pick2) else None

    # Both invalid
    if w1 is None and w2 is None:
        if len(eligible) >= 2:
            a, b = rng.sample(eligible, 2)
            return a, b
        if len(eligible) == 1:
            return eligible[0], eligible[0]
        return pd.NA, pd.NA

    # One valid, one invalid
    if w1 is not None and w2 is None:
        remaining = [c for c in eligible if c != w1]
        w2 = rng.choice(remaining) if remaining else w1
        return w1, w2

    if w1 is None and w2 is not None:
        remaining = [c for c in eligible if c != w2]
        w1 = rng.choice(remaining) if remaining else w2
        return w1, w2

    # Both valid; enforce distinct if possible
    if w1 == w2:
        remaining = [c for c in eligible if c != w1]
        if remaining:
            w2 = rng.choice(remaining)
    return w1, w2


# =============================================================================
# Model helpers
# =============================================================================

def build_prompt(input_word: str, candidate_words: List[str]) -> str:
    return BASE_PROMPT_TEMPLATE.format(
        n_candidates=len(candidate_words),
        n_picks=N_PICKS,
        input_word=input_word,
        candidate_list=", ".join(candidate_words),
    )


def build_repair_prompt(input_word: str, candidate_words: List[str], bad_output: object) -> str:
    """
    Deterministic retry prompt that explicitly flags the previous answer as invalid
    and re-states the exact output format requirement.
    """
    bad = "" if bad_output is None else str(bad_output).strip()
    return (
        "Your last answer was invalid because it did not follow the rules.\n"
        f"Your task is to select exactly {N_PICKS} words from the list that are most\n"
        "similar or closely related to the input word.\n\n"
        "You MUST follow the rules below exactly.\n\n"
        "Rules:\n"
        f"- Select exactly {N_PICKS} words.\n"
        "- Both selected words must come from the provided candidate list.\n"
        "- Do not select the input word.\n"
        f"- Output must contain only the {N_PICKS} chosen words.\n"
        "- Use the format: output: word1, word2\n"
        "- Do not add any explanation, reasoning, commentary, or extra text.\n"
        "- Do not change spelling or number of words.\n\n"
        f"Your previous answer was:\n{bad}\n\n"
        "Now follow the same format.\n\n"
        f"input word: {input_word}\n"
        f"candidates: [{', '.join(candidate_words)}]\n"
        "output: "
    )


def is_compliant(input_word: str, candidates: List[str], output_text: object) -> bool:
    """
    A trial is compliant iff both validated picks are valid (non-NA) AND distinct (if possible).
    Note: validate_picks already blanks out the 2nd pick if both picks are identical.
    """
    p1, p2 = extract_picks(output_text, input_word)
    v1, v2 = validate_picks(input_word=input_word, candidates=candidates, p1=p1, p2=p2)
    return isinstance(v1, str) and isinstance(v2, str)


def wrap_as_chat(tokenizer, prompts: List[str]) -> List[str]:
    wrapped = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        wrapped.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return wrapped


def generate_batch_with_params(
    model,
    tokenizer,
    prompts: List[str],
    *,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> List[str]:
    chat_prompts = wrap_as_chat(tokenizer, prompts)
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        gen_kwargs: Dict[str, object] = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
        out = model.generate(
            **inputs,
            **gen_kwargs,
            eos_token_id=TERMINATORS,
        )

    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def generate_batch(model, tokenizer, prompts: List[str]) -> List[str]:
    # Main deterministic pass (unchanged behavior)
    return generate_batch_with_params(model, tokenizer, prompts, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)


def _seed_for_retry(round_id: int, trial_id: int, input_word: str) -> int:
    """Stable seed for reproducible sampled retries."""
    return _stable_int_seed(str(round_id), str(trial_id), input_word)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# =============================================================================
# forced_choice core
# =============================================================================

def init_pools(vocab_words: List[str]) -> Dict[str, List[str]]:
    """
    Round 1 pool: for each input word, all other words in a deterministic shuffle.
    This guarantees same candidates per input in round 1 across models (given same vocab, BASE_SEED).
    """
    pools: Dict[str, List[str]] = {}
    N = len(vocab_words)

    for input_idx, input_word in enumerate(vocab_words):
        idxs = list(range(N))
        idxs.pop(input_idx)
        rng = random.Random(BASE_SEED + input_idx)
        rng.shuffle(idxs)
        pools[input_word] = [vocab_words[j] for j in idxs]
    return pools



def chunk_list_balanced(lst: List[str], k: int) -> List[List[str]]:
    """
    Split lst into groups with sizes as even as possible, each size <= k.

    Example: n=98, k=16
      naive: 16,16,16,16,16,16,2
      balanced: 14,14,14,14,14,14,14
    """
    n = len(lst)
    if n == 0:
        return []
    if n <= k:
        return [lst]

    m = math.ceil(n / k)          # number of groups/trials
    base = n // m                 # minimum group size
    r = n % m                     # first r groups get +1

    groups = []
    idx = 0
    for gi in range(m):
        size = base + (1 if gi < r else 0)
        groups.append(lst[idx:idx + size])
        idx += size
    return groups


def round_file_paths(round_id: int) -> Tuple[str, str]:
    raw_path = os.path.join(RAW_DIR, f"{MODEL_ALIAS}_forced_choice_round{round_id:02d}_raw_{TIMESTAMP}.csv")
    proc_path = os.path.join(PROCESSED_DIR, f"{MODEL_ALIAS}_forced_choice_round{round_id:02d}_processed_{TIMESTAMP}.csv")
    return raw_path, proc_path


def run_generation_round(
    round_id: int,
    pools: Dict[str, List[str]],
    model,
    tokenizer,
) -> str:
    """
    Generates raw trials for one round and writes to CSV.
    Returns raw_csv_path.
    Stores EVERY TRIAL ROW generated in the CSV.
    """
    raw_path, _ = round_file_paths(round_id)

    # Resume safety: if raw exists, skip regeneration
    if os.path.exists(raw_path):
        print(f"[Round {round_id}] Raw exists, skipping generation: {raw_path}")
        return raw_path

    trials = []
    trial_id = 0

    for input_word, pool in pools.items():
        if len(pool) <= STOP_POOL_SIZE:
            continue
        groups = chunk_list_balanced(pool, N_CANDIDATES)
        for g in groups:
            if len(g) < 2:
                continue
            prompt = build_prompt(input_word, g)
            trials.append((trial_id, round_id, input_word, g, prompt))
            trial_id += 1

    if not trials:
        pd.DataFrame(columns=["trial_id", "round", "input", "candidates", "output"]).to_csv(
            raw_path, index=False
        )
        print(f"[Round {round_id}] No trials to run. Wrote empty raw: {raw_path}")
        return raw_path

    print(f"[Round {round_id}] Generating {len(trials)} trials ...")

    # Ensure we start from a clean file (prevents weird partial leftovers)
    if os.path.exists(raw_path):
        os.remove(raw_path)

    buffer_meta = []
    buffer_prompts = []

    first_write = True  # guarantees header exactly once

    # Compliance bookkeeping (round-level)
    total_trials = 0
    initial_compliant = 0
    after_repair_noncompliant = 0
    final_compliant = 0
    sampled_retries_used = 0
    still_noncompliant_trial_ids: List[int] = []

    def flush_buffer() -> None:
        nonlocal first_write, buffer_meta, buffer_prompts, total_trials, initial_compliant, after_repair_noncompliant, final_compliant, sampled_retries_used, still_noncompliant_trial_ids

        if not buffer_prompts:
            return

        outs0 = generate_batch(model, tokenizer, buffer_prompts)
        total_trials += len(outs0)

        compliant0 = [
            is_compliant(inp2, cand2, out)
            for (tid2, rid2, inp2, cand2), out in zip(buffer_meta, outs0)
        ]
        initial_compliant += int(sum(compliant0))

        # Start with original outputs; only overwrite when a retry produces a compliant output.
        final_outs: List[str] = list(outs0)

        # Deterministic repair retry for non-compliant rows (still do_sample=False)
        bad_idxs = [i for i, ok in enumerate(compliant0) if not ok]
        remaining_after_repair: List[int] = bad_idxs

        if bad_idxs:
            repair_prompts: List[str] = []
            repair_meta: List[Tuple[int, int, int, str, List[str], str]] = []
            for i in bad_idxs:
                tid2, rid2, inp2, cand2 = buffer_meta[i]
                repair_prompts.append(build_repair_prompt(inp2, cand2, outs0[i]))
                repair_meta.append((i, int(tid2), int(rid2), inp2, cand2, outs0[i]))

            outs1 = generate_batch_with_params(
                model,
                tokenizer,
                repair_prompts,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            remaining_after_repair = []
            for (i, tid2, rid2, inp2, cand2, orig_out), out1 in zip(repair_meta, outs1):
                if is_compliant(inp2, cand2, out1):
                    final_outs[i] = out1
                else:
                    final_outs[i] = out1
                    remaining_after_repair.append(i)

        after_repair_noncompliant += int(len(remaining_after_repair))

        # Sampled retries for still non-compliant rows (BATCHED)
        for r in range(MAX_RETRIES):
            if not remaining_after_repair:
                break
            
            retry_prompts = []
            retry_meta_indices = []
            for i in remaining_after_repair:
                tid2, rid2, inp2, cand2 = buffer_meta[i]
                # Use current output as bad_output for the next prompt
                retry_prompts.append(build_repair_prompt(inp2, cand2, final_outs[i]))
                retry_meta_indices.append(i)

            # Use a batch-level seed for this retry attempt
            _set_torch_seed(BASE_SEED + round_id + r + int(buffer_meta[0][0]))

            outs_retry = generate_batch_with_params(
                model,
                tokenizer,
                retry_prompts,
                do_sample=True,
                temperature=RETRY_TEMPERATURE,
                top_p=RETRY_TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            new_remaining = []
            for i, out_try in zip(retry_meta_indices, outs_retry):
                sampled_retries_used += 1
                tid2, rid2, inp2, cand2 = buffer_meta[i]
                if is_compliant(inp2, cand2, out_try):
                    final_outs[i] = out_try
                else:
                    final_outs[i] = out_try
                    new_remaining.append(i)
            remaining_after_repair = new_remaining

        if remaining_after_repair:
            for i in remaining_after_repair:
                tid2, _, _, _ = buffer_meta[i]
                still_noncompliant_trial_ids.append(int(tid2))

        # Final compliance for this flush
        for (tid2, rid2, inp2, cand2), out_final in zip(buffer_meta, final_outs):
            if is_compliant(inp2, cand2, out_final):
                final_compliant += 1

        rows = []
        for (tid2, rid2, inp2, cand2), out in zip(buffer_meta, final_outs):
            rows.append(
                {
                    "trial_id": tid2,
                    "round": rid2,
                    "input": inp2,
                    "candidates": ", ".join(cand2),
                    "output": out.strip(),
                }
            )

        mode = "w" if first_write else "a"
        header = True if first_write else False
        pd.DataFrame(rows).to_csv(raw_path, mode=mode, header=header, index=False)

        first_write = False
        buffer_meta.clear()
        buffer_prompts.clear()

    for (tid, rid, inp, cand_list, prompt) in tqdm(trials, desc=f"Round {round_id} trials"):
        buffer_meta.append((tid, rid, inp, cand_list))
        buffer_prompts.append(prompt)

        if len(buffer_prompts) >= BATCH_SIZE:
            flush_buffer()

    # Flush remaining
    flush_buffer()

    # Round-level logging
    non0 = total_trials - initial_compliant
    print(f"[Round {round_id}] Initial generation: compliant={initial_compliant}/{total_trials} noncompliant={non0}")
    if non0 > 0:
        print(f"[Round {round_id}] After deterministic repair: remaining noncompliant={after_repair_noncompliant}")
        print(
            f"[Round {round_id}] After sampled retries: compliant={final_compliant}/{total_trials} "
            f"({final_compliant / max(1, total_trials):.1%}), sampled_retries_used={sampled_retries_used}"
        )
        if still_noncompliant_trial_ids:
            print(
                f"[Round {round_id}] WARNING: {len(still_noncompliant_trial_ids)} trials still noncompliant "
                f"after all retries (keeping original outputs). Example trial_ids: "
                f"{still_noncompliant_trial_ids[:10]}"
            )

    print(f"[Round {round_id}] Wrote raw: {raw_path}")
    return raw_path


def postprocess_round(round_id: int, raw_csv_path: str) -> str:
    """
    Reads raw round CSV.
    Writes processed CSV with:
      - extracted 1/2 (stage 1)
      - pick 1/2 validated (stage 2)
      - winner 1/2 with fallback (stage 3)
      - candidate 1..N_CANDIDATES columns (optional, for convenience)
    """
    _, proc_path = round_file_paths(round_id)

    # Resume safety: if processed exists, skip
    if os.path.exists(proc_path):
        print(f"[Round {round_id}] Processed exists, skipping: {proc_path}")
        return proc_path

    df_raw = pd.read_csv(raw_csv_path, engine="python")

    # Drop "header rows" that accidentally ended up in the file
    for col in ["trial_id", "round"]:
        if col in df_raw.columns:
            df_raw = df_raw[df_raw[col].astype(str).str.lower() != col]

    # Now coerce round/trial_id safely
    df_raw["round"] = pd.to_numeric(df_raw["round"], errors="coerce")
    df_raw["trial_id"] = pd.to_numeric(df_raw["trial_id"], errors="coerce")
    df_raw = df_raw.dropna(subset=["round", "trial_id"]).copy()
    df_raw["round"] = df_raw["round"].astype(int)
    df_raw["trial_id"] = df_raw["trial_id"].astype(int)
    
    if df_raw.empty:
        df_raw.to_csv(proc_path, index=False)
        print(f"[Round {round_id}] Raw empty. Wrote processed empty: {proc_path}")
        return proc_path

    # Stage 1: extract
    extracted = df_raw.apply(
        lambda r: extract_picks(r["output"], r["input"]),
        axis=1
    )

    df_raw["extracted 1"] = extracted.map(lambda t: t[0])
    df_raw["extracted 2"] = extracted.map(lambda t: t[1])


    cand_lists = df_raw["candidates"].apply(split_candidates)

    # Stage 2: validate picks
    validated = []
    for i, row in df_raw.iterrows():
        vp1, vp2 = validate_picks(
            input_word=row["input"],
            candidates=cand_lists.iloc[i],
            p1=row["extracted 1"],
            p2=row["extracted 2"],
        )
        validated.append((vp1, vp2))

    df_raw["pick 1"] = [v[0] for v in validated]
    df_raw["pick 2"] = [v[1] for v in validated]

    # Stage 3: winners with fallback
    winners = []
    for i, row in df_raw.iterrows():
        w1, w2 = pick_winners(
            input_word=row["input"],
            candidates=cand_lists.iloc[i],
            pick1=row["pick 1"],
            pick2=row["pick 2"],
            round_id=int(row["round"]),
            trial_id=int(row["trial_id"]) if "trial_id" in row else int(i),
        )
        winners.append((w1, w2))

    df_raw["winner 1"] = [w[0] for w in winners]
    df_raw["winner 2"] = [w[1] for w in winners]

    # Expand candidates into candidate 1..N_CANDIDATES
    for j in range(N_CANDIDATES):
        df_raw[f"candidate {j + 1}"] = cand_lists.apply(lambda lst, idx=j: lst[idx] if idx < len(lst) else pd.NA)

    ordered_cols = (
        ["trial_id", "round", "input", "candidates", "output",
         "extracted 1", "extracted 2",
         "pick 1", "pick 2",
         "winner 1", "winner 2"]
        + [f"candidate {j}" for j in range(1, N_CANDIDATES + 1)]
    )
    df_proc = df_raw[ordered_cols]
    df_proc.to_csv(proc_path, index=False)
    print(f"[Round {round_id}] Wrote processed: {proc_path}")
    return proc_path


def build_next_pools(
    vocab_words: List[str],
    current_pools: Dict[str, List[str]],
    processed_csv_path: str,
) -> Dict[str, List[str]]:
    """
    Next pool per input = unique winners from all trials for that input (preserve order).
    If an input had no trials (already stopped), carry its pool forward unchanged.
    """
    df = pd.read_csv(processed_csv_path)
    next_pools: Dict[str, List[str]] = {}

    if df.empty:
        return dict(current_pools)

    grouped = df.groupby("input", sort=False)

    for input_word in vocab_words:
        pool = current_pools[input_word]

        # If already stopped, carry
        if len(pool) <= STOP_POOL_SIZE:
            next_pools[input_word] = pool
            continue

        if input_word not in grouped.groups:
            next_pools[input_word] = pool
            continue

        g = grouped.get_group(input_word)

        winners_seq: List[str] = []
        for _, row in g.iterrows():
            for col in ["winner 1", "winner 2"]:
                w = row[col]
                if isinstance(w, str) and w and w != input_word:
                    winners_seq.append(w)

        # Unique, preserve order
        seen = set()
        uniq = []
        for w in winners_seq:
            if w not in seen:
                seen.add(w)
                uniq.append(w)

        next_pools[input_word] = uniq

    return next_pools


def forced_choice_done(pools: Dict[str, List[str]]) -> bool:
    """Stop if all pools are <= STOP_POOL_SIZE."""
    return all(len(pool) <= STOP_POOL_SIZE for pool in pools.values())


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    vocab_df = pd.read_csv(VOCAB_PATH)
    vocab_words: List[str] = vocab_df["word"].astype(str).tolist()
    print(f"Loaded {len(vocab_words)} vocabulary words.")

    with open(HF_TOKEN_PATH, "r") as f:
        hf_token = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # --- Qwen end-of-turn terminators ---
    global TERMINATORS
    TERMINATORS = [tokenizer.eos_token_id]
    for tok in ["<|endoftext|>", "<|im_end|>", "<|end_of_turn|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            TERMINATORS.append(tid)
    TERMINATORS = list(dict.fromkeys(TERMINATORS))
    print("TERMINATORS:", TERMINATORS)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(DEVICE)
    model.eval()

    pools = init_pools(vocab_words)

    for round_id in range(1, MAX_ROUNDS + 1):
        if forced_choice_done(pools):
            print(f"All pools <= {STOP_POOL_SIZE}. Stopping at round {round_id - 1}.")
            break

        raw_path = run_generation_round(round_id, pools, model, tokenizer)
        proc_path = postprocess_round(round_id, raw_path)

        pools = build_next_pools(vocab_words, pools, proc_path)

        sizes = [len(p) for p in pools.values()]
        print(
            f"[After Round {round_id}] pool size: "
            f"min={min(sizes)}, median={int(pd.Series(sizes).median())}, max={max(sizes)}"
        )


    final_path = os.path.join(PROCESSED_DIR, "final_candidates/", f"{MODEL_ALIAS}_forced_choice_FINAL_pools_{TIMESTAMP}.csv")
    rows = []
    for inp, pool in pools.items():
        rows.append({
            "input": inp,
            "final_pool_size": len(pool),
            "final_candidates": ", ".join(pool),
        })
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(final_path, index=False)
    print(f"Wrote final pools summary: {final_path}")


if __name__ == "__main__":
    main()
