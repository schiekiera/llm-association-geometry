import os
import gc
import time
import shutil
import tempfile
from typing import Optional, List, Dict
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np

# =============================================================================
# Configuration
# =============================================================================

# Paths
# Input sentences for context (and vocabulary source)
SENTENCES_PATH = "data/vocabulary/04_sentences/sentences_c4_final.csv"
VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"
HF_TOKEN_PATH = "data/token/hf_token.txt"
HF_CACHE_ROOT = "cache/huggingface"

HIDDEN_STATE_DIR_BASE = "data/01_hidden_state_embeddings/final_hidden_states"
os.makedirs(HIDDEN_STATE_DIR_BASE, exist_ok=True)

# Safer saving
DEFAULT_LOCAL_TMP_DIR = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
SAVE_RETRIES = 3
SAVE_RETRY_SLEEP_SEC = 3.0
# Reduce file size: cosine matrices dominate size (L, N, N). float16 halves storage.
SAVE_COSINE_FLOAT16 = True
SAVE_LEGACY_TORCH_FORMAT = False  # set True to use _use_new_zipfile_serialization=False


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def safe_torch_save(
    obj: dict,
    final_path: str,
    *,
    local_tmp_dir: str = DEFAULT_LOCAL_TMP_DIR,
    validate: bool = True,
    retries: int = SAVE_RETRIES,
    use_legacy_format: bool = SAVE_LEGACY_TORCH_FORMAT,
) -> None:
    """
    Save `obj` to `final_path` robustly:
      - write into local tmp first,
      - validate by loading (optional),
      - copy into final dir as .tmpcopy,
      - fsync and atomic replace.
    """
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    remote_tmp_path = final_path + ".tmpcopy"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        local_tmp_path = None
        try:
            os.makedirs(local_tmp_dir, exist_ok=True)
            fd, local_tmp_path = tempfile.mkstemp(prefix="hs_", suffix=".pt", dir=local_tmp_dir)
            os.close(fd)

            # 1) Write locally
            torch.save(
                obj,
                local_tmp_path,
                _use_new_zipfile_serialization=(not use_legacy_format),
            )

            # 2) Validate locally (fast sanity check)
            if validate:
                try:
                    _ = torch.load(local_tmp_path, map_location="cpu", weights_only=False)
                except TypeError:
                    _ = torch.load(local_tmp_path, map_location="cpu")
                # minimal key check
                if not isinstance(_, dict) or ("cosine_similarity_matrices" not in _):
                    raise RuntimeError("Validation failed: missing cosine_similarity_matrices in saved object")
                del _

            # 3) Copy to remote tmp and fsync
            _safe_remove(remote_tmp_path)
            shutil.copyfile(local_tmp_path, remote_tmp_path)
            try:
                with open(remote_tmp_path, "rb") as f:
                    os.fsync(f.fileno())
            except Exception:
                # fsync can fail on some setups; best effort
                pass

            # 4) Atomic replace into final path
            os.replace(remote_tmp_path, final_path)

            # Cleanup local tmp
            _safe_remove(local_tmp_path)
            return
        except Exception as e:
            last_err = e
            _safe_remove(remote_tmp_path)
            if local_tmp_path is not None:
                _safe_remove(local_tmp_path)
            if attempt < retries:
                time.sleep(SAVE_RETRY_SLEEP_SEC * attempt)
                continue
            raise
    if last_err is not None:
        raise last_err


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models to process
MODELS = [
    {"name": "google/gemma-2-9b-it", "alias": "gemma-2-9b-it"},
    {"name": "microsoft/phi-4", "alias": "phi-4"},
    {"name": "tiiuae/Falcon3-10B-Instruct", "alias": "Falcon3-10B-Instruct"},
    {"name": "Qwen/Qwen2.5-7B-Instruct", "alias": "Qwen2.5-7B-Instruct"},
    {"name": "mistralai/Mistral-Nemo-Instruct-2407", "alias": "Mistral-Nemo-Instruct-v1"},
    {"name": "mistralai/Mistral-7B-Instruct-v0.2", "alias": "Mistral-7B-Instruct-v0.2"},
    {"name": "meta-llama/Meta-Llama-3.1-8B-Instruct", "alias": "Llama-3.1-8B-Instruct"},
    {"name": "EssentialAI/rnj-1-instruct", "alias": "rnj-1-instruct"},
]

# Extraction Methods
METHODS = ["isolated", "template", "averaged", "forced_choice", "free_association"]

# Prompts
# {n_picks} is hardcoded to 2 for consistency with the example provided
FORCED_CHOICE_TEMPLATE = (
    "You will be given one input word and a list of candidate words.\n"
    "Your task is to select exactly 2 words from the list that are most\n"
    "similar or closely related to the input word.\n\n"
    "Rules:\n"
    "- Select exactly 2 words.\n"
    "- Both selected words must come from the provided candidate list.\n"
    "- Do not select the input word.\n"
    "- Output must contain only the 2 chosen words.\n"
    "- Use the format: output: word1, word2\n"
    "- Do not add any explanation, reasoning, commentary, or extra text.\n"
    "- Do not change spelling or number of words.\n\n"
    "Example:\n"
    "input word: dog\n"
    "candidates: [banana, violin, therapy, beer, tango, paper, cat, kiwi, jeans, car, vacation, note, leash, bath, ceiling, ivy]\n"
    "output: cat, leash\n\n"
    "Now follow the same format.\n\n"
    "input word: {input_word}\n"
)

FREE_ASSOCIATION_TEMPLATE = (
    "You will be given one input word.\n"
    "Produce exactly five different single-word associations.\n\n"
    "Rules:\n"
    "- Output only five associated words.\n"
    "- Each must be a single word (no spaces or punctuation inside a word).\n"
    "- All five words must be different from each other.\n"
    "- Do not repeat the input word.\n"
    "- Order the words by how quickly they come to mind (first = strongest).\n"
    "- Format your answer as a single line starting with 'output:'.\n"
    "- Separate the five words with commas and a space.\n"
    "- End the line with a period.\n"
    "- Do not add any explanations or extra text.\n"
    "Example:\n"
    "input: dog.\n"
    "output: bark, leash, pet, animal, cat.\n\n"
    "input: {input_word}"
)

# =============================================================================
# Hugging Face cache setup
# =============================================================================

os.makedirs(HF_CACHE_ROOT, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_ROOT

with open(HF_TOKEN_PATH, "r") as f:
    HF_TOKEN = f.read().strip()

# =============================================================================
# Load vocabulary and sentences
# =============================================================================

# 1. Load Original Target Vocabulary (The source of truth)
# VOCAB_PATH defined in config section above
print(f"Loading target vocabulary from {VOCAB_PATH}...")
vocab_df = pd.read_csv(VOCAB_PATH)

# Robustly find the word column
vocab_col = "word" if "word" in vocab_df.columns else vocab_df.columns[0]
if "word" in vocab_df.columns: 
    vocab_col = "word"
elif "word" in vocab_df.columns:
    vocab_col = "word"
else:
    vocab_col = vocab_df.columns[0] # Fallback

# Ensure all 5k target words are in the list
target_cue_words = sorted(vocab_df[vocab_col].astype(str).tolist())
print(f"Target vocabulary size: {len(target_cue_words)}")

# 2. Load Found Sentences
print(f"Loading sentences from {SENTENCES_PATH}...")
sentences_df = pd.read_csv(SENTENCES_PATH)

if "word" not in sentences_df.columns or "sentence" not in sentences_df.columns:
    raise ValueError(f"Sentences file must contain 'word' and 'sentence' columns.")

# Group found sentences
word_to_sentences = sentences_df.groupby("word")["sentence"].apply(list).to_dict()

# Identify missing words
found_words = set(word_to_sentences.keys())
missing_words = set(target_cue_words) - found_words
print(f"Found sentences for {len(found_words)} words.")
print(f"Missing sentences for {len(missing_words)} words.")
if missing_words:
    print(f"Example missing: {list(missing_words)[:5]}")
    print("Strategy: For missing words, the 'averaged' method will fallback to 'isolated' (0-shot).")

# Use the full target list to ensure perfect alignment with behavioral tasks
cue_words = target_cue_words

# =============================================================================
# Helper: mean hidden state per layer for target word within a prompt
# =============================================================================

def get_layer_means_for_prompt(
    prompt: str,
    model,
    tokenizer,
    target_substring: Optional[str] = None,
) -> list[torch.Tensor]:
    """
    Runs inference for a single prompt.
    Identifies tokens corresponding to `target_substring`.
    Returns list of tensors (one per layer), where each tensor is the mean embedding of the target tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    input_ids = inputs["input_ids"][0]

    special_ids = set(tokenizer.all_special_ids)
    
    # Default: use all content tokens if no substring match
    content_positions = [
        i for i, tok_id in enumerate(input_ids)
        if tok_id.item() not in special_ids
    ]

    if target_substring is not None:
        encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = encoding["offset_mapping"]
        
        # Simple rfind alignment (last occurrence)
        start_char = prompt.rfind(target_substring)
        
        if start_char != -1:
            end_char = start_char + len(target_substring)
            aligned_positions = []
            for i, (start, end) in enumerate(offsets):
                if i >= len(input_ids): break
                if input_ids[i].item() in special_ids: continue
                # Token must be strictly within the substring range? 
                # Or overlapping? "start < end_char and end > start_char"
                # Existing script logic: not (end <= start_char or start >= end_char)
                if not (end <= start_char or start >= end_char):
                    aligned_positions.append(i)
            if aligned_positions:
                content_positions = aligned_positions

    if not content_positions:
        # Fallback to all tokens if alignment fails
        content_positions = list(range(input_ids.size(0)))

    layer_means: list[torch.Tensor] = []
    for h in hidden_states:
        # h: (1, seq_len, hidden_dim)
        token_hidden = h[0, content_positions, :]
        mean_vec = token_hidden.mean(dim=0)
        layer_means.append(mean_vec)

    return layer_means

def compute_cosine_matrices(hidden_states_tensor):
    """
    hidden_states_tensor: (N, L, H)
    Returns: (L, N, N) cosine similarity matrices
    """
    print(f"  Computing cosine similarity matrices for {hidden_states_tensor.shape[1]} layers...")
    hidden_states_by_layer = hidden_states_tensor.permute(1, 0, 2) # (L, N, H)
    L_dim, N_dim, H_dim = hidden_states_by_layer.shape
    cosine_matrices = []
    
    for l_idx in range(L_dim):
        layer_vecs = hidden_states_by_layer[l_idx].to(torch.float32) # Ensure float32 for precision
        norms = layer_vecs.norm(dim=1, keepdim=True)
        layer_vecs_norm = layer_vecs / (norms + 1e-12)
        # Dot product of normalized vectors
        sim_matrix = torch.mm(layer_vecs_norm, layer_vecs_norm.t()) # (N, N)
        cosine_matrices.append(sim_matrix.cpu()) # Keep on CPU
        
    return torch.stack(cosine_matrices) # (L, N, N)

# =============================================================================
# Main Loop
# =============================================================================

def process_model(model_info: Dict[str, str]):
    model_name = model_info["name"]
    model_alias = model_info["alias"]
    
    print(f"\nChecking model: {model_name} ({model_alias})")
    
    methods_to_extract = []
    methods_to_update = []
    
    # 1. Check status of each method
    for method in METHODS:
        output_path = os.path.join(HIDDEN_STATE_DIR_BASE, model_alias, method, "hidden_states.pt")
        
        if not os.path.exists(output_path):
            methods_to_extract.append(method)
        else:
            # Check for cosine matrices
            try:
                try:
                    data = torch.load(output_path, map_location="cpu", weights_only=False)
                except TypeError:
                    data = torch.load(output_path, map_location="cpu")
                
                if "cosine_similarity_matrices" not in data:
                    print(f"  Found file but missing cosine matrices: {method}")
                    methods_to_update.append(method)
                else:
                    # check if cue_words match (optional safety, but let's assume valid)
                    pass
                del data
            except Exception as e:
                print(f"  Error reading {output_path}, will re-extract: {e}")
                methods_to_extract.append(method)

    # 2. Handle Updates (No model needed)
    for method in methods_to_update:
        output_path = os.path.join(HIDDEN_STATE_DIR_BASE, model_alias, method, "hidden_states.pt")
        print(f"  Updating cosine matrices for: {method}")
        
        try:
            try:
                data = torch.load(output_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(output_path, map_location="cpu")
            
            hidden_states_tensor = data["hidden_states"]
            cosine_matrices_tensor = compute_cosine_matrices(hidden_states_tensor)
            
            data["cosine_similarity_matrices"] = cosine_matrices_tensor
            
            # Safer save (local tmp -> remote atomic replace)
            safe_torch_save(data, output_path)
            
            print(f"  Updated: {output_path}")
            
            del data, hidden_states_tensor, cosine_matrices_tensor
            gc.collect()
            
        except Exception as e:
            print(f"  Failed to update {method}: {e}")

    # 3. Handle Extractions (Needs model)
    if not methods_to_extract:
        if not methods_to_update:
            print(f"  All methods up to date for {model_alias}.")
        return

    print(f"  Extracting methods: {methods_to_extract}")
    
    # Load model and tokenizer
    tokenizer_kwargs = {"token": HF_TOKEN, "trust_remote_code": True}
    if "Mistral-Nemo" in model_name:
        tokenizer_kwargs["fix_mistral_regex"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "token": HF_TOKEN,
        "output_hidden_states": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(DEVICE)
    model.eval()

    for method in methods_to_extract:
        target_dir = os.path.join(HIDDEN_STATE_DIR_BASE, model_alias, method)
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, "hidden_states.pt")
        
        print(f"  Processing method: '{method}'")
        
        all_word_layer_means: list[torch.Tensor] = []
        
        for word in tqdm(cue_words, desc=f"{model_alias} [{method}]"):
            
            # Determine prompts based on method
            prompts = []
            target_subs = []
            
            if method == "isolated":
                prompts = [word]
                target_subs = [word]
                
            elif method == "template":
                prompts = [f"What is the meaning of the word {word}?"]
                target_subs = [word]
                
            elif method == "averaged":
                # Get sentences for this word
                sents = word_to_sentences.get(word, [])
                
                if not sents:
                    # FALLBACK STRATEGY:
                    # If we have no sentences, we fallback to the isolated word.
                    # This acts as a prior.
                    # print(f"  Fallback: No sentences for '{word}', using isolated prompt.")
                    prompts = [word]
                    target_subs = [word]
                else:
                    prompts = sents
                    target_subs = [word] * len(sents)
            
            elif method == "forced_choice":
                prompts = [FORCED_CHOICE_TEMPLATE.format(input_word=word)]
                target_subs = [word]
                
            elif method == "free_association":
                prompts = [FREE_ASSOCIATION_TEMPLATE.format(input_word=word)]
                target_subs = [word]
            
            # Run inference for this word (1 or 50 times)
            # We collect the mean vector for each context
            context_vectors = [] 
            
            for p, sub in zip(prompts, target_subs):
                layer_means_list = get_layer_means_for_prompt(
                    prompt=p,
                    model=model,
                    tokenizer=tokenizer,
                    target_substring=sub,
                )
                # layer_means_list is [Layer1_vec, Layer2_vec, ...]
                # Stack to (L, H)
                layer_stack = torch.stack(layer_means_list).cpu() 
                context_vectors.append(layer_stack)
            
            # Average across contexts
            # Stack contexts: (N_contexts, L, H)
            if context_vectors:
                contexts_tensor = torch.stack(context_vectors)
                # Mean over dim 0 (contexts) -> (L, H)
                word_mean_tensor = contexts_tensor.mean(dim=0)
            else:
                # Should not be reached
                word_mean_tensor = torch.zeros((model.config.num_hidden_layers + 1, model.config.hidden_size))
            
            all_word_layer_means.append(word_mean_tensor)

        # Stack all words -> (N_words, L, H)
        hidden_states_tensor = torch.stack(all_word_layer_means)
        
        # Compute cosine similarity matrices per layer
        cosine_matrices_tensor = compute_cosine_matrices(hidden_states_tensor)
        if SAVE_COSINE_FLOAT16:
            cosine_matrices_tensor = cosine_matrices_tensor.to(torch.float16)

        payload = {
                "model_name": model_name,
                "cue_words": cue_words,
                "hidden_states": hidden_states_tensor,
                "cosine_similarity_matrices": cosine_matrices_tensor,
                "method": method,
        }

        print(f"  Saving (safe) to: {output_path} (via local tmp: {DEFAULT_LOCAL_TMP_DIR}) ...")
        safe_torch_save(payload, output_path)
        print(f"  Saved: {output_path}")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def main():
    print(f"Vocabulary: {len(cue_words)} words.")
    for m in MODELS:
        try:
            process_model(m)
        except Exception as e:
            print(f"Error processing {m['alias']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
