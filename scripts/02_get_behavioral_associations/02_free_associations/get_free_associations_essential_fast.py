import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any

# =============================================================================
# Configuration
# =============================================================================

VOCAB_PATH = "data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv"
HF_TOKEN_PATH = "data/token/hf_token.txt"
HF_CACHE_ROOT = "cache/huggingface"

ASSOC_OUTPUT_DIR = "data/02_behavioral_associations/02_free_associations/01_raw"
os.makedirs(ASSOC_OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "EssentialAI/rnj-1-instruct"
MODEL_ALIAS = "rnj-1-instruct"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_PROMPT = (
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
    "input: "
)

# Generation settings
N_RUNS = 126
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 25
BASE_SEED = 12345
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Utils
# =============================================================================

def chunks(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i+k]

def wrap_as_chat(tokenizer, prompts: List[str]) -> List[str]:
    wrapped = []
    for p in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}
        ]
        wrapped.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return wrapped

# =============================================================================
# Main
# =============================================================================

def main():
    os.environ["HF_HOME"] = HF_CACHE_ROOT
    print(f"Loading {MODEL_ALIAS}...")
    
    with open(HF_TOKEN_PATH, "r") as f:
        hf_token = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    terminators = [tokenizer.eos_token_id]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    vocab_df = pd.read_csv(VOCAB_PATH)
    cue_words_all = vocab_df["word"].astype(str).tolist()

    for run_idx in range(N_RUNS):
        run_seed = BASE_SEED + run_idx
        torch.manual_seed(run_seed)
        
        out_path = os.path.join(ASSOC_OUTPUT_DIR, f"{MODEL_ALIAS}/{MODEL_ALIAS}_FA_run{run_idx:02d}_{TIMESTAMP}.csv")
        buffer_rows = []
        wrote_header = False

        print(f"\n--- Run {run_idx+1}/{N_RUNS} (seed={run_seed}) ---")

        for batch_words in tqdm(list(chunks(cue_words_all, BATCH_SIZE))):
            prompts = [f"{BASE_PROMPT}{w}.\n output: " for w in batch_words]
            chat_prompts = wrap_as_chat(tokenizer, prompts)
            
            inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True).to(DEVICE)
            
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    eos_token_id=terminators,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
            responses = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for w, resp in zip(batch_words, responses):
                buffer_rows.append({
                    "run_idx": run_idx,
                    "seed": run_seed,
                    "cue_word": w,
                    "response": resp.strip()
                })

            if len(buffer_rows) >= 1000:
                pd.DataFrame(buffer_rows).to_csv(out_path, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                buffer_rows = []

        if buffer_rows:
            pd.DataFrame(buffer_rows).to_csv(out_path, mode="a", header=not wrote_header, index=False)

if __name__ == "__main__":
    main()

