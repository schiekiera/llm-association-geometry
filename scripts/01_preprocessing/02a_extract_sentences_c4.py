import argparse
import os
import csv
import re
import sys
import random
from collections import defaultdict
from typing import Set, Dict, List
from datetime import datetime
import pandas as pd

import nltk
from tqdm import tqdm
from datasets import load_dataset

# Ensure NLTK data is available (for sentence splitting)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def get_args():
    parser = argparse.ArgumentParser(description="Stream C4 dataset and extract sentences for target words.")
    parser.add_argument("--vocab", required=True, help="Path to vocabulary CSV (must have 'lemma' column)")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--dataset_name", default="allenai/c4", help="HF dataset name")
    parser.add_argument("--subset", default="en", help="Subset (en or realnewslike)")
    parser.add_argument("--max_sentences_per_word", type=int, default=50, help="Target number of sentences per word")
    parser.add_argument("--max_docs", type=int, default=1000000, help="Maximum number of C4 documents to scan")
    parser.add_argument("--min_sent_len", type=int, default=5, help="Min words in a sentence")
    parser.add_argument("--max_sent_len", type=int, default=100, help="Max words in a sentence")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    return parser.parse_args()

def load_vocab(path: str, shard_id: int = 0, num_shards: int = 1) -> Set[str]:
    print(f"Loading vocabulary from {path}...")
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_csv(path, sep="\t")
    
    if "word" in df.columns:
        all_words = df["word"].astype(str).str.lower().tolist()
        print("Using 'word' column for targets.")
    elif "lemma" in df.columns:
        all_words = df["lemma"].astype(str).str.lower().tolist()
        print("Using 'lemma' column for targets.")
    else:
        # Fallback if no header or different name, try first column
        all_words = df.iloc[:, 0].astype(str).str.lower().tolist()
    
    # Sort to ensure deterministic sharding
    all_words.sort()
    unique_words = sorted(list(set(all_words)))
    
    # Calculate shard slice
    if num_shards > 1:
        chunk_size = len(unique_words) // num_shards
        start_idx = shard_id * chunk_size
        # Last shard takes the remainder
        end_idx = (shard_id + 1) * chunk_size if shard_id < num_shards - 1 else len(unique_words)
        
        words = set(unique_words[start_idx:end_idx])
        print(f"Shard {shard_id}/{num_shards}: processing {len(words)} words (indices {start_idx}-{end_idx})")
    else:
        words = set(unique_words)
        print(f"Loaded {len(words)} unique target words.")
        
    return words

def clean_sentence(text: str) -> str:
    """Basic cleanup to remove excess whitespace."""
    return " ".join(text.split())

def main():
    args = get_args()
    
    # Needs pandas for easy CSV reading
    import pandas as pd
    
    # 1. Load Targets
    target_words = load_vocab(args.vocab, args.shard_id, args.num_shards)
    
    # Reservoirs: word -> list of sentences
    reservoirs = defaultdict(list)
    # Track which words are "done" to optimize
    completed_words = set()
    
    # 2. Setup Dataset Stream
    print(f"Initializing stream for {args.dataset_name} ({args.subset})...")
    ds = load_dataset(args.dataset_name, args.subset, split="train", streaming=True)
    
    # Robustness: Check for existing output to resume
    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        print(f"Found existing output at {args.output}. Resuming...")
        try:
            existing_df = pd.read_csv(args.output)
            # Ensure columns are correct
            if "word" in existing_df.columns and "sentence" in existing_df.columns:
                for _, row in existing_df.iterrows():
                    w = str(row["word"])
                    s = str(row["sentence"])
                    if w in target_words:
                        reservoirs[w].append(s)
            
            # Update completed status
            for w, sents in reservoirs.items():
                if len(sents) >= args.max_sentences_per_word:
                    completed_words.add(w)
            
            print(f"Resumed state: {len(completed_words)} words already completed.")
            print(f"Total sentences already collected: {sum(len(s) for s in reservoirs.values())}")
        except Exception as e:
            print(f"Error reading existing output: {e}. Starting fresh (or appending if file valid).")

    doc_count = 0
    total_sentences_found = sum(len(s) for s in reservoirs.values())
    start_time = datetime.now()
    
    # Regex for whole word matching (simple boundary)
    # Pre-compiling a massive regex for 30k words is slow/impossible.
    # Instead, we'll tokenize sentences and check set membership.
    
    print("Starting scan...")
    
    # Open file in append mode for continuous saving
    # If file didn't exist, write header
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["word", "sentence"])

        try:
            for sample in ds:
                text = sample.get("text", "")
                if not text:
                    continue
                    
                # Split into sentences (NLTK is robust for this)
                sentences = nltk.sent_tokenize(text)
                
                for sent in sentences:
                    # Basic length filtering
                    words = sent.split()
                    if len(words) < args.min_sent_len or len(words) > args.max_sent_len:
                        continue
                    
                    # Lowercase set for fast lookup
                    # removing punctuation for matching
                    sent_lower = sent.lower()
                    # Basic tokenization for matching: keep letters only? 
                    # Let's simple split by non-alphanumeric to find tokens
                    sent_tokens = set(re.split(r'[^a-z0-9]+', sent_lower))
                    
                    # Intersection finds targets present in this sentence
                    # Only check words that are NOT completed yet
                    present_targets = sent_tokens.intersection(target_words - completed_words)
                    
                    if present_targets:
                        clean_sent = clean_sentence(sent)
                        
                        for target in present_targets:
                            # Reservoir sampling / simple filling
                            if len(reservoirs[target]) < args.max_sentences_per_word:
                                reservoirs[target].append(clean_sent)
                                total_sentences_found += 1
                                
                                # Write immediately to disk and FLUSH
                                writer.writerow([target, clean_sent])
                                f.flush()
                                
                                if len(reservoirs[target]) >= args.max_sentences_per_word:
                                    completed_words.add(target)
                
                doc_count += 1
                
                # Progress logging
                if doc_count % 1000 == 0:
                    elapsed = datetime.now() - start_time
                    print(f"Scanned {doc_count} docs (this run). Found {total_sentences_found} sentences total. "
                          f"Completed words: {len(completed_words)}/{len(target_words)}. "
                          f"Time: {elapsed}", flush=True)
                
                # Global stop conditions
                if doc_count >= args.max_docs:
                    print("Reached maximum document limit.")
                    break
                
                if len(completed_words) == len(target_words):
                    print("All target words satisfied!")
                    break
                    
        except Exception as e:
            print(f"Stream interrupted or error: {e}")
            import traceback
            traceback.print_exc()
    
    # No need to save at the end as we streamed writes
    print("Done.")

if __name__ == "__main__":
    main()
