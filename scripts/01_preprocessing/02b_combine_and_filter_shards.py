import pandas as pd
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine shards and filter top 5000 words.")
    parser.add_argument("--input_dir", default="data/vocabulary/04_sentences", help="Directory with shard CSVs")
    parser.add_argument("--vocab_path", default="data/vocabulary/03_stimulus_list/subtlex_stimuli_6k.csv", help="Original 6k vocab to preserve frequency order")
    parser.add_argument("--output_sentences", default="data/vocabulary/04_sentences/sentences_c4_final.csv", help="Final combined sentences file")
    parser.add_argument("--output_vocab", default="data/vocabulary/03_stimulus_list/subtlex_stimuli_5k_final.csv", help="Final list of 5000 words")
    args = parser.parse_args()

    print(f"Reading shards from {args.input_dir}...")
    all_files = glob.glob(os.path.join(args.input_dir, "sentences_c4_shard_*.csv"))
    
    df_list = []
    for f in all_files:
        try:
            # Skip empty files
            if os.path.getsize(f) > 0:
                df = pd.read_csv(f, on_bad_lines='skip')
                df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        print("No data found!")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Total sentences extracted: {len(full_df)}")

    # Group by word to count
    word_counts = full_df['word'].value_counts()
    
    # Filter words with >= 50 sentences
    valid_words = word_counts[word_counts >= 50].index.tolist()
    print(f"Words with >= 50 sentences: {len(valid_words)}")

    # Load original vocab to prioritize high-frequency words
    print(f"Loading original vocab from {args.vocab_path}...")
    vocab_df = pd.read_csv(args.vocab_path)
    
    # Determine correct column name (surface form 'word' preferred)
    if "word" in vocab_df.columns:
        vocab_col = "word"
    elif "lemma" in vocab_df.columns:
        vocab_col = "lemma"
    else:
        vocab_col = vocab_df.columns[0]
        
    # Create an ordered list of candidates based on original frequency
    ordered_candidates = vocab_df[vocab_col].astype(str).tolist()
    
    final_5k_words = []
    for word in ordered_candidates:
        if word in valid_words:
            final_5k_words.append(word)
            if len(final_5k_words) >= 5000:
                break
    
    print(f"Selected {len(final_5k_words)} words for the final dataset.")

    if len(final_5k_words) < 5000:
        print("WARNING: Could not find 5000 valid words! Using all available.")

    # Filter the full dataframe to keep only these 5000 words
    final_df = full_df[full_df['word'].isin(final_5k_words)]
    
    # Downsample to exactly 50 sentences per word
    print("Downsampling to 50 sentences per word...")
    final_df = final_df.groupby('word').head(50)
    
    # Sort by word (optional, but nice)
    final_df = final_df.sort_values(by=['word'])

    # Save
    print(f"Saving final sentences to {args.output_sentences}...")
    os.makedirs(os.path.dirname(args.output_sentences), exist_ok=True)
    
    # final_df.to_csv(args.output_sentences, index=False)
    import csv
    try:
        final_df.to_csv(args.output_sentences, index=False, chunksize=10000)
    except OSError:
        print("Pandas write failed, falling back to standard csv module...")
        with open(args.output_sentences, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(final_df.columns)
            for row in final_df.itertuples(index=False):
                writer.writerow(row)
    
    # Save the new vocab list
    print(f"Saving final vocab list to {args.output_vocab}...")
    # Filter original vocab df to keep metadata (frequency etc.)
    final_vocab_df = vocab_df[vocab_df[vocab_col].isin(final_5k_words)].head(5000)
    final_vocab_df.to_csv(args.output_vocab, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
