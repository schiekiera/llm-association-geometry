# -*- coding: utf-8 -*-
"""
Python translation of the R script with:
1. Noun filtering
2. Removal of contraction-fragment pseudo-words
3. Lemmatization of all words
4. Removal of lemma duplicates, keeping the most frequent lemma row
"""

import pandas as pd
import spacy

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
PATH = "data/vocabulary/01_full_vocab/subtlex_US.csv"
STIMULUS_LIST_SIZE = 6000


# Load spaCy English model (install first if needed: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = pd.read_csv(PATH)

# Inspect columns (optional)
print(df.columns)
print(df.head())

# -------------------------------------------------------------------
# FILTER ONLY NOUNS (Dom_PoS_SUBTLEX == "Noun")
# -------------------------------------------------------------------
df_noun = df[df["Dom_PoS_SUBTLEX"] == "Noun"].copy()

# -------------------------------------------------------------------
# SORT BY FREQUENCY (descending, like order(-FREQcount))
# -------------------------------------------------------------------
df_noun = df_noun.sort_values("FREQcount", ascending=False)

# -------------------------------------------------------------------
# REMOVE CONTRACTION-FRAGMENT PSEUDO-WORDS
# -------------------------------------------------------------------
bad_contractions = [
    # 't contractions
    "isn", "aren", "wasn", "weren", "doesn", "don", "didn",
    "couldn", "shouldn", "wouldn", "mustn", "mightn", "needn",
    "hasn", "haven", "hadn", "ain",

    # 'll contractions
    "ll", "itll", "thatll", "therell", "theyll", "youl", "youll", "iwill",

    # 're contractions
    "re", "yre", "theyr",

    # 've contractions
    "ve", "theve", "youve", "theyve",

    # 'd contractions
    "d", "itd", "theyd", "youd", "hed",

    # 's contractions (safe removals only)
    "ts", "hes", "shes",
    
    # weli, misspelling of well
    "weli", 
]

df_noun = df_noun[~df_noun["Word"].isin(bad_contractions)].copy()

# -------------------------------------------------------------------
# LEMMATIZE ALL WORDS
# -------------------------------------------------------------------
# We lemmatize the "Word" column and create a new "lemma" column.
# Using nlp.pipe for efficiency.
words = df_noun["Word"].astype(str).tolist()
lemmas = []

for doc in nlp.pipe(words, batch_size=1000):
    # For single-word entries, take the lemma of the first token;
    # if something goes wrong, fall back to the original text
    if len(doc) > 0:
        lemmas.append(doc[0].lemma_)
    else:
        lemmas.append("")

df_noun["lemma"] = lemmas

# -------------------------------------------------------------------
# REMOVE LEMMA DUPLICATES, KEEPING MOST FREQUENT ROW
# -------------------------------------------------------------------
# We already sorted df_noun by FREQcount in descending order,
# so drop_duplicates(keep="first") will keep the row with highest FREQcount
# for each lemma.
df_noun = df_noun.drop_duplicates(subset="lemma", keep="first").copy()

# -------------------------------------------------------------------
# ADD INDEX COLUMN (1-based) AS FIRST COLUMN
# -------------------------------------------------------------------
df_noun = df_noun.reset_index(drop=True)
df_noun.insert(0, "index", df_noun.index + 1)

# delete all non string entries in word
df_noun = df_noun[df_noun["Word"].apply(lambda x: isinstance(x, str))]

# delete all words with only one or two letters
df_noun = df_noun[df_noun["Word"].apply(lambda x: len(x) > 2)]

# keep only lemma, Word, FREQcount, Dom_PoS_SUBTLEX and rename them for consistency
df_noun = df_noun[["lemma", "Word", "FREQcount"]]
df_noun.columns = ["lemma", "word", "freq"]

# -------------------------------------------------------------------
# SAVE RESULT: FILTERED VOCABULARY
# -------------------------------------------------------------------
out_path = "data/vocabulary/02_filtered_vocab/subtlex_US_nouns_lemma_dedup.csv"
df_noun.to_csv(out_path, index=False)

print("Final shape:", df_noun.shape)
print("Saved to:", out_path)


# -------------------------------------------------------------------
# SAVE RESULT: STIMULUS LIST
# -------------------------------------------------------------------

out_path_stimuli = "data/vocabulary/03_stimulus_list/subtlex_stimuli_6k.csv"

# select the first 500 rows
df_noun_stimuli = df_noun.head(STIMULUS_LIST_SIZE)

# save the first 500 rows
df_noun_stimuli.to_csv(out_path_stimuli, index=False)

print("Final shape:", df_noun_stimuli.shape)
print("Saved to:", out_path_stimuli)