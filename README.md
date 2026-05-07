---
license: cc-by-4.0
language:
- en
pretty_name: LLM Behavioral Association Dataset
tags:
- psychology
- behavior
- cognitive
- interpretability
- networks
- association
size_categories:
- 10M<n<100M
configs:
  # Forced choice: one config per model
  - config_name: "forced_choice/Falcon3-10B-Instruct"
    data_files:
      - split: train
        path: "forced_choice/Falcon3-10B-Instruct.parquet"
  - config_name: "forced_choice/gemma-2-9b-it"
    data_files:
      - split: train
        path: "forced_choice/gemma-2-9b-it.parquet"
  - config_name: "forced_choice/Llama-3.1-8B-Instruct"
    data_files:
      - split: train
        path: "forced_choice/Llama-3.1-8B-Instruct.parquet"
  - config_name: "forced_choice/Mistral-7B-Instruct-v0.2"
    data_files:
      - split: train
        path: "forced_choice/Mistral-7B-Instruct-v0.2.parquet"
  - config_name: "forced_choice/Mistral-Nemo-Instruct-v1"
    data_files:
      - split: train
        path: "forced_choice/Mistral-Nemo-Instruct-v1.parquet"
  - config_name: "forced_choice/phi-4"
    data_files:
      - split: train
        path: "forced_choice/phi-4.parquet"
  - config_name: "forced_choice/Qwen2.5-7B-Instruct"
    data_files:
      - split: train
        path: "forced_choice/Qwen2.5-7B-Instruct.parquet"
  - config_name: "forced_choice/rnj-1-instruct"
    data_files:
      - split: train
        path: "forced_choice/rnj-1-instruct.parquet"

  # Free association: one config per model
  - config_name: "free_association/Falcon3-10B-Instruct"
    data_files:
      - split: train
        path: "free_association/Falcon3-10B-Instruct.parquet"
  - config_name: "free_association/gemma-2-9b-it"
    data_files:
      - split: train
        path: "free_association/gemma-2-9b-it.parquet"
  - config_name: "free_association/Llama-3.1-8B-Instruct"
    data_files:
      - split: train
        path: "free_association/Llama-3.1-8B-Instruct.parquet"
  - config_name: "free_association/Mistral-7B-Instruct-v0.2"
    data_files:
      - split: train
        path: "free_association/Mistral-7B-Instruct-v0.2.parquet"
  - config_name: "free_association/Mistral-Nemo-Instruct-v1"
    data_files:
      - split: train
        path: "free_association/Mistral-Nemo-Instruct-v1.parquet"
  - config_name: "free_association/phi-4"
    data_files:
      - split: train
        path: "free_association/phi-4.parquet"
  - config_name: "free_association/Qwen2.5-7B-Instruct"
    data_files:
      - split: train
        path: "free_association/Qwen2.5-7B-Instruct.parquet"
  - config_name: "free_association/rnj-1-instruct"
    data_files:
      - split: train
        path: "free_association/rnj-1-instruct.parquet"
---

# From Associations to Activations
### Comparing Behavioral and Hidden-State Semantic Geometry in LLMs

> **Accepted at ICML 2026** · [Paper](https://arxiv.org/abs/2602.00628) · [Dataset on Hugging Face](https://huggingface.co/datasets/schiekiera/llm-association-geometry) · [Blog post](https://schiekiera.github.io/blog/2026/neural-semantic-geometry/)


In cognitive science, semantic knowledge is treated as a latent structure: we cannot observe a speaker's meaning representations directly, but we can probe them through behavior. Word-association paradigms use exactly this logic — when a participant sees a cue (e.g., *dog*), the associations they produce (*cat*, *leash*, *bark*) are constrained by their underlying semantic organization, and aggregated responses yield a similarity matrix that approximates the geometry of an otherwise unobserved system.

We transfer this measurement logic to large language models. Unlike humans, LLMs make *both* behavior and internal representations observable, so we can ask directly: how well does an LLM's behavioral output reveal its own internal semantic geometry?

<p align="center">
  <img src="img/conceptual.png" alt="Conceptual overview" width="780">
</p>

**The framework.** Over a shared vocabulary of 5,000 high-frequency English nouns, we (i) extract layer-wise word representations to form hidden-state similarity matrices, and (ii) collect behavioral associations to build behavioral similarity matrices. Representational similarity analysis (RSA) then correlates the two geometries to quantify behavior–activation alignment.

**Two psycholinguistic paradigms.** We probe each model under two classic tasks: **forced choice (FC)**, where the model selects the two most related words from a candidate set of 16, and **free association (FA)**, where the model generates five associates from a single cue. Cue–response counts are reweighted by PPMI, and a cue–cue similarity matrix is derived by cosine similarity. In total, we collected over **17.5 million trials** across eight instruction-tuned transformers (7B–14B params).

<p align="center">
  <img src="img/both_paradigms.png" alt="Forced choice and free association paradigms" width="780">
</p>

**Key findings.**
- **Forced choice aligns substantially more than free association.** Mean FC RSA reaches *r* = .463 under task-aligned hidden-state extraction, compared to *r* = .199 for FA. The advantage is consistent across all eight models.
- **Behavioral similarity predicts unseen hidden-state structure.** A held-out-words ridge regression shows that adding FC similarity on top of lexical baselines and a cross-model consensus geometry improves test *R*² by +.022, with peak *R*² = .844 for Llama-3.1-8B-Instruct.
- **Measurement protocol matters.** Whether behavior *reveals* internal structure is not a generic property of "behavior" — it depends critically on how responses are constrained and aggregated. Constrained paradigms like FC concentrate observations and yield higher signal-to-noise estimates of semantic geometry.

<p align="center">
  <img src="rsa_line_plot_1x2_grid_fc_fa.png" alt="Forced choice aligns substantially more with hidden states than free association" width="780">
</p>

For interpretability research, this means behavioral probing can serve as a practical tool for understanding internal representations under black-box access, *if* the probe is designed with sufficient response constraint.

---

## Citation

If you find this work helpful for your research, please cite our paper:

```bibtex
@inproceedings{schiekiera2026associations,
  title={From Associations to Activations: Comparing Behavioral and
         Hidden-State Semantic Geometry in {LLMs}},
  author={Schiekiera, Louis and Zimmer, Max and Roux, Christophe
          and Pokutta, Sebastian and G{\"u}nther, Fritz},
  booktitle={Proceedings of the 43rd International Conference on
             Machine Learning (ICML)},
  year={2026},
  publisher={PMLR},
}
```

---

# Repository contents

This repository accompanies the paper and contains the **behavioral association outputs** from eight instruction-tuned LLMs collected under both paradigms, plus the analysis scripts.

- **Forced choice (FC):** given an input word and a candidate set of 16 words, the model selects the two most related options.
- **Free association (FA):** given an input word, the model generates a set of 5 associated words.

**Scope:** This release includes **processed** data only (Parquet).

## Quick links
- [Folder structure](#folder-structure)
- [Dataset size](#dataset-size)
- [Data formats and schemas](#data-formats-and-schemas)
- [How to load](#how-to-load-using-hugging-face-datasets-)
- [License (dataset)](#license-dataset)
- [Upstream model terms](#upstream-model-terms)
- [Citation](#citation)

### Folder structure
```text
forced_choice/
  Falcon3-10B-Instruct.parquet
  gemma-2-9b-it.parquet
  Llama-3.1-8B-Instruct.parquet
  Mistral-7B-Instruct-v0.2.parquet
  Mistral-Nemo-Instruct-v1.parquet
  phi-4.parquet
  Qwen2.5-7B-Instruct.parquet
  rnj-1-instruct.parquet

free_association/
  Falcon3-10B-Instruct.parquet
  gemma-2-9b-it.parquet
  Llama-3.1-8B-Instruct.parquet
  Mistral-7B-Instruct-v0.2.parquet
  Mistral-Nemo-Instruct-v1.parquet
  phi-4.parquet
  Qwen2.5-7B-Instruct.parquet
  rnj-1-instruct.parquet

meta_data/
  models.json
  vocab.csv
```

## Dataset size

All files are Parquet.

### Forced choice (FC)
- 8 files, **1,565,000 trials per model** (≈ 12.52M trials total)
    - Each row corresponds to a single trial
- All models together on disk (Parquet): **902MB**

### Free association (FA)
- 8 files, **630,000 trials per model** (≈ 5.04M trials total)
    - Each row corresponds to a single association = **≈3.07M–3.15M rows per model**
- All models together on disk (Parquet): **58MB**

### Included models
See `meta_data/models.json` for canonical Hugging Face model IDs and model specs.

---

## Data formats and schemas

All data are provided as **Parquet** tables.

#### Forced choice (`forced_choice/*.parquet`)
Each row corresponds to a **single forced-choice trial**.

**Columns (8):**

- `trial_id` *(int)*: unique trial identifier within a model
- `input` *(string)*: cue word
- `candidates` *(string)*: candidate set (separated by commas)
- `output` *(string)*: model's raw output string
- `extracted_1`, `extracted_2` *(string)*: parsed/normalized extracted responses
- `pick_1`, `pick_2` *(string)*: final picks (exclusion of non-candidate words and input words)

#### Free association (`free_association/*.parquet`)
Each row corresponds to a **single association** produced by a model.

**Columns (4):**

- `run` *(int)*: run number extracted from the source filename (e.g., `_run14_`)
- `input` *(string)*: cue word
- `association` *(string)*: generated associate word
- `position` *(int)*: ordinal position of the associate within the response (1 = strongest/first)

---

## How to load using Hugging Face Datasets 🤗
You can load the Parquet files directly via `data_files`.

```python
from datasets import load_dataset

ds_fc = load_dataset(
    "schiekiera/llm-association-geometry",
    data_files="forced_choice/*.parquet"
)

ds_fa = load_dataset(
    "schiekiera/llm-association-geometry",
    data_files="free_association/*.parquet"
)
```



---

## License (dataset)

This dataset is released under Creative Commons Attribution 4.0 International (CC BY 4.0).
- You may share and adapt the data for any purpose, including commercial use.
- You must give appropriate credit, link the license, and indicate changes.

- **Creative Commons deed**: `https://creativecommons.org/licenses/by/4.0/`
- **Attribution best practices (TASL)**: `https://wiki.creativecommons.org/wiki/Recommended_practices_for_attribution`

### Suggested attribution (TASL)
- **Title**: From Associations to Activations — LLM Behavioral Association Dataset
- **Author**: Louis Schiekiera / Humboldt-Universität zu Berlin
- **Source**: huggingface.co/schiekiera/llm-association-geometry
- **License**: CC BY 4.0

---

## Upstream model terms

This repository distributes model-generated outputs (words).
Users are responsible for complying with upstream model licenses and acceptable-use policies where applicable.

### Falcon (TII Falcon / Falcon3)

TII's Falcon terms include an Acceptable Use Policy (AUP) requirement and state you may not use the work/derivatives or any output to create other works for any purpose that conflicts with the AUP.
See:
- `https://falconllm.tii.ae/falcon-terms-and-conditions.html`
- `https://falconllm.tii.ae/falcon3/falcon-3-acceptable-use-policy.html`

### Llama 3.1 (Meta)

The Llama 3.1 Community License includes requirements for certain distributions and also states that if you use the Llama materials or any outputs/results to create/train/fine-tune/improve an AI model that is distributed, you must include "Llama" at the beginning of that model name.
See:
- `https://www.llama.com/llama3_1/license/` (also mirrored on the model card)

### Gemma (Google)
This dataset includes outputs generated by Gemma models. Gemma's terms include an Acceptable Use Policy and distinguish between "Outputs" (which may generally be used and shared) and "Model Derivatives" (e.g., using outputs to build a model intended to replicate Gemma's capabilities).  
Users training or distributing models using this dataset should review the Gemma terms.
See: `https://ai.google.dev/gemma/terms`


### Mistral (Mistral-7B-Instruct-v0.2, Mistral-Nemo-Instruct-2407)

Both models are released under Apache-2.0 on Hugging Face.  ￼
Practical implication: Apache-2.0 is permissive (keep required notices/license when redistributing the model or derivatives). It typically does not impose special restrictions on sharing outputs.

### Phi (microsoft/phi-4)

Phi-4 is released under the MIT License on Hugging Face.  ￼
Practical implication: MIT is permissive for research/commercial use and redistribution (subject to keeping the license notice when redistributing the software/model). It typically does not add special "output use" constraints.

### Qwen (Qwen2.5-7B-Instruct)

Qwen2.5 models use the Qwen LICENSE AGREEMENT (not Apache-2.0 on HF).  ￼
Notable clauses to flag for downstream users:
•	If you use the Materials or any outputs/results to create/train/fine-tune/improve a model that you distribute or make available, you must display "Built with Qwen" or "Improved using Qwen" in the related documentation.  ￼
•	If commercially using the Materials and your product/service exceeds 100M monthly active users, you must request a separate license.  ￼

### rnj (EssentialAI/rnj-1-instruct)
rnj-1-instruct is listed under Apache-2.0 on Hugging Face.  ￼
Practical implication: similar to Mistral—permissive, usually no special constraints on outputs beyond standard Apache requirements for redistribution of the model/derivatives.

