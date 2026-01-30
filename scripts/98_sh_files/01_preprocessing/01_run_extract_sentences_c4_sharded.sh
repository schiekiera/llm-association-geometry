#!/bin/bash
#SBATCH --job-name=extract_c4_shard
#SBATCH --output=logs/01_preprocessing/extract_c4_shard_%a-%A.out
#SBATCH --error=logs/01_preprocessing/extract_c4_shard_%a-%A.err
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-19  # 20 shards

# Ensure log directory exists
mkdir -p logs/01_preprocessing

echo "Starting C4 extraction for shard ${SLURM_ARRAY_TASK_ID} at: $(date)"

# Define paths
VOCAB_FILE="data/vocabulary/03_stimulus_list/subtlex_stimuli_6k.csv"
OUTPUT_DIR="data/vocabulary/04_sentences"
mkdir -p ${OUTPUT_DIR}

# Output file specific to this shard
OUTPUT_FILE="${OUTPUT_DIR}/sentences_c4_shard_${SLURM_ARRAY_TASK_ID}.csv"

python3 scripts/01_preprocessing/02a_extract_sentences_c4.py \
    --vocab ${VOCAB_FILE} \
    --output ${OUTPUT_FILE} \
    --dataset_name allenai/c4 \
    --subset en \
    --max_docs 10000000 \
    --max_sentences_per_word 500 \
    --shard_id ${SLURM_ARRAY_TASK_ID} \
    --num_shards 20

echo "Finished extraction for shard ${SLURM_ARRAY_TASK_ID} at: $(date)"
