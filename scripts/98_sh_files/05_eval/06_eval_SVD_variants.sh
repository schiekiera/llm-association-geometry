#!/bin/bash
#SBATCH --job-name=eval_fc_variants
#SBATCH --output=logs/05_eval/03_SVD_variants/eval_variants-%j.out
#SBATCH --error=logs/05_eval/03_SVD_variants/eval_variants-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7%4

mkdir -p logs/05_eval/03_SVD_variants

echo "Starting CENTERED FC and FA SVD variants evaluation at: $(date)"

# Activate environment
source "$HOME/.bashrc"
conda activate conda_env_py311

# Settings
PAIR_SAMPLE_SIZE=500000
PAIR_SAMPLE_SEED=0

python3 scripts/05_eval/03_SVD_variants/01_compare_SVD_variants.py \
  --pair-sample-size "${PAIR_SAMPLE_SIZE}" \
  --seed "${PAIR_SAMPLE_SEED}" \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --job-count "${SLURM_ARRAY_TASK_COUNT}"

echo "Finished CENTERED FC and FA SVD variants evaluation at: $(date)"
