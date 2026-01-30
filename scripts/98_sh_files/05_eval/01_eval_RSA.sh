#!/bin/bash
#SBATCH --job-name=eval_rsa
#SBATCH --output=logs/05_eval/01_RSA/eval_rsa-%A_%a.out
#SBATCH --error=logs/05_eval/01_RSA/eval_rsa-%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=standard
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-31%8

set -euo pipefail

mkdir -p logs/05_eval/01_RSA

start_time=$(date +%s)

echo "Starting CENTERED RSA evaluation at: $(date)"
echo "Job: ${SLURM_JOB_ID:-NA} | Array: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $(hostname) | CPUs: $(nproc) | SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

# Record start time for timing
START_TIME=$(date +%s)

# Sampling settings for CPU computation
# - PAIR_SAMPLE_SIZE=0 uses all pairs (~12.5M pairs, very slow on CPU)
# - PAIR_SAMPLE_SIZE=500000 uses 500k random pairs
# - PAIR_SAMPLE_SIZE=100000 uses 100k random pairs
PAIR_SAMPLE_SIZE=500000
PAIR_SAMPLE_SEED=0

echo "Using pair sampling: ${PAIR_SAMPLE_SIZE} pairs"

# Set threading for CPU computation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source "$HOME/.bashrc"
conda activate conda_env_py311

echo "Python script starting at: $(date)"
PYTHON_START_TIME=$(date +%s)

# Run RSA evaluation (one model/prompt per array task)
JOB_COUNT=32
python scripts/05_eval/01_RSA/01_RSA.py \
  --pair-sample-size "${PAIR_SAMPLE_SIZE}" \
  --seed "${PAIR_SAMPLE_SEED}" \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --job-count "${JOB_COUNT}" \
  --no-summary

# Calculate and report timing
PYTHON_END_TIME=$(date +%s)
END_TIME=$(date +%s)

PYTHON_DURATION=$((PYTHON_END_TIME - PYTHON_START_TIME))
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "Python script completed at: $(date)"
echo "Python execution time: ${PYTHON_DURATION}s ($(($PYTHON_DURATION / 60))m $(($PYTHON_DURATION % 60))s)"
echo "Total job time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo "Finished CENTERED RSA evaluation at: $(date)"
