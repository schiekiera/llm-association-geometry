#!/bin/bash
#SBATCH --job-name=eval_nn
#SBATCH --output=logs/05_eval/02_NN/eval_nn-%A_%a.out
#SBATCH --error=logs/05_eval/02_NN/eval_nn-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7%2

set -euo pipefail

mkdir -p logs/05_eval/02_NN

echo "Starting CENTERED NN evaluation at: $(date)"
echo "Job: ${SLURM_JOB_ID:-NA} | Array: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $(hostname) | CPUs: $(nproc) | SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

# NN@k settings for CPU computation
NN_KS=(5 10 20 50 100 200)

echo "Using NN@k values: ${NN_KS[*]}"

# Set threading for CPU computation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source "$HOME/.bashrc"
conda activate conda_env_py311

echo "Python script starting at: $(date)"
PYTHON_START_TIME=$(date +%s)

# Run NN evaluation (one model per array task)
JOB_COUNT=8
python scripts/05_eval/02_NN/01_NN.py \
  --k "${NN_KS[@]}" \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --job-count "${JOB_COUNT}" \
  --device cuda \
  --chunk-size 1024 \
  --no-summary

PYTHON_END_TIME=$(date +%s)

PYTHON_DURATION=$((PYTHON_END_TIME - PYTHON_START_TIME))

echo "Python script completed at: $(date)"
echo "Python execution time: ${PYTHON_DURATION}s ($(($PYTHON_DURATION / 60))m $(($PYTHON_DURATION % 60))s)"
echo "Finished CENTERED NN evaluation at: $(date)"
