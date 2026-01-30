#!/bin/bash
#SBATCH --job-name=pred_cosine_std
#SBATCH --output=logs/06_predict/pred_cosine_std-%A_%a.out
#SBATCH --error=logs/06_predict/pred_cosine_std-%A_%a.err
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-31%10

set -euo pipefail

mkdir -p logs/06_predict

echo "Starting STANDARD held-out-words cosine prediction at: $(date)"
echo "Job: ${SLURM_JOB_ID:-NA} ArrayTask: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $(hostname) | CPUs: $(nproc) | SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"

START_TIME=$(date +%s)

# Hyperparameters
TRAIN_PAIR_SAMPLE_SIZE=100000
SEEDS="0"
BATCH_SIZE=50000

PROMPTS=(template averaged forced_choice free_association)

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

source "$HOME/.bashrc"
conda activate conda_env_py311

PROJECT_ROOT="projects/icml_project"
FC_ROOT="${PROJECT_ROOT}/data/02_behavioral_associations/01_forced_choice/03_postprocessed"

mapfile -t MODELS < <(ls -1 "${FC_ROOT}" 2>/dev/null | sort)

N_MODELS=${#MODELS[@]}
N_PROMPTS=${#PROMPTS[@]}
N_TASKS=$((N_MODELS * N_PROMPTS))

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -ge "${N_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} out of range for N_TASKS=${N_TASKS}. Exiting."
  exit 0
fi

MODEL_IDX=$((TASK_ID / N_PROMPTS))
PROMPT_IDX=$((TASK_ID % N_PROMPTS))

MODEL="${MODELS[MODEL_IDX]}"
PROMPT="${PROMPTS[PROMPT_IDX]}"

echo "Selected: model='${MODEL}' prompt='${PROMPT}'"
echo "Hyperparameters: TRAIN_PAIR_SAMPLE_SIZE=${TRAIN_PAIR_SAMPLE_SIZE} | SEEDS=${SEEDS} | BATCH_SIZE=${BATCH_SIZE}"
echo "Python script starting at: $(date)"
PYTHON_START_TIME=$(date +%s)

python3 scripts/06_predict/02_held-out_words_ablation/01_predict_held-out_words_ablation.py \
  --model "${MODEL}" \
  --prompt "${PROMPT}" \
  --pair-sample-size "${TRAIN_PAIR_SAMPLE_SIZE}" \
  --seeds "${SEEDS}" \
  --batch-size "${BATCH_SIZE}"

PYTHON_END_TIME=$(date +%s)
END_TIME=$(date +%s)

PYTHON_DURATION=$((PYTHON_END_TIME - PYTHON_START_TIME))
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "Python script completed at: $(date)"
echo "Python execution time: ${PYTHON_DURATION}s ($(($PYTHON_DURATION / 60))m $(($PYTHON_DURATION % 60))s)"
echo "Total job time: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo "Finished STANDARD held-out-words cosine prediction at: $(date)"
