#!/bin/bash
#SBATCH --job-name=post_fc_all
#SBATCH --output=logs/04_postprocess/01_FC/forced_choice_all-%j.out
#SBATCH --error=logs/04_postprocess/01_FC/forced_choice_all-%j.err
#SBATCH --partition=standard,interactive
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -eo pipefail

# Keep math libs from oversubscribing threads
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Create logs directory if it doesn't exist
mkdir -p logs/04_postprocess/

# Load environment
source miniforge3/etc/profile.d/conda.sh
conda activate conda_env_py311

# Change to the project directory
cd scripts/04_postprocessing/01_forced_choice/

# Start timer
echo "SLURM job started at: $(date)"
start_time=$(date +%s)

echo "Starting automated multi-model postprocessing..."

# This script auto-detects all models in output/02_behavioral_associations/01_forced_choice/02_processed
python -u 01_postprocess_forced_choice_all.py

# Stop timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "SLURM job completed at: $(date)"
echo "Total elapsed time: $elapsed seconds"





