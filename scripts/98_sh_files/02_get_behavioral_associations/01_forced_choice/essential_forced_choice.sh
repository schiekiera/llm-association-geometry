#!/bin/bash
#SBATCH --job-name=essential_hier_sim
#SBATCH --output=logs/02_get_behavioral_associations/01_forced_choice/essential/essential_hier_sim-%j.out
#SBATCH --error=logs/02_get_behavioral_associations/01_forced_choice/essential/essential_hier_sim-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -eo pipefail


# Create logs directory if it doesn't exist
mkdir -p logs/02_get_behavioral_associations/01_forced_choice/essential/

# Load environment
source miniforge3/etc/profile.d/conda.sh
conda activate conda_env_py311

# Change to the project directory
cd scripts/02_get_behavioral_associations/01_forced_choice/

# Start timer
echo "SLURM job started at: $(date)"
start_time=$(date +%s)

# Run the Python script
echo "Starting Python script execution..."
SCRIPT="forced_choice_essential.py"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

python -u "${SCRIPT}" "${TASK_ID}"

# Stop timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "SLURM job completed at: $(date)"
echo "Elapsed time: $elapsed seconds"

