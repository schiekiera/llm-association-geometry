#!/bin/bash
#SBATCH --job-name=gemma_free_associations
#SBATCH --output=logs/02_get_behavioral_associations/02_free_associations/gemma/gemma_free_associations-%j.out
#SBATCH --error=logs/02_get_behavioral_associations/02_free_associations/gemma/gemma_free_associations-%j.err
#SBATCH --partition=gpu_l40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

echo "SLURM job started at: $(date)"
start_time=$(date +%s)

set -eo pipefail


# Create logs directory if it doesn't exist
mkdir -p logs/02_get_behavioral_associations/02_free_associations/gemma/

# Load environment
source miniforge3/etc/profile.d/conda.sh
conda activate conda_env_py311

# Change to the project directory
cd 

# Run the Python script
python -u scripts/02_get_behavioral_associations/02_free_associations/get_free_associations_gemma_fast.py

# Stop timer
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "SLURM job completed at: $(date)"
echo "Elapsed time: $elapsed seconds"