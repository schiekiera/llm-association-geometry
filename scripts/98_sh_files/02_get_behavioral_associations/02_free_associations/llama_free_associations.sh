#!/bin/bash
#SBATCH --job-name=llama_FA
#SBATCH --output=logs/02_get_behavioral_associations/02_free_associations/llama/llama_FA-%j.out
#SBATCH --error=logs/02_get_behavioral_associations/02_free_associations/llama/llama_FA-%j.err
#SBATCH --partition=gpu_l40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

echo "SLURM job started at: $(date)"
start_time=$(date +%s)

set -eo pipefail

mkdir -p logs/02_get_behavioral_associations/02_free_associations/llama/

source miniforge3/etc/profile.d/conda.sh
conda activate conda_env_py311

cd 

python -u scripts/02_get_behavioral_associations/02_free_associations/get_free_associations_llama_fast.py

echo "SLURM job completed at: $(date)"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: $elapsed seconds"