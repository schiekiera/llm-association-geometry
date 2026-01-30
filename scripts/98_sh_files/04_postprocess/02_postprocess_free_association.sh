#!/bin/bash
#SBATCH --job-name=process_fa
#SBATCH --output=logs/04_postprocess/02_FA/process_fa-%j.out
#SBATCH --error=logs/04_postprocess/02_FA/process_fa-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard,interactive

mkdir -p logs/04_postprocess/02_FA

echo "Starting FA postprocessing at: $(date)"
# Activate environment
source "$HOME/.bashrc"
conda activate conda_env_py311

python3 scripts/04_postprocessing/02_free_associations/01_postprocess_fa_all.py --matrix-chunksize 300000

echo "Finished FA postprocessing at: $(date)"

echo "Elapsed time: $elapsed seconds"