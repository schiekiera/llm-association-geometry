#!/bin/bash
#SBATCH --job-name=proc_benchmarks
#SBATCH --output=logs/01_preprocessing/proc_benchmarks-%j.out
#SBATCH --error=logs/01_preprocessing/proc_benchmarks-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

mkdir -p logs/01_preprocessing

echo "Starting benchmark processing at: $(date)"

# Activate environment
source "$HOME/.bashrc"
conda activate conda_env_py311

# Process FastText (CPU heavy, but quick)
echo "--- Processing FastText ---"
python scripts/01_preprocessing/03_process_benchmarks.py --only fasttext --overwrite

# Process BERT (GPU)
echo "--- Processing BERT ---"
python scripts/01_preprocessing/03_process_benchmarks.py --only bert --overwrite

echo "Finished benchmark processing at: $(date)"
