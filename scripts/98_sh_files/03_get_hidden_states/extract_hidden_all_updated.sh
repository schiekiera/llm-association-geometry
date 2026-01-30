#!/bin/bash
#SBATCH --job-name=extract_hidden_methods
#SBATCH --output=logs/03_get_hidden_states/extract_hidden_methods-%j.out
#SBATCH --error=logs/03_get_hidden_states/extract_hidden_methods-%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

mkdir -p logs/03_get_hidden_states

echo "Starting extraction at: $(date)"

# New script with isolated, template, and averaged methods
python scripts/03_get_hidden_states/get_hidden_states_all_prompts.py

echo "Finished extraction at: $(date)"
