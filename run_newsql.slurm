#!/bin/bash
#SBATCH --job-name=tokenprob
#SBATCH --output=token_%j.out
#SBATCH --error=token_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=lovelace
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32

# Print GPU information
nvidia-smi

# Change to project directory
cd /u/gpinna/phd_projects/tokenprob/tokenprob

# Set CUDA environment variables for stability
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# uv run python test_visualizations.py
#uv run python regenerate_humaneval_visualizations.py

rm -f factorial_buggy_forced.json factorial_correct_forced.json
rm -f quick_test_result.json generation_analysis.json
rm -f test_enhanced_analysis.json

# Rimuovi vecchie visualizzazioni forced generation
rm -rf forced_visualizations/
rm -rf forced_visualizations_backup/
rm -rf forced_visualizations_organized/

# Rimuovi vecchie visualizzazioni HumanEval
rm -rf humaneval_analysis/

# Rimuovi altri file di test obsoleti
rm -f test_visualization_*.html

# Rimuovi risultati di analisi precedenti
rm -rf analysis_results/
rm -rf multi_model_analysis/
rm -rf model_visualizations/

uv run python test_forced_generation.py

uv run python test_forced_visualization.py

uv run python humaneval_runner.py --max-problems 5
