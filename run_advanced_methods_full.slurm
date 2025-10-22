#!/bin/bash
#SBATCH --job-name=tokenprob_6methods
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/tokenprob_full_%j.out
#SBATCH --error=logs/tokenprob_full_%j.err

# ============================================================================
# Token Probability Analysis - Full 6-Method Comparison
# ============================================================================
# This script runs comprehensive error detection analysis using 6 methods:
#   1. LecPrompt (baseline log-probability)
#   2. Semantic Energy (logits-based uncertainty)
#   3. Conformal Prediction (with calibration - statistical guarantees)
#   4. Attention Anomaly (attention pattern analysis)
#   5. Semantic Context (line-level semantic coherence) - NEW
#   6. Masked Token Replacement (direct prediction matching) - NEW
#
# Usage:
#   sbatch run_advanced_methods_full.slurm
#
# Output:
#   - Results in advanced_methods_comparison_full/
#   - Interactive HTML visualizations
#   - Comprehensive comparison reports
#   - Method ranking and statistics
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    Token Probability Analysis - Full 6-Method Comparison       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Analysis Configuration:"
echo "  • Model: StarCoder2-7B (default)"
echo "  • Methods: 6 (all available)"
echo "    1. LecPrompt (baseline)"
echo "    2. Semantic Energy"
echo "    3. Conformal Prediction (with calibration)"
echo "    4. Attention Anomaly"
echo "    5. Semantic Context"
echo "    6. Masked Token Replacement"
echo "  • Test Examples: 8 (from test set after calibration)"
echo "  • Calibration Examples: 2"
echo "  • Sensitivity Factor: 1.5"
echo "  • Conformal Alpha: 0.1 (90% coverage)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export TRANSFORMERS_CACHE=/scratch/$USER/transformers_cache
# export HF_HOME=/scratch/$USER/hf_home

# Create cache directories
# mkdir -p $TRANSFORMERS_CACHE
# mkdir -p $HF_HOME

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Starting Full 6-Method Analysis"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run the comprehensive analysis
# Using StarCoder2-7B as it's well-suited for code analysis and fits in A100 memory
uv run python test_advanced_methods.py \
    --model starcoder2-7b \
    --sensitivity 1.5 \
    --conformal-alpha 0.1 \
    --output advanced_methods_comparison_full

ANALYSIS_EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    echo ""
    echo "Generated Output:"
    echo "  📁 Results Directory: advanced_methods_comparison_full/"
    echo "  📊 Comparison Results: complete_comparison_results.json"
    echo "  📈 Visualizations: advanced_visualizations/"
    echo ""
    echo "View Results:"
    echo "  1. Open: advanced_methods_comparison_full/advanced_visualizations/index.html"
    echo "  2. Interactive Explorer: interactive_method_explorer.html"
    echo "  3. Calibration Info: Displayed in index.html"
    echo ""
    echo "Key Features:"
    echo "  ✓ Conformal Prediction with formal calibration"
    echo "  ✓ Semantic Context analysis (line-level coherence)"
    echo "  ✓ Masked Token Replacement (direct prediction matching)"
    echo "  ✓ 6×6 agreement matrix between all methods"
    echo "  ✓ Method ranking with weighted scoring"
    echo ""
else
    echo "❌ Analysis failed with exit code: $ANALYSIS_EXIT_CODE"
    echo ""
    echo "Check error log: logs/tokenprob_full_${SLURM_JOB_ID}.err"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
echo "End time: $(date)"
echo "════════════════════════════════════════════════════════════════"

# Print resource usage summary
echo ""
echo "Resource Usage Summary:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State

exit $ANALYSIS_EXIT_CODE
