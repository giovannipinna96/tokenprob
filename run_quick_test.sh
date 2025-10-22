#!/bin/bash
# ============================================================================
# Quick Test Script - Token Probability Analysis
# ============================================================================
# Runs a quick test with CodeBERT (smallest model) on a single example
# to verify all 6 methods are working correctly.
#
# Usage:
#   bash run_quick_test.sh
#
# This is useful for:
#   - Testing the installation
#   - Debugging issues
#   - Quick verification before submitting SLURM job
# ============================================================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Token Probability Analysis - Quick Test                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MODEL="codebert"
EXAMPLE="binary_search_missing_bounds"
OUTPUT="quick_test_results"

echo "Configuration:"
echo "  • Model: $MODEL (lightweight, ~500MB)"
echo "  • Test Example: $EXAMPLE"
echo "  • Output: $OUTPUT/"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' command not found"
    echo "Please install uv or use: python test_advanced_methods.py"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Running Quick Test..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run the test
uv run python test_advanced_methods.py \
    --model "$MODEL" \
    --example "$EXAMPLE" \
    --sensitivity 1.5 \
    --conformal-alpha 0.1 \
    --output "$OUTPUT"

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Quick test completed successfully!"
    echo ""
    echo "Methods tested:"
    echo "  1. ✓ LecPrompt (baseline)"
    echo "  2. ✓ Semantic Energy"
    echo "  3. ✓ Conformal Prediction (calibrated)"
    echo "  4. ✓ Attention Anomaly"
    echo "  5. ✓ Semantic Context (if available)"
    echo "  6. ✓ Masked Token Replacement (if available)"
    echo ""
    echo "📁 Results saved to: $OUTPUT/"
    echo "🌐 View visualization: $OUTPUT/advanced_visualizations/index.html"
    echo ""
    echo "Next steps:"
    echo "  • Open the HTML file in your browser"
    echo "  • If everything looks good, submit the full SLURM job:"
    echo "    sbatch run_advanced_methods_full.slurm"
else
    echo "❌ Quick test failed with exit code: $EXIT_CODE"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check error messages above"
    echo "  • Ensure dependencies are installed:"
    echo "    uv sync"
    echo "  • For Semantic Context, install:"
    echo "    pip install sentence-transformers"
fi

echo "════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
