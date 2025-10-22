#!/usr/bin/env python3
"""
Generate Detailed Visualizations

Standalone script to generate detailed visualizations from benchmark results.
Can be run after completing the benchmark to create comprehensive visual analysis.

Usage:
    # Generate from default location
    python generate_detailed_visualizations.py

    # Generate from custom results directory
    python generate_detailed_visualizations.py --input comparison_study

    # Specify custom output directory
    python generate_detailed_visualizations.py --input comparison_study --output my_visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from comparison.detailed_comparison_visualizer import DetailedComparisonVisualizer


def load_results(results_dir: str) -> dict:
    """
    Load benchmark results from directory.

    Args:
        results_dir: Directory containing results

    Returns:
        Dictionary with results and stats
    """
    results_file = os.path.join(results_dir, "complete_benchmark_results.json")
    stats_file = os.path.join(results_dir, "statistical_analysis.json")

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    print(f"Loading results from: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    stats = None
    if os.path.exists(stats_file):
        print(f"Loading statistics from: {stats_file}")
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    else:
        print(f"Warning: Statistics file not found: {stats_file}")
        stats = {}

    return results, stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate detailed visualizations from multi-model comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from default comparison_study directory
  python generate_detailed_visualizations.py

  # Generate from custom input directory
  python generate_detailed_visualizations.py --input my_results

  # Specify custom output directory
  python generate_detailed_visualizations.py --input comparison_study --output custom_viz

Notes:
  - Input directory must contain complete_benchmark_results.json
  - Statistical_analysis.json is optional but recommended
  - Output will be created in detailed_visualizations/ subdirectory
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        default="comparison_study",
        help="Input directory containing benchmark results (default: comparison_study)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <input>/detailed_visualizations)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Load results
    try:
        results, stats = load_results(args.input)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = args.input

    print("\n" + "="*80)
    print("DETAILED VISUALIZATION GENERATOR")
    print("="*80)
    print(f"\nInput directory: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Models in results: {results['metadata']['num_models']}")
    print(f"Test examples: {results['metadata']['num_examples']}")

    # Create visualizer
    visualizer = DetailedComparisonVisualizer()

    # Generate all visualizations
    try:
        visualizer.create_all_detailed_visualizations(results, output_dir)
    except Exception as e:
        print(f"\nError generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print success message
    vis_dir = os.path.join(output_dir, "detailed_visualizations")
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*80)
    print(f"\nVisualizations saved to: {vis_dir}/")
    print("\nGenerated files:")
    print("  - index.html (navigation page)")
    print("  - dashboard.html (interactive dashboard)")
    print("  - performance_matrix.html")
    print("  - anomaly_counts.html")
    print("  - line_detection.html")
    print("  - probability_distributions.html")
    print("  - token_comparison_<example>.html (per example)")

    # Try to get absolute path for opening
    try:
        abs_path = os.path.abspath(os.path.join(vis_dir, "index.html"))
        print(f"\n📂 Open in browser: file://{abs_path}")
    except:
        pass

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
