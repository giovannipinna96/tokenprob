#!/usr/bin/env python3
"""
Multi-Model Comparison Study Runner

Complete script for running comparative error detection study across multiple
state-of-the-art code models.

Usage:
    # Run full comparison on all models
    python test_multi_model_comparison.py

    # Run on specific models only
    python test_multi_model_comparison.py --models starcoder2-7b codet5p-2b

    # Use different sensitivity factor
    python test_multi_model_comparison.py --sensitivity 2.0

    # Custom output directory
    python test_multi_model_comparison.py --output my_results
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from comparison.benchmark_runner import MultiModelBenchmark
from comparison.statistical_analyzer import StatisticalAnalyzer
from comparison.comparison_visualizer import ComparisonVisualizer
from comparison.detailed_comparison_visualizer import DetailedComparisonVisualizer
from comparison.report_generator import ReportGenerator


def main():
    """Run multi-model comparison study."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Error Detection Comparison Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comparison on all models
  python test_multi_model_comparison.py

  # Run on specific models
  python test_multi_model_comparison.py --models starcoder2-7b codet5p-2b deepseek-6.7b

  # Adjust sensitivity factor
  python test_multi_model_comparison.py --sensitivity 2.0

  # Custom output directory
  python test_multi_model_comparison.py --output custom_results

Available Models:
  - starcoder2-7b: BigCode StarCoder2 7B (Causal LM)
  - codet5p-2b: Salesforce CodeT5+ 2B (Encoder-Decoder)
  - deepseek-6.7b: DeepSeek-Coder 6.7B (Causal LM)
  - codebert: Microsoft CodeBERT (MLM)
  - qwen-7b: Qwen 2.5 Coder 7B (Causal LM)
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to test (default: all enabled models)"
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="Sensitivity factor k for threshold τ = μ - k×σ (default: 1.5)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="comparison_study",
        help="Output directory for results (default: comparison_study)"
    )

    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip creating visualizations (faster)"
    )

    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip markdown report generation"
    )

    parser.add_argument(
        "--detailed-visualizations",
        action="store_true",
        help="Generate detailed token-level and per-example visualizations"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-MODEL ERROR DETECTION COMPARISON STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Models: {args.models if args.models else 'all enabled'}")
    print(f"  Sensitivity factor (k): {args.sensitivity}")
    print(f"  Output directory: {args.output}")
    print(f"  Skip visualizations: {args.skip_visualizations}")
    print(f"  Skip report: {args.skip_report}")
    print(f"  Detailed visualizations: {args.detailed_visualizations}")
    print()

    # Initialize benchmark runner
    benchmark = MultiModelBenchmark(sensitivity_factor=args.sensitivity)

    # Run benchmark
    print("\n" + "=" * 80)
    print("PHASE 1: RUNNING BENCHMARK")
    print("=" * 80)
    results = benchmark.run_full_benchmark(
        models=args.models,
        output_dir=args.output
    )

    # Statistical analysis
    print("\n" + "=" * 80)
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("=" * 80)
    analyzer = StatisticalAnalyzer()
    stats = analyzer.analyze(results['model_results'])

    # Save statistical analysis
    import json
    stats_file = os.path.join(args.output, "statistical_analysis.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistical analysis saved to: {stats_file}")

    # Create visualizations
    if not args.skip_visualizations:
        print("\n" + "=" * 80)
        print("PHASE 3: CREATING VISUALIZATIONS")
        print("=" * 80)
        visualizer = ComparisonVisualizer()
        visualizer.create_all_visualizations(results, stats, args.output)

    # Generate markdown report
    if not args.skip_report:
        print("\n" + "=" * 80)
        print("PHASE 4: GENERATING REPORT")
        print("=" * 80)
        report_gen = ReportGenerator()
        report_gen.generate_markdown_report(results, stats, args.output)

    # Generate detailed visualizations
    if args.detailed_visualizations:
        print("\n" + "=" * 80)
        print("PHASE 5: GENERATING DETAILED VISUALIZATIONS")
        print("=" * 80)
        detailed_visualizer = DetailedComparisonVisualizer()
        detailed_visualizer.create_all_detailed_visualizations(results, args.output)

    # Final summary
    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output}/")
    print("\nGenerated files:")
    print(f"  - complete_benchmark_results.json (full results)")
    print(f"  - statistical_analysis.json (statistical tests)")
    if not args.skip_report:
        print(f"  - comparison_report.md (comprehensive report)")
    if not args.skip_visualizations:
        print(f"  - confirmation_rates.html")
        print(f"  - bug_type_heatmap.html")
        print(f"  - radar_comparison.html")
        print(f"  - significance_matrix.html")
        print(f"  - example_breakdown.html")
        print(f"  - anomaly_difference_heatmap.html")
    if args.detailed_visualizations:
        print(f"\n  Detailed visualizations:")
        print(f"  - detailed_visualizations/index.html (navigation)")
        print(f"  - detailed_visualizations/dashboard.html (interactive)")
        print(f"  - detailed_visualizations/performance_matrix.html")
        print(f"  - detailed_visualizations/anomaly_counts.html")
        print(f"  - detailed_visualizations/line_detection.html")
        print(f"  - detailed_visualizations/probability_distributions.html")
        print(f"  - detailed_visualizations/token_comparison_*.html")

    # Print top model
    if stats.get('ranking'):
        print(f"\n🏆 Best performing model: {stats['ranking'][0]}")
        best_rate = stats['confirmation_rates'].get(stats['ranking'][0], 0)
        print(f"   Confirmation rate: {best_rate:.1%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
