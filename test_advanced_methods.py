#!/usr/bin/env python3
"""
Advanced Methods Testing Script

Main entry point for testing and comparing advanced error detection methods:
1. LecPrompt (baseline log-probability)
2. Semantic Energy (logits-based)
3. Conformal Prediction (statistical guarantees)
4. Attention Anomaly (attention pattern analysis)

Usage:
    # Run all methods on all examples
    python test_advanced_methods.py

    # Test specific example
    python test_advanced_methods.py --example binary_search_missing_bounds

    # Test specific methods
    python test_advanced_methods.py --methods lecprompt semantic_energy

    # Use specific model
    python test_advanced_methods.py --model starcoder2-7b

    # Generate only visualizations
    python test_advanced_methods.py --visualize-only --input advanced_methods_comparison

    # Custom output directory
    python test_advanced_methods.py --output my_results
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from comparison.advanced_comparison_runner import AdvancedMethodsComparisonRunner
from comparison.advanced_visualizer import AdvancedMethodsVisualizer
from comparison.token_level_visualizer import TokenLevelVisualizer


def load_model_and_detector(model_key: str = 'starcoder2-7b', sensitivity_factor: float = 1.5):
    """
    Load model and baseline detector.

    Args:
        model_key: Model identifier
        sensitivity_factor: k parameter

    Returns:
        Tuple of (model, tokenizer, baseline_detector)
    """
    print(f"\nLoading model: {model_key}")

    if model_key == 'starcoder2-7b':
        from detectors.starcoder2_detector import StarCoder2ErrorDetector
        detector = StarCoder2ErrorDetector(sensitivity_factor=sensitivity_factor)
        return detector.model, detector.tokenizer, detector

    elif model_key == 'deepseek-6.7b':
        from detectors.deepseek_detector import DeepSeekErrorDetector
        detector = DeepSeekErrorDetector(sensitivity_factor=sensitivity_factor)
        return detector.model, detector.tokenizer, detector

    elif model_key == 'codet5p-2b':
        from detectors.codet5_detector import CodeT5ErrorDetector
        detector = CodeT5ErrorDetector(sensitivity_factor=sensitivity_factor)
        return detector.model, detector.tokenizer, detector

    elif model_key == 'codebert':
        from codebert_error_detector import CodeBERTErrorDetector
        detector = CodeBERTErrorDetector(sensitivity_factor=sensitivity_factor)
        return detector.model, detector.tokenizer, detector

    elif model_key == 'qwen-7b':
        from logical_error_detector import LogicalErrorDetector
        detector = LogicalErrorDetector(
            model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            sensitivity_factor=sensitivity_factor
        )
        return detector.model, detector.tokenizer, detector

    else:
        raise ValueError(f"Unknown model: {model_key}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Error Detection Methods Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on all examples
  python test_advanced_methods.py

  # Test specific example
  python test_advanced_methods.py --example binary_search_missing_bounds

  # Test specific methods (not yet implemented - all 4 methods are always run)
  python test_advanced_methods.py --methods lecprompt semantic_energy

  # Use specific model
  python test_advanced_methods.py --model starcoder2-7b

  # Generate only visualizations (from existing results)
  python test_advanced_methods.py --visualize-only --input advanced_methods_comparison

  # Custom output directory
  python test_advanced_methods.py --output my_advanced_results

  # Adjust sensitivity factor
  python test_advanced_methods.py --sensitivity 2.0

Available Models:
  - starcoder2-7b: StarCoder2 7B (default)
  - deepseek-6.7b: DeepSeek-Coder 6.7B
  - codet5p-2b: CodeT5+ 2B
  - codebert: CodeBERT
  - qwen-7b: Qwen 2.5 Coder 7B
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="starcoder2-7b",
        choices=["starcoder2-7b", "deepseek-6.7b", "codet5p-2b", "codebert", "qwen-7b"],
        help="Model to use for analysis (default: starcoder2-7b)"
    )

    parser.add_argument(
        "--example",
        type=str,
        default=None,
        help="Test only specific example (default: all)"
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to test (currently all 4 are always run)"
    )

    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="Sensitivity factor k for threshold (default: 1.5)"
    )

    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.1,
        help="Conformal prediction significance level (default: 0.1 = 90%% coverage)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="advanced_methods_comparison",
        help="Output directory (default: advanced_methods_comparison)"
    )

    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only generate visualizations (requires existing results)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory for visualize-only mode"
    )

    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation (both comparative and token-level)"
    )

    parser.add_argument(
        "--skip-token-visualizations",
        action="store_true",
        help="Skip token-level visualization generation (generate only comparative visualizations)"
    )

    args = parser.parse_args()

    print("="*80)
    print("ADVANCED ERROR DETECTION METHODS COMPARISON")
    print("="*80)

    # Visualize-only mode
    if args.visualize_only:
        input_dir = args.input or args.output

        if not os.path.exists(input_dir):
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        results_file = os.path.join(input_dir, "complete_comparison_results.json")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)

        print(f"\nLoading results from: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        visualizer = AdvancedMethodsVisualizer()
        visualizer.create_all_visualizations(results, input_dir)

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nVisualizations saved to: {input_dir}/advanced_visualizations/")

        return

    # Full comparison mode
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Sensitivity factor (k): {args.sensitivity}")
    print(f"  Conformal alpha: {args.conformal_alpha}")
    print(f"  Output directory: {args.output}")
    print(f"  Skip visualizations: {args.skip_visualizations}")

    if args.example:
        print(f"  Testing example: {args.example}")
    else:
        print(f"  Testing: all examples")

    if args.methods:
        print(f"  Methods: {', '.join(args.methods)}")
        print(f"  Note: Currently all 4 methods are always run")

    # Load model and detector
    try:
        model, tokenizer, baseline_detector = load_model_and_detector(
            args.model,
            args.sensitivity
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize comparison runner
    runner = AdvancedMethodsComparisonRunner(
        model=model,
        tokenizer=tokenizer,
        baseline_detector=baseline_detector,
        sensitivity_factor=args.sensitivity,
        conformal_alpha=args.conformal_alpha
    )

    # Run comparison
    try:
        examples_list = [args.example] if args.example else None
        results = runner.run_full_comparison(
            output_dir=args.output,
            examples=examples_list
        )
    except Exception as e:
        print(f"\nError running comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate visualizations
    if not args.skip_visualizations:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Generate comparative visualizations
        try:
            print("\n1. Generating comparative visualizations...")
            visualizer = AdvancedMethodsVisualizer()
            visualizer.create_all_visualizations(results, args.output)
            print("✓ Comparative visualizations complete")
        except Exception as e:
            print(f"\n✗ Error generating comparative visualizations: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Comparison results were saved, but comparative visualizations failed")

        # Generate token-level visualizations
        if not args.skip_token_visualizations:
            try:
                print("\n2. Generating token-level visualizations...")
                token_visualizer = TokenLevelVisualizer()
                token_visualizer.generate_all_token_visualizations(results, args.output)
                print("✓ Token-level visualizations complete")
            except Exception as e:
                print(f"\n✗ Error generating token-level visualizations: {e}")
                import traceback
                traceback.print_exc()
                print("\nNote: Comparison results were saved, but token-level visualizations failed")
        else:
            print("\n2. Skipping token-level visualizations (--skip-token-visualizations)")
    else:
        print("\n✓ Skipping all visualizations (--skip-visualizations)")

    # Print final summary
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output}/")
    print("\nGenerated files:")
    print(f"  - complete_comparison_results.json")
    print(f"  - index.html (main navigation)")

    if not args.skip_visualizations:
        print(f"\n  Comparative visualizations in: advanced_visualizations/")
        print(f"    - index.html (navigation)")
        print(f"    - interactive_method_explorer.html")
        print(f"    - methods_comparison_heatmap.html")
        print(f"    - anomaly_counts_comparison.html")
        print(f"    - method_agreement_matrix.html")
        print(f"    - method_performance_radar.html")
        print(f"    - venn_diagram_overlap.html")
        print(f"    - token_level_multimethod_view_*.html")

        if not args.skip_token_visualizations:
            print(f"\n  Token-level visualizations in: token_visualizations/")
            print(f"    - by_example/index.html (browse by example)")
            print(f"    - by_example/{{example}}/index.html (per-example index)")
            print(f"    - by_example/{{example}}/{{method}}_{{buggy|correct}}.html (80 files)")

    # Print method ranking
    if 'method_ranking' in results:
        print("\n" + "="*80)
        print("METHOD RANKING")
        print("="*80)

        for i, method_info in enumerate(results['method_ranking'], 1):
            method_name = method_info['method']
            total_score = method_info['total_score']
            conf_rate = method_info['confirmation_rate']

            print(f"\n{i}. {method_name.upper()}")
            print(f"   Overall Score: {total_score:.3f}")
            print(f"   Confirmation Rate: {conf_rate:.1%}")
            print(f"   Avg Execution Time: {method_info['avg_execution_time']:.2f}s")

    # Print best method per metric
    if 'aggregate_statistics' in results:
        stats = results['aggregate_statistics']

        print("\n" + "="*80)
        print("BEST METHOD PER METRIC")
        print("="*80)

        # Highest confirmation rate
        best_conf = max(
            stats.items(),
            key=lambda x: x[1].get('confirmation_rate', 0) if isinstance(x[1], dict) else 0
        )
        if isinstance(best_conf[1], dict):
            print(f"\n✓ Highest Confirmation Rate: {best_conf[0].upper()}")
            print(f"  {best_conf[1]['confirmation_rate']:.1%}")

        # Fastest execution
        fastest = min(
            [(k, v) for k, v in stats.items() if isinstance(v, dict) and 'avg_execution_time' in v],
            key=lambda x: x[1]['avg_execution_time']
        )
        print(f"\n⚡ Fastest Execution: {fastest[0].upper()}")
        print(f"  {fastest[1]['avg_execution_time']:.2f}s average")

        # Most anomalies detected
        most_anom = max(
            [(k, v) for k, v in stats.items() if isinstance(v, dict) and 'avg_buggy_anomalies' in v],
            key=lambda x: x[1]['avg_buggy_anomalies']
        )
        print(f"\n🔍 Most Anomalies Detected: {most_anom[0].upper()}")
        print(f"  {most_anom[1]['avg_buggy_anomalies']:.1f} average in buggy code")

    print("\n" + "="*80)

    # Free memory
    del model
    del baseline_detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
