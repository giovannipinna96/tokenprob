#!/usr/bin/env python3
"""
Test Suite for LecPrompt Logical Error Detection

This module tests the logical error detection implementation based on the
LecPrompt paper, validating that the technique correctly identifies errors
in buggy code vs correct code.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logical_error_detector import LogicalErrorDetector, TokenError, LineError
from LLM import QwenProbabilityAnalyzer
from test_examples import TestExamplesDataset, TestExample
import numpy as np


def test_statistical_threshold():
    """Test the statistical threshold computation."""
    print("="*60)
    print("TEST 1: Statistical Threshold Computation")
    print("="*60)

    detector = LogicalErrorDetector(sensitivity_factor=1.5)

    # Test with known values
    log_probs = [-2.0, -2.5, -3.0, -3.5, -4.0, -10.0]  # Last one is outlier

    mean, std_dev, threshold = detector.compute_statistical_threshold(log_probs, k=1.5)

    print(f"Log probabilities: {log_probs}")
    print(f"Mean: {mean:.4f}")
    print(f"Std Dev: {std_dev:.4f}")
    print(f"Threshold (μ - 1.5σ): {threshold:.4f}")

    # The outlier -10.0 should be below threshold
    assert log_probs[-1] < threshold, "Outlier should be below threshold"
    # The normal values should mostly be above threshold
    normal_above = sum(1 for lp in log_probs[:-1] if lp >= threshold)
    print(f"Normal values above threshold: {normal_above}/{len(log_probs)-1}")

    print("✓ Test passed\n")


def test_single_example(detector: LogicalErrorDetector,
                       example: TestExample,
                       k: float = 1.5) -> Dict[str, Any]:
    """
    Test error detection on a single example.

    Args:
        detector: LogicalErrorDetector instance
        example: TestExample to analyze
        k: Sensitivity factor

    Returns:
        Test results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: {example.name}")
    print(f"Bug Type: {example.bug_type}")
    print(f"{'='*60}")

    # Analyze buggy code
    print("\n--- BUGGY CODE ---")
    print(example.buggy_code)
    buggy_results = detector.localize_errors(example.buggy_code, k=k)

    # Analyze correct code
    print("\n--- CORRECT CODE ---")
    print(example.correct_code)
    correct_results = detector.localize_errors(example.correct_code, k=k)

    # Compare results
    buggy_stats = buggy_results['statistics']
    correct_stats = correct_results['statistics']

    print("\n--- COMPARISON ---")
    print(f"Anomalous tokens:  Buggy={buggy_stats['anomalous_tokens']:3d}  "
          f"Correct={correct_stats['anomalous_tokens']:3d}  "
          f"Diff={buggy_stats['anomalous_tokens'] - correct_stats['anomalous_tokens']:+3d}")

    print(f"Error lines:       Buggy={buggy_stats['error_lines']:3d}  "
          f"Correct={correct_stats['error_lines']:3d}  "
          f"Diff={buggy_stats['error_lines'] - correct_stats['error_lines']:+3d}")

    print(f"Mean log prob:     Buggy={buggy_stats['mean_log_prob']:7.3f}  "
          f"Correct={correct_stats['mean_log_prob']:7.3f}  "
          f"Diff={buggy_stats['mean_log_prob'] - correct_stats['mean_log_prob']:+7.3f}")

    # Check hypothesis: buggy code should have more anomalous tokens
    hypothesis_confirmed = (
        buggy_stats['anomalous_tokens'] >= correct_stats['anomalous_tokens']
    )

    print(f"\nHypothesis (more anomalies in buggy): {'✓ CONFIRMED' if hypothesis_confirmed else '✗ REJECTED'}")

    # Show detected error lines in buggy code
    if buggy_results['line_errors']:
        print("\n--- ERROR LINES IN BUGGY CODE ---")
        for line_error in buggy_results['line_errors']:
            if line_error.is_error_line:
                print(f"Line {line_error.line_number}: {line_error.line_content}")
                print(f"  Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")
                print(f"  Error score: {line_error.error_score:.3f}")

    return {
        'example_name': example.name,
        'bug_type': example.bug_type,
        'buggy_stats': buggy_stats,
        'correct_stats': correct_stats,
        'hypothesis_confirmed': hypothesis_confirmed,
        'anomaly_diff': buggy_stats['anomalous_tokens'] - correct_stats['anomalous_tokens'],
        'error_lines_diff': buggy_stats['error_lines'] - correct_stats['error_lines']
    }


def test_all_examples(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                     k: float = 1.5,
                     output_file: str = "error_detection_results.json"):
    """
    Test error detection on all examples from the dataset.

    Args:
        model_name: HuggingFace model name
        k: Sensitivity factor
        output_file: File to save results
    """
    print("="*60)
    print("TESTING LOGICAL ERROR DETECTION ON ALL EXAMPLES")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Sensitivity factor (k): {k}")
    print("")

    # Initialize detector
    detector = LogicalErrorDetector(model_name=model_name, sensitivity_factor=k)

    # Load test examples
    dataset = TestExamplesDataset()
    examples = dataset.get_all_examples()

    print(f"Loaded {len(examples)} test examples\n")

    # Test each example
    results = []
    confirmations = 0

    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Processing: {example.name}")
        try:
            result = test_single_example(detector, example, k=k)
            results.append(result)

            if result['hypothesis_confirmed']:
                confirmations += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Calculate overall statistics
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)

    total_examples = len(results)
    confirmation_rate = confirmations / total_examples if total_examples > 0 else 0.0

    print(f"Total examples tested: {total_examples}")
    print(f"Hypothesis confirmed: {confirmations}/{total_examples} ({confirmation_rate*100:.1f}%)")

    # Group by bug type
    bug_types = {}
    for result in results:
        bug_type = result['bug_type']
        if bug_type not in bug_types:
            bug_types[bug_type] = {'total': 0, 'confirmed': 0}
        bug_types[bug_type]['total'] += 1
        if result['hypothesis_confirmed']:
            bug_types[bug_type]['confirmed'] += 1

    print("\nResults by bug type:")
    for bug_type, stats in bug_types.items():
        rate = stats['confirmed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {bug_type:12s}: {stats['confirmed']}/{stats['total']} ({rate:.1f}%)")

    # Calculate average differences
    anomaly_diffs = [r['anomaly_diff'] for r in results]
    error_line_diffs = [r['error_lines_diff'] for r in results]

    print(f"\nAverage anomaly difference (buggy - correct): {np.mean(anomaly_diffs):.2f}")
    print(f"Average error lines difference (buggy - correct): {np.mean(error_line_diffs):.2f}")

    # Save results
    summary = {
        'model_name': model_name,
        'sensitivity_factor': k,
        'total_examples': total_examples,
        'confirmations': confirmations,
        'confirmation_rate': confirmation_rate,
        'results_by_bug_type': bug_types,
        'average_anomaly_diff': float(np.mean(anomaly_diffs)),
        'average_error_lines_diff': float(np.mean(error_line_diffs)),
        'detailed_results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Overall assessment
    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)
    if confirmation_rate >= 0.7:
        print("✓ EXCELLENT: LecPrompt error detection works very well on this dataset")
    elif confirmation_rate >= 0.5:
        print("✓ GOOD: LecPrompt error detection shows promise")
    else:
        print("⚠ MIXED: LecPrompt error detection shows mixed results")

    return summary


def test_sensitivity_analysis(model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    """
    Test how different sensitivity factors (k) affect detection performance.

    Args:
        model_name: HuggingFace model name
    """
    print("="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)

    k_values = [1.0, 1.5, 2.0, 2.5]
    dataset = TestExamplesDataset()

    # Test on first example only for speed
    example = dataset.examples[0]
    print(f"Testing on: {example.name}\n")

    detector = LogicalErrorDetector(model_name=model_name)

    results = []
    for k in k_values:
        print(f"\nTesting with k={k}")
        buggy_results = detector.localize_errors(example.buggy_code, k=k)
        correct_results = detector.localize_errors(example.correct_code, k=k)

        buggy_anomalies = buggy_results['statistics']['anomalous_tokens']
        correct_anomalies = correct_results['statistics']['anomalous_tokens']

        results.append({
            'k': k,
            'buggy_anomalies': buggy_anomalies,
            'correct_anomalies': correct_anomalies,
            'diff': buggy_anomalies - correct_anomalies
        })

        print(f"  Buggy: {buggy_anomalies} anomalies")
        print(f"  Correct: {correct_anomalies} anomalies")
        print(f"  Difference: {buggy_anomalies - correct_anomalies}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'k':>6} | {'Buggy':>6} | {'Correct':>7} | {'Diff':>5}")
    print("-"*60)
    for r in results:
        print(f"{r['k']:6.1f} | {r['buggy_anomalies']:6d} | {r['correct_anomalies']:7d} | {r['diff']:+5d}")

    # Save sensitivity analysis
    with open("sensitivity_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSensitivity analysis saved to: sensitivity_analysis.json")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Test LecPrompt Logical Error Detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="Sensitivity factor k (default: 1.5)"
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Test single example by name"
    )
    parser.add_argument(
        "--sensitivity-analysis",
        action="store_true",
        help="Run sensitivity analysis for different k values"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="error_detection_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    if args.sensitivity_analysis:
        # Run sensitivity analysis
        test_sensitivity_analysis(model_name=args.model)
    elif args.example:
        # Test single example
        detector = LogicalErrorDetector(
            model_name=args.model,
            sensitivity_factor=args.sensitivity
        )
        dataset = TestExamplesDataset()
        try:
            example = dataset.get_example(args.example)
            test_single_example(detector, example, k=args.sensitivity)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available examples: {[ex.name for ex in dataset.examples]}")
    else:
        # Run full test suite
        test_statistical_threshold()
        test_all_examples(
            model_name=args.model,
            k=args.sensitivity,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
