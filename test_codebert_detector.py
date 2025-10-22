#!/usr/bin/env python3
"""
Test Suite for CodeBERT-based Logical Error Detection

This module tests the CodeBERT implementation which uses masked language
modeling (MLM) to compute token probabilities, following the original
LecPrompt paper more closely.
"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from codebert_error_detector import CodeBERTErrorDetector
from test_examples import TestExamplesDataset
import numpy as np


def compare_codebert_vs_causal():
    """
    Compare CodeBERT (MLM) approach vs Causal LM (Qwen) approach.
    """
    print("="*60)
    print("COMPARISON: CodeBERT (MLM) vs Causal LM (Qwen)")
    print("="*60)

    from logical_error_detector import LogicalErrorDetector

    # Load first example
    dataset = TestExamplesDataset()
    example = dataset.examples[0]

    print(f"\nTesting on: {example.name}")
    print(f"Buggy code:\n{example.buggy_code}\n")

    # CodeBERT approach
    print("--- CodeBERT (MLM) Analysis ---")
    codebert_detector = CodeBERTErrorDetector(sensitivity_factor=1.5)
    codebert_results = codebert_detector.localize_errors(example.buggy_code)

    # Causal LM approach
    print("\n--- Causal LM (Qwen) Analysis ---")
    causal_detector = LogicalErrorDetector(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        sensitivity_factor=1.5
    )
    causal_results = causal_detector.localize_errors(example.buggy_code)

    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"{'Metric':<30} {'CodeBERT':<15} {'Causal LM':<15}")
    print("-"*60)
    print(f"{'Anomalous tokens':<30} {codebert_results['statistics']['anomalous_tokens']:<15} {causal_results['statistics']['anomalous_tokens']:<15}")
    print(f"{'Error lines':<30} {codebert_results['statistics']['error_lines']:<15} {causal_results['statistics']['error_lines']:<15}")
    print(f"{'Mean log prob':<30} {codebert_results['statistics']['mean_log_prob']:<15.3f} {causal_results['statistics']['mean_log_prob']:<15.3f}")
    print(f"{'Std dev':<30} {codebert_results['statistics']['std_dev']:<15.3f} {causal_results['statistics']['std_dev']:<15.3f}")

    print("\nKey Difference:")
    print("  • CodeBERT: Masks each token and predicts (BERT-style MLM)")
    print("  • Causal LM: Autoregressive prediction (GPT-style)")


def test_codebert_on_example(example_name: str, k: float = 1.5):
    """
    Test CodeBERT detector on a specific example.
    """
    print("="*60)
    print(f"CodeBERT Error Detection: {example_name}")
    print("="*60)

    # Load example
    dataset = TestExamplesDataset()
    try:
        example = dataset.get_example(example_name)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable examples:")
        for ex in dataset.examples:
            print(f"  - {ex.name}")
        return

    print(f"Description: {example.description}")
    print(f"Bug type: {example.bug_type}\n")

    # Initialize detector
    detector = CodeBERTErrorDetector(sensitivity_factor=k)

    # Analyze buggy code
    print("--- BUGGY CODE ---")
    print(example.buggy_code)
    buggy_results = detector.localize_errors(example.buggy_code, k=k)

    print("\nDetected error lines:")
    for line_error in buggy_results['line_errors']:
        if line_error.is_error_line:
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    • Error score: {line_error.error_score:.3f}")
            print(f"    • Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")
            # Show which tokens are anomalous
            for token in line_error.anomalous_tokens:
                print(f"      - '{token.token}' (log_prob={token.log_probability:.3f}, deviation={token.deviation_score:.2f}σ)")

    # Analyze correct code
    print("\n--- CORRECT CODE ---")
    print(example.correct_code)
    correct_results = detector.localize_errors(example.correct_code, k=k)

    print("\nDetected error lines:")
    found_errors = False
    for line_error in correct_results['line_errors']:
        if line_error.is_error_line:
            found_errors = True
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    • Error score: {line_error.error_score:.3f}")
            print(f"    • Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")

    if not found_errors:
        print("  No error lines detected ✓")

    # Comparison
    print("\n--- COMPARISON ---")
    buggy_stats = buggy_results['statistics']
    correct_stats = correct_results['statistics']

    print(f"Anomalous tokens:  Buggy={buggy_stats['anomalous_tokens']:3d}  "
          f"Correct={correct_stats['anomalous_tokens']:3d}  "
          f"Diff={buggy_stats['anomalous_tokens'] - correct_stats['anomalous_tokens']:+3d}")

    print(f"Error lines:       Buggy={buggy_stats['error_lines']:3d}  "
          f"Correct={correct_stats['error_lines']:3d}  "
          f"Diff={buggy_stats['error_lines'] - correct_stats['error_lines']:+3d}")

    hypothesis_confirmed = (
        buggy_stats['anomalous_tokens'] > correct_stats['anomalous_tokens']
    )
    print(f"\nHypothesis (buggy has more anomalies): {'✓ CONFIRMED' if hypothesis_confirmed else '✗ REJECTED'}")


def test_all_examples_with_codebert(k: float = 1.5, output_file: str = "codebert_results.json"):
    """
    Test CodeBERT on all examples from the dataset.
    """
    print("="*60)
    print("CodeBERT Error Detection - Full Test Suite")
    print("="*60)
    print(f"Sensitivity factor (k): {k}\n")

    # Initialize detector
    detector = CodeBERTErrorDetector(sensitivity_factor=k)

    # Load test examples
    dataset = TestExamplesDataset()
    examples = dataset.get_all_examples()

    print(f"Loaded {len(examples)} test examples\n")

    # Test each example
    results = []
    confirmations = 0

    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] {example.name}")

        try:
            # Analyze buggy code
            buggy_results = detector.localize_errors(example.buggy_code, k=k)

            # Analyze correct code
            correct_results = detector.localize_errors(example.correct_code, k=k)

            # Compare
            buggy_stats = buggy_results['statistics']
            correct_stats = correct_results['statistics']

            hypothesis_confirmed = (
                buggy_stats['anomalous_tokens'] >= correct_stats['anomalous_tokens']
            )

            if hypothesis_confirmed:
                confirmations += 1

            result = {
                'example_name': example.name,
                'bug_type': example.bug_type,
                'buggy_anomalies': buggy_stats['anomalous_tokens'],
                'correct_anomalies': correct_stats['anomalous_tokens'],
                'buggy_error_lines': buggy_stats['error_lines'],
                'correct_error_lines': correct_stats['error_lines'],
                'hypothesis_confirmed': hypothesis_confirmed
            }

            results.append(result)

            print(f"  Buggy: {buggy_stats['anomalous_tokens']} anomalies, {buggy_stats['error_lines']} error lines")
            print(f"  Correct: {correct_stats['anomalous_tokens']} anomalies, {correct_stats['error_lines']} error lines")
            print(f"  Result: {'✓' if hypothesis_confirmed else '✗'}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
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

    # Save results
    summary = {
        'model': 'CodeBERT (microsoft/codebert-base)',
        'approach': 'Masked Language Modeling (MLM)',
        'sensitivity_factor': k,
        'total_examples': total_examples,
        'confirmations': confirmations,
        'confirmation_rate': confirmation_rate,
        'results_by_bug_type': bug_types,
        'detailed_results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Assessment
    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)
    if confirmation_rate >= 0.7:
        print("✓ EXCELLENT: CodeBERT error detection works very well")
    elif confirmation_rate >= 0.5:
        print("✓ GOOD: CodeBERT error detection shows promise")
    else:
        print("⚠ MIXED: CodeBERT error detection shows mixed results")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Test CodeBERT-based Logical Error Detection"
    )
    parser.add_argument(
        "--example",
        type=str,
        help="Test on a specific example by name"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="Sensitivity factor k (default: 1.5)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare CodeBERT vs Causal LM approach"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test on all examples"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="codebert_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    if args.compare:
        # Compare approaches
        compare_codebert_vs_causal()
    elif args.example:
        # Test single example
        test_codebert_on_example(args.example, k=args.sensitivity)
    elif args.all:
        # Test all examples
        test_all_examples_with_codebert(k=args.sensitivity, output_file=args.output)
    else:
        # Show help
        print("CodeBERT-based Error Detection Test Suite")
        print("\nAvailable options:")
        print("  --example <name>   Test on a specific example")
        print("  --all              Test on all examples")
        print("  --compare          Compare CodeBERT vs Causal LM")
        print("  --sensitivity <k>  Set sensitivity factor (default: 1.5)")
        print("\nExamples:")
        print("  python test_codebert_detector.py --example factorial_recursion_base_case")
        print("  python test_codebert_detector.py --all")
        print("  python test_codebert_detector.py --compare")


if __name__ == "__main__":
    main()
