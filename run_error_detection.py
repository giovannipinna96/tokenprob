#!/usr/bin/env python3
"""
Run Error Detection Analysis Script

This script provides a convenient interface to run logical error detection
using the LecPrompt technique on code snippets or test examples.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logical_error_detector import LogicalErrorDetector
from codebert_error_detector import CodeBERTErrorDetector
from LLM import QwenProbabilityAnalyzer
from test_examples import TestExamplesDataset
from visualizer import TokenVisualizer, TokenVisualizationMode


def run_error_detection_on_example(example_name: str,
                                  model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                                  k: float = 1.5,
                                  output_dir: str = "error_detection_results",
                                  use_codebert: bool = False):
    """
    Run error detection analysis on a specific test example.

    Args:
        example_name: Name of the example from test_examples.py
        model_name: HuggingFace model name
        k: Sensitivity factor
        output_dir: Directory to save results
    """
    print("="*60)
    print("LOGICAL ERROR DETECTION ANALYSIS")
    print("="*60)
    print(f"Example: {example_name}")
    print(f"Model: {'CodeBERT (MLM)' if use_codebert else model_name}")
    print(f"Approach: {'Masked Language Modeling' if use_codebert else 'Causal LM'}")
    print(f"Sensitivity (k): {k}")
    print("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load example
    dataset = TestExamplesDataset()
    try:
        example = dataset.get_example(example_name)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable examples:")
        for ex in dataset.examples:
            print(f"  - {ex.name} ({ex.bug_type}): {ex.description}")
        return

    print(f"Description: {example.description}")
    print(f"Bug type: {example.bug_type}\n")

    # Initialize detector based on model type
    if use_codebert:
        detector = CodeBERTErrorDetector(sensitivity_factor=k)
    else:
        detector = LogicalErrorDetector(model_name=model_name, sensitivity_factor=k)

    # Analyze buggy code
    print("="*60)
    print("ANALYZING BUGGY CODE")
    print("="*60)
    print(example.buggy_code)
    print("")

    buggy_results = detector.localize_errors(example.buggy_code, k=k)

    print("\nDetected error lines:")
    for line_error in buggy_results['line_errors']:
        if line_error.is_error_line:
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    • Error score: {line_error.error_score:.3f}")
            print(f"    • Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")
            print(f"    • Avg log prob: {line_error.avg_log_prob:.3f}")

    # Analyze correct code
    print("\n" + "="*60)
    print("ANALYZING CORRECT CODE")
    print("="*60)
    print(example.correct_code)
    print("")

    correct_results = detector.localize_errors(example.correct_code, k=k)

    print("\nDetected error lines:")
    error_lines_found = False
    for line_error in correct_results['line_errors']:
        if line_error.is_error_line:
            error_lines_found = True
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    • Error score: {line_error.error_score:.3f}")
            print(f"    • Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")

    if not error_lines_found:
        print("  No error lines detected ✓")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    buggy_stats = buggy_results['statistics']
    correct_stats = correct_results['statistics']

    print(f"Metric                  Buggy    Correct   Difference")
    print("-"*60)
    print(f"Anomalous tokens        {buggy_stats['anomalous_tokens']:5d}    {correct_stats['anomalous_tokens']:7d}   {buggy_stats['anomalous_tokens'] - correct_stats['anomalous_tokens']:+10d}")
    print(f"Error lines             {buggy_stats['error_lines']:5d}    {correct_stats['error_lines']:7d}   {buggy_stats['error_lines'] - correct_stats['error_lines']:+10d}")
    print(f"Mean log prob           {buggy_stats['mean_log_prob']:5.2f}    {correct_stats['mean_log_prob']:7.2f}   {buggy_stats['mean_log_prob'] - correct_stats['mean_log_prob']:+10.2f}")

    hypothesis_confirmed = (
        buggy_stats['anomalous_tokens'] > correct_stats['anomalous_tokens']
    )

    print(f"\nHypothesis (buggy has more anomalies): {'✓ CONFIRMED' if hypothesis_confirmed else '✗ REJECTED'}")

    # Save results
    output_file = os.path.join(output_dir, f"{example_name}_error_detection.json")
    result = {
        'timestamp': datetime.now().isoformat(),
        'example': {
            'name': example.name,
            'description': example.description,
            'bug_type': example.bug_type,
            'buggy_code': example.buggy_code,
            'correct_code': example.correct_code
        },
        'model_name': model_name,
        'sensitivity_factor': k,
        'buggy_analysis': {
            'statistics': buggy_stats,
            'error_lines': [
                {
                    'line_number': le.line_number,
                    'line_content': le.line_content,
                    'error_score': le.error_score,
                    'num_anomalous_tokens': le.num_anomalous_tokens,
                    'num_tokens': le.num_tokens
                }
                for le in buggy_results['line_errors'] if le.is_error_line
            ]
        },
        'correct_analysis': {
            'statistics': correct_stats,
            'error_lines': [
                {
                    'line_number': le.line_number,
                    'line_content': le.line_content,
                    'error_score': le.error_score,
                    'num_anomalous_tokens': le.num_anomalous_tokens,
                    'num_tokens': le.num_tokens
                }
                for le in correct_results['line_errors'] if le.is_error_line
            ]
        },
        'comparison': {
            'hypothesis_confirmed': hypothesis_confirmed,
            'anomaly_diff': buggy_stats['anomalous_tokens'] - correct_stats['anomalous_tokens'],
            'error_lines_diff': buggy_stats['error_lines'] - correct_stats['error_lines'],
            'mean_log_prob_diff': buggy_stats['mean_log_prob'] - correct_stats['mean_log_prob']
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_file}")


def run_error_detection_on_code(code: str,
                               model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
                               k: float = 1.5,
                               use_codebert: bool = False):
    """
    Run error detection on arbitrary code snippet.

    Args:
        code: Python code to analyze
        model_name: HuggingFace model name
        k: Sensitivity factor
    """
    print("="*60)
    print("LOGICAL ERROR DETECTION ON CODE SNIPPET")
    print("="*60)
    print(f"Model: {'CodeBERT (MLM)' if use_codebert else model_name}")
    print(f"Approach: {'Masked Language Modeling' if use_codebert else 'Causal LM'}")
    print(f"Sensitivity (k): {k}")
    print("")

    print("Code to analyze:")
    print("-"*60)
    print(code)
    print("-"*60)

    # Initialize detector based on model type
    if use_codebert:
        detector = CodeBERTErrorDetector(sensitivity_factor=k)
    else:
        detector = LogicalErrorDetector(model_name=model_name, sensitivity_factor=k)

    # Analyze code
    results = detector.localize_errors(code, k=k)

    # Display results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)

    stats = results['statistics']
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Anomalous tokens: {stats['anomalous_tokens']} ({stats['anomalous_tokens']/stats['total_tokens']*100:.1f}%)")
    print(f"Total lines: {stats['total_lines']}")
    print(f"Error lines: {stats['error_lines']}")
    print(f"Mean log prob: {stats['mean_log_prob']:.4f}")
    print(f"Std dev: {stats['std_dev']:.4f}")
    print(f"Threshold: {stats['threshold']:.4f}")

    print("\n" + "="*60)
    print("DETECTED ERROR LINES")
    print("="*60)

    error_found = False
    for line_error in results['line_errors']:
        if line_error.is_error_line:
            error_found = True
            print(f"\nLine {line_error.line_number}: {line_error.line_content}")
            print(f"  • Error score: {line_error.error_score:.3f}")
            print(f"  • Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")
            print(f"  • Avg log prob: {line_error.avg_log_prob:.3f}")
            print(f"  • Min log prob: {line_error.min_log_prob:.3f}")

            # Show anomalous tokens
            if line_error.anomalous_tokens:
                print(f"  • Anomalous tokens:")
                for tok in line_error.anomalous_tokens:
                    print(f"    - '{tok.token}' (log_prob={tok.log_probability:.3f}, "
                          f"deviation={tok.deviation_score:.2f}σ)")

    if not error_found:
        print("No error lines detected. Code appears clean! ✓")

    return results


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Run Logical Error Detection Analysis (LecPrompt)"
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
        help="Run on a specific test example by name"
    )
    parser.add_argument(
        "--code-file",
        type=str,
        help="Run on code from a file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="error_detection_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--use-codebert",
        action="store_true",
        help="Use CodeBERT (MLM approach) instead of Causal LM"
    )

    args = parser.parse_args()

    if args.example:
        # Run on test example
        run_error_detection_on_example(
            example_name=args.example,
            model_name=args.model,
            k=args.sensitivity,
            output_dir=args.output_dir,
            use_codebert=args.use_codebert
        )
    elif args.code_file:
        # Run on code from file
        with open(args.code_file, 'r', encoding='utf-8') as f:
            code = f.read()
        run_error_detection_on_code(
            code=code,
            model_name=args.model,
            k=args.sensitivity,
            use_codebert=args.use_codebert
        )
    else:
        # Show available examples
        print("Please specify --example or --code-file")
        print("\nAvailable test examples:")
        dataset = TestExamplesDataset()
        for ex in dataset.examples:
            print(f"  - {ex.name:30s} ({ex.bug_type:10s}): {ex.description}")
        print("\nExample usage:")
        print(f"  {sys.argv[0]} --example binary_search_missing_bounds")
        print(f"  {sys.argv[0]} --example factorial_recursion_base_case --use-codebert")
        print(f"  {sys.argv[0]} --code-file mycode.py")
        print(f"  {sys.argv[0]} --code-file mycode.py --use-codebert")


if __name__ == "__main__":
    main()
