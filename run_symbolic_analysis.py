#!/usr/bin/env python3
"""
Symbolic Analysis Runner

This script demonstrates how to use the symbolic execution analyzer to compare
buggy and correct code implementations. It serves as both a demo and a practical
tool for analyzing code correctness.

Usage:
    python run_symbolic_analysis.py                    # Analyze all test examples
    python run_symbolic_analysis.py --example <name>   # Analyze specific example
    python run_symbolic_analysis.py --custom           # Analyze custom code pair
    python run_symbolic_analysis.py --test             # Quick test with factorial
"""

import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symbolic_analyzer import (
    SymbolicExecutionAnalyzer,
    BehavioralComparison,
    compare_codes,
    analyze_all_test_examples,
    TEST_EXAMPLES_AVAILABLE,
    Z3_AVAILABLE
)

if TEST_EXAMPLES_AVAILABLE:
    from test_examples import TestExamplesDataset


def format_comparison_result(result: BehavioralComparison, verbose: bool = True) -> str:
    """Format a behavioral comparison result for display."""
    output = []
    output.append(f"Function: {result.function_name}")
    output.append(f"Execution successful: {result.execution_successful}")

    if result.error_message:
        output.append(f"Error: {result.error_message}")
        return "\n".join(output)

    output.append(f"Behavioral distance: {result.behavioral_distance:.4f}")
    output.append(f"Paths explored: {result.paths_explored}")
    output.append(f"Paths divergent: {result.paths_divergent}")
    output.append(f"Path divergence ratio: {result.path_divergence_ratio:.4f}")
    output.append(f"Execution time: {result.execution_time_ms:.2f}ms")

    output.append(f"Divergent inputs: {len(result.divergent_inputs)}")
    output.append(f"Error-triggering inputs: {len(result.error_triggering_inputs)}")

    if verbose and result.divergent_inputs:
        output.append("\nDivergent inputs (first 5):")
        for i, inp in enumerate(result.divergent_inputs[:5]):
            output.append(f"  {i+1}. {inp}")

    if verbose and result.error_triggering_inputs:
        output.append("\nError-triggering inputs:")
        for i, inp in enumerate(result.error_triggering_inputs[:3]):
            output.append(f"  {i+1}. {inp}")

    if verbose and result.output_differences:
        output.append("\nOutput differences (first 3):")
        for i, diff in enumerate(result.output_differences[:3]):
            output.append(f"  {i+1}. Input: {diff['input']}")
            output.append(f"     Buggy output: {diff['buggy_output']}")
            output.append(f"     Correct output: {diff['correct_output']}")
            if diff.get('buggy_error'):
                output.append(f"     Buggy error: {diff['buggy_error']}")
            if diff.get('correct_error'):
                output.append(f"     Correct error: {diff['correct_error']}")

    return "\n".join(output)


def analyze_specific_example(example_name: str) -> Optional[BehavioralComparison]:
    """Analyze a specific test example by name."""
    if not TEST_EXAMPLES_AVAILABLE:
        print("Error: test_examples.py not available")
        return None

    dataset = TestExamplesDataset()

    # Find the example
    target_example = None
    for example in dataset.examples:
        if example.name == example_name:
            target_example = example
            break

    if not target_example:
        print(f"Error: Example '{example_name}' not found")
        print("Available examples:")
        for example in dataset.examples:
            print(f"  - {example.name}")
        return None

    print(f"Analyzing example: {target_example.name}")
    print(f"Description: {target_example.description}")
    print(f"Bug type: {target_example.bug_type}")
    print()

    analyzer = SymbolicExecutionAnalyzer()
    result = analyzer.analyze_test_example(target_example)

    print("Analysis Results:")
    print("=" * 50)
    print(format_comparison_result(result, verbose=True))

    return result


def analyze_custom_code():
    """Analyze custom code pair provided by user."""
    print("Enter buggy code (end with '###' on a new line):")
    buggy_lines = []
    while True:
        line = input()
        if line.strip() == "###":
            break
        buggy_lines.append(line)

    buggy_code = "\n".join(buggy_lines)

    print("\nEnter correct code (end with '###' on a new line):")
    correct_lines = []
    while True:
        line = input()
        if line.strip() == "###":
            break
        correct_lines.append(line)

    correct_code = "\n".join(correct_lines)

    print("\nAnalyzing custom code...")
    result = compare_codes(buggy_code, correct_code)

    print("Analysis Results:")
    print("=" * 50)
    print(format_comparison_result(result, verbose=True))

    return result


def run_quick_test():
    """Run a quick test with the factorial example."""
    print("Running quick test with factorial example...")

    buggy = '''def factorial(n):
    if n == 1:  # Bug: missing base case for n=0
        return 1
    return n * factorial(n - 1)'''

    correct = '''def factorial(n):
    if n == 0 or n == 1:  # Correct base cases
        return 1
    return n * factorial(n - 1)'''

    result = compare_codes(buggy, correct)

    print("Quick Test Results:")
    print("=" * 50)
    print(format_comparison_result(result, verbose=True))

    return result


def analyze_all_examples(save_results: bool = True) -> List[BehavioralComparison]:
    """Analyze all available test examples."""
    if not TEST_EXAMPLES_AVAILABLE:
        print("Error: test_examples.py not available")
        return []

    print("Analyzing all test examples...")
    print("=" * 50)

    results = analyze_all_test_examples()

    print(f"\nAnalyzed {len(results)} examples")
    print("\nSummary:")
    print("-" * 30)

    for result in results:
        print(f"{result.function_name:25} | Distance: {result.behavioral_distance:.3f} | "
              f"Divergent: {len(result.divergent_inputs):2d} | "
              f"Errors: {len(result.error_triggering_inputs):2d}")

    if save_results and results:
        # Create analyzer to get summary and save results
        analyzer = SymbolicExecutionAnalyzer()
        analyzer.analysis_history = results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"symbolic_analysis_results_{timestamp}.json"
        analyzer.save_analysis_results(filename)

        print(f"\nDetailed results saved to: {filename}")

        # Print summary statistics
        summary = analyzer.get_analysis_summary()
        print("\nOverall Statistics:")
        print("-" * 30)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:30}: {value:.4f}")
            else:
                print(f"{key:30}: {value}")

    return results


def create_comparison_report(results: List[BehavioralComparison],
                           output_file: Optional[str] = None) -> str:
    """Create a detailed comparison report."""
    if not results:
        return "No results to report."

    report_lines = []
    report_lines.append("Symbolic Execution Analysis Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total analyses: {len(results)}")
    report_lines.append("")

    # Summary statistics
    distances = [r.behavioral_distance for r in results]
    times = [r.execution_time_ms for r in results]

    report_lines.append("Summary Statistics:")
    report_lines.append("-" * 30)
    report_lines.append(f"Average behavioral distance: {sum(distances)/len(distances):.4f}")
    report_lines.append(f"Min behavioral distance: {min(distances):.4f}")
    report_lines.append(f"Max behavioral distance: {max(distances):.4f}")
    report_lines.append(f"Average execution time: {sum(times)/len(times):.2f}ms")
    report_lines.append("")

    # Individual results
    report_lines.append("Individual Results:")
    report_lines.append("-" * 30)

    for i, result in enumerate(results, 1):
        report_lines.append(f"\n{i}. {result.function_name}")
        report_lines.append("-" * (len(str(i)) + 2 + len(result.function_name)))
        report_lines.append(format_comparison_result(result, verbose=False))

    report_content = "\n".join(report_lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"Report saved to: {output_file}")

    return report_content


def main():
    """Main function to handle command line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Symbolic Execution Analyzer for Code Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_symbolic_analysis.py                          # Analyze all examples
  python run_symbolic_analysis.py --test                   # Quick factorial test
  python run_symbolic_analysis.py --example factorial_recursion_base_case
  python run_symbolic_analysis.py --custom                 # Interactive custom input
  python run_symbolic_analysis.py --all --report report.txt
        """
    )

    parser.add_argument('--example', type=str,
                       help='Analyze specific example by name')
    parser.add_argument('--custom', action='store_true',
                       help='Analyze custom code pair (interactive)')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test with factorial example')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all available test examples')
    parser.add_argument('--report', type=str,
                       help='Generate detailed report and save to file')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Print system information
    print("Symbolic Execution Analyzer")
    print("=" * 40)
    print(f"Z3 Solver available: {Z3_AVAILABLE}")
    print(f"Test Examples available: {TEST_EXAMPLES_AVAILABLE}")
    print()

    results = []

    if args.test:
        result = run_quick_test()
        if result:
            results.append(result)

    elif args.example:
        result = analyze_specific_example(args.example)
        if result:
            results.append(result)

    elif args.custom:
        result = analyze_custom_code()
        if result:
            results.append(result)

    elif args.all or not any([args.test, args.example, args.custom]):
        # Default behavior: analyze all examples
        results = analyze_all_examples(save_results=args.save)

    # Generate report if requested
    if args.report and results:
        create_comparison_report(results, args.report)

    # Save results if requested and not already saved
    if args.save and results and not args.all:
        analyzer = SymbolicExecutionAnalyzer()
        analyzer.analysis_history = results

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"symbolic_analysis_results_{timestamp}.json"
        analyzer.save_analysis_results(filename)

    if not results:
        print("No analysis performed. Use --help for usage information.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)