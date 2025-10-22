#!/usr/bin/env python3
"""
Test Script for Forced Generation Analyzer

This script demonstrates the forced generation system with various examples,
including both buggy and correct code implementations to test the hypothesis
that forced generation of buggy code results in lower confidence scores.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forced_generation_analyzer import ForcedGenerationAnalyzer, ForcedGenerationResult


class ForcedGenerationTester:
    """Test suite for the forced generation analyzer."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        """Initialize the tester with specified model."""
        self.model_name = model_name
        self.analyzer = ForcedGenerationAnalyzer(model_name=model_name)
        self.test_examples = self._create_test_examples()

    def _create_test_examples(self) -> List[Dict[str, Any]]:
        """Create a set of test examples with problems and solutions."""
        return [
            {
                "name": "factorial_basic",
                "problem": "Write a Python function to calculate the factorial of a number using recursion.",
                "correct_code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
                "buggy_code": """def factorial(n):
    if n == 1:  # Bug: missing n == 0 case
        return 1
    return n * factorial(n - 1)""",
                "description": "Factorial function with missing base case"
            },

            {
                "name": "fibonacci_simple",
                "problem": "Create a function that returns the nth Fibonacci number using recursion.",
                "correct_code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)""",
                "buggy_code": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-1)  # Bug: should be n-2""",
                "description": "Fibonacci with wrong recursive call"
            },

            {
                "name": "binary_search",
                "problem": "Implement binary search to find an element in a sorted array.",
                "correct_code": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1""",
                "buggy_code": """def binary_search(arr, target):
    left, right = 0, len(arr)  # Bug: should be len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1""",
                "description": "Binary search with incorrect initial bounds"
            },

            {
                "name": "find_max",
                "problem": "Write a function to find the maximum element in a list.",
                "correct_code": """def find_max(numbers):
    if not numbers:
        return None

    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num

    return max_val""",
                "buggy_code": """def find_max(numbers):
    max_val = numbers[0]  # Bug: no check for empty list
    for num in numbers[1:]:
        if num > max_val:
            max_val = num

    return max_val""",
                "description": "Find max without empty list validation"
            },

            {
                "name": "is_prime",
                "problem": "Implement a function to check if a number is prime.",
                "correct_code": """def is_prime(n):
    if n < 2:
        return False

    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False

    return True""",
                "buggy_code": """def is_prime(n):
    if n < 2:
        return False

    for i in range(2, n):  # Bug: inefficient, should check up to sqrt(n)
        if n % i == 0:
            return False

    return True""",
                "description": "Prime check with inefficient algorithm"
            }
        ]

    def run_single_test(self, example: Dict[str, Any], code_type: str = "correct") -> ForcedGenerationResult:
        """
        Run forced generation test on a single example.

        Args:
            example: Test example dictionary
            code_type: "correct" or "buggy"

        Returns:
            Forced generation analysis result
        """
        code = example[f"{code_type}_code"]
        problem = example["problem"]

        print(f"\n{'='*60}")
        print(f"Testing: {example['name']} ({code_type})")
        print(f"Problem: {problem}")
        print(f"Target code:")
        print(code)
        print(f"{'='*60}")

        result = self.analyzer.force_generation_with_logits(
            problem_description=problem,
            target_code=code,
            verbose=True
        )

        return result

    def run_comparative_test(self, example: Dict[str, Any], output_dir: str = "forced_generation_results") -> Dict[str, Any]:
        """
        Run comparative test between correct and buggy code.

        Args:
            example: Test example
            output_dir: Directory to save results

        Returns:
            Comparison results
        """
        print(f"\n🔍 COMPARATIVE TEST: {example['name']}")
        print(f"Description: {example['description']}")

        # Test correct code
        correct_result = self.run_single_test(example, "correct")

        # Test buggy code
        buggy_result = self.run_single_test(example, "buggy")

        # Compare results
        comparison = {
            "example_name": example["name"],
            "description": example["description"],
            "correct_analysis": {
                "avg_probability": correct_result.average_probability,
                "avg_rank": correct_result.average_rank,
                "avg_surprisal": correct_result.average_surprisal,
                "high_uncertainty_tokens": correct_result.high_uncertainty_tokens,
                "total_tokens": correct_result.total_tokens
            },
            "buggy_analysis": {
                "avg_probability": buggy_result.average_probability,
                "avg_rank": buggy_result.average_rank,
                "avg_surprisal": buggy_result.average_surprisal,
                "high_uncertainty_tokens": buggy_result.high_uncertainty_tokens,
                "total_tokens": buggy_result.total_tokens
            },
            "comparison_metrics": {
                "probability_difference": buggy_result.average_probability - correct_result.average_probability,
                "rank_difference": buggy_result.average_rank - correct_result.average_rank,
                "surprisal_difference": buggy_result.average_surprisal - correct_result.average_surprisal,
                "uncertainty_token_difference": buggy_result.high_uncertainty_tokens - correct_result.high_uncertainty_tokens
            },
            "hypothesis_test": {
                "buggy_lower_probability": bool(buggy_result.average_probability < correct_result.average_probability),
                "buggy_higher_rank": bool(buggy_result.average_rank > correct_result.average_rank),
                "buggy_higher_surprisal": bool(buggy_result.average_surprisal > correct_result.average_surprisal),
                "buggy_more_uncertainty": bool(buggy_result.high_uncertainty_tokens > correct_result.high_uncertainty_tokens)
            }
        }

        # Print comparison summary
        print(f"\n📊 COMPARISON RESULTS for {example['name']}:")
        print(f"Correct Code - Avg Prob: {correct_result.average_probability:.3f}, Avg Rank: {correct_result.average_rank:.1f}")
        print(f"Buggy Code   - Avg Prob: {buggy_result.average_probability:.3f}, Avg Rank: {buggy_result.average_rank:.1f}")
        print(f"Hypothesis Tests:")
        print(f"  ✓ Buggy has lower probability: {comparison['hypothesis_test']['buggy_lower_probability']}")
        print(f"  ✓ Buggy has higher rank: {comparison['hypothesis_test']['buggy_higher_rank']}")
        print(f"  ✓ Buggy has higher surprisal: {comparison['hypothesis_test']['buggy_higher_surprisal']}")
        print(f"  ✓ Buggy has more uncertainty: {comparison['hypothesis_test']['buggy_more_uncertainty']}")

        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)

        # Save individual results
        self.analyzer.save_analysis(correct_result, f"{output_dir}/{example['name']}_correct.json")
        self.analyzer.save_analysis(buggy_result, f"{output_dir}/{example['name']}_buggy.json")

        # Save comparison
        comparison_file = f"{output_dir}/{example['name']}_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_dir}/")

        return comparison

    def run_all_comparative_tests(self, output_dir: str = "forced_generation_results") -> Dict[str, Any]:
        """
        Run comparative tests on all examples.

        Args:
            output_dir: Directory to save results

        Returns:
            Complete test results with summary
        """
        print(f"🚀 RUNNING FORCED GENERATION TESTS ON {len(self.test_examples)} EXAMPLES")
        print(f"Model: {self.model_name}")

        results = {
            "model_name": self.model_name,
            "total_examples": len(self.test_examples),
            "individual_results": [],
            "summary_statistics": {}
        }

        for i, example in enumerate(self.test_examples, 1):
            print(f"\n[{i}/{len(self.test_examples)}] Processing {example['name']}...")

            try:
                comparison = self.run_comparative_test(example, output_dir)
                results["individual_results"].append(comparison)
            except Exception as e:
                print(f"❌ Error processing {example['name']}: {e}")
                continue

        # Calculate summary statistics
        if results["individual_results"]:
            hypothesis_confirmations = {
                "buggy_lower_probability": 0,
                "buggy_higher_rank": 0,
                "buggy_higher_surprisal": 0,
                "buggy_more_uncertainty": 0
            }

            total_valid = len(results["individual_results"])

            for result in results["individual_results"]:
                for key in hypothesis_confirmations:
                    if result["hypothesis_test"][key]:
                        hypothesis_confirmations[key] += 1

            results["summary_statistics"] = {
                "hypothesis_confirmation_rates": {
                    key: (count / total_valid) * 100 for key, count in hypothesis_confirmations.items()
                },
                "overall_confirmation_rate": sum(hypothesis_confirmations.values()) / (total_valid * 4) * 100
            }

        # Save complete results
        os.makedirs(output_dir, exist_ok=True)
        complete_file = f"{output_dir}/complete_forced_generation_analysis.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print final summary
        print(f"\n{'='*80}")
        print(f"🎯 FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total examples processed: {len(results['individual_results'])}")

        if results["individual_results"]:
            summary = results["summary_statistics"]["hypothesis_confirmation_rates"]
            print(f"Hypothesis confirmation rates:")
            print(f"  • Buggy code has lower probability: {summary['buggy_lower_probability']:.1f}%")
            print(f"  • Buggy code has higher rank: {summary['buggy_higher_rank']:.1f}%")
            print(f"  • Buggy code has higher surprisal: {summary['buggy_higher_surprisal']:.1f}%")
            print(f"  • Buggy code has more uncertainty: {summary['buggy_more_uncertainty']:.1f}%")
            print(f"Overall confirmation rate: {results['summary_statistics']['overall_confirmation_rate']:.1f}%")

        print(f"Complete results saved to: {complete_file}")

        return results

    def test_natural_vs_forced_comparison(self, example_name: str = "factorial_basic"):
        """
        Compare natural generation vs forced generation for debugging.

        Args:
            example_name: Name of example to test
        """
        example = next((ex for ex in self.test_examples if ex["name"] == example_name), None)
        if not example:
            print(f"Example {example_name} not found!")
            return

        print(f"\n🔄 NATURAL vs FORCED GENERATION COMPARISON")
        print(f"Example: {example['name']}")

        comparison = self.analyzer.compare_with_natural_generation(
            problem_description=example["problem"],
            target_code=example["correct_code"],
            max_tokens=150
        )

        print(f"\nResults:")
        print(f"Forced generation: {comparison['forced_generation']['code'][:100]}...")
        print(f"Natural generation: {comparison['natural_generation']['code'][:100]}...")
        print(f"Codes match: {comparison['comparison']['codes_match']}")
        print(f"Probability difference: {comparison['comparison']['probability_difference']:.3f}")
        print(f"Rank difference: {comparison['comparison']['rank_difference']:.1f}")

        return comparison


def main():
    """Main function to run tests."""
    print("🧪 FORCED GENERATION ANALYZER TESTING SUITE")

    # Initialize tester
    tester = ForcedGenerationTester()

    # Test 1: Run all comparative tests
    print("\n📋 Test 1: Running all comparative tests...")
    all_results = tester.run_all_comparative_tests()

    # Test 2: Natural vs forced comparison
    print("\n📋 Test 2: Natural vs forced generation comparison...")
    tester.test_natural_vs_forced_comparison("factorial_basic")

    print("\n✅ All tests completed!")
    print("Check the 'forced_generation_results' directory for detailed outputs.")


if __name__ == "__main__":
    main()