#!/usr/bin/env python3
"""
Test Examples Dataset for LLM Token Probability Analysis

This module contains structured test examples with:
- Prompts (textual requests)
- Buggy Python code (intentionally flawed)
- Correct Python code (ground truth)

Each example is designed to test the hypothesis that low token probabilities
correlate with areas prone to bugs or errors.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestExample:
    """Structure for a test example with prompt, buggy code, and correct code."""
    name: str
    prompt: str
    buggy_code: str
    correct_code: str
    description: str
    bug_type: str  # Type of bug: logic, syntax, edge_case, etc.

class TestExamplesDataset:
    """Collection of test examples for analysis."""

    def __init__(self):
        self.examples = self._create_examples()

    def _create_examples(self) -> List[TestExample]:
        """Create the dataset of test examples."""
        return [
            TestExample(
                name="binary_search_missing_bounds",
                prompt="Write a Python function that implements binary search to find an element in a sorted array.",
                buggy_code='''def binary_search(arr, target):
    left = 0
    right = len(arr)  # Bug: should be len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1''',
                correct_code='''def binary_search(arr, target):
    left = 0
    right = len(arr) - 1  # Correct bounds

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1''',
                description="Binary search with incorrect array bounds",
                bug_type="logic"
            ),

            TestExample(
                name="factorial_recursion_base_case",
                prompt="Write a recursive Python function to calculate the factorial of a number.",
                buggy_code='''def factorial(n):
    if n == 1:  # Bug: missing base case for n=0
        return 1
    return n * factorial(n - 1)''',
                correct_code='''def factorial(n):
    if n == 0 or n == 1:  # Correct base cases
        return 1
    return n * factorial(n - 1)''',
                description="Factorial function missing edge case for n=0",
                bug_type="edge_case"
            ),

            TestExample(
                name="bubble_sort_inner_loop",
                prompt="Implement bubble sort algorithm in Python.",
                buggy_code='''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1):  # Bug: should be range(n-1-i)
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr''',
                correct_code='''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1-i):  # Correct optimization
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr''',
                description="Bubble sort with inefficient inner loop bounds",
                bug_type="logic"
            ),

            TestExample(
                name="list_max_empty_check",
                prompt="Write a function to find the maximum element in a list.",
                buggy_code='''def find_max(numbers):
    max_val = numbers[0]  # Bug: no check for empty list
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val''',
                correct_code='''def find_max(numbers):
    if not numbers:  # Check for empty list
        raise ValueError("Cannot find max of empty list")
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val''',
                description="Find max function without empty list validation",
                bug_type="edge_case"
            ),

            TestExample(
                name="fibonacci_negative_input",
                prompt="Create a function that returns the nth Fibonacci number.",
                buggy_code='''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # Bug: no validation for negative n''',
                correct_code='''def fibonacci(n):
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
                description="Fibonacci function without negative input validation",
                bug_type="edge_case"
            ),

            TestExample(
                name="string_reverse_indexing",
                prompt="Write a function to reverse a string without using built-in reverse methods.",
                buggy_code='''def reverse_string(s):
    result = ""
    for i in range(len(s)):  # Bug: should iterate backwards
        result += s[i]
    return result''',
                correct_code='''def reverse_string(s):
    result = ""
    for i in range(len(s)-1, -1, -1):  # Correct backwards iteration
        result += s[i]
    return result''',
                description="String reversal with incorrect iteration direction",
                bug_type="logic"
            ),

            TestExample(
                name="prime_check_optimization",
                prompt="Implement a function to check if a number is prime.",
                buggy_code='''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):  # Bug: inefficient, should check up to sqrt(n)
        if n % i == 0:
            return False
    return True''',
                correct_code='''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):  # Optimized to sqrt(n)
        if n % i == 0:
            return False
    return True''',
                description="Prime check function with inefficient algorithm",
                bug_type="logic"
            ),

            TestExample(
                name="merge_arrays_index_bounds",
                prompt="Write a function to merge two sorted arrays into one sorted array.",
                buggy_code='''def merge_arrays(arr1, arr2):
    result = []
    i = j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Bug: missing remaining elements from both arrays
    return result''',
                correct_code='''def merge_arrays(arr1, arr2):
    result = []
    i = j = 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # Add remaining elements
    while i < len(arr1):
        result.append(arr1[i])
        i += 1

    while j < len(arr2):
        result.append(arr2[j])
        j += 1

    return result''',
                description="Merge sorted arrays with incomplete implementation",
                bug_type="logic"
            ),

            TestExample(
                name="count_vowels_case_sensitivity",
                prompt="Create a function to count vowels in a string.",
                buggy_code='''def count_vowels(text):
    vowels = "aeiou"
    count = 0
    for char in text:
        if char in vowels:  # Bug: doesn't handle uppercase vowels
            count += 1
    return count''',
                correct_code='''def count_vowels(text):
    vowels = "aeiouAEIOU"  # Include both cases
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count''',
                description="Vowel counting function missing case handling",
                bug_type="logic"
            ),

            TestExample(
                name="division_zero_check",
                prompt="Write a function that calculates the average of a list of numbers.",
                buggy_code='''def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)  # Bug: no check for empty list (division by zero)''',
                correct_code='''def calculate_average(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    total = sum(numbers)
    return total / len(numbers)''',
                description="Average calculation without zero division protection",
                bug_type="edge_case"
            )
        ]

    def get_example(self, name: str) -> TestExample:
        """Get a specific example by name."""
        for example in self.examples:
            if example.name == name:
                return example
        raise ValueError(f"Example '{name}' not found")

    def get_all_examples(self) -> List[TestExample]:
        """Get all examples."""
        return self.examples

    def get_examples_by_bug_type(self, bug_type: str) -> List[TestExample]:
        """Get examples filtered by bug type."""
        return [ex for ex in self.examples if ex.bug_type == bug_type]

    def get_calibration_set(self, exclude_examples: Optional[List[str]] = None) -> List[TestExample]:
        """
        Get calibration set for conformal prediction.

        Returns the first 2 examples (binary_search and factorial) for calibration,
        UNLESS they conflict with the requested test examples. In that case, uses
        alternative examples to avoid data leakage.

        Args:
            exclude_examples: List of example names to exclude from calibration
                            (typically the examples being tested)

        Returns:
            List of 2 TestExample objects for calibration
        """
        if exclude_examples is None:
            # Default: Use first 2 examples: binary_search (logic) and factorial (edge_case)
            return self.examples[:2]

        # Dynamic calibration: exclude requested test examples to avoid data leakage
        available_for_calibration = [ex for ex in self.examples if ex.name not in exclude_examples]

        if len(available_for_calibration) < 2:
            raise ValueError(f"Not enough examples for calibration after excluding {exclude_examples}. Need at least 2.")

        # Try to get diverse bug types for robust calibration
        logic_examples = [ex for ex in available_for_calibration if ex.bug_type == "logic"]
        edge_case_examples = [ex for ex in available_for_calibration if ex.bug_type == "edge_case"]

        # Prefer one logic + one edge_case for diversity
        if logic_examples and edge_case_examples:
            return [logic_examples[0], edge_case_examples[0]]
        else:
            # Fallback: just use first 2 available
            return available_for_calibration[:2]

    def get_test_set(self, requested_examples: Optional[List[str]] = None) -> List[TestExample]:
        """
        Get test set for evaluation after calibration.

        Args:
            requested_examples: If provided, return only these specific examples.
                              Otherwise, return all examples except calibration set.

        Returns:
            List of TestExample objects for testing
        """
        if requested_examples is None:
            # Use remaining 8 examples for testing (excluding calibration set)
            return self.examples[2:]
        else:
            # Return only requested examples
            return [ex for ex in self.examples if ex.name in requested_examples]

    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get information about the calibration/test split.

        Returns:
            Dictionary with split metadata
        """
        cal_set = self.get_calibration_set()
        test_set = self.get_test_set()

        return {
            "total_examples": len(self.examples),
            "calibration_count": len(cal_set),
            "test_count": len(test_set),
            "calibration_examples": [ex.name for ex in cal_set],
            "test_examples": [ex.name for ex in test_set],
            "calibration_bug_types": [ex.bug_type for ex in cal_set],
            "test_bug_types": [ex.bug_type for ex in test_set]
        }

    def save_to_json(self, filename: str = "test_examples.json"):
        """Save examples to JSON file."""
        data = {
            "examples": [
                {
                    "name": ex.name,
                    "prompt": ex.prompt,
                    "buggy_code": ex.buggy_code,
                    "correct_code": ex.correct_code,
                    "description": ex.description,
                    "bug_type": ex.bug_type
                }
                for ex in self.examples
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Test examples saved to {filename}")

    def load_from_json(self, filename: str):
        """Load examples from JSON file."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.examples = [
            TestExample(
                name=ex["name"],
                prompt=ex["prompt"],
                buggy_code=ex["buggy_code"],
                correct_code=ex["correct_code"],
                description=ex["description"],
                bug_type=ex["bug_type"]
            )
            for ex in data["examples"]
        ]

        print(f"Test examples loaded from {filename}")

def create_comprehensive_test_prompts() -> List[str]:
    """Create prompts for testing both buggy and correct code."""
    dataset = TestExamplesDataset()
    prompts = []

    for example in dataset.get_all_examples():
        # Prompt for buggy code
        buggy_prompt = f"{example.prompt}\n\nGenerate the following code:\n{example.buggy_code}"
        prompts.append({
            "type": "buggy",
            "example_name": example.name,
            "prompt": buggy_prompt,
            "expected_code": example.buggy_code,
            "bug_type": example.bug_type
        })

        # Prompt for correct code
        correct_prompt = f"{example.prompt}\n\nGenerate the following code:\n{example.correct_code}"
        prompts.append({
            "type": "correct",
            "example_name": example.name,
            "prompt": correct_prompt,
            "expected_code": example.correct_code,
            "bug_type": example.bug_type
        })

    return prompts

if __name__ == "__main__":
    # Create and save the dataset
    dataset = TestExamplesDataset()
    dataset.save_to_json()

    print(f"\nCreated {len(dataset.examples)} test examples:")
    for ex in dataset.examples:
        print(f"  • {ex.name} ({ex.bug_type}): {ex.description}")

    # Create comprehensive test prompts
    prompts = create_comprehensive_test_prompts()

    with open("test_prompts.json", 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"\nCreated {len(prompts)} test prompts saved to 'test_prompts.json'")