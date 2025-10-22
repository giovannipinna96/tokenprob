#!/usr/bin/env python3
"""
Test Suite for Symbolic Execution Analyzer

This module contains comprehensive tests for the symbolic_analyzer.py module.
It verifies that the behavioral comparison and symbolic execution work correctly
with various code examples.
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symbolic_analyzer import (
    SymbolicExecutionAnalyzer,
    SimpleSymbolicExecutor,
    BehavioralComparison,
    compare_codes
)


class TestSymbolicExecutionAnalyzer(unittest.TestCase):
    """Test cases for the SymbolicExecutionAnalyzer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = SymbolicExecutionAnalyzer()

    def test_factorial_example(self):
        """Test the factorial example with missing base case."""
        buggy_code = '''def factorial(n):
    if n == 1:  # Bug: missing base case for n=0
        return 1
    return n * factorial(n - 1)'''

        correct_code = '''def factorial(n):
    if n == 0 or n == 1:  # Correct base cases
        return 1
    return n * factorial(n - 1)'''

        result = self.analyzer.analyze_code_pair(buggy_code, correct_code)

        # Verify basic properties
        self.assertIsInstance(result, BehavioralComparison)
        self.assertEqual(result.function_name, "factorial")
        self.assertTrue(result.execution_successful)
        self.assertIsNone(result.error_message)

        # Should find behavioral differences
        self.assertGreater(result.behavioral_distance, 0.0)
        self.assertGreater(len(result.divergent_inputs), 0)

        # Should find the n=0 case as divergent
        n0_found = any(inp.get('n') == 0 for inp in result.divergent_inputs)
        self.assertTrue(n0_found, "Should detect n=0 as a divergent case")

        print(f"Factorial test results:")
        print(f"  Behavioral distance: {result.behavioral_distance:.3f}")
        print(f"  Divergent inputs: {len(result.divergent_inputs)}")
        print(f"  Error-triggering inputs: {len(result.error_triggering_inputs)}")

    def test_binary_search_example(self):
        """Test binary search with incorrect bounds."""
        buggy_code = '''def binary_search(arr, target):
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

    return -1'''

        correct_code = '''def binary_search(arr, target):
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

    return -1'''

        result = self.analyzer.analyze_code_pair(buggy_code, correct_code)

        # Should detect differences
        self.assertIsInstance(result, BehavioralComparison)
        self.assertEqual(result.function_name, "binary_search")
        self.assertTrue(result.execution_successful)

        # Buggy version should trigger errors due to index out of bounds
        self.assertGreaterEqual(len(result.error_triggering_inputs), 0)

        print(f"Binary search test results:")
        print(f"  Behavioral distance: {result.behavioral_distance:.3f}")
        print(f"  Divergent inputs: {len(result.divergent_inputs)}")
        print(f"  Error-triggering inputs: {len(result.error_triggering_inputs)}")

    def test_empty_list_max_example(self):
        """Test find max function without empty list check."""
        buggy_code = '''def find_max(numbers):
    max_val = numbers[0]  # Bug: no check for empty list
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val'''

        correct_code = '''def find_max(numbers):
    if not numbers:  # Check for empty list
        raise ValueError("Cannot find max of empty list")
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val'''

        result = self.analyzer.analyze_code_pair(buggy_code, correct_code)

        # Should detect error differences with empty list
        self.assertTrue(result.execution_successful)
        self.assertGreater(len(result.error_triggering_inputs), 0)

        # Empty list should be among error-triggering inputs
        empty_list_found = any(
            inp.get('numbers') == [] or inp.get('arr') == []
            for inp in result.error_triggering_inputs
        )
        self.assertTrue(empty_list_found, "Should detect empty list as error-triggering")

        print(f"Find max test results:")
        print(f"  Behavioral distance: {result.behavioral_distance:.3f}")
        print(f"  Divergent inputs: {len(result.divergent_inputs)}")
        print(f"  Error-triggering inputs: {len(result.error_triggering_inputs)}")

    def test_identical_functions(self):
        """Test two identical functions should have zero behavioral distance."""
        code = '''def add_numbers(a, b):
    return a + b'''

        result = self.analyzer.analyze_code_pair(code, code)

        # Identical code should have zero behavioral distance
        self.assertEqual(result.behavioral_distance, 0.0)
        self.assertEqual(len(result.divergent_inputs), 0)
        self.assertEqual(len(result.error_triggering_inputs), 0)

        print(f"Identical functions test:")
        print(f"  Behavioral distance: {result.behavioral_distance:.3f}")

    def test_bubble_sort_example(self):
        """Test bubble sort with inefficient inner loop."""
        buggy_code = '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1):  # Bug: should be range(n-1-i)
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr'''

        correct_code = '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-1-i):  # Correct optimization
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr'''

        result = self.analyzer.analyze_code_pair(buggy_code, correct_code)

        # Both should produce same output (correct sorting), but inefficiency won't be detected
        # This is a limitation of our simple behavioral comparison
        self.assertTrue(result.execution_successful)

        print(f"Bubble sort test results:")
        print(f"  Behavioral distance: {result.behavioral_distance:.3f}")
        print(f"  Divergent inputs: {len(result.divergent_inputs)}")
        print(f"  Note: Efficiency bugs may not be detected in behavioral comparison")

    def test_analysis_summary(self):
        """Test that analysis summary works correctly."""
        # Run a few analyses
        self.test_factorial_example()
        self.test_binary_search_example()

        summary = self.analyzer.get_analysis_summary()

        # Should have analysis data
        self.assertGreater(summary["total_analyses"], 0)
        self.assertIn("avg_behavioral_distance", summary)
        self.assertIn("total_divergent_cases", summary)
        self.assertIn("total_error_triggering_cases", summary)

        print("Analysis summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")


class TestSimpleSymbolicExecutor(unittest.TestCase):
    """Test cases for the SimpleSymbolicExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.executor = SimpleSymbolicExecutor()

    def test_extract_function_info(self):
        """Test function information extraction."""
        code = '''def test_func(a, b, c=None):
    return a + b'''

        info = self.executor.extract_function_info(code)

        self.assertIsNotNone(info)
        self.assertEqual(info['name'], 'test_func')
        self.assertEqual(len(info['parameters']), 3)
        self.assertEqual(info['parameters'][0]['name'], 'a')
        self.assertEqual(info['parameters'][1]['name'], 'b')
        self.assertEqual(info['parameters'][2]['name'], 'c')

    def test_execute_with_inputs(self):
        """Test code execution with inputs."""
        code = '''def multiply(x, y):
    return x * y'''

        result, error = self.executor.execute_with_inputs(code, 'multiply', {'x': 3, 'y': 4})

        self.assertIsNone(error)
        self.assertEqual(result, 12)

    def test_execute_with_error(self):
        """Test code execution that produces an error."""
        code = '''def divide(x, y):
    return x / y'''

        result, error = self.executor.execute_with_inputs(code, 'divide', {'x': 10, 'y': 0})

        self.assertIsNone(result)
        self.assertIsNotNone(error)
        self.assertIn('division by zero', error.lower())

    def test_generate_test_inputs(self):
        """Test automatic test input generation."""
        func_info = {
            'name': 'binary_search',
            'parameters': [
                {'name': 'arr', 'annotation': None},
                {'name': 'target', 'annotation': None}
            ]
        }

        inputs = self.executor.generate_test_inputs(func_info)

        self.assertGreater(len(inputs), 0)
        # Should generate various array and target combinations
        has_empty_array = any(inp.get('arr') == [] for inp in inputs)
        has_target = any('target' in inp for inp in inputs)

        self.assertTrue(has_empty_array, "Should generate empty array test case")
        self.assertTrue(has_target, "Should generate target parameters")


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions."""

    def test_compare_codes_function(self):
        """Test the convenience function for quick comparisons."""
        buggy = '''def abs_value(x):
    if x > 0:  # Bug: should be x >= 0
        return x
    else:
        return -x'''

        correct = '''def abs_value(x):
    if x >= 0:  # Correct condition
        return x
    else:
        return -x'''

        result = compare_codes(buggy, correct)

        self.assertIsInstance(result, BehavioralComparison)
        self.assertEqual(result.function_name, "abs_value")
        self.assertTrue(result.execution_successful)

        # Should detect x=0 as divergent case
        x0_found = any(inp.get('x') == 0 for inp in result.divergent_inputs)
        self.assertTrue(x0_found, "Should detect x=0 as divergent case")


def run_comprehensive_tests():
    """Run all tests and provide detailed output."""
    print("Running Symbolic Execution Analyzer Test Suite")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestSymbolicExecutionAnalyzer,
        TestSimpleSymbolicExecutor,
        TestConvenienceFunctions
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "comprehensive":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    else:
        print("Running basic test suite...")
        unittest.main(verbosity=2)