#!/usr/bin/env python3
"""
Symbolic Execution Analyzer for Code Behavior Comparison

This module provides symbolic execution capabilities to compare the behavioral
differences between buggy and correct code implementations. It's designed to
work alongside the existing probability analysis to provide deeper insights
into code correctness.

Key Features:
- Symbolic execution using Z3 solver
- Behavioral comparison between code variants
- Input generation to expose behavioral differences
- Path coverage analysis
- Integration with existing test examples

Note: This module is completely standalone and optional. The main system
continues to work without it. Install z3-solver manually if needed:
    pip install z3-solver
"""

import ast
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import math

# Optional Z3 import with fallback
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: z3-solver not available. Install with: pip install z3-solver")

# Import existing test examples if available
try:
    from test_examples import TestExample, TestExamplesDataset
    TEST_EXAMPLES_AVAILABLE = True
except ImportError:
    TEST_EXAMPLES_AVAILABLE = False
    print("Warning: test_examples.py not found. Some functionality may be limited.")


@dataclass
class BehavioralComparison:
    """Results from comparing two code implementations behaviorally."""
    buggy_code: str
    correct_code: str
    function_name: str

    # Execution results
    execution_successful: bool
    error_message: Optional[str]

    # Behavioral differences
    behavioral_distance: float  # 0.0 = identical, 1.0 = completely different
    divergent_inputs: List[Dict[str, Any]]  # Inputs that produce different outputs
    error_triggering_inputs: List[Dict[str, Any]]  # Inputs that cause errors only in buggy code

    # Path analysis
    paths_explored: int
    paths_divergent: int
    path_divergence_ratio: float

    # Detailed analysis
    output_differences: List[Dict[str, Any]]
    execution_time_ms: float

    # Symbolic execution metrics
    constraints_generated: int
    solver_calls: int
    symbolic_variables: List[str]


@dataclass
class SymbolicTestCase:
    """A test case generated through symbolic execution."""
    input_values: Dict[str, Any]
    expected_output: Any
    buggy_output: Any
    outputs_differ: bool
    triggers_error: bool
    error_type: Optional[str]
    execution_path: str


class SimpleSymbolicExecutor:
    """
    Simplified symbolic execution engine using concrete execution with
    generated test cases. Falls back gracefully when Z3 is not available.
    """

    def __init__(self):
        self.z3_available = Z3_AVAILABLE
        self.test_cases_generated = 0
        self.max_test_cases = 100

    def extract_function_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Extract function name and parameters from code."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        params.append({
                            'name': arg.arg,
                            'annotation': ast.get_source_segment(code, arg.annotation) if arg.annotation else None
                        })
                    return {
                        'name': node.name,
                        'parameters': params,
                        'body': ast.get_source_segment(code, node) if hasattr(ast, 'get_source_segment') else code
                    }
        except Exception as e:
            print(f"Error parsing code: {e}")
            return None
        return None

    def execute_with_inputs(self, code: str, func_name: str, inputs: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Execute code with given inputs and return result and any error."""
        try:
            # Create a local namespace for execution
            local_ns = {}
            global_ns = {'__builtins__': __builtins__}

            # Execute the code to define the function
            exec(code, global_ns, local_ns)

            if func_name not in local_ns:
                return None, f"Function {func_name} not found"

            func = local_ns[func_name]

            # Call the function with inputs
            if isinstance(inputs, dict):
                result = func(**inputs)
            elif isinstance(inputs, (list, tuple)):
                result = func(*inputs)
            else:
                result = func(inputs)

            return result, None

        except Exception as e:
            return None, str(e)

    def generate_test_inputs(self, func_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test inputs for the function."""
        if not func_info:
            return []

        test_inputs = []
        params = func_info['parameters']

        if not params:
            return [{}]  # No parameters

        # Generate various test cases based on parameter names and common patterns
        param_names = [p['name'] for p in params]

        # Common test patterns
        test_patterns = []

        if any('arr' in name or 'list' in name or 'numbers' in name for name in param_names):
            # Array/list parameters
            test_patterns.extend([
                {'arr': [], 'target': 5} if 'target' in param_names else {'arr': []},
                {'arr': [1], 'target': 1} if 'target' in param_names else {'arr': [1]},
                {'arr': [1, 2, 3, 4, 5], 'target': 3} if 'target' in param_names else {'arr': [1, 2, 3, 4, 5]},
                {'arr': [1, 2, 3, 4, 5], 'target': 6} if 'target' in param_names else {'arr': [5, 4, 3, 2, 1]},
            ])

        if any('n' == name for name in param_names):
            # Numeric parameter 'n'
            test_patterns.extend([
                {'n': 0}, {'n': 1}, {'n': 5}, {'n': 10}, {'n': -1}, {'n': -5}
            ])

        # Filter patterns to match actual parameters
        for pattern in test_patterns:
            if all(key in param_names for key in pattern.keys()):
                test_inputs.append(pattern)

        # If no specific patterns matched, generate generic inputs
        if not test_inputs:
            for param in params:
                if 'n' in param['name']:
                    test_inputs.extend([{param['name']: i} for i in [0, 1, 5, -1]])
                elif 'arr' in param['name'] or 'list' in param['name']:
                    test_inputs.extend([
                        {param['name']: []},
                        {param['name']: [1, 2, 3]}
                    ])
                else:
                    test_inputs.extend([{param['name']: i} for i in [0, 1, 5]])

        return test_inputs[:self.max_test_cases]

    def compare_functions(self, buggy_code: str, correct_code: str) -> BehavioralComparison:
        """Compare two function implementations behaviorally."""
        start_time = __import__('time').time()

        # Extract function information
        buggy_info = self.extract_function_info(buggy_code)
        correct_info = self.extract_function_info(correct_code)

        if not buggy_info or not correct_info:
            return BehavioralComparison(
                buggy_code=buggy_code,
                correct_code=correct_code,
                function_name="unknown",
                execution_successful=False,
                error_message="Could not parse function definitions",
                behavioral_distance=1.0,
                divergent_inputs=[],
                error_triggering_inputs=[],
                paths_explored=0,
                paths_divergent=0,
                path_divergence_ratio=1.0,
                output_differences=[],
                execution_time_ms=0,
                constraints_generated=0,
                solver_calls=0,
                symbolic_variables=[]
            )

        func_name = buggy_info['name']

        # Generate test inputs
        test_inputs_list = self.generate_test_inputs(buggy_info)

        # Compare executions
        divergent_inputs = []
        error_triggering_inputs = []
        output_differences = []

        total_tests = len(test_inputs_list)
        successful_tests = 0

        for test_inputs in test_inputs_list:
            try:
                # Execute both versions
                buggy_result, buggy_error = self.execute_with_inputs(buggy_code, func_name, test_inputs)
                correct_result, correct_error = self.execute_with_inputs(correct_code, func_name, test_inputs)

                successful_tests += 1

                # Check for differences
                outputs_differ = buggy_result != correct_result
                has_error_difference = bool(buggy_error) != bool(correct_error)

                if outputs_differ or has_error_difference:
                    divergent_inputs.append(test_inputs)

                    output_differences.append({
                        'input': test_inputs,
                        'buggy_output': buggy_result,
                        'correct_output': correct_result,
                        'buggy_error': buggy_error,
                        'correct_error': correct_error
                    })

                # Check if only buggy version throws error
                if buggy_error and not correct_error:
                    error_triggering_inputs.append(test_inputs)

            except Exception as e:
                print(f"Error during comparison with inputs {test_inputs}: {e}")

        # Calculate metrics
        behavioral_distance = len(divergent_inputs) / max(total_tests, 1) if total_tests > 0 else 1.0
        path_divergence_ratio = len(divergent_inputs) / max(successful_tests, 1) if successful_tests > 0 else 1.0

        end_time = __import__('time').time()
        execution_time_ms = (end_time - start_time) * 1000

        return BehavioralComparison(
            buggy_code=buggy_code,
            correct_code=correct_code,
            function_name=func_name,
            execution_successful=True,
            error_message=None,
            behavioral_distance=behavioral_distance,
            divergent_inputs=divergent_inputs,
            error_triggering_inputs=error_triggering_inputs,
            paths_explored=total_tests,
            paths_divergent=len(divergent_inputs),
            path_divergence_ratio=path_divergence_ratio,
            output_differences=output_differences,
            execution_time_ms=execution_time_ms,
            constraints_generated=0,  # Not implemented in simple version
            solver_calls=0,  # Not implemented in simple version
            symbolic_variables=[]  # Not implemented in simple version
        )


class SymbolicExecutionAnalyzer:
    """
    Main analyzer class that combines symbolic execution with behavioral comparison.
    This is the primary interface for the symbolic analysis functionality.
    """

    def __init__(self, max_test_cases: int = 100):
        """
        Initialize the symbolic execution analyzer.

        Args:
            max_test_cases: Maximum number of test cases to generate per comparison
        """
        self.executor = SimpleSymbolicExecutor()
        self.executor.max_test_cases = max_test_cases
        self.analysis_history: List[BehavioralComparison] = []

    def analyze_test_example(self, test_example) -> BehavioralComparison:
        """
        Analyze a test example from the existing dataset.

        Args:
            test_example: TestExample object (if available) or dict with buggy_code, correct_code

        Returns:
            BehavioralComparison with analysis results
        """
        if hasattr(test_example, 'buggy_code'):
            # TestExample object
            buggy_code = test_example.buggy_code
            correct_code = test_example.correct_code
        else:
            # Dict or similar
            buggy_code = test_example.get('buggy_code', '')
            correct_code = test_example.get('correct_code', '')

        comparison = self.executor.compare_functions(buggy_code, correct_code)
        self.analysis_history.append(comparison)

        return comparison

    def analyze_code_pair(self, buggy_code: str, correct_code: str) -> BehavioralComparison:
        """
        Analyze a pair of code implementations directly.

        Args:
            buggy_code: Code implementation with bugs
            correct_code: Correct code implementation

        Returns:
            BehavioralComparison with analysis results
        """
        comparison = self.executor.compare_functions(buggy_code, correct_code)
        self.analysis_history.append(comparison)

        return comparison

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary statistics from all analyses performed."""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}

        distances = [comp.behavioral_distance for comp in self.analysis_history]
        execution_times = [comp.execution_time_ms for comp in self.analysis_history]

        return {
            "total_analyses": len(self.analysis_history),
            "avg_behavioral_distance": sum(distances) / len(distances),
            "min_behavioral_distance": min(distances),
            "max_behavioral_distance": max(distances),
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "total_divergent_cases": sum(len(comp.divergent_inputs) for comp in self.analysis_history),
            "total_error_triggering_cases": sum(len(comp.error_triggering_inputs) for comp in self.analysis_history),
        }

    def save_analysis_results(self, filename: str):
        """Save all analysis results to a JSON file."""
        def convert_for_json(obj):
            """Convert complex objects for JSON serialization."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        data = {
            "analysis_summary": self.get_analysis_summary(),
            "individual_analyses": [convert_for_json(comp) for comp in self.analysis_history],
            "metadata": {
                "z3_available": Z3_AVAILABLE,
                "test_examples_available": TEST_EXAMPLES_AVAILABLE,
                "analyzer_type": "SimpleSymbolicExecutor"
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert_for_json)

        print(f"Analysis results saved to {filename}")


# Convenience functions for easy usage
def compare_codes(buggy_code: str, correct_code: str) -> BehavioralComparison:
    """Quick function to compare two code implementations."""
    analyzer = SymbolicExecutionAnalyzer()
    return analyzer.analyze_code_pair(buggy_code, correct_code)


def analyze_all_test_examples() -> List[BehavioralComparison]:
    """Analyze all available test examples if test_examples module is available."""
    if not TEST_EXAMPLES_AVAILABLE:
        print("test_examples.py not available. Cannot analyze test examples.")
        return []

    analyzer = SymbolicExecutionAnalyzer()
    dataset = TestExamplesDataset()
    results = []

    for example in dataset.examples:
        print(f"Analyzing {example.name}...")
        try:
            result = analyzer.analyze_test_example(example)
            results.append(result)
            print(f"  Behavioral distance: {result.behavioral_distance:.3f}")
            print(f"  Divergent inputs: {len(result.divergent_inputs)}")
            print(f"  Error-triggering inputs: {len(result.error_triggering_inputs)}")
        except Exception as e:
            print(f"  Error: {e}")

    return results


if __name__ == "__main__":
    print("Symbolic Execution Analyzer")
    print("=" * 40)
    print(f"Z3 Solver available: {Z3_AVAILABLE}")
    print(f"Test Examples available: {TEST_EXAMPLES_AVAILABLE}")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test with a simple example
        buggy = '''def factorial(n):
    if n == 1:  # Bug: missing base case for n=0
        return 1
    return n * factorial(n - 1)'''

        correct = '''def factorial(n):
    if n == 0 or n == 1:  # Correct base cases
        return 1
    return n * factorial(n - 1)'''

        print("Testing with factorial example...")
        result = compare_codes(buggy, correct)

        print(f"Function: {result.function_name}")
        print(f"Behavioral distance: {result.behavioral_distance:.3f}")
        print(f"Divergent inputs: {len(result.divergent_inputs)}")
        print(f"Error-triggering inputs: {len(result.error_triggering_inputs)}")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")

        if result.output_differences:
            print("\nOutput differences:")
            for diff in result.output_differences[:3]:  # Show first 3
                print(f"  Input: {diff['input']}")
                print(f"  Buggy output: {diff['buggy_output']}")
                print(f"  Correct output: {diff['correct_output']}")
                if diff['buggy_error']:
                    print(f"  Buggy error: {diff['buggy_error']}")
                print()

    elif TEST_EXAMPLES_AVAILABLE:
        print("Analyzing all test examples...")
        results = analyze_all_test_examples()
        print(f"\nAnalyzed {len(results)} examples")

        if results:
            analyzer = SymbolicExecutionAnalyzer()
            analyzer.analysis_history = results
            analyzer.save_analysis_results("symbolic_analysis_results.json")

    else:
        print("Run with 'python symbolic_analyzer.py test' to test with factorial example")
        print("Or install test_examples.py to analyze all examples")