"""
Use Case: Python Code Generation with Bug Analysis

This module demonstrates a practical use case for the LLM probability analysis system.
It focuses on Python code generation tasks where the model's uncertainty might
correlate with potential bugs or problematic code sections.
"""

import subprocess
import tempfile
import os
import sys
from typing import List, Dict, Tuple, Any
from LLM import QwenProbabilityAnalyzer, TokenAnalysis
from visualizer import TokenVisualizer, TokenVisualizationMode


class CodeGenerationUseCase:
    """
    A comprehensive use case for analyzing LLM behavior during Python code generation.
    
    This class implements a scenario where we ask the LLM to generate Python code
    and then analyze which parts the model was uncertain about, potentially
    identifying areas prone to bugs.
    """
    
    def __init__(self):
        """Initialize the use case with analyzer and visualizer."""
        self.analyzer = QwenProbabilityAnalyzer()
        self.visualizer = TokenVisualizer()
        
        # Define the coding prompt and expected solution
        self.prompt = """Write a Python function that implements a binary search algorithm to find the position of a target element in a sorted array. The function should:

1. Take a sorted array and a target value as parameters
2. Return the index of the target if found, or -1 if not found
3. Use proper binary search technique (divide and conquer)
4. Handle edge cases appropriately
5. Include proper error handling for invalid inputs

Please write clean, well-commented code with proper variable names."""

        # Reference correct implementation
        self.correct_solution = '''def binary_search(arr, target):
    """
    Perform binary search on a sorted array to find the target element.
    
    Args:
        arr (list): A sorted list of comparable elements
        target: The element to search for
        
    Returns:
        int: Index of target if found, -1 otherwise
        
    Raises:
        TypeError: If arr is not a list or None
        ValueError: If arr is not sorted (in ascending order)
    """
    # Input validation
    if arr is None:
        raise TypeError("Array cannot be None")
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    if len(arr) == 0:
        return -1
    
    # Check if array is sorted
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            raise ValueError("Array must be sorted in ascending order")
    
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid potential overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1'''

        # Test cases for validation
        self.test_cases = [
            # (array, target, expected_result, description)
            ([1, 2, 3, 4, 5], 3, 2, "Target in middle of array"),
            ([1, 2, 3, 4, 5], 1, 0, "Target at beginning"),
            ([1, 2, 3, 4, 5], 5, 4, "Target at end"),
            ([1, 2, 3, 4, 5], 6, -1, "Target not in array (larger)"),
            ([1, 2, 3, 4, 5], 0, -1, "Target not in array (smaller)"),
            ([10], 10, 0, "Single element array - found"),
            ([10], 5, -1, "Single element array - not found"),
            ([], 5, -1, "Empty array"),
            ([1, 3, 5, 7, 9, 11, 13], 7, 3, "Odd length array"),
            ([2, 4, 6, 8, 10, 12], 8, 3, "Even length array"),
            ([1, 1, 1, 1, 1], 1, 0, "Array with duplicates"),  # Returns first occurrence
            ([-5, -2, 0, 3, 7], -2, 1, "Array with negative numbers"),
            ([1.1, 2.2, 3.3, 4.4], 3.3, 2, "Array with float numbers"),
        ]
        
        # Error test cases
        self.error_test_cases = [
            (None, 5, TypeError, "None array should raise TypeError"),
            ("not a list", 5, TypeError, "String instead of list should raise TypeError"),
            ([3, 1, 2], 2, ValueError, "Unsorted array should raise ValueError"),
            ([5, 4, 3, 2, 1], 3, ValueError, "Reverse sorted array should raise ValueError"),
        ]
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete use case analysis including generation, testing, and visualization.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("="*60)
        print("PYTHON CODE GENERATION USE CASE ANALYSIS")
        print("="*60)
        print(f"Prompt: {self.prompt}")
        print("\n" + "="*60)
        print("GENERATING CODE WITH PROBABILITY ANALYSIS...")
        print("="*60)
        
        # Generate code with analysis
        generated_code, token_analyses = self.analyzer.generate_with_analysis(
            prompt=self.prompt,
            max_new_tokens=300,
            temperature=0.1,  # Lower temperature for more deterministic code generation
            do_sample=True
        )
        
        print("\nGenerated Code:")
        print("-" * 40)
        print(generated_code)
        
        # Analyze the generation
        generation_stats = self.analyzer.get_generation_stats()
        
        print("\n" + "="*60)
        print("ANALYZING GENERATED CODE...")
        print("="*60)
        
        # Test the generated code
        test_results = self._test_generated_code(generated_code)
        
        # Identify problematic regions
        low_confidence_regions = self.visualizer.identify_low_confidence_regions(
            token_analyses, threshold_percentile=20
        )
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # HTML visualization with different modes
        visualizations = {}
        for mode in [TokenVisualizationMode.PROBABILITY, TokenVisualizationMode.RANK, 
                    TokenVisualizationMode.ENTROPY, TokenVisualizationMode.SURPRISAL]:
            visualizations[mode] = self.visualizer.create_html_visualization(
                token_analyses, mode=mode, 
                title=f"Code Generation Analysis - {mode.title()}"
            )
        
        # Generate comprehensive report
        analysis_report = self.visualizer.generate_analysis_report(token_analyses)
        
        # Correlate low confidence regions with code issues
        correlation_analysis = self._correlate_confidence_with_errors(
            token_analyses, test_results, low_confidence_regions
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return {
            "prompt": self.prompt,
            "generated_code": generated_code,
            "correct_solution": self.correct_solution,
            "token_analyses": token_analyses,
            "generation_stats": generation_stats,
            "test_results": test_results,
            "low_confidence_regions": low_confidence_regions,
            "visualizations": visualizations,
            "analysis_report": analysis_report,
            "correlation_analysis": correlation_analysis
        }
    
    def _test_generated_code(self, generated_code: str) -> Dict[str, Any]:
        """
        Test the generated code against the test cases.
        
        Args:
            generated_code: The code generated by the LLM
            
        Returns:
            Dictionary with test results
        """
        print("Testing generated code against test cases...")
        
        # Create a temporary file with the generated code
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(generated_code)
                temp_file = f.name
            
            # Import the function dynamically
            spec = __import__('importlib.util').util.spec_from_file_location("generated_module", temp_file)
            module = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to find the function (common names)
            function_names = ['binary_search', 'binarySearch', 'search', 'find']
            test_function = None
            
            for name in function_names:
                if hasattr(module, name):
                    test_function = getattr(module, name)
                    break
            
            if test_function is None:
                return {
                    "success": False,
                    "error": "Could not find binary search function in generated code",
                    "passed_tests": 0,
                    "total_tests": len(self.test_cases),
                    "test_details": []
                }
            
            # Run test cases
            passed_tests = 0
            test_details = []
            
            for i, (arr, target, expected, description) in enumerate(self.test_cases):
                try:
                    result = test_function(arr.copy() if arr else arr, target)
                    passed = (result == expected)
                    passed_tests += passed
                    
                    test_details.append({
                        "test_id": i,
                        "description": description,
                        "input": (arr, target),
                        "expected": expected,
                        "actual": result,
                        "passed": passed,
                        "error": None
                    })
                    
                except Exception as e:
                    test_details.append({
                        "test_id": i,
                        "description": description,
                        "input": (arr, target),
                        "expected": expected,
                        "actual": None,
                        "passed": False,
                        "error": str(e)
                    })
            
            # Test error cases
            error_tests_passed = 0
            for i, (arr, target, expected_error, description) in enumerate(self.error_test_cases):
                try:
                    result = test_function(arr, target)
                    # If we get here, no exception was raised when one was expected
                    test_details.append({
                        "test_id": len(self.test_cases) + i,
                        "description": description,
                        "input": (arr, target),
                        "expected": f"Should raise {expected_error.__name__}",
                        "actual": f"Returned {result} instead of raising exception",
                        "passed": False,
                        "error": None
                    })
                except expected_error:
                    # Correct exception was raised
                    error_tests_passed += 1
                    test_details.append({
                        "test_id": len(self.test_cases) + i,
                        "description": description,
                        "input": (arr, target),
                        "expected": f"Should raise {expected_error.__name__}",
                        "actual": f"Correctly raised {expected_error.__name__}",
                        "passed": True,
                        "error": None
                    })
                except Exception as e:
                    # Wrong exception was raised
                    test_details.append({
                        "test_id": len(self.test_cases) + i,
                        "description": description,
                        "input": (arr, target),
                        "expected": f"Should raise {expected_error.__name__}",
                        "actual": f"Raised {type(e).__name__}: {str(e)}",
                        "passed": False,
                        "error": str(e)
                    })
            
            total_passed = passed_tests + error_tests_passed
            total_tests = len(self.test_cases) + len(self.error_test_cases)
            
            return {
                "success": True,
                "function_name": test_function.__name__,
                "passed_tests": total_passed,
                "total_tests": total_tests,
                "pass_rate": total_passed / total_tests,
                "test_details": test_details
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute generated code: {str(e)}",
                "passed_tests": 0,
                "total_tests": len(self.test_cases) + len(self.error_test_cases),
                "test_details": []
            }
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _correlate_confidence_with_errors(self, 
                                        token_analyses: List[TokenAnalysis],
                                        test_results: Dict[str, Any],
                                        low_confidence_regions: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Analyze correlation between model confidence and code correctness.
        
        Args:
            token_analyses: List of token analyses from generation
            test_results: Results from testing the generated code
            low_confidence_regions: Regions where model had low confidence
            
        Returns:
            Dictionary with correlation analysis
        """
        print("Analyzing correlation between model confidence and code correctness...")
        
        # Extract metrics for analysis
        probabilities = [analysis.probability for analysis in token_analyses]
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0
        min_probability = min(probabilities) if probabilities else 0
        
        # Analyze test performance
        pass_rate = test_results.get("pass_rate", 0)
        failed_tests = [t for t in test_results.get("test_details", []) if not t["passed"]]
        
        # Hypothesis: Lower model confidence correlates with bugs
        confidence_score = avg_probability
        error_count = len(failed_tests)
        
        # Generate insights
        insights = []
        
        if pass_rate < 0.5:
            insights.append("❌ Code has significant issues (pass rate < 50%)")
        elif pass_rate < 0.8:
            insights.append("⚠️ Code has some issues (pass rate < 80%)")
        else:
            insights.append("✅ Code performs well (pass rate ≥ 80%)")
        
        if confidence_score < 0.3:
            insights.append("🔴 Very low model confidence - high likelihood of errors")
        elif confidence_score < 0.5:
            insights.append("🟡 Moderate model confidence - some uncertainty")
        else:
            insights.append("🟢 High model confidence")
        
        if len(low_confidence_regions) > 3:
            insights.append(f"⚠️ Many low-confidence regions ({len(low_confidence_regions)}) detected")
        elif len(low_confidence_regions) > 0:
            insights.append(f"⚠️ {len(low_confidence_regions)} low-confidence region(s) detected")
        else:
            insights.append("✅ No significant low-confidence regions")
        
        # Correlation hypothesis
        if pass_rate < 0.7 and confidence_score < 0.4:
            correlation_hypothesis = "STRONG: Low confidence strongly correlates with poor performance"
        elif pass_rate < 0.8 and confidence_score < 0.5:
            correlation_hypothesis = "MODERATE: Some correlation between confidence and performance"
        elif pass_rate > 0.8 and confidence_score > 0.6:
            correlation_hypothesis = "POSITIVE: High confidence correlates with good performance"
        else:
            correlation_hypothesis = "UNCLEAR: No clear correlation pattern observed"
        
        return {
            "confidence_score": confidence_score,
            "min_confidence": min_probability,
            "pass_rate": pass_rate,
            "error_count": error_count,
            "low_confidence_regions_count": len(low_confidence_regions),
            "insights": insights,
            "correlation_hypothesis": correlation_hypothesis,
            "failed_tests": failed_tests
        }
    
    def save_complete_analysis(self, results: Dict[str, Any], filename: str = "complete_analysis.html"):
        """
        Save the complete analysis as an HTML report.
        
        Args:
            results: Results dictionary from run_complete_analysis
            filename: Output filename
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Code Generation Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 30px 0; }}
                .code {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; 
                         font-family: monospace; overflow-x: auto; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; text-align: center; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .test-passed {{ background-color: #d4edda; }}
                .test-failed {{ background-color: #f8d7da; }}
                .insight {{ margin: 10px 0; padding: 10px; background-color: #fff3cd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 LLM Code Generation Analysis Report</h1>
                <p><strong>Model:</strong> Qwen 2.5 Coder 7B Instruct</p>
                <p><strong>Task:</strong> Binary Search Implementation</p>
                <p><strong>Analysis Date:</strong> {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📝 Prompt</h2>
                <div class="code">{results['prompt']}</div>
            </div>
            
            <div class="section">
                <h2>🤖 Generated Code</h2>
                <div class="code">{results['generated_code']}</div>
            </div>
            
            <div class="section">
                <h2>📊 Generation Statistics</h2>
                <div class="stats">
        """
        
        for key, value in results['generation_stats'].items():
            html_content += f"""
                    <div class="stat-box">
                        <h4>{key.replace('_', ' ').title()}</h4>
                        <p>{value:.4f if isinstance(value, float) else value}</p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🧪 Test Results</h2>
        """
        
        if results['test_results']['success']:
            html_content += f"""
                <p><strong>Pass Rate:</strong> {results['test_results']['pass_rate']:.1%} 
                ({results['test_results']['passed_tests']}/{results['test_results']['total_tests']} tests passed)</p>
            """
            
            for test in results['test_results']['test_details']:
                status_class = "test-passed" if test['passed'] else "test-failed"
                status_icon = "✅" if test['passed'] else "❌"
                html_content += f"""
                <div class="test-result {status_class}">
                    <p>{status_icon} <strong>{test['description']}</strong></p>
                    <p>Input: {test['input']}, Expected: {test['expected']}, Actual: {test['actual']}</p>
                    {f"<p>Error: {test['error']}</p>" if test['error'] else ""}
                </div>
                """
        else:
            html_content += f"""
                <div class="test-result test-failed">
                    <p>❌ <strong>Testing Failed:</strong> {results['test_results']['error']}</p>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>🔍 Confidence Analysis</h2>
        """
        
        for insight in results['correlation_analysis']['insights']:
            html_content += f'<div class="insight">{insight}</div>'
        
        html_content += f"""
                <div class="insight">
                    <strong>Correlation Hypothesis:</strong> {results['correlation_analysis']['correlation_hypothesis']}
                </div>
            </div>
            
            <div class="section">
                <h2>🎨 Token Visualizations</h2>
                <h3>Probability-based Coloring</h3>
                {results['visualizations'][TokenVisualizationMode.PROBABILITY]}
                
                <h3>Rank-based Coloring</h3>
                {results['visualizations'][TokenVisualizationMode.RANK]}
                
                <h3>Entropy-based Coloring</h3>
                {results['visualizations'][TokenVisualizationMode.ENTROPY]}
            </div>
            
            <div class="section">
                <h2>📈 Detailed Analysis Report</h2>
                {results['analysis_report']}
            </div>
            
            <div class="section">
                <h2>✅ Reference Solution</h2>
                <div class="code">{results['correct_solution']}</div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Complete analysis saved to {filename}")


if __name__ == "__main__":
    # Run the complete use case
    use_case = CodeGenerationUseCase()
    results = use_case.run_complete_analysis()
    
    # Save the complete analysis
    use_case.save_complete_analysis(results, "binary_search_analysis.html")
    
    print("\n" + "="*60)
    print("USE CASE SUMMARY")
    print("="*60)
    print(f"Pass Rate: {results['test_results'].get('pass_rate', 0):.1%}")
    print(f"Confidence Score: {results['correlation_analysis']['confidence_score']:.4f}")
    print(f"Low Confidence Regions: {results['correlation_analysis']['low_confidence_regions_count']}")
    print(f"Correlation: {results['correlation_analysis']['correlation_hypothesis']}")
    print("\nDetailed report saved to 'binary_search_analysis.html'")