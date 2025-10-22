#!/usr/bin/env python3
"""
HumanEval Plus Runner with Token Probability Analysis

This script:
1. Loads HumanEval Plus dataset from HuggingFace
2. For each problem, generates solution with LLM (saving logits)
3. Tests generated code using HumanEval Plus test cases
4. If tests fail, creates HTML visualization comparing:
   - Forced generation (canonical solution)
   - Free generation (model's solution with saved logits)
5. Shows "Failed X/Y test cases" in HTML
6. Generates hierarchical index pages
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from datasets import load_dataset
import subprocess
import tempfile
import signal

from LLM import QwenProbabilityAnalyzer, TokenAnalysis
from forced_generation_analyzer import ForcedGenerationAnalyzer, ForcedGenerationResult, ForcedTokenAnalysis
from forced_visualizer import ForcedGenerationVisualizer, ForcedVisualizationMode
from codet5_validator import CodeT5Validator
from nomic_validator import NomicCodeValidator


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Code execution timed out")


class HumanEvalRunner:
    """Runner for HumanEval Plus benchmark with token probability analysis."""

    def __init__(self, output_dir: str = "humaneval_analysis", timeout: int = 10):
        """
        Initialize the HumanEval runner.

        Args:
            output_dir: Base directory for all outputs
            timeout: Timeout in seconds for test execution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.dataset = None
        self.visualizer = ForcedGenerationVisualizer()
        self.codet5_validator = None  # Lazy loading to save memory
        self.nomic_validator = None  # Lazy loading to save memory

    def load_dataset(self):
        """Load HumanEval Plus dataset from HuggingFace."""
        print("Loading HumanEval Plus dataset from HuggingFace...")
        self.dataset = load_dataset("evalplus/humanevalplus", split="test")
        print(f"Loaded {len(self.dataset)} problems")
        return self.dataset

    def get_codet5_validator(self):
        """Lazy-load CodeT5 validator to save memory."""
        if self.codet5_validator is None:
            print("Loading CodeT5 validator...")
            self.codet5_validator = CodeT5Validator(model_name="Salesforce/codet5-base")
        return self.codet5_validator

    def validate_tokens_with_codet5(self, code: str, token_analyses: List[TokenAnalysis]) -> List[TokenAnalysis]:
        """
        Validate tokens using CodeT5 and update TokenAnalysis objects.

        Args:
            code: Full generated code
            token_analyses: List of TokenAnalysis objects to validate

        Returns:
            Updated list of TokenAnalysis objects with CodeT5 metrics
        """
        validator = self.get_codet5_validator()

        print(f"Validating {len(token_analyses)} tokens with CodeT5...")

        # Validate each token
        for analysis in token_analyses:
            try:
                result = validator.validate_token(code, analysis.token, analysis.position)

                # Update analysis with CodeT5 metrics
                analysis.codet5_validation_score = result["validation_score"]
                analysis.codet5_alternatives = result["alternatives"]
                analysis.codet5_predicted_token = result["predicted_token"]
                analysis.codet5_matches = result["matches_prediction"]
            except Exception as e:
                print(f"Warning: CodeT5 validation failed for token at position {analysis.position}: {e}")
                # Set default values on error
                analysis.codet5_validation_score = 0.0
                analysis.codet5_alternatives = []
                analysis.codet5_predicted_token = ""
                analysis.codet5_matches = False

        return token_analyses

    def validate_forced_tokens_with_codet5(self, code: str, forced_result: ForcedGenerationResult) -> ForcedGenerationResult:
        """
        Validate forced generation tokens using CodeT5.

        Args:
            code: Full generated code
            forced_result: ForcedGenerationResult to validate

        Returns:
            Updated ForcedGenerationResult with CodeT5 metrics
        """
        validator = self.get_codet5_validator()

        print(f"Validating {len(forced_result.token_analyses)} forced tokens with CodeT5...")

        # Validate each forced token
        for analysis in forced_result.token_analyses:
            try:
                result = validator.validate_token(code, analysis.token, analysis.position)

                # Update analysis with CodeT5 metrics
                analysis.codet5_validation_score = result["validation_score"]
                analysis.codet5_alternatives = result["alternatives"]
                analysis.codet5_predicted_token = result["predicted_token"]
                analysis.codet5_matches = result["matches_prediction"]
            except Exception as e:
                print(f"Warning: CodeT5 validation failed for forced token at position {analysis.position}: {e}")
                # Set default values on error
                analysis.codet5_validation_score = 0.0
                analysis.codet5_alternatives = []
                analysis.codet5_predicted_token = ""
                analysis.codet5_matches = False

        return forced_result

    def get_nomic_validator(self):
        """Lazy-load Nomic validator to save memory."""
        if self.nomic_validator is None:
            print("Loading Nomic validator...")
            self.nomic_validator = NomicCodeValidator(model_name="nomic-ai/nomic-embed-code")
        return self.nomic_validator

    def validate_tokens_with_nomic(self, code: str, token_analyses: List[TokenAnalysis]) -> List[TokenAnalysis]:
        """
        Validate tokens using Nomic-embed-code and update TokenAnalysis objects.

        Args:
            code: Full generated code
            token_analyses: List of TokenAnalysis objects to validate

        Returns:
            Updated list of TokenAnalysis objects with Nomic metrics
        """
        validator = self.get_nomic_validator()

        print(f"Validating {len(token_analyses)} tokens with Nomic-embed-code...")

        # Validate each token
        for analysis in token_analyses:
            try:
                result = validator.validate_token(code, analysis.token, analysis.position)

                # Update analysis with Nomic metrics
                analysis.nomic_coherence_score = result["coherence_score"]
                analysis.nomic_similarity_drop = result["similarity_drop"]
                analysis.nomic_context_similarity = result["context_similarity"]
            except Exception as e:
                print(f"Warning: Nomic validation failed for token at position {analysis.position}: {e}")
                # Set default values on error
                analysis.nomic_coherence_score = 0.5
                analysis.nomic_similarity_drop = 0.0
                analysis.nomic_context_similarity = 0.5

        return token_analyses

    def validate_forced_tokens_with_nomic(self, code: str, forced_result: ForcedGenerationResult) -> ForcedGenerationResult:
        """
        Validate forced generation tokens using Nomic-embed-code.

        Args:
            code: Full generated code
            forced_result: ForcedGenerationResult to validate

        Returns:
            Updated ForcedGenerationResult with Nomic metrics
        """
        validator = self.get_nomic_validator()

        print(f"Validating {len(forced_result.token_analyses)} forced tokens with Nomic-embed-code...")

        # Validate each forced token
        for analysis in forced_result.token_analyses:
            try:
                result = validator.validate_token(code, analysis.token, analysis.position)

                # Update analysis with Nomic metrics
                analysis.nomic_coherence_score = result["coherence_score"]
                analysis.nomic_similarity_drop = result["similarity_drop"]
                analysis.nomic_context_similarity = result["context_similarity"]
            except Exception as e:
                print(f"Warning: Nomic validation failed for forced token at position {analysis.position}: {e}")
                # Set default values on error
                analysis.nomic_coherence_score = 0.5
                analysis.nomic_similarity_drop = 0.0
                analysis.nomic_context_similarity = 0.5

        return forced_result

    def extract_function_from_generation(self, generated_text: str, entry_point: str) -> str:
        """
        Extract function code from LLM generation.

        Args:
            generated_text: Full text generated by LLM
            entry_point: Function name to extract

        Returns:
            Extracted function code
        """
        # Try to find the function definition
        lines = generated_text.split('\n')
        function_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            # Start capturing when we find "def entry_point"
            if f"def {entry_point}" in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                function_lines.append(line)
            elif in_function:
                # Check if we're still in the function (based on indentation)
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                    # We've exited the function
                    break
                function_lines.append(line)

        if function_lines:
            return '\n'.join(function_lines)

        # Fallback: return the entire generation
        return generated_text

    def run_tests_safe(self,
                      problem_prompt: str,
                      generated_code: str,
                      test_code: str,
                      entry_point: str) -> Dict[str, Any]:
        """
        Run HumanEval test cases on generated code safely.

        Args:
            problem_prompt: Original problem prompt
            generated_code: Code generated by LLM
            test_code: Test code from HumanEval Plus dataset
            entry_point: Function name to test

        Returns:
            Dictionary with test results: {passed, failed, total, details, error}
        """
        # Extract just the function from generated code
        function_code = self.extract_function_from_generation(generated_code, entry_point)

        # Create test script
        test_script = f"""
{function_code}

{test_code}

# Run the check function
try:
    check({entry_point})
    print("ALL_TESTS_PASSED")
except AssertionError as e:
    print(f"TEST_FAILED: {{e}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""

        try:
            # Write test script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_file = f.name

            # Run with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            # Clean up
            os.unlink(temp_file)

            # Parse output
            output = result.stdout + result.stderr

            if "ALL_TESTS_PASSED" in output:
                return {
                    "passed": 1,
                    "failed": 0,
                    "total": 1,
                    "details": ["All tests passed"],
                    "error": None
                }
            elif "TEST_FAILED" in output:
                error_msg = output.split("TEST_FAILED:")[-1].strip()
                return {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "details": [f"Test failed: {error_msg}"],
                    "error": error_msg
                }
            else:
                error_msg = output.strip()
                return {
                    "passed": 0,
                    "failed": 1,
                    "total": 1,
                    "details": [f"Execution error: {error_msg}"],
                    "error": error_msg
                }

        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 1,
                "total": 1,
                "details": ["Timeout"],
                "error": f"Execution timeout ({self.timeout}s)"
            }
        except Exception as e:
            return {
                "passed": 0,
                "failed": 1,
                "total": 1,
                "details": [f"Exception: {str(e)}"],
                "error": str(e)
            }

    def process_problem(self,
                       problem: Dict[str, Any],
                       model_name: str,
                       analyzer: QwenProbabilityAnalyzer,
                       forced_analyzer: ForcedGenerationAnalyzer,
                       model_dir: Path) -> Dict[str, Any]:
        """
        Process a single HumanEval problem.

        Args:
            problem: Problem from HumanEval Plus dataset
            model_name: Name of the model being tested
            analyzer: Probability analyzer for free generation
            forced_analyzer: Analyzer for forced generation
            model_dir: Directory for model outputs

        Returns:
            Dictionary with processing results
        """
        task_id = problem["task_id"].replace("/", "_")
        problem_dir = model_dir / task_id
        problem_dir.mkdir(exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Processing: {problem['task_id']}")
        print(f"{'='*80}")

        result = {
            "task_id": problem["task_id"],
            "status": "pending",
            "generated_code": None,
            "test_results": None,
            "visualization_created": False,
            "error": None
        }

        try:
            # 1. Generate solution with free generation (save logits)
            # Using output_scores=True from HuggingFace transformers
            print("Generating solution with output_scores=True...")
            generated_text, token_analyses = analyzer.generate_with_output_scores(
                prompt=problem["prompt"],
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True
            )

            # Post-process: remove markdown code blocks
            clean_code = generated_text.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[9:]  # Remove ```python
            if clean_code.startswith("```"):
                clean_code = clean_code[3:]  # Remove ```
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3]  # Remove trailing ```
            clean_code = clean_code.strip()

            result["generated_code"] = clean_code

            # Validate tokens with CodeT5
            print("Validating generated tokens with CodeT5...")
            token_analyses = self.validate_tokens_with_codet5(clean_code, token_analyses)

            # Validate tokens with Nomic-embed-code
            print("Validating generated tokens with Nomic-embed-code...")
            token_analyses = self.validate_tokens_with_nomic(clean_code, token_analyses)

            # Save generation results
            generation_file = problem_dir / "generated_solution.json"
            with open(generation_file, 'w') as f:
                json.dump({
                    "task_id": problem["task_id"],
                    "prompt": problem["prompt"],
                    "generated_code": generated_text,
                    "token_count": len(token_analyses),
                    "tokens": [
                        {
                            "token": a.token,
                            "position": a.position,
                            "probability": float(a.probability),
                            "logit": float(a.logit),
                            "rank": a.rank
                        }
                        for a in token_analyses
                    ]
                }, f, indent=2)

            # 2. Run tests
            print("Running tests...")
            test_results = self.run_tests_safe(
                problem["prompt"],
                clean_code,
                problem["test"],
                problem["entry_point"]
            )

            result["test_results"] = test_results

            # Save test results
            test_file = problem_dir / "test_results.json"
            with open(test_file, 'w') as f:
                json.dump(test_results, f, indent=2)

            # 3. If failed, create visualization
            if test_results["failed"] > 0:
                print(f"Tests failed ({test_results['failed']}/{test_results['total']}). Creating visualization...")

                # Forced generation on canonical solution
                print("Running forced generation on canonical solution...")
                forced_result = forced_analyzer.force_generation_with_logits(
                    problem["prompt"],
                    problem["canonical_solution"],
                    verbose=False
                )

                # Validate forced tokens with CodeT5
                print("Validating forced tokens with CodeT5...")
                forced_result = self.validate_forced_tokens_with_codet5(
                    problem["canonical_solution"],
                    forced_result
                )

                # Validate forced tokens with Nomic-embed-code
                print("Validating forced tokens with Nomic-embed-code...")
                forced_result = self.validate_forced_tokens_with_nomic(
                    problem["canonical_solution"],
                    forced_result
                )

                # Save forced generation results
                forced_file = problem_dir / "forced_canonical.json"
                forced_analyzer.save_analysis(forced_result, str(forced_file))

                # Create comparison visualization with test results
                print("Creating comparison visualization...")
                self.create_comparison_with_test_results(
                    forced_result=forced_result,
                    free_analyses=token_analyses,
                    free_code=clean_code,
                    test_results=test_results,
                    problem=problem,
                    output_file=problem_dir / "comparison.html"
                )

                result["visualization_created"] = True
                print("✅ Visualization created")
            else:
                print(f"✅ All tests passed ({test_results['passed']}/{test_results['total']})")

            result["status"] = "success"

        except Exception as e:
            print(f"❌ Error processing problem: {e}")
            traceback.print_exc()
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def create_comparison_with_test_results(self,
                                          forced_result: ForcedGenerationResult,
                                          free_analyses: List[TokenAnalysis],
                                          free_code: str,
                                          test_results: Dict[str, Any],
                                          problem: Dict[str, Any],
                                          output_file: Path):
        """
        Create HTML comparison visualization with test results banner.

        Args:
            forced_result: Forced generation analysis result
            free_analyses: Free generation token analyses
            free_code: Generated code from free generation
            test_results: Test execution results
            problem: Original problem from dataset
            output_file: Output HTML file path
        """
        # Create test results banner
        passed = test_results["passed"]
        failed = test_results["failed"]
        total = test_results["total"]
        pass_rate = (passed / total * 100) if total > 0 else 0

        test_banner = f"""
        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                    color: white; padding: 25px; border-radius: 10px; margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <h2 style='margin: 0 0 15px 0; color: white; font-size: 28px;'>
                ❌ Test Results: Failed {failed}/{total} test cases
            </h2>
            <div style='background: rgba(255,255,255,0.2); border-radius: 10px; padding: 15px;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 10px;'>
                    <span style='font-size: 18px;'><strong>Passed:</strong> {passed}</span>
                    <span style='font-size: 18px;'><strong>Failed:</strong> {failed}</span>
                    <span style='font-size: 18px;'><strong>Pass Rate:</strong> {pass_rate:.1f}%</span>
                </div>
                <div style='height: 30px; background: #f8f9fa; border-radius: 15px; overflow: hidden; border: 2px solid white;'>
                    <div style='height: 100%; background: linear-gradient(90deg, #51cf66 0%, #40c057 100%);
                                width: {pass_rate}%; transition: width 0.5s ease;'></div>
                </div>
            </div>
            <div style='margin-top: 15px; font-size: 14px; opacity: 0.9;'>
                <strong>Error Details:</strong> {test_results.get('error', 'See test details below')}
            </div>
        </div>
        """

        # Create problem info section
        problem_info = f"""
        <div style='background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #339af0;'>
            <h3 style='margin-top: 0; color: #339af0;'>📋 Problem Information</h3>
            <p><strong>Task ID:</strong> {problem['task_id']}</p>
            <p><strong>Entry Point:</strong> <code>{problem['entry_point']}</code></p>
            <div style='margin-top: 15px;'>
                <strong>Problem Description:</strong>
                <pre style='background: white; padding: 15px; border-radius: 5px; overflow-x: auto;'>{problem['prompt']}</pre>
            </div>
        </div>
        """

        # Create comparison visualization for multiple modes
        comparison_html_parts = []

        for mode in [ForcedVisualizationMode.FORCED_LOGITS,
                     ForcedVisualizationMode.FORCED_PROBABILITY,
                     ForcedVisualizationMode.FORCED_LOG_PROBABILITY,
                     ForcedVisualizationMode.FORCED_CODET5_VALIDATION,
                     ForcedVisualizationMode.FORCED_NOMIC_COHERENCE]:

            # Get visualization scheme
            scheme = self.visualizer.color_schemes[mode]

            # Create forced visualization
            forced_viz = self.visualizer.create_forced_logits_visualization(
                forced_result,
                mode=mode,
                title=f"🟢 Canonical Solution (Forced): {scheme['label']}"
            )

            # Create free visualization (convert TokenAnalysis to format compatible with visualizer)
            # For now, create a simple HTML for free generation
            free_viz = self.create_free_generation_visualization(
                free_analyses,
                free_code,
                mode,
                f"🔴 Generated Solution (Free): {scheme['label']}"
            )

            comparison_html_parts.append(f"""
            <div style='margin: 40px 0;'>
                <h2 style='text-align: center; color: #495057; border-bottom: 3px solid #dee2e6; padding-bottom: 10px;'>
                    Comparison: {scheme['label']}
                </h2>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;'>
                    <div style='border: 3px solid #51cf66; border-radius: 10px; padding: 20px; background: #f8fff8;'>
                        {forced_viz}
                    </div>
                    <div style='border: 3px solid #ff6b6b; border-radius: 10px; padding: 20px; background: #fff5f5;'>
                        {free_viz}
                    </div>
                </div>
            </div>
            """)

        # Combine all parts
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HumanEval Analysis: {problem['task_id']}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }}
        .container {{ max-width: 1800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 4px solid #3498db; padding-bottom: 15px; }}
        pre {{ overflow-x: auto; margin: 0; }}
        code {{ font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace; font-size: 14px; line-height: 1.6; }}
        .code-block {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 10px 0; }}
        .token-code {{ font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace; font-size: 15px; line-height: 2.0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 HumanEval Plus Analysis: {problem['task_id']}</h1>

        {test_banner}
        {problem_info}
        {''.join(comparison_html_parts)}

        <div style='margin-top: 40px; padding: 20px; background: #e9ecef; border-radius: 8px;'>
            <h3>💡 About This Analysis</h3>
            <p><strong>Forced Generation (Left):</strong> The model was forced to generate the canonical solution token-by-token.
            The logits show how confident the model was about each forced token.</p>
            <p><strong>Free Generation (Right):</strong> The model freely generated its own solution.
            The logits show the model's natural confidence in its choices.</p>
            <p><strong>Hypothesis:</strong> If the model's free generation has lower logits/probabilities in buggy regions,
            it suggests the model has implicit knowledge of code quality.</p>
        </div>
    </div>
    <script>
        // Initialize syntax highlighting
        hljs.highlightAll();
    </script>
</body>
</html>"""

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)

        print(f"Comparison visualization saved: {output_file}")

    def create_free_generation_visualization(self,
                                            analyses: List[TokenAnalysis],
                                            generated_code: str,
                                            mode: str,
                                            title: str) -> str:
        """Create HTML visualization for free generation."""
        from visualizer import TokenVisualizer, TokenVisualizationMode

        visualizer = TokenVisualizer()

        # Map forced modes to regular modes
        mode_mapping = {
            ForcedVisualizationMode.FORCED_LOGITS: TokenVisualizationMode.LOGITS,
            ForcedVisualizationMode.FORCED_PROBABILITY: TokenVisualizationMode.PROBABILITY,
            ForcedVisualizationMode.FORCED_LOG_PROBABILITY: TokenVisualizationMode.LOG_PROBABILITY,
            ForcedVisualizationMode.FORCED_RANK: TokenVisualizationMode.RANK,
            ForcedVisualizationMode.FORCED_SURPRISAL: TokenVisualizationMode.SURPRISAL,
            ForcedVisualizationMode.FORCED_CODET5_VALIDATION: TokenVisualizationMode.CODET5_VALIDATION,
            ForcedVisualizationMode.FORCED_NOMIC_COHERENCE: TokenVisualizationMode.NOMIC_COHERENCE,
        }

        viz_mode = mode_mapping.get(mode, TokenVisualizationMode.PROBABILITY)

        return visualizer.create_html_visualization(
            analyses,
            mode=viz_mode,
            title=title
        )

    def generate_model_index(self, model_dir: Path, model_name: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate model-specific index page.

        Args:
            model_dir: Model directory path
            model_name: Model name
            results: List of problem results

        Returns:
            Path to generated index.html
        """
        # Calculate stats
        total = len(results)
        passed = len([r for r in results if r.get("test_results", {}).get("failed", 1) == 0])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HumanEval Analysis - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 4px solid #667eea; padding-bottom: 15px; }}
        .stats {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
        .stat-item {{ text-align: center; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        .problems-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin: 30px 0; }}
        .problem-card {{ border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; transition: all 0.3s ease; }}
        .problem-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .problem-card.passed {{ border-left: 6px solid #51cf66; background: #f8fff8; }}
        .problem-card.failed {{ border-left: 6px solid #ff6b6b; background: #fff5f5; }}
        .problem-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }}
        .test-result {{ font-size: 16px; margin: 10px 0; padding: 8px; border-radius: 5px; }}
        .test-result.pass {{ background: #d3f9d8; color: #2b8a3e; }}
        .test-result.fail {{ background: #ffe3e3; color: #c92a2a; }}
        .view-link {{ display: inline-block; padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 25px; margin-top: 10px; transition: background 0.3s ease; }}
        .view-link:hover {{ background: #764ba2; }}
        .back-link {{ display: inline-block; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px; margin-bottom: 20px; }}
        .back-link:hover {{ background: #5a6268; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">← Back to Main Index</a>
        <h1>🤖 {model_name}</h1>

        <div class="stats">
            <h2 style="margin-top: 0; color: white;">📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Problems Tested</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{passed}</div>
                    <div class="stat-label">Passed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{failed}</div>
                    <div class="stat-label">Failed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{pass_rate:.1f}%</div>
                    <div class="stat-label">Pass Rate</div>
                </div>
            </div>
        </div>

        <h2>📝 Problems</h2>
        <div class="problems-grid">"""

        # Add problem cards
        for result in results:
            task_id = result["task_id"]
            task_id_safe = task_id.replace("/", "_")
            test_results = result.get("test_results", {})
            passed_tests = test_results.get("passed", 0)
            failed_tests = test_results.get("failed", 1)
            total_tests = test_results.get("total", 1)

            card_class = "passed" if failed_tests == 0 else "failed"
            result_class = "pass" if failed_tests == 0 else "fail"
            icon = "✅" if failed_tests == 0 else "❌"
            test_text = f"{icon} Passed {passed_tests}/{total_tests} tests" if failed_tests == 0 else f"{icon} Failed {failed_tests}/{total_tests} tests"

            html += f"""
            <div class="problem-card {card_class}">
                <div class="problem-title">{task_id}</div>
                <div class="test-result {result_class}">{test_text}</div>"""

            if failed_tests > 0:
                html += f"""
                <a href="{task_id_safe}/comparison.html" class="view-link" target="_blank">View Analysis →</a>"""

            html += """
            </div>"""

        html += """
        </div>
    </div>
</body>
</html>"""

        # Save
        index_path = model_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Model index generated: {index_path}")
        return str(index_path)

    def run_model(self, model_name: str, max_problems: Optional[int] = None) -> Dict[str, Any]:
        """
        Run analysis for a single model on all HumanEval problems.

        Args:
            model_name: Name of model to test
            max_problems: Maximum number of problems to process (None for all)

        Returns:
            Dictionary with model results
        """
        print(f"\n{'#'*80}")
        print(f"Starting analysis for model: {model_name}")
        print(f"{'#'*80}\n")

        # Create model directory
        model_safe_name = model_name.replace("/", "_").replace("-", "_").replace(":", "_")
        model_dir = self.output_dir / model_safe_name
        model_dir.mkdir(exist_ok=True)

        # Initialize analyzers
        start_time = time.time()

        try:
            print("Loading model...")
            analyzer = QwenProbabilityAnalyzer(model_name=model_name)
            forced_analyzer = ForcedGenerationAnalyzer(model_name=model_name)

            # Process problems
            problems = list(self.dataset)
            if max_problems:
                problems = problems[:max_problems]

            results = []
            for i, problem in enumerate(problems):
                print(f"\nProgress: {i+1}/{len(problems)}")
                result = self.process_problem(
                    problem,
                    model_name,
                    analyzer,
                    forced_analyzer,
                    model_dir
                )
                results.append(result)

            # Calculate statistics
            processing_time = time.time() - start_time
            successful = len([r for r in results if r["status"] == "success"])
            passed = len([r for r in results if r.get("test_results", {}).get("failed", 1) == 0])
            visualizations = len([r for r in results if r.get("visualization_created", False)])

            model_result = {
                "model_name": model_name,
                "model_safe_name": model_safe_name,
                "status": "success",
                "problems_processed": len(results),
                "successful": successful,
                "problems_passed": passed,
                "visualizations_created": visualizations,
                "processing_time": processing_time,
                "results": results
            }

            # Save model results
            results_file = model_dir / "model_results.json"
            with open(results_file, 'w') as f:
                json.dump(model_result, f, indent=2)

            # Generate model index
            self.generate_model_index(model_dir, model_name, results)

            print(f"\n{'='*80}")
            print(f"Model {model_name} completed:")
            print(f"  - Problems processed: {len(results)}")
            print(f"  - Successful: {successful}")
            print(f"  - Tests passed: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")
            print(f"  - Visualizations created: {visualizations}")
            print(f"  - Time: {processing_time:.1f}s")
            print(f"{'='*80}\n")

            return model_result

        except Exception as e:
            print(f"❌ Error with model {model_name}: {e}")
            traceback.print_exc()
            return {
                "model_name": model_name,
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }


    def generate_main_index(self, all_results: List[Dict[str, Any]]) -> str:
        """
        Generate main index page for all models.

        Args:
            all_results: List of model results

        Returns:
            Path to generated index.html
        """
        # Calculate aggregate stats
        total_models = len(all_results)
        successful_models = len([r for r in all_results if r["status"] == "success"])

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HumanEval Plus Analysis - Multi-Model Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 40px; border-radius: 20px; box-shadow: 0 15px 50px rgba(0,0,0,0.3); }}
        h1 {{ color: #2c3e50; text-align: center; font-size: 42px; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #7f8c8d; font-size: 18px; margin-bottom: 30px; }}
        .overview {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; margin: 25px 0; }}
        .overview h2 {{ margin-top: 0; font-size: 28px; }}
        .models-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin: 30px 0; }}
        .model-card {{ border: 3px solid #e0e0e0; border-radius: 15px; padding: 25px; background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); transition: all 0.4s ease; position: relative; overflow: hidden; }}
        .model-card::before {{ content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 5px; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); }}
        .model-card:hover {{ transform: translateY(-8px); box-shadow: 0 15px 40px rgba(0,0,0,0.2); border-color: #f5576c; }}
        .model-name {{ font-size: 24px; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
        .model-stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid #dee2e6; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #f5576c; }}
        .stat-label {{ font-size: 14px; color: #6c757d; margin-top: 5px; }}
        .view-button {{ display: inline-block; padding: 12px 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; text-decoration: none; border-radius: 30px; margin-top: 15px; font-weight: bold; transition: all 0.3s ease; text-align: center; }}
        .view-button:hover {{ transform: scale(1.05); box-shadow: 0 8px 20px rgba(245,87,108,0.4); }}
        .status-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold; margin-bottom: 10px; }}
        .status-success {{ background: #d3f9d8; color: #2b8a3e; }}
        .status-error {{ background: #ffe3e3; color: #c92a2a; }}
        .timestamp {{ text-align: center; color: #adb5bd; margin-top: 30px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 HumanEval Plus Analysis</h1>
        <div class="subtitle">Multi-Model Token Probability Analysis with Forced vs Free Generation Comparison</div>

        <div class="overview">
            <h2>📊 Overview</h2>
            <p style="font-size: 18px; margin: 10px 0;"><strong>Models Tested:</strong> {total_models}</p>
            <p style="font-size: 18px; margin: 10px 0;"><strong>Successful Runs:</strong> {successful_models}/{total_models}</p>
            <p style="font-size: 16px; margin-top: 20px; opacity: 0.9;">This analysis evaluates large language models on the HumanEval Plus benchmark, comparing forced generation (canonical solution) with free generation to understand model confidence patterns in code generation.</p>
        </div>

        <h2 style="margin-top: 40px; color: #2c3e50;">🤖 Models</h2>
        <div class="models-grid">"""

        # Add model cards
        for result in all_results:
            model_name = result["model_name"]
            model_safe_name = result.get("model_safe_name", model_name.replace("/", "_").replace("-", "_"))
            status = result["status"]

            status_badge_class = "status-success" if status == "success" else "status-error"
            status_icon = "✅" if status == "success" else "❌"
            status_text = "Success" if status == "success" else f"Error"

            html += f"""
            <div class="model-card">
                <span class="status-badge {status_badge_class}">{status_icon} {status_text}</span>
                <div class="model-name">{model_name}</div>"""

            if status == "success":
                problems_processed = result["problems_processed"]
                problems_passed = result["problems_passed"]
                pass_rate = (problems_passed / problems_processed * 100) if problems_processed > 0 else 0
                visualizations = result["visualizations_created"]
                time_taken = result["processing_time"]

                html += f"""
                <div class="model-stats">
                    <div class="stat-box">
                        <div class="stat-value">{problems_processed}</div>
                        <div class="stat-label">Problems Tested</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{pass_rate:.1f}%</div>
                        <div class="stat-label">Pass Rate</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{problems_passed}</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{visualizations}</div>
                        <div class="stat-label">Visualizations</div>
                    </div>
                </div>
                <p style="color: #6c757d; font-size: 14px; margin: 10px 0;">Processing Time: {time_taken:.1f}s</p>
                <a href="{model_safe_name}/index.html" class="view-button">View Detailed Results →</a>"""
            else:
                error_msg = result.get("error", "Unknown error")
                html += f"""
                <p style="color: #c92a2a; margin: 15px 0;">Error: {error_msg[:100]}...</p>"""

            html += """
            </div>"""

        html += f"""
        </div>

        <div style="margin-top: 50px; padding: 25px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #f5576c;">
            <h3 style="color: #2c3e50; margin-top: 0;">🔍 About This Analysis</h3>
            <p><strong>HumanEval Plus</strong> is an extended version of OpenAI's HumanEval benchmark with 80x more test cases for rigorous code generation evaluation.</p>
            <p><strong>Forced vs Free Generation:</strong> For each failed problem, we compare:</p>
            <ul>
                <li><strong>Forced Generation:</strong> Model forced to generate the canonical solution token-by-token</li>
                <li><strong>Free Generation:</strong> Model's own generated solution</li>
            </ul>
            <p>By analyzing token probabilities and logits, we can understand where models struggle and whether they have implicit knowledge of code correctness.</p>
        </div>

        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""

        # Save
        index_path = self.output_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n🎉 Main index generated: {index_path}")
        return str(index_path)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run HumanEval Plus analysis with token probability visualization")
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-2-2b-it"
    ], help="Models to test")
    parser.add_argument("--output-dir", default="humaneval_analysis", help="Output directory")
    parser.add_argument("--max-problems", type=int, default=None, help="Maximum problems per model (None for all 164)")
    parser.add_argument("--timeout", type=int, default=10, help="Timeout for test execution (seconds)")

    args = parser.parse_args()

    # Initialize runner
    runner = HumanEvalRunner(output_dir=args.output_dir, timeout=args.timeout)

    # Load dataset
    runner.load_dataset()

    # Run analysis for each model
    all_results = []
    for model_name in args.models:
        model_result = runner.run_model(model_name, max_problems=args.max_problems)
        all_results.append(model_result)

    # Generate main index
    runner.generate_main_index(all_results)

    print("\n" + "="*80)
    print("ALL MODELS COMPLETED")
    print("="*80)
    for result in all_results:
        print(f"\n{result['model_name']}:")
        print(f"  Status: {result['status']}")
        if result['status'] == 'success':
            print(f"  Pass Rate: {result['problems_passed']}/{result['problems_processed']}")
            print(f"  Visualizations: {result['visualizations_created']}")

    print(f"\n🎉 All results saved to: {args.output_dir}/")
    print(f"📂 Open {args.output_dir}/index.html to view the analysis")


if __name__ == "__main__":
    main()
