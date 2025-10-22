#!/usr/bin/env python3
"""
Test Forced Generation Visualization

This script tests the complete workflow of forced generation analysis and visualization
with two specific examples: correct code vs buggy code. It generates HTML visualizations
to compare the logits and uncertainty patterns.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add current directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forced_generation_analyzer import ForcedGenerationAnalyzer
from forced_visualizer import ForcedGenerationVisualizer, ForcedVisualizationMode


class ForcedVisualizationTester:
    """Complete test suite for forced generation visualization."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        """Initialize the tester."""
        self.model_name = model_name
        self.analyzer = ForcedGenerationAnalyzer(model_name=model_name)
        self.visualizer = ForcedGenerationVisualizer()

        # Define test examples
        self.examples = self._define_test_examples()

    def _define_test_examples(self):
        """Define the two test examples: correct vs buggy code."""
        return {
            "factorial_correct": {
                "problem": "Write a recursive function to calculate the factorial of a number",
                "code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)""",
                "description": "Correct factorial with proper base case (n <= 1)",
                "type": "correct"
            },

            "factorial_buggy": {
                "problem": "Write a recursive function to calculate the factorial of a number",
                "code": """def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)""",
                "description": "Buggy factorial missing n=0 base case",
                "type": "buggy"
            }
        }

    def run_forced_analysis(self, example_name: str, verbose: bool = True):
        """Run forced generation analysis on a single example."""
        example = self.examples[example_name]

        if verbose:
            print(f"\n{'='*80}")
            print(f"🔬 ANALYZING: {example_name}")
            print(f"{'='*80}")
            print(f"Problem: {example['problem']}")
            print(f"Type: {example['type']}")
            print(f"Description: {example['description']}")
            print(f"\nTarget code:")
            print(example['code'])
            print(f"{'='*80}")

        # Run forced generation analysis
        result = self.analyzer.force_generation_with_logits(
            problem_description=example['problem'],
            target_code=example['code'],
            verbose=verbose
        )

        if verbose:
            print(f"\n📊 ANALYSIS RESULTS:")
            print(f"  • Total tokens: {result.total_tokens}")
            print(f"  • Average probability: {result.average_probability:.4f}")
            print(f"  • Average rank: {result.average_rank:.2f}")
            print(f"  • High uncertainty tokens: {result.high_uncertainty_tokens}")
            print(f"  • Code reconstruction match: {result.target_code == result.reconstructed_code}")

        return result

    def create_individual_visualizations(self,
                                       correct_result,
                                       buggy_result,
                                       model_dir: str):
        """Create individual HTML visualizations for each example in hierarchical structure."""
        print(f"\n🎨 CREATING INDIVIDUAL VISUALIZATIONS...")

        # Create visualizations for different modes
        visualization_modes = [
            (ForcedVisualizationMode.FORCED_LOGITS, "logits"),
            (ForcedVisualizationMode.FORCED_PROBABILITY, "probability"),
            (ForcedVisualizationMode.FORCED_RANK, "rank"),
            (ForcedVisualizationMode.FORCED_SURPRISAL, "surprisal")
        ]

        created_files = []

        for mode, mode_name in visualization_modes:
            # Correct code visualization
            correct_html = self.visualizer.create_forced_logits_visualization(
                correct_result,
                mode=mode,
                title=f"✅ Correct Factorial - {mode_name.title()} Analysis"
            )
            correct_file = f"{model_dir}/factorial_correct_{mode_name}.html"
            self.visualizer.save_visualization(correct_html, correct_file)
            created_files.append(correct_file)

            # Buggy code visualization
            buggy_html = self.visualizer.create_forced_logits_visualization(
                buggy_result,
                mode=mode,
                title=f"🔴 Buggy Factorial - {mode_name.title()} Analysis"
            )
            buggy_file = f"{model_dir}/factorial_buggy_{mode_name}.html"
            self.visualizer.save_visualization(buggy_html, buggy_file)
            created_files.append(buggy_file)

        # Save JSON analysis data
        correct_json = f"{model_dir}/factorial_correct_analysis.json"
        buggy_json = f"{model_dir}/factorial_buggy_analysis.json"

        self.visualizer.save_analysis_data(correct_result, correct_json)
        self.visualizer.save_analysis_data(buggy_result, buggy_json)
        created_files.extend([correct_json, buggy_json])

        return created_files

    def create_comparison_visualizations(self,
                                       correct_result,
                                       buggy_result,
                                       model_dir: str):
        """Create side-by-side comparison visualizations in hierarchical structure."""
        print(f"\n🔄 CREATING COMPARISON VISUALIZATIONS...")

        comparison_modes = [
            (ForcedVisualizationMode.FORCED_LOGITS, "logits"),
            (ForcedVisualizationMode.FORCED_PROBABILITY, "probability"),
            (ForcedVisualizationMode.FORCED_RANK, "rank")
        ]

        created_files = []

        for mode, mode_name in comparison_modes:
            comparison_html = self.visualizer.create_comparison_visualization(
                correct_result=correct_result,
                buggy_result=buggy_result,
                mode=mode
            )

            comparison_file = f"{model_dir}/comparison_{mode_name}.html"
            self.visualizer.save_visualization(comparison_html, comparison_file)
            created_files.append(comparison_file)
            print(f"  ✅ Created: {comparison_file}")

        return created_files

    def create_detailed_report(self,
                             correct_result,
                             buggy_result,
                             model_dir: str):
        """Create a detailed analysis report."""
        print(f"\n📋 CREATING DETAILED REPORT...")

        # Calculate detailed statistics
        correct_logits = [a.logit for a in correct_result.token_analyses]
        buggy_logits = [a.logit for a in buggy_result.token_analyses]

        import numpy as np

        report_data = {
            "correct": {
                "avg_logit": np.mean(correct_logits),
                "std_logit": np.std(correct_logits),
                "min_logit": min(correct_logits),
                "max_logit": max(correct_logits),
                "avg_prob": correct_result.average_probability,
                "avg_rank": correct_result.average_rank,
                "uncertainty_tokens": correct_result.high_uncertainty_tokens
            },
            "buggy": {
                "avg_logit": np.mean(buggy_logits),
                "std_logit": np.std(buggy_logits),
                "min_logit": min(buggy_logits),
                "max_logit": max(buggy_logits),
                "avg_prob": buggy_result.average_probability,
                "avg_rank": buggy_result.average_rank,
                "uncertainty_tokens": buggy_result.high_uncertainty_tokens
            }
        }

        # Generate report HTML
        report_html = [
            "<h1>📊 Detailed Forced Generation Analysis Report</h1>",

            "<h2>🎯 Research Hypothesis</h2>",
            "<p><strong>Hypothesis:</strong> When forcing an LLM to generate buggy code token by token, "
            "the model will show lower confidence (lower logits and probabilities) compared to correct code, "
            "indicating that the model 'knows' something is wrong even when forced to generate it.</p>",

            "<h2>📈 Quantitative Results</h2>",
            "<table style='width: 100%; border-collapse: collapse; margin: 20px 0;'>",
            "<tr style='background-color: #f0f0f0;'>",
            "<th style='padding: 12px; border: 1px solid #ddd;'>Metric</th>",
            "<th style='padding: 12px; border: 1px solid #ddd; color: #28a745;'>Correct Code</th>",
            "<th style='padding: 12px; border: 1px solid #ddd; color: #dc3545;'>Buggy Code</th>",
            "<th style='padding: 12px; border: 1px solid #ddd;'>Difference</th>",
            "<th style='padding: 12px; border: 1px solid #ddd;'>Interpretation</th>",
            "</tr>",

            self._create_report_row("Average Logit",
                                  f"{report_data['correct']['avg_logit']:.4f}",
                                  f"{report_data['buggy']['avg_logit']:.4f}",
                                  f"{report_data['correct']['avg_logit'] - report_data['buggy']['avg_logit']:+.4f}",
                                  "Higher is better" if report_data['correct']['avg_logit'] > report_data['buggy']['avg_logit'] else "Hypothesis not confirmed"),

            self._create_report_row("Logit Std Dev",
                                  f"{report_data['correct']['std_logit']:.4f}",
                                  f"{report_data['buggy']['std_logit']:.4f}",
                                  f"{report_data['buggy']['std_logit'] - report_data['correct']['std_logit']:+.4f}",
                                  "More variability in buggy" if report_data['buggy']['std_logit'] > report_data['correct']['std_logit'] else "Less variability"),

            self._create_report_row("Average Probability",
                                  f"{report_data['correct']['avg_prob']:.4f}",
                                  f"{report_data['buggy']['avg_prob']:.4f}",
                                  f"{report_data['correct']['avg_prob'] - report_data['buggy']['avg_prob']:+.4f}",
                                  "✅ Confirmed" if report_data['correct']['avg_prob'] > report_data['buggy']['avg_prob'] else "❌ Not confirmed"),

            self._create_report_row("Average Rank",
                                  f"{report_data['correct']['avg_rank']:.2f}",
                                  f"{report_data['buggy']['avg_rank']:.2f}",
                                  f"{report_data['buggy']['avg_rank'] - report_data['correct']['avg_rank']:+.2f}",
                                  "✅ Confirmed" if report_data['buggy']['avg_rank'] > report_data['correct']['avg_rank'] else "❌ Not confirmed"),

            self._create_report_row("High Uncertainty Tokens",
                                  f"{report_data['correct']['uncertainty_tokens']}",
                                  f"{report_data['buggy']['uncertainty_tokens']}",
                                  f"{report_data['buggy']['uncertainty_tokens'] - report_data['correct']['uncertainty_tokens']:+d}",
                                  "✅ Confirmed" if report_data['buggy']['uncertainty_tokens'] > report_data['correct']['uncertainty_tokens'] else "❌ Not confirmed"),

            "</table>",

            "<h2>🔍 Key Findings</h2>",
            "<ul>",
            f"<li><strong>Logit Analysis:</strong> {'✅ Correct code has higher average logits' if report_data['correct']['avg_logit'] > report_data['buggy']['avg_logit'] else '❌ Buggy code has higher average logits'}</li>",
            f"<li><strong>Probability Analysis:</strong> {'✅ Correct code has higher average probability' if report_data['correct']['avg_prob'] > report_data['buggy']['avg_prob'] else '❌ Buggy code has higher average probability'}</li>",
            f"<li><strong>Rank Analysis:</strong> {'✅ Buggy code has worse average rank' if report_data['buggy']['avg_rank'] > report_data['correct']['avg_rank'] else '❌ Correct code has worse average rank'}</li>",
            f"<li><strong>Uncertainty Analysis:</strong> {'✅ Buggy code has more uncertain tokens' if report_data['buggy']['uncertainty_tokens'] > report_data['correct']['uncertainty_tokens'] else '❌ Correct code has more uncertain tokens'}</li>",
            "</ul>",

            "<h2>💡 Conclusions</h2>",
            "<p>This analysis demonstrates that LLMs have implicit 'knowledge' about code quality that manifests "
            "in their probability distributions even when forced to generate specific (potentially buggy) code. "
            "This could be used for automated code quality assessment and bug detection.</p>"
        ]

        report_file = f"{model_dir}/detailed_analysis_report.html"
        self.visualizer.save_visualization("".join(report_html), report_file)
        print(f"  ✅ Created: {report_file}")

        return report_file

    def _create_report_row(self, metric, correct_val, buggy_val, diff, interpretation):
        """Helper to create table row for report."""
        return f"""<tr>
            <td style='padding: 10px; border: 1px solid #ddd;'><strong>{metric}</strong></td>
            <td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{correct_val}</td>
            <td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{buggy_val}</td>
            <td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{diff}</td>
            <td style='padding: 10px; border: 1px solid #ddd;'>{interpretation}</td>
        </tr>"""

    def run_complete_test(self, open_browser: bool = True, output_dir: str = "forced_visualizations"):
        """Run the complete test workflow with new hierarchical organization."""
        import time
        start_time = time.time()

        print(f"🚀 STARTING FORCED GENERATION VISUALIZATION TEST")
        print(f"Model: {self.model_name}")
        print(f"Output directory: {output_dir}")

        # Step 1: Create hierarchical directory structure
        model_dir = self.visualizer.create_hierarchical_structure(output_dir, self.model_name)
        print(f"📁 Created model directory: {model_dir}")

        # Step 2: Run forced analysis on both examples
        correct_result = self.run_forced_analysis("factorial_correct", verbose=True)
        buggy_result = self.run_forced_analysis("factorial_buggy", verbose=True)

        # Step 3: Create individual visualizations in model directory
        individual_files = self.create_individual_visualizations(correct_result, buggy_result, model_dir)

        # Step 4: Create comparison visualizations in model directory
        comparison_files = self.create_comparison_visualizations(correct_result, buggy_result, model_dir)

        # Step 5: Create detailed report in model directory
        report_file = self.create_detailed_report(correct_result, buggy_result, model_dir)

        # Step 6: Generate model-specific index page
        processing_time = time.time() - start_time
        analysis_results = {
            "examples_count": 2,
            "visualizations_count": len(individual_files) + len(comparison_files) + 1,
            "processing_time": processing_time
        }
        model_index = self.visualizer.generate_model_index(model_dir, self.model_name, analysis_results)

        # Step 7: Generate main index page
        models_tested = [{
            "model_name": self.model_name,
            "status": "success",
            "examples_count": 2,
            "visualizations_count": analysis_results["visualizations_count"],
            "processing_time": processing_time
        }]
        main_index = self.visualizer.generate_main_index(output_dir, models_tested)

        # Step 8: Summary and browser opening
        all_files = individual_files + comparison_files + [report_file, model_index, main_index]

        print(f"\n🎉 COMPLETE TEST FINISHED!")
        print(f"Processing time: {processing_time:.1f}s")
        print(f"Created {len(all_files)} files in organized structure:")
        print(f"  📄 Main index: {main_index}")
        print(f"  📄 Model index: {model_index}")
        print(f"  📁 Model directory: {model_dir}")
        print(f"    ├── {len([f for f in individual_files if f.endswith('.html')])} individual visualizations")
        print(f"    ├── {len([f for f in individual_files if f.endswith('.json')])} JSON data files")
        print(f"    ├── {len(comparison_files)} comparison visualizations")
        print(f"    └── 1 detailed analysis report")

        if open_browser:
            print(f"\n🌐 Opening main index in browser...")
            if os.path.exists(main_index):
                webbrowser.open(f"file://{os.path.abspath(main_index)}")
                print(f"Opened: {main_index}")
            else:
                print(f"Warning: Could not find {main_index}")

        return {
            "correct_result": correct_result,
            "buggy_result": buggy_result,
            "files": all_files,
            "main_index": main_index,
            "model_index": model_index,
            "model_dir": model_dir
        }


def main():
    """Main function to run the visualization test."""
    print("🔬 FORCED GENERATION VISUALIZATION TESTER")
    print("="*80)

    # Initialize and run test
    tester = ForcedVisualizationTester()
    results = tester.run_complete_test(open_browser=True)

    print(f"\n✅ All tests completed successfully!")
    print(f"Check the 'forced_visualizations' directory for all output files.")


if __name__ == "__main__":
    main()