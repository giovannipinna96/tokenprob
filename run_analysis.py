#!/usr/bin/env python3
"""
Analysis Runner for Test Examples

This script runs the LLM token probability analysis on the test examples dataset,
comparing buggy vs correct code to validate the hypothesis that low probability
tokens correlate with problematic code areas.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLM import QwenProbabilityAnalyzer
from visualizer import TokenVisualizer, TokenVisualizationMode
from test_examples import TestExamplesDataset, TestExample

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

class AnalysisRunner:
    """Runs analysis on test examples and compares results."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        """Initialize the analysis runner."""
        self.model_name = model_name
        self.analyzer = QwenProbabilityAnalyzer(model_name=model_name)
        self.visualizer = TokenVisualizer()
        self.dataset = TestExamplesDataset()

    def analyze_example(self, example: TestExample, code_type: str) -> Dict[str, Any]:
        """
        Analyze a single example (either buggy or correct code).

        Args:
            example: Test example to analyze
            code_type: "buggy" or "correct"

        Returns:
            Analysis results dictionary
        """
        code = example.buggy_code if code_type == "buggy" else example.correct_code
        prompt = f"{example.prompt}\n\nCode to analyze:\n{code}"

        print(f"Analyzing {example.name} ({code_type})...")

        # Generate with analysis
        generated_text, token_analyses = self.analyzer.generate_with_analysis(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.1  # Low temperature for more deterministic analysis
        )

        # Get statistics
        stats = self.analyzer.get_generation_stats()

        # Identify low confidence regions
        low_conf_regions = self.visualizer.identify_low_confidence_regions(
            token_analyses, threshold_percentile=20
        )

        return {
            "example_name": example.name,
            "code_type": code_type,
            "bug_type": example.bug_type,
            "prompt": prompt,
            "generated_text": generated_text,
            "statistics": stats,
            "low_confidence_regions": len(low_conf_regions),
            "low_confidence_tokens": [
                {
                    "position": region[0],  # start position
                    "end_position": region[1],  # end position
                    "avg_probability": region[2],  # average probability in region
                    "tokens": [
                        {
                            "position": i,
                            "token": token_analyses[i].token,
                            "probability": token_analyses[i].probability
                        }
                        for i in range(region[0], min(region[1] + 1, len(token_analyses)))
                    ]
                }
                for region in low_conf_regions
            ],
            "token_analyses": [
                {
                    "token": analysis.token,
                    "position": analysis.position,
                    "probability": analysis.probability,
                    "rank": analysis.rank,
                    "entropy": analysis.entropy,
                    "surprisal": analysis.surprisal,
                    "top_10_tokens": [
                        {
                            "token_id": token_id,
                            "probability": prob,
                            "token_text": self.analyzer.tokenizer.decode([token_id], skip_special_tokens=False)
                        }
                        for token_id, prob in analysis.top_k_probs[:10]
                    ]
                }
                for analysis in token_analyses
            ]
        }

    def run_full_analysis(self, output_dir: str = "analysis_results") -> Dict[str, Any]:
        """
        Run analysis on all examples in the dataset.

        Args:
            output_dir: Directory to save results

        Returns:
            Complete analysis results
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "examples": []
        }

        print(f"Running analysis on {len(self.dataset.examples)} examples...")
        print(f"Model: {self.model_name}")
        print("="*60)

        for i, example in enumerate(self.dataset.examples, 1):
            print(f"\n[{i}/{len(self.dataset.examples)}] Processing: {example.name}")

            try:
                # Analyze buggy code
                buggy_analysis = self.analyze_example(example, "buggy")

                # Analyze correct code
                correct_analysis = self.analyze_example(example, "correct")

                # Compare results
                comparison = self.compare_analyses(buggy_analysis, correct_analysis)

                example_result = {
                    "example": {
                        "name": example.name,
                        "description": example.description,
                        "bug_type": example.bug_type,
                        "prompt": example.prompt,
                        "buggy_code": example.buggy_code,
                        "correct_code": example.correct_code
                    },
                    "buggy_analysis": buggy_analysis,
                    "correct_analysis": correct_analysis,
                    "comparison": comparison
                }

                results["examples"].append(example_result)

                # Save individual example result
                example_file = os.path.join(output_dir, f"{example.name}_analysis.json")
                with open(example_file, 'w', encoding='utf-8') as f:
                    json.dump(example_result, f, indent=2, ensure_ascii=False)

                print(f"  Saved individual result to {example_file}")

            except Exception as e:
                print(f"  Error analyzing {example.name}: {e}")
                continue

        # Save complete results
        complete_file = os.path.join(output_dir, "complete_analysis.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Generate summary report
        summary = self.generate_summary_report(results)
        summary_file = os.path.join(output_dir, "analysis_summary.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"  • Complete analysis: complete_analysis.json")
        print(f"  • Summary report: analysis_summary.md")
        print(f"  • Individual results: {len(results['examples'])} files")

        return results

    def compare_analyses(self, buggy: Dict[str, Any], correct: Dict[str, Any]) -> Dict[str, Any]:
        """Compare buggy vs correct analysis results."""
        buggy_stats = buggy["statistics"]
        correct_stats = correct["statistics"]

        return {
            "probability_difference": {
                "buggy_avg": buggy_stats["avg_probability"],
                "correct_avg": correct_stats["avg_probability"],
                "difference": buggy_stats["avg_probability"] - correct_stats["avg_probability"],
                "hypothesis_confirmed": buggy_stats["avg_probability"] < correct_stats["avg_probability"]
            },
            "low_confidence_comparison": {
                "buggy_regions": buggy["low_confidence_regions"],
                "correct_regions": correct["low_confidence_regions"],
                "difference": buggy["low_confidence_regions"] - correct["low_confidence_regions"],
                "hypothesis_confirmed": buggy["low_confidence_regions"] > correct["low_confidence_regions"]
            },
            "entropy_comparison": {
                "buggy_avg": buggy_stats["avg_entropy"],
                "correct_avg": correct_stats["avg_entropy"],
                "difference": buggy_stats["avg_entropy"] - correct_stats["avg_entropy"],
                "hypothesis_confirmed": buggy_stats["avg_entropy"] > correct_stats["avg_entropy"]
            }
        }

    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a markdown summary report."""
        total_examples = len(results["examples"])

        # Calculate overall statistics
        prob_confirmations = 0
        region_confirmations = 0
        entropy_confirmations = 0

        for example in results["examples"]:
            comp = example["comparison"]
            if comp["probability_difference"]["hypothesis_confirmed"]:
                prob_confirmations += 1
            if comp["low_confidence_comparison"]["hypothesis_confirmed"]:
                region_confirmations += 1
            if comp["entropy_comparison"]["hypothesis_confirmed"]:
                entropy_confirmations += 1

        report = f"""# LLM Token Probability Analysis Report

**Model:** {results["model_name"]}
**Timestamp:** {results["timestamp"]}
**Total Examples:** {total_examples}

## Hypothesis Validation

The hypothesis states that **low probability tokens correlate with buggy code areas**.

### Results Summary

| Metric | Confirmations | Percentage | Status |
|--------|--------------|------------|---------|
| **Average Probability** | {prob_confirmations}/{total_examples} | {prob_confirmations/total_examples*100:.1f}% | {'✅ CONFIRMED' if prob_confirmations/total_examples > 0.6 else '❌ REJECTED'} |
| **Low Confidence Regions** | {region_confirmations}/{total_examples} | {region_confirmations/total_examples*100:.1f}% | {'✅ CONFIRMED' if region_confirmations/total_examples > 0.6 else '❌ REJECTED'} |
| **Average Entropy** | {entropy_confirmations}/{total_examples} | {entropy_confirmations/total_examples*100:.1f}% | {'✅ CONFIRMED' if entropy_confirmations/total_examples > 0.6 else '❌ REJECTED'} |

### Overall Hypothesis Status

**{'✅ HYPOTHESIS CONFIRMED' if (prob_confirmations + region_confirmations + entropy_confirmations) / (total_examples * 3) > 0.6 else '❌ HYPOTHESIS REJECTED'}**

## Detailed Results by Example

"""

        for example in results["examples"]:
            comp = example["comparison"]
            ex_data = example["example"]

            report += f"""### {ex_data["name"]} ({ex_data["bug_type"]})

**Description:** {ex_data["description"]}

| Metric | Buggy Code | Correct Code | Difference | Confirmed |
|--------|------------|--------------|------------|-----------|
| Avg Probability | {comp["probability_difference"]["buggy_avg"]:.3f} | {comp["probability_difference"]["correct_avg"]:.3f} | {comp["probability_difference"]["difference"]:.3f} | {'✅' if comp["probability_difference"]["hypothesis_confirmed"] else '❌'} |
| Low Conf Regions | {comp["low_confidence_comparison"]["buggy_regions"]} | {comp["low_confidence_comparison"]["correct_regions"]} | {comp["low_confidence_comparison"]["difference"]} | {'✅' if comp["low_confidence_comparison"]["hypothesis_confirmed"] else '❌'} |
| Avg Entropy | {comp["entropy_comparison"]["buggy_avg"]:.3f} | {comp["entropy_comparison"]["correct_avg"]:.3f} | {comp["entropy_comparison"]["difference"]:.3f} | {'✅' if comp["entropy_comparison"]["hypothesis_confirmed"] else '❌'} |

"""

        report += """
## Interpretation

- **Average Probability**: Lower values for buggy code suggest the model is less confident
- **Low Confidence Regions**: More regions in buggy code indicate uncertainty hotspots
- **Average Entropy**: Higher entropy in buggy code shows the model considers more alternatives

## Recommendations

Based on these results, consider:
1. Using token probability as a code quality indicator
2. Flagging low-confidence regions for human review
3. Developing automated tools based on these patterns
"""

        return report

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run LLM Token Probability Analysis on Test Examples")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                       help="HuggingFace model name to use")
    parser.add_argument("--output-dir", type=str, default="analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--example", type=str, help="Run analysis on specific example only")

    args = parser.parse_args()

    # Create analysis runner
    runner = AnalysisRunner(model_name=args.model)

    if args.example:
        # Analyze single example
        try:
            example = runner.dataset.get_example(args.example)
            print(f"Analyzing single example: {args.example}")

            buggy_analysis = runner.analyze_example(example, "buggy")
            correct_analysis = runner.analyze_example(example, "correct")
            comparison = runner.compare_analyses(buggy_analysis, correct_analysis)

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            result_file = os.path.join(args.output_dir, f"{args.example}_single_analysis.json")

            result = {
                "example": example.__dict__,
                "buggy_analysis": buggy_analysis,
                "correct_analysis": correct_analysis,
                "comparison": comparison
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(result), f, indent=2, ensure_ascii=False)

            print(f"Results saved to {result_file}")

        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available examples: {[ex.name for ex in runner.dataset.examples]}")

    else:
        # Run full analysis
        runner.run_full_analysis(args.output_dir)

if __name__ == "__main__":
    main()