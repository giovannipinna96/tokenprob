#!/usr/bin/env python3
"""
Regenerate HumanEval Visualizations with All Methods

This script regenerates all visualizations for HumanEval problems with ALL
available visualization methods, including the new advanced methods from
METHODS_OVERVIEW.md:
- Semantic Energy (Method 2)
- Conformal Prediction Score (Method 3)
- Attention metrics (Method 4) - placeholder for now

For each failed problem, it creates visualizations comparing:
- Forced generation (canonical solution)
- Free generation (model's solution)

With all available modes.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import html

from forced_generation_analyzer import ForcedGenerationResult
from forced_visualizer import ForcedGenerationVisualizer, ForcedVisualizationMode
from visualizer import TokenVisualizer, TokenVisualizationMode
from LLM import TokenAnalysis


class HumanEvalVisualizationRegenerator:
    """Regenerate HumanEval visualizations with all available methods."""

    def __init__(self, base_dir: str = "humaneval_analysis"):
        """
        Initialize the regenerator.

        Args:
            base_dir: Base directory containing HumanEval analysis results
        """
        self.base_dir = Path(base_dir)
        self.visualizer = ForcedGenerationVisualizer()
        self.token_visualizer = TokenVisualizer()

        # All available forced visualization modes
        self.forced_modes = [
            ForcedVisualizationMode.FORCED_LOGITS,
            ForcedVisualizationMode.FORCED_PROBABILITY,
            ForcedVisualizationMode.FORCED_LOG_PROBABILITY,
            ForcedVisualizationMode.FORCED_RANK,
            ForcedVisualizationMode.FORCED_SURPRISAL,
            ForcedVisualizationMode.PROBABILITY_MARGIN,
            ForcedVisualizationMode.FORCED_CODET5_VALIDATION,
            ForcedVisualizationMode.FORCED_NOMIC_COHERENCE,
            # Advanced methods from METHODS_OVERVIEW.md
            ForcedVisualizationMode.SEMANTIC_ENERGY,
            ForcedVisualizationMode.CONFORMAL_SCORE,
            # Note: Attention methods are placeholders for now
            # ForcedVisualizationMode.ATTENTION_ENTROPY,
            # ForcedVisualizationMode.ATTENTION_SELF_ATTENTION,
            # ForcedVisualizationMode.ATTENTION_VARIANCE,
            # ForcedVisualizationMode.ATTENTION_ANOMALY_SCORE,
        ]

        # Corresponding free visualization modes
        self.free_mode_mapping = {
            ForcedVisualizationMode.FORCED_LOGITS: TokenVisualizationMode.LOGITS,
            ForcedVisualizationMode.FORCED_PROBABILITY: TokenVisualizationMode.PROBABILITY,
            ForcedVisualizationMode.FORCED_LOG_PROBABILITY: TokenVisualizationMode.LOG_PROBABILITY,
            ForcedVisualizationMode.FORCED_RANK: TokenVisualizationMode.RANK,
            ForcedVisualizationMode.FORCED_SURPRISAL: TokenVisualizationMode.SURPRISAL,
            ForcedVisualizationMode.PROBABILITY_MARGIN: TokenVisualizationMode.PROBABILITY_MARGIN,
            ForcedVisualizationMode.FORCED_CODET5_VALIDATION: TokenVisualizationMode.CODET5_VALIDATION,
            ForcedVisualizationMode.FORCED_NOMIC_COHERENCE: TokenVisualizationMode.NOMIC_COHERENCE,
            ForcedVisualizationMode.SEMANTIC_ENERGY: TokenVisualizationMode.SEMANTIC_ENERGY,
            ForcedVisualizationMode.CONFORMAL_SCORE: TokenVisualizationMode.CONFORMAL_SCORE,
        }

    def load_forced_result(self, json_path: Path) -> Optional[ForcedGenerationResult]:
        """Load forced generation result from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Reconstruct ForcedGenerationResult from JSON
            # This is a simplified version - you may need to adjust based on actual JSON structure
            from forced_generation_analyzer import ForcedTokenAnalysis

            token_analyses = []
            for token_data in data.get("per_token_analysis", []):
                # Convert top_alternatives format
                top_alternatives = []
                for alt in token_data.get("top_alternatives", []):
                    top_alternatives.append((
                        alt["token_id"],
                        alt["probability"],
                        alt["token_text"]
                    ))

                # Create ForcedTokenAnalysis (parameters must match the dataclass definition)
                analysis = ForcedTokenAnalysis(
                    token=token_data["token"],
                    token_id=token_data["token_id"],
                    position=token_data["position"],
                    probability=token_data["probability"],
                    logit=token_data["logit"],
                    rank=token_data["rank"],
                    entropy=token_data["distribution_metrics"]["entropy"],  # Required!
                    surprisal=token_data["surprisal"],
                    perplexity=token_data["distribution_metrics"]["perplexity"],
                    confidence_score=token_data["confidence_score"],
                    top_k_alternatives=top_alternatives,
                    max_probability=token_data["distribution_metrics"]["max_probability"],
                    probability_margin=token_data["distribution_metrics"]["probability_margin"],
                    distribution_entropy=token_data["distribution_metrics"]["entropy"],
                    codet5_validation_score=token_data.get("codet5_validation", {}).get("validation_score"),
                    codet5_alternatives=[(alt["token"], alt["probability"]) for alt in token_data.get("codet5_validation", {}).get("alternatives", [])],
                    codet5_predicted_token=token_data.get("codet5_validation", {}).get("predicted_token"),
                    codet5_matches=token_data.get("codet5_validation", {}).get("matches", False),
                    nomic_coherence_score=token_data.get("nomic_validation", {}).get("coherence_score"),
                    nomic_similarity_drop=token_data.get("nomic_validation", {}).get("similarity_drop"),
                    nomic_context_similarity=token_data.get("nomic_validation", {}).get("context_similarity"),
                )
                token_analyses.append(analysis)

            # Create ForcedGenerationResult (parameters must match the dataclass definition)
            result = ForcedGenerationResult(
                original_prompt=data["metadata"]["original_prompt"],
                target_code=data["metadata"]["target_code"],
                reconstructed_code=data["metadata"]["reconstructed_code"],
                model_name=data["metadata"]["model_name"],
                token_analyses=token_analyses,
                average_probability=data["summary_statistics"]["average_probability"],
                average_rank=data["summary_statistics"]["average_rank"],
                average_surprisal=data["summary_statistics"]["average_surprisal"],
                average_confidence=data["summary_statistics"]["average_confidence"],
                total_tokens=data["summary_statistics"]["total_tokens"],
                high_uncertainty_tokens=data["summary_statistics"]["high_uncertainty_tokens"],
                model_preferred_alternative=data.get("model_alternatives", {}).get("what_model_would_generate", "")
            )

            return result

        except Exception as e:
            print(f"Error loading forced result from {json_path}: {e}")
            return None

    def load_free_generation_tokens(self, json_path: Path) -> Optional[List[TokenAnalysis]]:
        """Load free generation token analyses from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            token_analyses = []
            for token_data in data.get("tokens", []):
                analysis = TokenAnalysis(
                    token=token_data["token"],
                    token_id=0,  # Not stored in current format
                    position=token_data["position"],
                    probability=token_data["probability"],
                    logit=token_data["logit"],
                    rank=token_data["rank"],
                    perplexity=0.0,  # Calculate if needed
                    entropy=0.0,  # Calculate if needed
                    surprisal=0.0,  # Calculate if needed
                    top_k_probs=[],
                    max_probability=token_data["probability"],
                    probability_margin=0.0,
                    shannon_entropy=0.0,
                    local_perplexity=0.0,
                    sequence_improbability=0.0,
                    confidence_score=token_data["probability"],
                )
                token_analyses.append(analysis)

            return token_analyses

        except Exception as e:
            print(f"Error loading free generation from {json_path}: {e}")
            return None

    def create_free_generation_visualization(self,
                                            analyses: List[TokenAnalysis],
                                            generated_code: str,
                                            mode: str,
                                            title: str) -> str:
        """Create HTML visualization for free generation."""
        viz_mode = self.free_mode_mapping.get(mode, TokenVisualizationMode.PROBABILITY)

        return self.token_visualizer.create_html_visualization(
            analyses,
            mode=viz_mode,
            title=title
        )

    def create_comparison_html(self,
                              forced_result: ForcedGenerationResult,
                              free_analyses: List[TokenAnalysis],
                              free_code: str,
                              test_results: Dict[str, Any],
                              problem: Dict[str, Any],
                              output_file: Path):
        """
        Create comprehensive comparison HTML with ALL visualization modes.

        Args:
            forced_result: Forced generation analysis result
            free_analyses: Free generation token analyses
            free_code: Generated code from free generation
            test_results: Test execution results
            problem: Original problem information
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
            <p><strong>Entry Point:</strong> <code>{problem.get('entry_point', 'N/A')}</code></p>
            <div style='margin-top: 15px;'>
                <strong>Problem Description:</strong>
                <pre style='background: white; padding: 15px; border-radius: 5px; overflow-x: auto;'>{html.escape(problem.get('prompt', ''))}</pre>
            </div>
        </div>
        """

        # Create comparison visualizations for ALL modes
        comparison_html_parts = []

        for mode in self.forced_modes:
            # Get visualization scheme
            scheme = self.visualizer.color_schemes.get(mode, {})
            if not scheme:
                continue

            # Create forced visualization
            forced_viz = self.visualizer.create_forced_logits_visualization(
                forced_result,
                mode=mode,
                title=f"🟢 Canonical Solution (Forced): {scheme['label']}"
            )

            # Create free visualization
            free_viz = self.create_free_generation_visualization(
                free_analyses,
                free_code,
                mode,
                f"🔴 Generated Solution (Free): {scheme['label']}"
            )

            comparison_html_parts.append(f"""
            <div style='margin: 40px 0; page-break-inside: avoid;'>
                <h2 style='text-align: center; color: #495057; border-bottom: 3px solid #dee2e6; padding-bottom: 10px;'>
                    Comparison: {scheme['label']}
                </h2>
                <p style='text-align: center; color: #6c757d; margin-bottom: 20px;'>
                    <em>{scheme['description']}</em>
                </p>
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
        @media print {{
            .container {{ box-shadow: none; }}
            body {{ background: white; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 HumanEval Plus Analysis: {problem['task_id']}</h1>

        {test_banner}
        {problem_info}

        <div style='margin: 30px 0; padding: 20px; background: #e7f5ff; border-radius: 8px; border-left: 5px solid #339af0;'>
            <h3 style='margin-top: 0; color: #1971c2;'>📊 Analysis Overview</h3>
            <p>This page shows comprehensive token-level analysis comparing:</p>
            <ul>
                <li><strong>Canonical Solution (Left/Green):</strong> Model forced to generate the correct solution token-by-token</li>
                <li><strong>Generated Solution (Right/Red):</strong> Model's own free generation that failed tests</li>
            </ul>
            <p>Each visualization mode reveals different aspects of model confidence and uncertainty:</p>
            <ul>
                <li><strong>Basic Metrics:</strong> Logits, Probability, Rank, Surprisal</li>
                <li><strong>Validation:</strong> CodeT5 validation, Nomic coherence</li>
                <li><strong>Advanced Methods:</strong> Semantic Energy (Method 2), Conformal Scores (Method 3)</li>
            </ul>
        </div>

        {''.join(comparison_html_parts)}

        <div style='margin-top: 40px; padding: 20px; background: #e9ecef; border-radius: 8px;'>
            <h3>💡 About This Analysis</h3>
            <p><strong>Forced Generation (Left):</strong> The model was forced to generate the canonical solution token-by-token.
            The metrics show how confident the model was about each forced token.</p>
            <p><strong>Free Generation (Right):</strong> The model freely generated its own solution.
            The metrics show the model's natural confidence in its choices.</p>
            <p><strong>Hypothesis:</strong> If the model's free generation has lower confidence (higher uncertainty) in buggy regions,
            it suggests the model has implicit knowledge of code quality that could be used for bug detection.</p>
            <p><strong>Methods:</strong></p>
            <ul>
                <li><strong>Method 1 (Baseline):</strong> LecPrompt - Statistical anomaly detection via log probabilities</li>
                <li><strong>Method 2:</strong> Semantic Energy - Pre-softmax logit energy (outperforms probabilities by 13% AUROC)</li>
                <li><strong>Method 3:</strong> Conformal Prediction - Statistically rigorous uncertainty quantification</li>
                <li><strong>Method 4:</strong> Attention Anomaly - Attention pattern analysis (requires attention weights)</li>
            </ul>
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

        print(f"✅ Regenerated visualization: {output_file}")

    def regenerate_model_visualizations(self, model_dir: Path):
        """Regenerate all visualizations for a specific model."""
        print(f"\n{'='*80}")
        print(f"Processing model: {model_dir.name}")
        print(f"{'='*80}\n")

        regenerated_count = 0

        # Find all problem directories
        for problem_dir in sorted(model_dir.iterdir()):
            if not problem_dir.is_dir():
                continue

            if problem_dir.name in ['index.html', 'model_results.json']:
                continue

            # Check if this problem has forced canonical data (meaning it failed tests)
            forced_json = problem_dir / "forced_canonical.json"
            free_json = problem_dir / "generated_solution.json"
            test_json = problem_dir / "test_results.json"

            if not forced_json.exists():
                print(f"⏭️  Skipping {problem_dir.name} (tests passed, no forced generation)")
                continue

            print(f"🔄 Regenerating {problem_dir.name}...")

            try:
                # Load data
                forced_result = self.load_forced_result(forced_json)
                if forced_result is None:
                    print(f"   ❌ Failed to load forced result")
                    continue

                free_analyses = self.load_free_generation_tokens(free_json)
                if free_analyses is None:
                    print(f"   ❌ Failed to load free generation")
                    continue

                with open(free_json, 'r') as f:
                    free_data = json.load(f)
                free_code = free_data.get("generated_code", "")

                with open(test_json, 'r') as f:
                    test_results = json.load(f)

                # Create problem info
                problem_info = {
                    'task_id': free_data.get("task_id", problem_dir.name),
                    'prompt': free_data.get("prompt", ""),
                    'entry_point': problem_dir.name.split('_')[-1] if '_' in problem_dir.name else "unknown"
                }

                # Generate new comprehensive comparison (overwrites old comparison.html)
                output_file = problem_dir / "comparison.html"
                self.create_comparison_html(
                    forced_result=forced_result,
                    free_analyses=free_analyses,
                    free_code=free_code,
                    test_results=test_results,
                    problem=problem_info,
                    output_file=output_file
                )

                regenerated_count += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n✅ Regenerated {regenerated_count} visualizations for {model_dir.name}\n")
        return regenerated_count

    def regenerate_all(self):
        """Regenerate visualizations for all models."""
        print(f"\n{'#'*80}")
        print(f"# Regenerating HumanEval Visualizations with ALL Methods")
        print(f"{'#'*80}\n")

        if not self.base_dir.exists():
            print(f"❌ Base directory not found: {self.base_dir}")
            return

        total_regenerated = 0

        # Process each model directory
        for model_dir in sorted(self.base_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            if model_dir.name == 'index.html':
                continue

            count = self.regenerate_model_visualizations(model_dir)
            total_regenerated += count

        print(f"\n{'#'*80}")
        print(f"# COMPLETE: Regenerated {total_regenerated} total visualizations")
        print(f"{'#'*80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate HumanEval visualizations with all available methods"
    )
    parser.add_argument(
        "--base-dir",
        default="humaneval_analysis",
        help="Base directory containing HumanEval analysis results"
    )
    parser.add_argument(
        "--model",
        help="Regenerate only for specific model (e.g., Qwen_Qwen2.5_Coder_7B_Instruct)"
    )

    args = parser.parse_args()

    regenerator = HumanEvalVisualizationRegenerator(base_dir=args.base_dir)

    if args.model:
        model_dir = regenerator.base_dir / args.model
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return
        regenerator.regenerate_model_visualizations(model_dir)
    else:
        regenerator.regenerate_all()


if __name__ == "__main__":
    main()
