#!/usr/bin/env python3
"""
Forced Generation Visualizer

Specialized visualizer for forced generation analysis that focuses on logits
and uncertainty patterns in forced token sequences. This visualizer is optimized
for comparing correct vs buggy code patterns.
"""

import numpy as np
import html
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from forced_generation_analyzer import ForcedTokenAnalysis, ForcedGenerationResult
from LLM import TokenAnalysis


class ForcedVisualizationMode:
    """Visualization modes specific to forced generation analysis"""
    FORCED_LOGITS = "forced_logits"
    FORCED_PROBABILITY = "forced_probability"
    FORCED_LOG_PROBABILITY = "forced_log_probability"
    FORCED_RANK = "forced_rank"
    FORCED_SURPRISAL = "forced_surprisal"
    FORCED_CONFIDENCE = "forced_confidence"
    PROBABILITY_MARGIN = "probability_margin"
    FORCED_CODET5_VALIDATION = "forced_codet5_validation"
    FORCED_NOMIC_COHERENCE = "forced_nomic_coherence"
    # Advanced methods from METHODS_OVERVIEW.md
    SEMANTIC_ENERGY = "semantic_energy"  # Method 2: pre-softmax logits energy
    CONFORMAL_SCORE = "conformal_score"  # Method 3: conformal prediction uncertainty
    ATTENTION_ENTROPY = "attention_entropy"  # Method 4: attention distribution entropy
    ATTENTION_SELF_ATTENTION = "attention_self_attention"  # Method 4: self-attention weight
    ATTENTION_VARIANCE = "attention_variance"  # Method 4: attention variance
    ATTENTION_ANOMALY_SCORE = "attention_anomaly_score"  # Method 4: combined attention anomaly


class ForcedGenerationVisualizer:
    """
    Specialized visualizer for forced generation analysis with focus on
    logits and model uncertainty patterns.
    """

    def __init__(self):
        """Initialize the forced generation visualizer."""
        self.color_schemes = {
            ForcedVisualizationMode.FORCED_LOGITS: {
                'colormap': 'RdYlGn',  # Red (low logits) to Green (high logits)
                'label': 'Forced Token Logits',
                'reverse': False,
                'description': 'Raw logit values for each forced token'
            },
            ForcedVisualizationMode.FORCED_PROBABILITY: {
                'colormap': 'RdYlGn',
                'label': 'Forced Token Probability',
                'reverse': False,
                'description': 'Probability the model assigned to each forced token'
            },
            ForcedVisualizationMode.FORCED_LOG_PROBABILITY: {
                'colormap': 'RdYlGn',
                'label': 'Log Probability',
                'reverse': True,  # More negative = worse
                'description': 'Log probability of the forced token (log(p))'
            },
            ForcedVisualizationMode.FORCED_RANK: {
                'colormap': 'RdYlGn',
                'label': 'Forced Token Rank',
                'reverse': True,  # Lower rank = better
                'description': 'Rank of forced token in model\'s preference (1=best)'
            },
            ForcedVisualizationMode.FORCED_SURPRISAL: {
                'colormap': 'RdYlBu',
                'label': 'Token Surprisal',
                'reverse': True,  # Higher surprisal = more unexpected
                'description': 'Unexpectedness of the forced token (-log2(prob))'
            },
            ForcedVisualizationMode.FORCED_CONFIDENCE: {
                'colormap': 'RdYlGn',
                'label': 'Model Confidence',
                'reverse': False,
                'description': 'Combined confidence score for forced token'
            },
            ForcedVisualizationMode.PROBABILITY_MARGIN: {
                'colormap': 'RdYlGn',
                'label': 'Probability Margin',
                'reverse': False,
                'description': 'Difference between top-1 and forced token probability'
            },
            ForcedVisualizationMode.FORCED_CODET5_VALIDATION: {
                'colormap': 'RdYlGn',
                'label': 'CodeT5 Validation Score',
                'reverse': False,
                'description': 'Probability that CodeT5 assigns to the forced token'
            },
            ForcedVisualizationMode.FORCED_NOMIC_COHERENCE: {
                'colormap': 'RdYlGn',
                'label': 'Nomic Coherence Score',
                'reverse': False,
                'description': 'Semantic coherence of the forced token in context'
            },
            # Advanced methods from METHODS_OVERVIEW.md
            ForcedVisualizationMode.SEMANTIC_ENERGY: {
                'colormap': 'RdYlBu',
                'label': 'Semantic Energy',
                'reverse': True,  # Higher energy = higher uncertainty
                'description': 'Energy = -logit(token), higher values indicate more uncertainty (Method 2)'
            },
            ForcedVisualizationMode.CONFORMAL_SCORE: {
                'colormap': 'RdYlBu',
                'label': 'Conformal Prediction Score',
                'reverse': True,  # Higher score = higher uncertainty
                'description': 'Conformal score = 1 - P(token), higher values indicate more uncertainty (Method 3)'
            },
            ForcedVisualizationMode.ATTENTION_ENTROPY: {
                'colormap': 'RdYlBu',
                'label': 'Attention Entropy',
                'reverse': True,  # Higher entropy = more uncertain
                'description': 'Entropy of attention distribution, higher values indicate uncertain attention (Method 4)'
            },
            ForcedVisualizationMode.ATTENTION_SELF_ATTENTION: {
                'colormap': 'RdYlGn',
                'label': 'Self-Attention Weight',
                'reverse': False,  # Higher self-attention = better
                'description': 'Self-attention weight, lower values may indicate anomalies (Method 4)'
            },
            ForcedVisualizationMode.ATTENTION_VARIANCE: {
                'colormap': 'RdYlBu',
                'label': 'Attention Variance',
                'reverse': True,  # Higher variance = more erratic
                'description': 'Variance of attention weights, higher values indicate erratic patterns (Method 4)'
            },
            ForcedVisualizationMode.ATTENTION_ANOMALY_SCORE: {
                'colormap': 'RdYlBu',
                'label': 'Attention Anomaly Score',
                'reverse': True,  # Higher score = more anomalous
                'description': 'Combined attention anomaly score (0-1), higher values indicate anomalies (Method 4)'
            }
        }

    def _convert_to_token_analysis(self, forced_analysis: ForcedTokenAnalysis) -> TokenAnalysis:
        """
        Convert ForcedTokenAnalysis to TokenAnalysis for compatibility with existing visualizer.
        """
        # Convert top alternatives format
        top_k_probs = [(token_id, prob) for token_id, prob, _ in forced_analysis.top_k_alternatives]

        return TokenAnalysis(
            token=forced_analysis.token,
            token_id=forced_analysis.token_id,
            position=forced_analysis.position,
            probability=forced_analysis.probability,
            logit=forced_analysis.logit,
            rank=forced_analysis.rank,
            perplexity=forced_analysis.perplexity,
            entropy=forced_analysis.entropy,
            surprisal=forced_analysis.surprisal,
            top_k_probs=top_k_probs,
            # Map forced-specific metrics to existing fields
            max_probability=forced_analysis.max_probability,
            probability_margin=forced_analysis.probability_margin,
            shannon_entropy=forced_analysis.distribution_entropy,
            local_perplexity=forced_analysis.perplexity,
            sequence_improbability=0.0,  # Not directly available in forced analysis
            confidence_score=forced_analysis.confidence_score,
            # CodeT5 validation metrics
            codet5_validation_score=forced_analysis.codet5_validation_score,
            codet5_alternatives=forced_analysis.codet5_alternatives,
            codet5_predicted_token=forced_analysis.codet5_predicted_token,
            codet5_matches=forced_analysis.codet5_matches,
            # Nomic validation metrics
            nomic_coherence_score=forced_analysis.nomic_coherence_score,
            nomic_similarity_drop=forced_analysis.nomic_similarity_drop,
            nomic_context_similarity=forced_analysis.nomic_context_similarity
        )

    def _normalize_values(self, values: List[float], mode: str) -> List[float]:
        """Normalize values to [0, 1] range for color mapping."""
        if not values:
            return []

        min_val, max_val = min(values), max(values)

        # Handle edge case where all values are the same
        if min_val == max_val:
            return [0.5] * len(values)

        # Special handling for different modes
        if mode == ForcedVisualizationMode.FORCED_LOGITS:
            # For logits, we might want to handle negative values specially
            # Normalize to [0, 1] but preserve the scale
            normalized = [(val - min_val) / (max_val - min_val) for val in values]
        elif mode == ForcedVisualizationMode.FORCED_RANK:
            # For ranks, lower is better, so we reverse the normalization
            normalized = [(max_val - val) / (max_val - min_val) for val in values]
        else:
            # Standard normalization
            normalized = [(val - min_val) / (max_val - min_val) for val in values]

        return normalized

    def _get_color_from_value(self, normalized_value: float, colormap: str) -> str:
        """Convert normalized value to hex color."""
        # Simple red-yellow-green gradient
        if colormap == 'RdYlGn':
            if normalized_value < 0.5:
                # Red to Yellow
                r = 255
                g = int(255 * (normalized_value * 2))
                b = 0
            else:
                # Yellow to Green
                r = int(255 * (2 - normalized_value * 2))
                g = 255
                b = 0
        elif colormap == 'RdYlBu':
            if normalized_value < 0.5:
                # Red to Yellow
                r = 255
                g = int(255 * (normalized_value * 2))
                b = 0
            else:
                # Yellow to Blue
                r = int(255 * (2 - normalized_value * 2))
                g = int(255 * (2 - normalized_value * 2))
                b = int(255 * (normalized_value * 2 - 1))
        else:
            # Default grayscale
            gray = int(255 * normalized_value)
            r, g, b = gray, gray, gray

        return f"#{r:02x}{g:02x}{b:02x}"

    def create_forced_logits_visualization(self,
                                         result: ForcedGenerationResult,
                                         mode: str = ForcedVisualizationMode.FORCED_LOGITS,
                                         title: Optional[str] = None) -> str:
        """
        Create HTML visualization specifically for forced generation logits.

        Args:
            result: Forced generation analysis result
            mode: Visualization mode
            title: Optional title override

        Returns:
            HTML string with the visualization
        """
        if not result.token_analyses:
            return "<p>No token analyses available for visualization.</p>"

        # Extract values based on mode
        if mode == ForcedVisualizationMode.FORCED_LOGITS:
            values = [analysis.logit for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_PROBABILITY:
            values = [analysis.probability for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_LOG_PROBABILITY:
            values = [np.log(analysis.probability) if analysis.probability > 0 else -100.0 for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_RANK:
            values = [analysis.rank for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_SURPRISAL:
            values = [analysis.surprisal for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_CONFIDENCE:
            values = [analysis.confidence_score for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.PROBABILITY_MARGIN:
            values = [analysis.probability_margin for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_CODET5_VALIDATION:
            values = [analysis.codet5_validation_score if analysis.codet5_validation_score is not None else 0.0 for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.FORCED_NOMIC_COHERENCE:
            values = [analysis.nomic_coherence_score if analysis.nomic_coherence_score is not None else 0.5 for analysis in result.token_analyses]
        # Advanced methods from METHODS_OVERVIEW.md
        elif mode == ForcedVisualizationMode.SEMANTIC_ENERGY:
            # Semantic Energy = -logit(token)
            values = [-analysis.logit for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.CONFORMAL_SCORE:
            # Conformal Score = 1 - P(token)
            values = [1.0 - analysis.probability for analysis in result.token_analyses]
        elif mode == ForcedVisualizationMode.ATTENTION_ENTROPY:
            # Note: attention metrics require attention weights which are not currently saved
            # These would need to be added to the data collection process
            values = [0.0 for analysis in result.token_analyses]  # Placeholder
        elif mode == ForcedVisualizationMode.ATTENTION_SELF_ATTENTION:
            values = [0.0 for analysis in result.token_analyses]  # Placeholder
        elif mode == ForcedVisualizationMode.ATTENTION_VARIANCE:
            values = [0.0 for analysis in result.token_analyses]  # Placeholder
        elif mode == ForcedVisualizationMode.ATTENTION_ANOMALY_SCORE:
            values = [0.0 for analysis in result.token_analyses]  # Placeholder
        else:
            values = [analysis.logit for analysis in result.token_analyses]

        # Normalize values for color mapping
        normalized_values = self._normalize_values(values, mode)

        # Get color scheme
        scheme = self.color_schemes.get(mode, self.color_schemes[ForcedVisualizationMode.FORCED_LOGITS])

        # Create title
        if title is None:
            title = f"Forced Generation Analysis: {scheme['label']}"

        # Start HTML
        html_parts = [
            f"<h2>{title}</h2>",
            f"<p><strong>Model:</strong> {result.model_name}</p>",
            f"<p><strong>Problem:</strong> {html.escape(result.original_prompt)}</p>",
            f"<p><strong>Visualization:</strong> {scheme['label']} - {scheme['description']}</p>",
            f"<p><strong>Total Tokens:</strong> {result.total_tokens} | "
            f"<strong>Avg Probability:</strong> {result.average_probability:.3f} | "
            f"<strong>High Uncertainty Tokens:</strong> {result.high_uncertainty_tokens}</p>",
            "<div style='font-family: monospace; font-size: 16px; line-height: 2.0; margin: 20px 0; padding: 15px; border: 1px solid #ccc; background-color: #fafafa;'>"
        ]

        # Add each token with color coding
        for i, (analysis, norm_val, original_val) in enumerate(zip(result.token_analyses, normalized_values, values)):
            color = self._get_color_from_value(norm_val, scheme['colormap'])

            # Handle token display
            token_display = html.escape(analysis.token)
            if token_display == '':
                token_display = '&lt;empty&gt;'
            elif token_display == ' ':
                token_display = '&nbsp;'
            elif token_display == '\n':
                token_display = '\\n<br>'

            # Create detailed tooltip
            tooltip_parts = [
                f"Token: '{html.escape(analysis.token)}'",
                f"Position: {analysis.position}",
                f"Token ID: {analysis.token_id}",
                "",
                "=== Forced Generation Metrics ===",
                f"Logit: {analysis.logit:.4f}",
                f"Probability: {analysis.probability:.6f}",
                f"Rank: {analysis.rank}",
                f"Surprisal: {analysis.surprisal:.4f}",
                f"Confidence: {analysis.confidence_score:.4f}",
                "",
                "=== Distribution Info ===",
                f"Max Probability: {analysis.max_probability:.6f}",
                f"Probability Margin: {analysis.probability_margin:.6f}",
                f"Distribution Entropy: {analysis.distribution_entropy:.4f}",
                "",
                "=== Top Alternatives ===",
            ]

            # Add top alternatives
            for j, (token_id, prob, token_text) in enumerate(analysis.top_k_alternatives[:5]):
                tooltip_parts.append(f"{j+1}. '{html.escape(token_text)}' (prob={prob:.6f})")

            tooltip = "\\n".join(tooltip_parts)

            # Add token span
            html_parts.append(
                f'<span style="background-color: {color}; padding: 4px 2px; margin: 1px; '
                f'border-radius: 4px; border: 1px solid #ddd; display: inline-block; '
                f'cursor: help;" title="{tooltip}">{token_display}</span>'
            )

        html_parts.append("</pre></div>")

        # Add color legend
        html_parts.append(self._create_color_legend(scheme, values))

        # Add summary statistics
        html_parts.append(self._create_summary_section(result))

        return "".join(html_parts)

    def _create_color_legend(self, scheme: Dict, values: List[float]) -> str:
        """Create color legend for the visualization."""
        min_val, max_val = min(values), max(values)

        legend_html = [
            "<div style='margin: 30px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>",
            f"<h3>Color Legend: {scheme['label']}</h3>",
            "<div style='display: flex; align-items: center; margin: 15px 0;'>"
        ]

        # Create gradient bar
        gradient_colors = []
        for i in range(20):  # Use fewer colors for smoother gradient
            norm_val = i / 19.0
            if scheme.get('reverse', False):
                norm_val = 1 - norm_val
            color = self._get_color_from_value(norm_val, scheme['colormap'])
            gradient_colors.append(color)

        gradient = "linear-gradient(to right, " + ", ".join(gradient_colors) + ")"

        legend_html.extend([
            f"<div style='width: 300px; height: 30px; background: {gradient}; border: 2px solid #333; margin-right: 15px; border-radius: 3px;'></div>",
            f"<div style='display: flex; flex-direction: column;'>",
            f"<span style='font-weight: bold;'>Min: {min_val:.4f}</span>",
            f"<span style='font-weight: bold;'>Max: {max_val:.4f}</span>",
            f"</div>",
            "</div>",
            f"<p><em>{scheme['description']}</em></p>",
            "</div>"
        ])

        return "".join(legend_html)

    def _create_summary_section(self, result: ForcedGenerationResult) -> str:
        """Create summary statistics section."""
        # Calculate additional statistics
        logits = [analysis.logit for analysis in result.token_analyses]
        probabilities = [analysis.probability for analysis in result.token_analyses]
        ranks = [analysis.rank for analysis in result.token_analyses]

        summary_html = [
            "<div style='margin: 30px 0; padding: 20px; background-color: #e9ecef; border-radius: 5px;'>",
            "<h3>Summary Statistics</h3>",
            "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>",

            # Left column
            "<div>",
            "<h4>Model Confidence</h4>",
            f"<p><strong>Average Probability:</strong> {result.average_probability:.4f}</p>",
            f"<p><strong>Average Rank:</strong> {result.average_rank:.2f}</p>",
            f"<p><strong>Average Surprisal:</strong> {result.average_surprisal:.4f}</p>",
            f"<p><strong>High Uncertainty Tokens:</strong> {result.high_uncertainty_tokens} / {result.total_tokens} ({result.high_uncertainty_tokens/result.total_tokens*100:.1f}%)</p>",
            "</div>",

            # Right column
            "<div>",
            "<h4>Logit Statistics</h4>",
            f"<p><strong>Min Logit:</strong> {min(logits):.4f}</p>",
            f"<p><strong>Max Logit:</strong> {max(logits):.4f}</p>",
            f"<p><strong>Average Logit:</strong> {np.mean(logits):.4f}</p>",
            f"<p><strong>Logit Std Dev:</strong> {np.std(logits):.4f}</p>",
            "</div>",

            "</div>",
            "</div>"
        ]

        return "".join(summary_html)

    def create_comparison_visualization(self,
                                      correct_result: ForcedGenerationResult,
                                      buggy_result: ForcedGenerationResult,
                                      mode: str = ForcedVisualizationMode.FORCED_LOGITS) -> str:
        """
        Create side-by-side comparison of correct vs buggy code visualizations.

        Args:
            correct_result: Analysis result for correct code
            buggy_result: Analysis result for buggy code
            mode: Visualization mode

        Returns:
            HTML string with comparison visualization
        """
        scheme = self.color_schemes.get(mode, self.color_schemes[ForcedVisualizationMode.FORCED_LOGITS])

        # Create individual visualizations
        correct_viz = self.create_forced_logits_visualization(
            correct_result, mode, f"✅ Correct Code: {scheme['label']}"
        )
        buggy_viz = self.create_forced_logits_visualization(
            buggy_result, mode, f"🔴 Buggy Code: {scheme['label']}"
        )

        # Comparison statistics
        comparison_stats = self._calculate_comparison_stats(correct_result, buggy_result)

        # Combine into comparison layout
        comparison_html = [
            "<h1>🔬 Forced Generation Comparison: Correct vs Buggy Code</h1>",
            "<div style='margin: 20px 0; padding: 20px; background-color: #fff3cd; border-radius: 5px; border-left: 5px solid #ffc107;'>",
            "<h3>📊 Comparison Summary</h3>",
            f"<p><strong>Hypothesis:</strong> Buggy code should have lower logits and higher uncertainty</p>",
            comparison_stats,
            "</div>",
            "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0;'>",
            f"<div style='border: 2px solid #28a745; border-radius: 10px; padding: 20px;'>{correct_viz}</div>",
            f"<div style='border: 2px solid #dc3545; border-radius: 10px; padding: 20px;'>{buggy_viz}</div>",
            "</div>"
        ]

        return "".join(comparison_html)

    def _calculate_comparison_stats(self, correct: ForcedGenerationResult, buggy: ForcedGenerationResult) -> str:
        """Calculate comparison statistics between correct and buggy results."""
        # Calculate differences
        prob_diff = correct.average_probability - buggy.average_probability
        rank_diff = buggy.average_rank - correct.average_rank  # Higher rank is worse
        uncertainty_diff = buggy.high_uncertainty_tokens - correct.high_uncertainty_tokens

        # Calculate logit differences
        correct_logits = [a.logit for a in correct.token_analyses]
        buggy_logits = [a.logit for a in buggy.token_analyses]
        logit_diff = np.mean(correct_logits) - np.mean(buggy_logits)

        stats_html = [
            "<table style='width: 100%; border-collapse: collapse;'>",
            "<tr style='background-color: #f8f9fa;'>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Metric</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Correct</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Buggy</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Difference</th>",
            "<th style='padding: 10px; border: 1px solid #ddd;'>Hypothesis</th>",
            "</tr>",

            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Avg Probability</strong></td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{correct.average_probability:.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{buggy.average_probability:.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{prob_diff:+.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{'✅' if prob_diff > 0 else '❌'}</td></tr>",

            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Avg Logits</strong></td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{np.mean(correct_logits):.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{np.mean(buggy_logits):.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{logit_diff:+.4f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{'✅' if logit_diff > 0 else '❌'}</td></tr>",

            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Avg Rank</strong></td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{correct.average_rank:.2f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{buggy.average_rank:.2f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{rank_diff:+.2f}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{'✅' if rank_diff > 0 else '❌'}</td></tr>",

            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>High Uncertainty Tokens</strong></td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{correct.high_uncertainty_tokens}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{buggy.high_uncertainty_tokens}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{uncertainty_diff:+d}</td>",
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{'✅' if uncertainty_diff > 0 else '❌'}</td></tr>",

            "</table>"
        ]

        return "".join(stats_html)

    def create_hierarchical_structure(self, base_dir: str, model_name: str) -> str:
        """
        Create hierarchical directory structure for organized visualizations.

        Args:
            base_dir: Base directory for forced visualizations
            model_name: Name of the model (will be sanitized for filesystem)

        Returns:
            Path to the model-specific directory
        """
        # Sanitize model name for filesystem
        model_safe_name = model_name.replace("/", "_").replace("-", "_").replace(":", "_")

        # Create directory structure
        base_path = Path(base_dir)
        model_path = base_path / model_safe_name

        base_path.mkdir(exist_ok=True)
        model_path.mkdir(exist_ok=True)

        return str(model_path)

    def save_visualization(self, html_content: str, filepath: str):
        """Save visualization to HTML file with complete HTML document structure."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create complete HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forced Generation Analysis Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #ffffff; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_html)

        print(f"Visualization saved to: {filepath}")

    def save_analysis_data(self, result: ForcedGenerationResult, filepath: str):
        """Save analysis data to JSON file for raw data access."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert result to JSON-serializable format
        analysis_data = {
            "metadata": {
                "model_name": result.model_name,
                "original_prompt": result.original_prompt,
                "target_code": result.target_code,
                "reconstructed_code": result.reconstructed_code,
                "total_tokens": result.total_tokens
            },
            "summary_statistics": {
                "average_probability": float(result.average_probability),
                "average_rank": float(result.average_rank),
                "average_surprisal": float(result.average_surprisal),
                "average_confidence": float(result.average_confidence),
                "high_uncertainty_tokens": result.high_uncertainty_tokens
            },
            "tokens": [
                {
                    "position": analysis.position,
                    "token": analysis.token,
                    "token_id": analysis.token_id,
                    "probability": float(analysis.probability),
                    "logit": float(analysis.logit),
                    "rank": analysis.rank,
                    "surprisal": float(analysis.surprisal),
                    "confidence_score": float(analysis.confidence_score),
                    "top_alternatives": [
                        {
                            "token_id": int(token_id),
                            "probability": float(prob),
                            "token_text": token_text
                        }
                        for token_id, prob, token_text in analysis.top_k_alternatives[:10]
                    ]
                }
                for analysis in result.token_analyses
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)

        print(f"Analysis data saved to: {filepath}")

    def generate_main_index(self, base_dir: str, models_tested: List[Dict[str, Any]]) -> str:
        """
        Generate main index page for forced visualizations (similar to model_visualizations).

        Args:
            base_dir: Base directory for forced visualizations
            models_tested: List of model test results

        Returns:
            Path to generated index.html
        """
        from datetime import datetime

        # Calculate summary statistics
        total_models = len(models_tested)
        successful_models = len([m for m in models_tested if m.get('status') == 'success'])
        total_examples = sum(m.get('examples_count', 0) for m in models_tested)
        total_visualizations = sum(m.get('visualizations_count', 0) for m in models_tested)

        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forced Generation Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f8ff; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #e74c3c; padding-bottom: 15px; }}
        .summary {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 30px 0; }}
        .model-card {{ border: 2px solid #ecf0f1; border-radius: 10px; padding: 20px; background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); transition: transform 0.3s ease; }}
        .model-card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }}
        .model-name {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .model-status {{ padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 15px; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-error {{ background: #f8d7da; color: #721c24; }}
        .model-link {{ display: inline-block; padding: 10px 20px; background: #e74c3c; color: white; text-decoration: none; border-radius: 25px; margin-top: 10px; transition: background 0.3s ease; }}
        .model-link:hover {{ background: #c0392b; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Forced Generation Analysis Results</h1>

        <div class="summary">
            <h2 style="margin-top: 0; color: white;">📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{total_models}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Models Tested</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{successful_models}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Successful</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{total_examples}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Examples Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{total_visualizations}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Visualizations</div>
                </div>
            </div>
            <p style="margin-bottom: 0; text-align: center; font-size: 14px; color: #ecf0f1;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>

        <h2>🤖 Model Results</h2>

        <div class="model-grid">"""

        # Add model cards
        for model in models_tested:
            model_safe_name = model['model_name'].replace("/", "_").replace("-", "_").replace(":", "_")
            status_class = "status-success" if model.get('status') == 'success' else "status-error"
            status_icon = "✅" if model.get('status') == 'success' else "❌"
            status_text = "Success" if model.get('status') == 'success' else f"Error: {model.get('error', 'Unknown')}"

            html_content += f"""
            <div class="model-card">
                <div class="model-name">{model['model_name']}</div>
                <div class="model-status {status_class}">{status_icon} {status_text}</div>

                <p><strong>Examples:</strong> {model.get('examples_count', 0)}</p>
                <p><strong>Visualizations:</strong> {model.get('visualizations_count', 0)}</p>
                <p><strong>Processing Time:</strong> {model.get('processing_time', 0):.1f}s</p>"""

            if model.get('status') == 'success':
                html_content += f"""
                <a href="{model_safe_name}/index.html" class="model-link">View Analysis</a>"""

            html_content += """
            </div>"""

        html_content += f"""
        </div>

        <h2>🔍 About Forced Generation Analysis</h2>
        <p>This analysis demonstrates a novel approach to understanding LLM behavior by forcing models to generate specific code token-by-token while capturing their uncertainty about each forced choice.</p>

        <h3>🎯 Research Hypothesis</h3>
        <p><strong>Core Hypothesis:</strong> When forcing an LLM to generate buggy code, the model will exhibit lower confidence (lower logits and probabilities) compared to generating correct code, indicating implicit knowledge of code quality.</p>

        <h3>📊 Analysis Types</h3>
        <h4>🔹 Individual Analysis</h4>
        <ul>
            <li><strong>Forced Logits:</strong> Raw logit values assigned to each forced token</li>
            <li><strong>Forced Probability:</strong> Probability the model assigned to each forced token</li>
            <li><strong>Forced Rank:</strong> Ranking of forced token among all vocabulary options</li>
            <li><strong>Surprisal:</strong> Unexpectedness of each forced token choice</li>
        </ul>

        <h4>🔹 Comparative Analysis</h4>
        <ul>
            <li><strong>Side-by-Side Comparisons:</strong> Direct visual comparison of correct vs buggy code metrics</li>
            <li><strong>Statistical Analysis:</strong> Quantitative validation of hypothesis with significance testing</li>
            <li><strong>Uncertainty Patterns:</strong> Identification of high-uncertainty regions in forced generation</li>
        </ul>

        <h3>📁 Directory Structure</h3>
        <ul>
            <li><code>index.html</code> - This main overview page</li>
            <li><code>[model_name]/</code> - Individual directories for each model tested</li>
            <li><code>[model_name]/index.html</code> - Model-specific analysis overview</li>
            <li><code>[model_name]/factorial_correct_[metric].html</code> - Individual visualizations for correct code</li>
            <li><code>[model_name]/factorial_buggy_[metric].html</code> - Individual visualizations for buggy code</li>
            <li><code>[model_name]/comparison_[metric].html</code> - Side-by-side comparison visualizations</li>
            <li><code>[model_name]/detailed_analysis_report.html</code> - Scientific analysis report</li>
            <li><code>[model_name]/factorial_[type]_analysis.json</code> - Raw analysis data</li>
        </ul>

        <h3>🧪 Methodology</h3>
        <p>Each model was presented with a programming problem and forced to generate both correct and buggy implementations token-by-token. The model's confidence in each forced token was measured through multiple metrics, enabling analysis of whether the model exhibits implicit knowledge about code quality even when constrained to generate specific outputs.</p>
    </div>
</body>
</html>"""

        # Save to file
        index_path = Path(base_dir) / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Main index generated: {index_path}")
        return str(index_path)

    def generate_model_index(self, model_dir: str, model_name: str, analysis_results: Dict[str, Any]) -> str:
        """
        Generate model-specific index page (similar to model_visualizations structure).

        Args:
            model_dir: Model-specific directory path
            model_name: Name of the model
            analysis_results: Dictionary containing analysis results and file info

        Returns:
            Path to generated model index.html
        """
        from datetime import datetime

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forced Generation Analysis - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #555; border-left: 4px solid #e74c3c; padding-left: 15px; }}
        .stats {{ background: #ffe8e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .stats p {{ margin: 5px 0; }}
        .example-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .example-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }}
        .example-card.correct {{ border-left: 5px solid #28a745; background: #f8fff8; }}
        .example-card.buggy {{ border-left: 5px solid #dc3545; background: #fff8f8; }}
        .example-title {{ font-weight: bold; color: #333; margin-bottom: 10px; font-size: 16px; }}
        .example-subtitle {{ color: #666; font-size: 14px; margin-bottom: 15px; }}
        .viz-links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .viz-link {{ padding: 8px 12px; color: white; text-decoration: none; border-radius: 4px; font-size: 14px; transition: all 0.3s ease; }}
        .viz-link:hover {{ transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .viz-link.logits {{ background: #007bff; }}
        .viz-link.probability {{ background: #28a745; }}
        .viz-link.rank {{ background: #ffc107; color: #333; }}
        .viz-link.surprisal {{ background: #6f42c1; }}
        .viz-link.comparison {{ background: #fd7e14; }}
        .viz-link.analysis {{ background: #20c997; }}
        .viz-link.data {{ background: #6c757d; }}
        .section {{ margin: 30px 0; }}
        .back-link {{ display: inline-block; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px; margin-bottom: 20px; }}
        .back-link:hover {{ background: #5a6268; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../index.html" class="back-link">← Back to Main Index</a>

        <h1>🔬 Forced Generation Analysis - {model_name}</h1>

        <div class="stats">
            <h3>📊 Analysis Summary</h3>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Examples Analyzed:</strong> {analysis_results.get('examples_count', 2)}</p>
            <p><strong>Total Visualizations:</strong> {analysis_results.get('visualizations_count', 0)}</p>
            <p><strong>Processing Time:</strong> {analysis_results.get('processing_time', 0):.1f}s</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <h2>📝 Examples and Analysis</h2>

        <div class="example-grid">
            <!-- Correct Code Example -->
            <div class="example-card correct">
                <div class="example-title">✅ Factorial (Correct Implementation)</div>
                <div class="example-subtitle">Proper base case: n &lt;= 1</div>
                <p><strong>Description:</strong> Correct recursive factorial with complete base case handling</p>

                <h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">Individual Visualizations</h4>
                <div class="viz-links" style="margin-bottom: 15px;">
                    <a href="factorial_correct_logits.html" class="viz-link logits" target="_blank">LOGITS</a>
                    <a href="factorial_correct_probability.html" class="viz-link probability" target="_blank">PROBABILITY</a>
                    <a href="factorial_correct_rank.html" class="viz-link rank" target="_blank">RANK</a>
                    <a href="factorial_correct_surprisal.html" class="viz-link surprisal" target="_blank">SURPRISAL</a>
                </div>

                <h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">Raw Data</h4>
                <a href="factorial_correct_analysis.json" class="viz-link data" target="_blank">Analysis JSON</a>
            </div>

            <!-- Buggy Code Example -->
            <div class="example-card buggy">
                <div class="example-title">🔴 Factorial (Buggy Implementation)</div>
                <div class="example-subtitle">Missing base case: n == 1 only</div>
                <p><strong>Description:</strong> Buggy factorial missing n=0 base case, will fail for edge cases</p>

                <h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">Individual Visualizations</h4>
                <div class="viz-links" style="margin-bottom: 15px;">
                    <a href="factorial_buggy_logits.html" class="viz-link logits" target="_blank">LOGITS</a>
                    <a href="factorial_buggy_probability.html" class="viz-link probability" target="_blank">PROBABILITY</a>
                    <a href="factorial_buggy_rank.html" class="viz-link rank" target="_blank">RANK</a>
                    <a href="factorial_buggy_surprisal.html" class="viz-link surprisal" target="_blank">SURPRISAL</a>
                </div>

                <h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">Raw Data</h4>
                <a href="factorial_buggy_analysis.json" class="viz-link data" target="_blank">Analysis JSON</a>
            </div>
        </div>

        <div class="section">
            <h2>🔄 Comparative Analysis</h2>
            <p>Side-by-side comparisons of correct vs buggy code to validate the research hypothesis:</p>

            <div class="viz-links">
                <a href="comparison_logits.html" class="viz-link comparison" target="_blank">LOGITS COMPARISON</a>
                <a href="comparison_probability.html" class="viz-link comparison" target="_blank">PROBABILITY COMPARISON</a>
                <a href="comparison_rank.html" class="viz-link comparison" target="_blank">RANK COMPARISON</a>
                <a href="detailed_analysis_report.html" class="viz-link analysis" target="_blank">DETAILED SCIENTIFIC REPORT</a>
            </div>
        </div>

        <div class="section">
            <h2>🔍 About Forced Generation</h2>
            <p>This analysis uses a novel <strong>forced generation</strong> approach where the model is constrained to produce specific code token-by-token, while we capture the model's confidence about each forced choice.</p>

            <h3>Key Insights:</h3>
            <ul>
                <li><strong>Logits:</strong> Raw model outputs showing genuine confidence levels</li>
                <li><strong>Probability:</strong> Normalized confidence the model had in each forced token</li>
                <li><strong>Rank:</strong> Position of forced token among all vocabulary options (lower = more preferred)</li>
                <li><strong>Surprisal:</strong> How unexpected each forced choice was to the model</li>
            </ul>

            <h3>Research Question:</h3>
            <p><em>Does the model exhibit lower confidence when forced to generate buggy code compared to correct code, suggesting implicit knowledge of code quality?</em></p>
        </div>
    </div>
</body>
</html>"""

        # Save to file
        index_path = Path(model_dir) / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Model index generated: {index_path}")
        return str(index_path)


if __name__ == "__main__":
    # Quick test
    print("Forced Generation Visualizer module loaded successfully!")