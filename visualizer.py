import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import List, Dict, Optional, Tuple
import html
from IPython.display import HTML, display
import seaborn as sns
from LLM import TokenAnalysis
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class TokenVisualizationMode:
    """Enumeration of visualization modes"""
    PROBABILITY = "probability"
    LOG_PROBABILITY = "log_probability"
    LOGITS = "logits"
    RANK = "rank"
    ENTROPY = "entropy"
    SURPRISAL = "surprisal"
    PERPLEXITY = "perplexity"
    # New advanced modes
    MAX_PROBABILITY = "max_probability"
    PROBABILITY_MARGIN = "probability_margin"
    SHANNON_ENTROPY = "shannon_entropy"
    LOCAL_PERPLEXITY = "local_perplexity"
    SEQUENCE_IMPROBABILITY = "sequence_improbability"
    CONFIDENCE_SCORE = "confidence_score"
    # CodeT5 validation mode
    CODET5_VALIDATION = "codet5_validation"
    # Nomic-embed-code validation mode
    NOMIC_COHERENCE = "nomic_coherence"
    # LecPrompt logical error detection modes
    LOGICAL_ERROR_DETECTION = "logical_error_detection"
    STATISTICAL_DEVIATION = "statistical_deviation"
    ERROR_LIKELIHOOD = "error_likelihood"
    # Advanced methods from METHODS_OVERVIEW.md
    SEMANTIC_ENERGY = "semantic_energy"  # Method 2: pre-softmax logits energy
    CONFORMAL_SCORE = "conformal_score"  # Method 3: conformal prediction uncertainty
    ATTENTION_ENTROPY = "attention_entropy"  # Method 4: attention distribution entropy
    ATTENTION_SELF_ATTENTION = "attention_self_attention"  # Method 4: self-attention weight
    ATTENTION_VARIANCE = "attention_variance"  # Method 4: attention variance
    ATTENTION_ANOMALY_SCORE = "attention_anomaly_score"  # Method 4: combined attention anomaly
    # Aggregated suspicion score
    SUSPICION_SCORE = "suspicion_score"  # Aggregated suspicion score (0-100)


class TokenVisualizer:
    """
    Visualizer for token generation analysis with multiple display modes
    and color-coding based on different metrics.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        # Color schemes for different metrics
        self.color_schemes = {
            TokenVisualizationMode.PROBABILITY: {
                'colormap': 'RdYlGn',  # Red (low) to Green (high)
                'label': 'Token Probability',
                'reverse': False
            },
            TokenVisualizationMode.LOG_PROBABILITY: {
                'colormap': 'RdYlGn',
                'label': 'Log Probability',
                'reverse': True,  # More negative is worse
                'description': 'Log probability of the token (log(p))'
            },
            TokenVisualizationMode.LOGITS: {
                'colormap': 'RdYlGn',
                'label': 'Logit Value',
                'reverse': False
            },
            TokenVisualizationMode.RANK: {
                'colormap': 'RdYlGn',
                'label': 'Token Rank (1=best)',
                'reverse': True  # Lower rank (1) should be green
            },
            TokenVisualizationMode.ENTROPY: {
                'colormap': 'RdYlBu',  # Red (high entropy) to Blue (low entropy)
                'label': 'Model Entropy',
                'reverse': True
            },
            TokenVisualizationMode.SURPRISAL: {
                'colormap': 'RdYlBu',
                'label': 'Token Surprisal',
                'reverse': True
            },
            TokenVisualizationMode.PERPLEXITY: {
                'colormap': 'RdYlBu',
                'label': 'Model Perplexity',
                'reverse': True
            },
            # New advanced visualization modes
            TokenVisualizationMode.MAX_PROBABILITY: {
                'colormap': 'RdYlGn',
                'label': 'Max Probability in Distribution',
                'reverse': False
            },
            TokenVisualizationMode.PROBABILITY_MARGIN: {
                'colormap': 'RdYlGn',
                'label': 'Probability Margin (Top1-Top2)',
                'reverse': False
            },
            TokenVisualizationMode.SHANNON_ENTROPY: {
                'colormap': 'RdYlBu',
                'label': 'Shannon Entropy',
                'reverse': True
            },
            TokenVisualizationMode.LOCAL_PERPLEXITY: {
                'colormap': 'RdYlBu',
                'label': 'Local Perplexity',
                'reverse': True
            },
            TokenVisualizationMode.SEQUENCE_IMPROBABILITY: {
                'colormap': 'RdPu',
                'label': 'Sequence Improbability',
                'reverse': True
            },
            TokenVisualizationMode.CONFIDENCE_SCORE: {
                'colormap': 'RdYlGn',
                'label': 'Confidence Score',
                'reverse': False
            },
            TokenVisualizationMode.CODET5_VALIDATION: {
                'colormap': 'RdYlGn',
                'label': 'CodeT5 Validation Score',
                'reverse': False,
                'description': 'Probability that CodeT5 assigns to this token (high=correct, low=suspicious)'
            },
            TokenVisualizationMode.NOMIC_COHERENCE: {
                'colormap': 'RdYlGn',
                'label': 'Nomic Coherence Score',
                'reverse': False,
                'description': 'Semantic coherence of token in context (high=coherent, low=inconsistent)'
            },
            # LecPrompt logical error detection modes
            TokenVisualizationMode.LOGICAL_ERROR_DETECTION: {
                'colormap': 'RdYlGn',
                'label': 'Logical Error Detection',
                'reverse': True,
                'description': 'Anomaly-based error detection (red=likely error, green=normal)'
            },
            TokenVisualizationMode.STATISTICAL_DEVIATION: {
                'colormap': 'RdYlBu',
                'label': 'Statistical Deviation (std devs)',
                'reverse': True,
                'description': 'Deviation from mean in standard deviations'
            },
            TokenVisualizationMode.ERROR_LIKELIHOOD: {
                'colormap': 'RdYlGn',
                'label': 'Error Likelihood Score',
                'reverse': True,
                'description': 'Likelihood of logical error (0=safe, 1=error)'
            },
            # Advanced methods from METHODS_OVERVIEW.md
            TokenVisualizationMode.SEMANTIC_ENERGY: {
                'colormap': 'RdYlBu',
                'label': 'Semantic Energy (pre-softmax)',
                'reverse': True,
                'description': 'Energy = -logit(token), higher = more uncertain (Farquhar et al., NeurIPS 2024)'
            },
            TokenVisualizationMode.CONFORMAL_SCORE: {
                'colormap': 'RdYlBu',
                'label': 'Conformal Prediction Score',
                'reverse': True,
                'description': 'Conformal score = 1 - P(token), higher = larger prediction set (Quach et al., ICLR 2024)'
            },
            TokenVisualizationMode.ATTENTION_ENTROPY: {
                'colormap': 'RdYlBu',
                'label': 'Attention Entropy',
                'reverse': True,
                'description': 'Entropy of attention distribution, higher = more uncertain (Ott et al., ICML 2018)'
            },
            TokenVisualizationMode.ATTENTION_SELF_ATTENTION: {
                'colormap': 'RdYlGn',
                'label': 'Self-Attention Weight',
                'reverse': False,
                'description': 'Self-attention weight (a_ii), lower = potentially anomalous'
            },
            TokenVisualizationMode.ATTENTION_VARIANCE: {
                'colormap': 'RdYlBu',
                'label': 'Attention Variance',
                'reverse': True,
                'description': 'Variance of attention weights, higher = erratic attention pattern'
            },
            TokenVisualizationMode.ATTENTION_ANOMALY_SCORE: {
                'colormap': 'RdYlGn',
                'label': 'Attention Anomaly Score',
                'reverse': True,
                'description': 'Combined attention anomaly: 0.5×H + 0.3×(1-SA) + 0.2×Var (Jesse et al., ICSE 2023)'
            },
            # Aggregated suspicion score
            TokenVisualizationMode.SUSPICION_SCORE: {
                'colormap': 'RdYlGn',
                'label': 'Suspicion Score (0-100)',
                'reverse': True,
                'description': 'Aggregated suspicion score combining rank, surprisal, entropy, margin. Score >= 60: HIGH risk, >= 40: MEDIUM risk, >= 20: LOW risk, < 20: Safe'
            }
        }
    
    def _normalize_values(self, values: List[float], mode: str) -> List[float]:
        """
        Normalize values for color mapping.

        Args:
            values: List of metric values
            mode: Visualization mode

        Returns:
            Normalized values between 0 and 1
        """
        values = np.array(values)

        if mode == TokenVisualizationMode.RANK:
            # For rank, we want to handle it specially (log scale for better visualization)
            values = np.log(values + 1)

        # Normalize to [0, 1]
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            # All values are the same - show warning
            print(f"⚠️  WARNING: All values for '{mode}' are identical ({min_val:.6f})")
            print(f"   This metric may not be available or calculated for this analysis.")
            print(f"   All tokens will be displayed with neutral color.")
            return [0.5] * len(values)  # All values are the same

        normalized = (values - min_val) / (max_val - min_val)

        # Reverse if needed
        if self.color_schemes[mode]['reverse']:
            normalized = 1 - normalized

        return normalized.tolist()
    
    def _get_color_from_value(self, normalized_value: float, colormap: str) -> str:
        """
        Get hex color from normalized value using the specified colormap.
        
        Args:
            normalized_value: Value between 0 and 1
            colormap: Name of matplotlib colormap
            
        Returns:
            Hex color string
        """
        cmap = plt.cm.get_cmap(colormap)
        rgba = cmap(normalized_value)
        return mcolors.rgb2hex(rgba[:3])
    
    def create_html_visualization(self, 
                                analyses: List[TokenAnalysis], 
                                mode: str = TokenVisualizationMode.PROBABILITY,
                                title: str = "Token Analysis Visualization") -> str:
        """
        Create HTML visualization of tokens with color coding.
        
        Args:
            analyses: List of token analyses
            mode: Visualization mode
            title: Title for the visualization
            
        Returns:
            HTML string for display
        """
        if not analyses:
            return "<p>No tokens to visualize</p>"
        
        # Extract values based on mode
        if mode == TokenVisualizationMode.PROBABILITY:
            values = [analysis.probability for analysis in analyses]
        elif mode == TokenVisualizationMode.LOG_PROBABILITY:
            values = [np.log(analysis.probability) if analysis.probability > 0 else -100.0 for analysis in analyses]
        elif mode == TokenVisualizationMode.LOGITS:
            values = [analysis.logit for analysis in analyses]
        elif mode == TokenVisualizationMode.RANK:
            values = [analysis.rank for analysis in analyses]
        elif mode == TokenVisualizationMode.ENTROPY:
            values = [analysis.entropy for analysis in analyses]
        elif mode == TokenVisualizationMode.SURPRISAL:
            values = [analysis.surprisal for analysis in analyses]
        elif mode == TokenVisualizationMode.PERPLEXITY:
            values = [analysis.perplexity for analysis in analyses]
        # New advanced modes
        elif mode == TokenVisualizationMode.MAX_PROBABILITY:
            values = [analysis.max_probability for analysis in analyses]
        elif mode == TokenVisualizationMode.PROBABILITY_MARGIN:
            values = [analysis.probability_margin for analysis in analyses]
        elif mode == TokenVisualizationMode.SHANNON_ENTROPY:
            values = [analysis.shannon_entropy for analysis in analyses]
        elif mode == TokenVisualizationMode.LOCAL_PERPLEXITY:
            values = [analysis.local_perplexity for analysis in analyses]
        elif mode == TokenVisualizationMode.SEQUENCE_IMPROBABILITY:
            values = [analysis.sequence_improbability for analysis in analyses]
        elif mode == TokenVisualizationMode.CONFIDENCE_SCORE:
            values = [analysis.confidence_score for analysis in analyses]
        elif mode == TokenVisualizationMode.CODET5_VALIDATION:
            values = [analysis.codet5_validation_score if analysis.codet5_validation_score is not None else 0.0 for analysis in analyses]
        elif mode == TokenVisualizationMode.NOMIC_COHERENCE:
            values = [analysis.nomic_coherence_score if analysis.nomic_coherence_score is not None else 0.5 for analysis in analyses]
        # LecPrompt logical error detection modes
        elif mode == TokenVisualizationMode.LOGICAL_ERROR_DETECTION:
            values = [1.0 - (analysis.error_likelihood if analysis.error_likelihood is not None else 0.5) for analysis in analyses]
        elif mode == TokenVisualizationMode.STATISTICAL_DEVIATION:
            values = [abs(analysis.statistical_score) if analysis.statistical_score is not None else 0.0 for analysis in analyses]
        elif mode == TokenVisualizationMode.ERROR_LIKELIHOOD:
            values = [1.0 - (analysis.error_likelihood if analysis.error_likelihood is not None else 0.0) for analysis in analyses]
        # Advanced methods from METHODS_OVERVIEW.md
        elif mode == TokenVisualizationMode.SEMANTIC_ENERGY:
            values = [analysis.semantic_energy if analysis.semantic_energy is not None else -analysis.logit for analysis in analyses]
        elif mode == TokenVisualizationMode.CONFORMAL_SCORE:
            values = [analysis.conformal_score if analysis.conformal_score is not None else 1.0 - analysis.probability for analysis in analyses]
        elif mode == TokenVisualizationMode.ATTENTION_ENTROPY:
            values = [analysis.attention_entropy if analysis.attention_entropy is not None else 0.0 for analysis in analyses]
        elif mode == TokenVisualizationMode.ATTENTION_SELF_ATTENTION:
            values = [analysis.attention_self_attention if analysis.attention_self_attention is not None else 0.5 for analysis in analyses]
        elif mode == TokenVisualizationMode.ATTENTION_VARIANCE:
            values = [analysis.attention_variance if analysis.attention_variance is not None else 0.0 for analysis in analyses]
        elif mode == TokenVisualizationMode.ATTENTION_ANOMALY_SCORE:
            values = [analysis.attention_anomaly_score if analysis.attention_anomaly_score is not None else 0.0 for analysis in analyses]
        # Aggregated suspicion score
        elif mode == TokenVisualizationMode.SUSPICION_SCORE:
            values = [analysis.suspicion_score if analysis.suspicion_score is not None else 0.0 for analysis in analyses]
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        # Normalize values
        normalized_values = self._normalize_values(values, mode)

        # Get color scheme
        scheme = self.color_schemes[mode]

        # Check if all values are uniform
        min_val, max_val = min(values), max(values)
        is_uniform = (max_val == min_val)

        # Create HTML
        html_parts = [
            f"<h2>{title}</h2>",
            f"<p><strong>Visualization Mode:</strong> {scheme['label']}</p>"
        ]

        # Add warning if values are uniform
        if is_uniform:
            html_parts.append(
                f"<div style='background-color: #fff3cd; border: 2px solid #ffc107; "
                f"padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                f"<strong>⚠️ Warning:</strong> All tokens have the same value ({min_val:.6f}) "
                f"for this metric. This metric may not be available or calculated for this analysis. "
                f"All tokens are displayed with neutral color."
                f"</div>"
            )

        html_parts.append("<div class='code-block'><pre class='token-code' style='margin: 0; white-space: pre-wrap; word-wrap: break-word;'>")
        
        for i, (analysis, norm_val, original_val) in enumerate(zip(analyses, normalized_values, values)):
            color = self._get_color_from_value(norm_val, scheme['colormap'])
            
            # Escape HTML characters in token
            token_display = html.escape(analysis.token).replace(' ', '&nbsp;')
            if token_display == '':
                token_display = '&lt;empty&gt;'
            
            # Create tooltip with detailed information
            tooltip_parts = [
                f"Token: {html.escape(analysis.token)}",
                f"Position: {analysis.position}",
                "",
                "=== Basic Metrics ===",
                f"Probability: {analysis.probability:.6f}",
                f"Rank: {analysis.rank}",
                f"Logit: {analysis.logit:.4f}",
                "",
                "=== Information Theory ===",
                f"Entropy: {analysis.entropy:.4f}",
                f"Surprisal: {analysis.surprisal:.4f}",
                f"Perplexity: {analysis.perplexity:.4f}",
                "",
                "=== Advanced Metrics ===",
                f"Max Probability: {analysis.max_probability:.6f}",
                f"Prob. Margin: {analysis.probability_margin:.6f}",
                f"Shannon Entropy: {analysis.shannon_entropy:.4f}",
                f"Local Perplexity: {analysis.local_perplexity:.4f}",
                f"Sequence Improbability: {analysis.sequence_improbability:.6f}",
                f"Confidence Score: {analysis.confidence_score:.4f}"
            ]

            # Add LM-Polygraph metrics if available
            if analysis.lm_polygraph_metrics:
                tooltip_parts.extend([
                    "",
                    "=== LM-Polygraph Metrics ==="
                ])
                for metric_name, metric_value in analysis.lm_polygraph_metrics.items():
                    tooltip_parts.append(f"{metric_name}: {metric_value:.4f}")

            # Add CodeT5 validation metrics if available
            if analysis.codet5_validation_score is not None:
                tooltip_parts.extend([
                    "",
                    "=== CodeT5 Validation ==="
                ])
                tooltip_parts.append(f"Validation Score: {analysis.codet5_validation_score:.6f}")
                if analysis.codet5_predicted_token:
                    tooltip_parts.append(f"Predicted Token: {analysis.codet5_predicted_token}")
                if analysis.codet5_matches is not None:
                    match_str = "✓ Matches" if analysis.codet5_matches else "✗ Differs"
                    tooltip_parts.append(f"Match: {match_str}")
                if analysis.codet5_alternatives:
                    tooltip_parts.append("Top Alternatives:")
                    for alt_token, alt_prob in analysis.codet5_alternatives[:3]:
                        tooltip_parts.append(f"  - {alt_token}: {alt_prob:.4f}")

            # Add Nomic-embed-code validation metrics if available
            if analysis.nomic_coherence_score is not None:
                tooltip_parts.extend([
                    "",
                    "=== Nomic Semantic Coherence ==="
                ])
                tooltip_parts.append(f"Coherence Score: {analysis.nomic_coherence_score:.6f}")
                if analysis.nomic_similarity_drop is not None:
                    tooltip_parts.append(f"Similarity Drop: {analysis.nomic_similarity_drop:.6f}")
                if analysis.nomic_context_similarity is not None:
                    tooltip_parts.append(f"Context Similarity: {analysis.nomic_context_similarity:.6f}")

            # Add LecPrompt error detection metrics if available
            if analysis.is_anomalous is not None:
                tooltip_parts.extend([
                    "",
                    "=== LecPrompt Error Detection ==="
                ])
                tooltip_parts.append(f"Is Anomalous: {'YES' if analysis.is_anomalous else 'NO'}")
                if analysis.statistical_score is not None:
                    tooltip_parts.append(f"Statistical Score: {analysis.statistical_score:.3f} σ")
                if analysis.error_likelihood is not None:
                    tooltip_parts.append(f"Error Likelihood: {analysis.error_likelihood:.3f}")

            # Add advanced methods metrics if available
            has_advanced_metrics = any([
                analysis.semantic_energy is not None,
                analysis.conformal_score is not None,
                analysis.attention_entropy is not None,
                analysis.attention_self_attention is not None,
                analysis.attention_variance is not None,
                analysis.attention_anomaly_score is not None
            ])
            if has_advanced_metrics:
                tooltip_parts.extend([
                    "",
                    "=== Advanced Detection Methods ==="
                ])
                if analysis.semantic_energy is not None:
                    tooltip_parts.append(f"Semantic Energy: {analysis.semantic_energy:.4f}")
                if analysis.conformal_score is not None:
                    tooltip_parts.append(f"Conformal Score: {analysis.conformal_score:.4f}")
                if analysis.attention_entropy is not None:
                    tooltip_parts.append(f"Attention Entropy: {analysis.attention_entropy:.4f}")
                if analysis.attention_self_attention is not None:
                    tooltip_parts.append(f"Self-Attention: {analysis.attention_self_attention:.4f}")
                if analysis.attention_variance is not None:
                    tooltip_parts.append(f"Attention Variance: {analysis.attention_variance:.4f}")
                if analysis.attention_anomaly_score is not None:
                    tooltip_parts.append(f"Attention Anomaly Score: {analysis.attention_anomaly_score:.4f}")

            tooltip = "\\n".join(tooltip_parts)
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 1px; margin: 0; '
                f'border-radius: 3px; display: inline-block;" '
                f'title="{tooltip}">{token_display}</span>'
            )
        
        html_parts.append("</pre></div>")
        
        # Add legend
        html_parts.append(self._create_color_legend(scheme, values))
        
        return "".join(html_parts)
    
    def _create_color_legend(self, scheme: Dict, values: List[float]) -> str:
        """
        Create a color legend for the visualization.
        
        Args:
            scheme: Color scheme dictionary
            values: Original values for min/max calculation
            
        Returns:
            HTML string for legend
        """
        min_val, max_val = min(values), max(values)
        
        legend_html = [
            "<div style='margin: 20px 0;'>",
            f"<p><strong>Legend ({scheme['label']}):</strong></p>",
            "<div style='display: flex; align-items: center; margin: 10px 0;'>"
        ]
        
        # Create gradient bar
        gradient_colors = []
        for i in range(100):
            norm_val = i / 99.0
            if scheme['reverse']:
                norm_val = 1 - norm_val
            color = self._get_color_from_value(norm_val, scheme['colormap'])
            gradient_colors.append(color)
        
        gradient = "linear-gradient(to right, " + ", ".join(gradient_colors) + ")"
        
        legend_html.extend([
            f"<div style='width: 200px; height: 20px; background: {gradient}; border: 1px solid #ccc; margin-right: 10px;'></div>",
            f"<span style='margin-right: 10px;'>{min_val:.4f}</span>",
            f"<span>{max_val:.4f}</span>",
            "</div>",
            "</div>"
        ])
        
        return "".join(legend_html)
    
    def create_matplotlib_visualization(self, 
                                      analyses: List[TokenAnalysis],
                                      mode: str = TokenVisualizationMode.PROBABILITY,
                                      figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """
        Create matplotlib visualization with detailed plots.
        
        Args:
            analyses: List of token analyses
            mode: Visualization mode
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not analyses:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No tokens to visualize', ha='center', va='center')
            return fig
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Token Generation Analysis', fontsize=16)
        
        positions = [analysis.position for analysis in analyses]
        
        # Plot 1: Main metric over time
        if mode == TokenVisualizationMode.PROBABILITY:
            values = [analysis.probability for analysis in analyses]
            ax1.set_ylabel('Probability')
        elif mode == TokenVisualizationMode.LOGITS:
            values = [analysis.logit for analysis in analyses]
            ax1.set_ylabel('Logit Value')
        elif mode == TokenVisualizationMode.RANK:
            values = [analysis.rank for analysis in analyses]
            ax1.set_ylabel('Rank')
            ax1.set_yscale('log')
        elif mode == TokenVisualizationMode.ENTROPY:
            values = [analysis.entropy for analysis in analyses]
            ax1.set_ylabel('Entropy')
        elif mode == TokenVisualizationMode.SURPRISAL:
            values = [analysis.surprisal for analysis in analyses]
            ax1.set_ylabel('Surprisal')
        elif mode == TokenVisualizationMode.PERPLEXITY:
            values = [analysis.perplexity for analysis in analyses]
            ax1.set_ylabel('Perplexity')
        # New advanced modes
        elif mode == TokenVisualizationMode.MAX_PROBABILITY:
            values = [analysis.max_probability for analysis in analyses]
            ax1.set_ylabel('Max Probability')
        elif mode == TokenVisualizationMode.PROBABILITY_MARGIN:
            values = [analysis.probability_margin for analysis in analyses]
            ax1.set_ylabel('Probability Margin')
        elif mode == TokenVisualizationMode.SHANNON_ENTROPY:
            values = [analysis.shannon_entropy for analysis in analyses]
            ax1.set_ylabel('Shannon Entropy')
        elif mode == TokenVisualizationMode.LOCAL_PERPLEXITY:
            values = [analysis.local_perplexity for analysis in analyses]
            ax1.set_ylabel('Local Perplexity')
        elif mode == TokenVisualizationMode.SEQUENCE_IMPROBABILITY:
            values = [analysis.sequence_improbability for analysis in analyses]
            ax1.set_ylabel('Sequence Improbability')
        elif mode == TokenVisualizationMode.CONFIDENCE_SCORE:
            values = [analysis.confidence_score for analysis in analyses]
            ax1.set_ylabel('Confidence Score')
        # Advanced methods from METHODS_OVERVIEW.md
        elif mode == TokenVisualizationMode.SEMANTIC_ENERGY:
            values = [analysis.semantic_energy if analysis.semantic_energy is not None else -analysis.logit for analysis in analyses]
            ax1.set_ylabel('Semantic Energy')
        elif mode == TokenVisualizationMode.CONFORMAL_SCORE:
            values = [analysis.conformal_score if analysis.conformal_score is not None else 1.0 - analysis.probability for analysis in analyses]
            ax1.set_ylabel('Conformal Score')
        elif mode == TokenVisualizationMode.ATTENTION_ENTROPY:
            values = [analysis.attention_entropy if analysis.attention_entropy is not None else 0.0 for analysis in analyses]
            ax1.set_ylabel('Attention Entropy')
        elif mode == TokenVisualizationMode.ATTENTION_SELF_ATTENTION:
            values = [analysis.attention_self_attention if analysis.attention_self_attention is not None else 0.5 for analysis in analyses]
            ax1.set_ylabel('Self-Attention Weight')
        elif mode == TokenVisualizationMode.ATTENTION_VARIANCE:
            values = [analysis.attention_variance if analysis.attention_variance is not None else 0.0 for analysis in analyses]
            ax1.set_ylabel('Attention Variance')
        elif mode == TokenVisualizationMode.ATTENTION_ANOMALY_SCORE:
            values = [analysis.attention_anomaly_score if analysis.attention_anomaly_score is not None else 0.0 for analysis in analyses]
            ax1.set_ylabel('Attention Anomaly Score')
        # Aggregated suspicion score
        elif mode == TokenVisualizationMode.SUSPICION_SCORE:
            values = [analysis.suspicion_score if analysis.suspicion_score is not None else 0.0 for analysis in analyses]
            ax1.set_ylabel('Suspicion Score (0-100)')

        ax1.plot(positions, values, 'b-', linewidth=2)
        ax1.set_xlabel('Token Position')
        ax1.set_title(f'{self.color_schemes[mode]["label"]} over Generation')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Probability distribution
        probabilities = [analysis.probability for analysis in analyses]
        ax2.hist(probabilities, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Token Probabilities')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rank distribution
        ranks = [analysis.rank for analysis in analyses]
        ax3.hist(ranks, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Token Ranks')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Entropy vs Surprisal
        entropies = [analysis.entropy for analysis in analyses]
        surprisals = [analysis.surprisal for analysis in analyses]
        scatter = ax4.scatter(entropies, surprisals, c=probabilities, cmap='RdYlGn', alpha=0.7)
        ax4.set_xlabel('Entropy')
        ax4.set_ylabel('Surprisal')
        ax4.set_title('Entropy vs Surprisal (colored by probability)')
        plt.colorbar(scatter, ax=ax4, label='Probability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plotly_visualization(self, 
                                              analyses: List[TokenAnalysis],
                                              mode: str = TokenVisualizationMode.PROBABILITY) -> go.Figure:
        """
        Create interactive Plotly visualization.
        
        Args:
            analyses: List of token analyses
            mode: Visualization mode
            
        Returns:
            Plotly figure
        """
        if not analyses:
            fig = go.Figure()
            fig.add_annotation(text="No tokens to visualize", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Token Metrics Over Time', 'Probability Distribution', 
                          'Token Rank Distribution', 'Metric Correlation'],
            specs=[[{"secondary_y": True}, {}], [{}, {"type": "scatter"}]]
        )
        
        positions = [analysis.position for analysis in analyses]
        probabilities = [analysis.probability for analysis in analyses]
        ranks = [analysis.rank for analysis in analyses]
        entropies = [analysis.entropy for analysis in analyses]
        surprisals = [analysis.surprisal for analysis in analyses]
        tokens = [analysis.token for analysis in analyses]
        
        # Plot 1: Main metric and probability over time
        if mode == TokenVisualizationMode.PROBABILITY:
            values = probabilities
            y_title = 'Probability'
        elif mode == TokenVisualizationMode.LOGITS:
            values = [analysis.logit for analysis in analyses]
            y_title = 'Logit Value'
        elif mode == TokenVisualizationMode.RANK:
            values = ranks
            y_title = 'Rank'
        elif mode == TokenVisualizationMode.ENTROPY:
            values = entropies
            y_title = 'Entropy'
        elif mode == TokenVisualizationMode.SURPRISAL:
            values = surprisals
            y_title = 'Surprisal'
        elif mode == TokenVisualizationMode.PERPLEXITY:
            values = [analysis.perplexity for analysis in analyses]
            y_title = 'Perplexity'
        # New advanced modes
        elif mode == TokenVisualizationMode.MAX_PROBABILITY:
            values = [analysis.max_probability for analysis in analyses]
            y_title = 'Max Probability'
        elif mode == TokenVisualizationMode.PROBABILITY_MARGIN:
            values = [analysis.probability_margin for analysis in analyses]
            y_title = 'Probability Margin'
        elif mode == TokenVisualizationMode.SHANNON_ENTROPY:
            values = [analysis.shannon_entropy for analysis in analyses]
            y_title = 'Shannon Entropy'
        elif mode == TokenVisualizationMode.LOCAL_PERPLEXITY:
            values = [analysis.local_perplexity for analysis in analyses]
            y_title = 'Local Perplexity'
        elif mode == TokenVisualizationMode.SEQUENCE_IMPROBABILITY:
            values = [analysis.sequence_improbability for analysis in analyses]
            y_title = 'Sequence Improbability'
        elif mode == TokenVisualizationMode.CONFIDENCE_SCORE:
            values = [analysis.confidence_score for analysis in analyses]
            y_title = 'Confidence Score'
        # Advanced methods from METHODS_OVERVIEW.md
        elif mode == TokenVisualizationMode.SEMANTIC_ENERGY:
            values = [analysis.semantic_energy if analysis.semantic_energy is not None else -analysis.logit for analysis in analyses]
            y_title = 'Semantic Energy'
        elif mode == TokenVisualizationMode.CONFORMAL_SCORE:
            values = [analysis.conformal_score if analysis.conformal_score is not None else 1.0 - analysis.probability for analysis in analyses]
            y_title = 'Conformal Score'
        elif mode == TokenVisualizationMode.ATTENTION_ENTROPY:
            values = [analysis.attention_entropy if analysis.attention_entropy is not None else 0.0 for analysis in analyses]
            y_title = 'Attention Entropy'
        elif mode == TokenVisualizationMode.ATTENTION_SELF_ATTENTION:
            values = [analysis.attention_self_attention if analysis.attention_self_attention is not None else 0.5 for analysis in analyses]
            y_title = 'Self-Attention Weight'
        elif mode == TokenVisualizationMode.ATTENTION_VARIANCE:
            values = [analysis.attention_variance if analysis.attention_variance is not None else 0.0 for analysis in analyses]
            y_title = 'Attention Variance'
        elif mode == TokenVisualizationMode.ATTENTION_ANOMALY_SCORE:
            values = [analysis.attention_anomaly_score if analysis.attention_anomaly_score is not None else 0.0 for analysis in analyses]
            y_title = 'Attention Anomaly Score'
        # Aggregated suspicion score
        elif mode == TokenVisualizationMode.SUSPICION_SCORE:
            values = [analysis.suspicion_score if analysis.suspicion_score is not None else 0.0 for analysis in analyses]
            y_title = 'Suspicion Score (0-100)'

        # Enhanced hover information with all metrics
        hover_text = []
        for analysis in analyses:
            hover_info = [
                f"Token: {analysis.token}",
                f"Position: {analysis.position}",
                f"Probability: {analysis.probability:.6f}",
                f"Rank: {analysis.rank}",
                f"Entropy: {analysis.entropy:.4f}",
                f"Surprisal: {analysis.surprisal:.4f}",
                f"Max Prob: {analysis.max_probability:.6f}",
                f"Prob Margin: {analysis.probability_margin:.6f}",
                f"Confidence: {analysis.confidence_score:.4f}"
            ]
            # Add advanced methods if available
            if analysis.semantic_energy is not None:
                hover_info.append(f"Semantic Energy: {analysis.semantic_energy:.4f}")
            if analysis.conformal_score is not None:
                hover_info.append(f"Conformal Score: {analysis.conformal_score:.4f}")
            if analysis.attention_entropy is not None:
                hover_info.append(f"Attention Entropy: {analysis.attention_entropy:.4f}")
            if analysis.attention_self_attention is not None:
                hover_info.append(f"Self-Attention: {analysis.attention_self_attention:.4f}")
            if analysis.attention_anomaly_score is not None:
                hover_info.append(f"Attention Anomaly: {analysis.attention_anomaly_score:.4f}")
            if analysis.suspicion_score is not None:
                hover_info.append(f"Suspicion Score: {analysis.suspicion_score:.1f}/100")
            hover_text.append("<br>".join(hover_info))

        fig.add_trace(
            go.Scatter(x=positions, y=values, mode='lines+markers', name=y_title,
                      text=hover_text, hovertemplate='%{text}<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=positions, y=probabilities, mode='lines', name='Probability',
                      line=dict(color='red', dash='dash'), yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: Probability histogram
        fig.add_trace(
            go.Histogram(x=probabilities, name='Probability Distribution', nbinsx=30),
            row=1, col=2
        )
        
        # Plot 3: Rank histogram
        fig.add_trace(
            go.Histogram(x=ranks, name='Rank Distribution', nbinsx=50),
            row=2, col=1
        )
        
        # Plot 4: Scatter plot
        fig.add_trace(
            go.Scatter(x=entropies, y=surprisals, mode='markers',
                      marker=dict(color=probabilities, colorscale='RdYlGn', 
                                showscale=True, colorbar=dict(title="Probability")),
                      text=tokens, name='Entropy vs Surprisal',
                      hovertemplate='Entropy: %{x}<br>Surprisal: %{y}<br>Token: %{text}<extra></extra>'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, title_text="Interactive Token Analysis")
        fig.update_xaxes(title_text="Token Position", row=1, col=1)
        fig.update_yaxes(title_text=y_title, row=1, col=1)
        fig.update_yaxes(title_text="Probability", secondary_y=True, row=1, col=1)
        fig.update_xaxes(title_text="Probability", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Rank", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Entropy", row=2, col=2)
        fig.update_yaxes(title_text="Surprisal", row=2, col=2)
        
        return fig
    
    def identify_low_confidence_regions(self, 
                                      analyses: List[TokenAnalysis],
                                      threshold_percentile: float = 20) -> List[Tuple[int, int, float]]:
        """
        Identify regions where the model has low confidence (potentially bug-prone areas).
        
        Args:
            analyses: List of token analyses
            threshold_percentile: Percentile below which tokens are considered low confidence
            
        Returns:
            List of (start_pos, end_pos, avg_probability) tuples for low confidence regions
        """
        if not analyses:
            return []
        
        probabilities = [analysis.probability for analysis in analyses]
        threshold = np.percentile(probabilities, threshold_percentile)
        
        low_confidence_regions = []
        current_region_start = None
        current_region_probs = []
        
        for i, analysis in enumerate(analyses):
            if analysis.probability <= threshold:
                if current_region_start is None:
                    current_region_start = i
                current_region_probs.append(analysis.probability)
            else:
                if current_region_start is not None:
                    # End of a low confidence region
                    avg_prob = np.mean(current_region_probs)
                    low_confidence_regions.append((current_region_start, i-1, avg_prob))
                    current_region_start = None
                    current_region_probs = []
        
        # Handle case where file ends with a low confidence region
        if current_region_start is not None:
            avg_prob = np.mean(current_region_probs)
            low_confidence_regions.append((current_region_start, len(analyses)-1, avg_prob))
        
        return low_confidence_regions
    
    def generate_analysis_report(self, analyses: List[TokenAnalysis]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            analyses: List of token analyses
            
        Returns:
            HTML report string
        """
        if not analyses:
            return "<p>No analysis data available</p>"
        
        # Calculate comprehensive statistics
        probabilities = [a.probability for a in analyses]
        ranks = [a.rank for a in analyses]
        entropies = [a.entropy for a in analyses]
        surprisals = [a.surprisal for a in analyses]

        # New advanced metrics
        max_probabilities = [a.max_probability for a in analyses]
        probability_margins = [a.probability_margin for a in analyses]
        shannon_entropies = [a.shannon_entropy for a in analyses]
        local_perplexities = [a.local_perplexity for a in analyses]
        sequence_improbabilities = [a.sequence_improbability for a in analyses]
        confidence_scores = [a.confidence_score for a in analyses]

        stats = {
            'total_tokens': len(analyses),
            # Basic metrics
            'avg_probability': np.mean(probabilities),
            'min_probability': np.min(probabilities),
            'max_probability': np.max(probabilities),
            'avg_rank': np.mean(ranks),
            'median_rank': np.median(ranks),
            'avg_entropy': np.mean(entropies),
            'avg_surprisal': np.mean(surprisals),
            'tokens_in_top_10': len([r for r in ranks if r <= 10]),
            'tokens_in_top_100': len([r for r in ranks if r <= 100]),
            # Advanced metrics
            'avg_max_probability': np.mean(max_probabilities),
            'avg_probability_margin': np.mean(probability_margins),
            'avg_shannon_entropy': np.mean(shannon_entropies),
            'avg_local_perplexity': np.mean(local_perplexities),
            'avg_sequence_improbability': np.mean(sequence_improbabilities),
            'avg_confidence_score': np.mean(confidence_scores),
            'min_confidence_score': np.min(confidence_scores),
            'max_confidence_score': np.max(confidence_scores)
        }

        # Add LM-Polygraph statistics if available
        lm_polygraph_stats = {}
        if analyses[0].lm_polygraph_metrics:
            for metric_name in analyses[0].lm_polygraph_metrics.keys():
                metric_values = [a.lm_polygraph_metrics.get(metric_name, 0.0) for a in analyses if a.lm_polygraph_metrics]
                if metric_values:
                    lm_polygraph_stats[f'avg_{metric_name}'] = np.mean(metric_values)

        # Add advanced methods statistics if available
        advanced_methods_stats = {}
        semantic_energies = [a.semantic_energy for a in analyses if a.semantic_energy is not None]
        if semantic_energies:
            advanced_methods_stats['avg_semantic_energy'] = np.mean(semantic_energies)
            advanced_methods_stats['min_semantic_energy'] = np.min(semantic_energies)
            advanced_methods_stats['max_semantic_energy'] = np.max(semantic_energies)

        conformal_scores = [a.conformal_score for a in analyses if a.conformal_score is not None]
        if conformal_scores:
            advanced_methods_stats['avg_conformal_score'] = np.mean(conformal_scores)
            advanced_methods_stats['min_conformal_score'] = np.min(conformal_scores)
            advanced_methods_stats['max_conformal_score'] = np.max(conformal_scores)

        attention_entropies = [a.attention_entropy for a in analyses if a.attention_entropy is not None]
        if attention_entropies:
            advanced_methods_stats['avg_attention_entropy'] = np.mean(attention_entropies)
            advanced_methods_stats['max_attention_entropy'] = np.max(attention_entropies)

        attention_self_attentions = [a.attention_self_attention for a in analyses if a.attention_self_attention is not None]
        if attention_self_attentions:
            advanced_methods_stats['avg_self_attention'] = np.mean(attention_self_attentions)
            advanced_methods_stats['min_self_attention'] = np.min(attention_self_attentions)

        attention_variances = [a.attention_variance for a in analyses if a.attention_variance is not None]
        if attention_variances:
            advanced_methods_stats['avg_attention_variance'] = np.mean(attention_variances)
            advanced_methods_stats['max_attention_variance'] = np.max(attention_variances)

        attention_anomaly_scores = [a.attention_anomaly_score for a in analyses if a.attention_anomaly_score is not None]
        if attention_anomaly_scores:
            advanced_methods_stats['avg_attention_anomaly'] = np.mean(attention_anomaly_scores)
            advanced_methods_stats['max_attention_anomaly'] = np.max(attention_anomaly_scores)
        
        # Identify low confidence regions
        low_conf_regions = self.identify_low_confidence_regions(analyses)
        
        # Generate HTML report
        report_html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
            <h1>Token Generation Analysis Report</h1>
            
            <h2>Summary Statistics</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
                </tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Total Tokens</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['total_tokens']}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Probability</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_probability']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Min/Max Probability</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['min_probability']:.4f} / {stats['max_probability']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Rank</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_rank']:.1f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Median Rank</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['median_rank']:.1f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Tokens in Top 10</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['tokens_in_top_10']} ({100*stats['tokens_in_top_10']/stats['total_tokens']:.1f}%)</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Tokens in Top 100</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['tokens_in_top_100']} ({100*stats['tokens_in_top_100']/stats['total_tokens']:.1f}%)</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Entropy</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_entropy']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Surprisal</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_surprisal']:.4f}</td></tr>
            </table>

            <h2>Advanced Metrics</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Advanced Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
                </tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Max Probability</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_max_probability']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Probability Margin</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_probability_margin']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Shannon Entropy</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_shannon_entropy']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Local Perplexity</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_local_perplexity']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Sequence Improbability</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_sequence_improbability']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Average Confidence Score</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['avg_confidence_score']:.4f}</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">Min/Max Confidence Score</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{stats['min_confidence_score']:.4f} / {stats['max_confidence_score']:.4f}</td></tr>
            </table>"""

        # Add LM-Polygraph metrics table if available
        if lm_polygraph_stats:
            report_html += f"""
            <h2>LM-Polygraph Metrics</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">LM-Polygraph Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Average Value</th>
                </tr>"""
            for metric_name, metric_value in lm_polygraph_stats.items():
                display_name = metric_name.replace('avg_', '').replace('_', ' ').title()
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;">{display_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{metric_value:.4f}</td></tr>"""
            report_html += "</table>"

        # Add advanced methods table if available
        if advanced_methods_stats:
            report_html += f"""
            <h2>Advanced Error Detection Methods (METHODS_OVERVIEW.md)</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Method & Metric</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: right;">Value</th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Reference</th>
                </tr>"""
            if 'avg_semantic_energy' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Semantic Energy</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_semantic_energy']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Farquhar et al., NeurIPS 2024</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Min/Max Energy</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['min_semantic_energy']:.4f} / {advanced_methods_stats['max_semantic_energy']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            if 'avg_conformal_score' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Conformal Prediction</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_conformal_score']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Quach et al., ICLR 2024</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Min/Max Score</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['min_conformal_score']:.4f} / {advanced_methods_stats['max_conformal_score']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            if 'avg_attention_entropy' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Attention Entropy</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_attention_entropy']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Ott et al., ICML 2018</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Max Entropy</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['max_attention_entropy']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            if 'avg_self_attention' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Self-Attention</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_self_attention']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Jesse et al., ICSE 2023</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Min Self-Attention</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['min_self_attention']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            if 'avg_attention_variance' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Attention Variance</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_attention_variance']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Allamanis et al., 2021</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Max Variance</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['max_attention_variance']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            if 'avg_attention_anomaly' in advanced_methods_stats:
                report_html += f"""
                <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Attention Anomaly Combined</strong> (avg)</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['avg_attention_anomaly']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">Combined Score</td></tr>
                <tr><td style="border: 1px solid #ddd; padding: 8px;">  Max Anomaly</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{advanced_methods_stats['max_attention_anomaly']:.4f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;"></td></tr>"""
            report_html += "</table>"

        report_html += f"""
            
            <h2>Low Confidence Regions (Potential Bug Areas)</h2>
            <p>These regions show low model confidence and might be prone to errors:</p>
        """
        
        if low_conf_regions:
            report_html += """
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px;">Region</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Tokens</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Avg Probability</th>
                </tr>
            """
            for i, (start, end, avg_prob) in enumerate(low_conf_regions):
                tokens_in_region = " ".join([html.escape(analyses[j].token) for j in range(start, end+1)])
                report_html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Positions {start}-{end}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; font-family: monospace;">{tokens_in_region}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{avg_prob:.4f}</td>
                </tr>
                """
            report_html += "</table>"
        else:
            report_html += "<p>No low confidence regions detected.</p>"
        
        report_html += """
            <h2>Interpretation</h2>
            <ul>
                <li><strong>High probability tokens (green):</strong> Model is confident in these choices</li>
                <li><strong>Low probability tokens (red):</strong> Model is uncertain, potential areas for bugs or errors</li>
                <li><strong>High rank tokens:</strong> Less likely choices that might indicate creative or risky generation</li>
                <li><strong>High entropy:</strong> Model sees many possible alternatives at this position</li>
                <li><strong>High surprisal:</strong> The chosen token was unexpected given the context</li>
            </ul>
        </div>
        """
        
        return report_html


if __name__ == "__main__":
    # Example usage with mock data
    from LLM import TokenAnalysis
    
    # Create some example analyses with all new metrics
    mock_analyses = [
        TokenAnalysis(
            token="def", token_id=123, position=0, probability=0.95, logit=8.2, rank=1,
            perplexity=2.1, entropy=3.5, surprisal=0.07, top_k_probs=[],
            max_probability=0.95, probability_margin=0.92, shannon_entropy=3.5,
            local_perplexity=2.1, sequence_improbability=0.05, confidence_score=0.89,
            lm_polygraph_metrics=None
        ),
        TokenAnalysis(
            token=" factorial", token_id=456, position=1, probability=0.85, logit=7.8, rank=2,
            perplexity=2.3, entropy=3.8, surprisal=0.23, top_k_probs=[],
            max_probability=0.85, probability_margin=0.72, shannon_entropy=3.8,
            local_perplexity=2.3, sequence_improbability=0.15, confidence_score=0.74,
            lm_polygraph_metrics=None
        ),
        TokenAnalysis(
            token="(", token_id=789, position=2, probability=0.92, logit=8.1, rank=1,
            perplexity=2.0, entropy=3.2, surprisal=0.12, top_k_probs=[],
            max_probability=0.92, probability_margin=0.87, shannon_entropy=3.2,
            local_perplexity=2.0, sequence_improbability=0.08, confidence_score=0.84,
            lm_polygraph_metrics=None
        ),
        TokenAnalysis(
            token="n", token_id=101, position=3, probability=0.45, logit=5.2, rank=15,
            perplexity=4.2, entropy=5.8, surprisal=1.15, top_k_probs=[],
            max_probability=0.65, probability_margin=0.20, shannon_entropy=5.8,
            local_perplexity=4.2, sequence_improbability=0.55, confidence_score=0.21,
            lm_polygraph_metrics=None
        ),
        TokenAnalysis(
            token="):", token_id=112, position=4, probability=0.78, logit=6.9, rank=3,
            perplexity=2.8, entropy=4.1, surprisal=0.36, top_k_probs=[],
            max_probability=0.78, probability_margin=0.58, shannon_entropy=4.1,
            local_perplexity=2.8, sequence_improbability=0.22, confidence_score=0.62,
            lm_polygraph_metrics=None
        ),
    ]
    
    visualizer = TokenVisualizer()
    
    # Create HTML visualization
    html_viz = visualizer.create_html_visualization(
        mock_analyses, 
        mode=TokenVisualizationMode.PROBABILITY,
        title="Example Token Analysis"
    )
    
    print("HTML Visualization created")
    print("="*50)
    print(html_viz[:500] + "...")  # Print first 500 characters
    
    # Generate analysis report
    report = visualizer.generate_analysis_report(mock_analyses)
    print("\nAnalysis Report created")
    print("="*50)
    print(report[:500] + "...")  # Print first 500 characters