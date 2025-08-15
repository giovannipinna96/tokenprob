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
    LOGITS = "logits"
    RANK = "rank"
    ENTROPY = "entropy"
    SURPRISAL = "surprisal"
    PERPLEXITY = "perplexity"


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
        else:
            raise ValueError(f"Unknown visualization mode: {mode}")
        
        # Normalize values
        normalized_values = self._normalize_values(values, mode)
        
        # Get color scheme
        scheme = self.color_schemes[mode]
        
        # Create HTML
        html_parts = [
            f"<h2>{title}</h2>",
            f"<p><strong>Visualization Mode:</strong> {scheme['label']}</p>",
            "<div style='font-family: monospace; font-size: 14px; line-height: 1.8; margin: 20px 0;'>"
        ]
        
        for i, (analysis, norm_val, original_val) in enumerate(zip(analyses, normalized_values, values)):
            color = self._get_color_from_value(norm_val, scheme['colormap'])
            
            # Escape HTML characters in token
            token_display = html.escape(analysis.token).replace(' ', '&nbsp;')
            if token_display == '':
                token_display = '&lt;empty&gt;'
            
            # Create tooltip with detailed information
            tooltip = (
                f"Token: {html.escape(analysis.token)}\\n"
                f"Position: {analysis.position}\\n"
                f"Probability: {analysis.probability:.6f}\\n"
                f"Rank: {analysis.rank}\\n"
                f"Logit: {analysis.logit:.4f}\\n"
                f"Entropy: {analysis.entropy:.4f}\\n"
                f"Surprisal: {analysis.surprisal:.4f}\\n"
                f"Perplexity: {analysis.perplexity:.4f}"
            )
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 1px; margin: 0; '
                f'border-radius: 3px; display: inline-block;" '
                f'title="{tooltip}">{token_display}</span>'
            )
        
        html_parts.append("</div>")
        
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
        
        fig.add_trace(
            go.Scatter(x=positions, y=values, mode='lines+markers', name=y_title,
                      text=tokens, hovertemplate=f'Position: %{{x}}<br>{y_title}: %{{y}}<br>Token: %{{text}}<extra></extra>'),
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
        
        # Calculate statistics
        probabilities = [a.probability for a in analyses]
        ranks = [a.rank for a in analyses]
        entropies = [a.entropy for a in analyses]
        surprisals = [a.surprisal for a in analyses]
        
        stats = {
            'total_tokens': len(analyses),
            'avg_probability': np.mean(probabilities),
            'min_probability': np.min(probabilities),
            'max_probability': np.max(probabilities),
            'avg_rank': np.mean(ranks),
            'median_rank': np.median(ranks),
            'avg_entropy': np.mean(entropies),
            'avg_surprisal': np.mean(surprisals),
            'tokens_in_top_10': len([r for r in ranks if r <= 10]),
            'tokens_in_top_100': len([r for r in ranks if r <= 100])
        }
        
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
    
    # Create some example analyses
    mock_analyses = [
        TokenAnalysis("def", 123, 0, 0.95, 8.2, 1, 2.1, 3.5, 0.07, []),
        TokenAnalysis(" factorial", 456, 1, 0.85, 7.8, 2, 2.3, 3.8, 0.23, []),
        TokenAnalysis("(", 789, 2, 0.92, 8.1, 1, 2.0, 3.2, 0.12, []),
        TokenAnalysis("n", 101, 3, 0.45, 5.2, 15, 4.2, 5.8, 1.15, []),
        TokenAnalysis("):", 112, 4, 0.78, 6.9, 3, 2.8, 4.1, 0.36, []),
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