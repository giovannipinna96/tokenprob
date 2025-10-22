#!/usr/bin/env python3
"""
Advanced Methods Visualizer

Creates 7 interactive visualizations for comparing the 4 error detection methods:
1. LecPrompt (baseline)
2. Semantic Energy
3. Conformal Prediction
4. Attention Anomaly

Visualizations:
- Methods comparison heatmap
- Anomaly counts comparison
- Method agreement matrix
- Token-level multi-method view
- Method performance radar
- Venn diagram overlap
- Interactive method explorer
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import html


class AdvancedMethodsVisualizer:
    """Create comprehensive visualizations for method comparison."""

    def __init__(self):
        """Initialize visualizer with color schemes."""
        self.method_colors = {
            'lecprompt': '#1f77b4',          # Blue
            'semantic_energy': '#ff7f0e',    # Orange
            'conformal': '#2ca02c',          # Green
            'attention': '#d62728',          # Red
            'semantic_context': '#9467bd',   # Purple (5th method)
            'masked_token_replacement': '#8c564b'  # Brown (6th method)
        }

        self.method_names = {
            'lecprompt': 'LecPrompt (Baseline)',
            'semantic_energy': 'Semantic Energy',
            'conformal': 'Conformal Prediction',
            'attention': 'Attention Anomaly',
            'semantic_context': 'Semantic Context',
            'masked_token_replacement': 'Masked Token Replacement'
        }

    @staticmethod
    def _convert_to_serializable(obj):
        """
        Convert numpy types to native Python types for JSON serialization.

        Args:
            obj: Object to convert (can be dict, list, numpy type, etc.)

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: AdvancedMethodsVisualizer._convert_to_serializable(value)
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [AdvancedMethodsVisualizer._convert_to_serializable(item)
                    for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return AdvancedMethodsVisualizer._convert_to_serializable(obj.tolist())
        else:
            return obj

    def create_all_visualizations(self, results: Dict, output_dir: str):
        """
        Create all 7 visualizations.

        Args:
            results: Complete comparison results
            output_dir: Directory to save visualizations
        """
        print("\n" + "="*80)
        print("CREATING ADVANCED VISUALIZATIONS")
        print("="*80)

        vis_dir = os.path.join(output_dir, "advanced_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Methods comparison heatmap
        self.plot_methods_comparison_heatmap(results, vis_dir)

        # 2. Anomaly counts comparison
        self.plot_anomaly_counts_comparison(results, vis_dir)

        # 3. Method agreement matrix
        self.plot_method_agreement_matrix(results, vis_dir)

        # 4. Token-level multi-method views (one per example)
        self.plot_token_level_multimethod_views(results, vis_dir)

        # 5. Method performance radar
        self.plot_method_performance_radar(results, vis_dir)

        # 6. Venn diagram overlap
        self.plot_venn_diagram_overlap(results, vis_dir)

        # 7. Interactive method explorer
        self.create_interactive_method_explorer(results, vis_dir)

        # 8. Create index page
        self._create_index_page(results, vis_dir)

        print(f"\nVisualizations saved to {vis_dir}/")

    def plot_methods_comparison_heatmap(self, results: Dict, output_dir: str):
        """
        Heatmap showing which methods confirm hypothesis for each example.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating methods comparison heatmap...")

        individual_results = results['individual_results']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Build matrix: examples × methods
        example_names = [r['example_name'] for r in individual_results if 'error' not in r]
        matrix = []
        hover_text = []

        for result in individual_results:
            if 'error' in result:
                continue

            row = []
            hover_row = []

            for method in methods:
                confirmed = result['hypothesis_confirmation'].get(method, False)
                row.append(1 if confirmed else 0)

                buggy_anom = result['buggy'][method]['num_anomalies']
                correct_anom = result['correct'][method]['num_anomalies']

                hover_row.append(
                    f"Method: {self.method_names[method]}<br>"
                    f"Example: {result['example_name']}<br>"
                    f"Confirmed: {'Yes' if confirmed else 'No'}<br>"
                    f"Buggy anomalies: {buggy_anom}<br>"
                    f"Correct anomalies: {correct_anom}<br>"
                    f"Differential: {buggy_anom - correct_anom}"
                )

            matrix.append(row)
            hover_text.append(hover_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[self.method_names[m] for m in methods],
            y=example_names,
            colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showscale=False
        ))

        # Add checkmarks/crosses
        for i, example_name in enumerate(example_names):
            for j, method in enumerate(methods):
                symbol = '✓' if matrix[i][j] == 1 else '✗'
                color = 'green' if matrix[i][j] == 1 else 'red'

                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"<b>{symbol}</b>",
                    showarrow=False,
                    font=dict(size=18, color=color)
                )

        fig.update_layout(
            title="Methods Comparison: Hypothesis Confirmation",
            xaxis_title="Detection Method",
            yaxis_title="Test Example",
            height=600,
            font=dict(size=12)
        )

        filename = os.path.join(output_dir, "methods_comparison_heatmap.html")
        fig.write_html(filename)
        print(f"    Created: methods_comparison_heatmap.html")

    def plot_anomaly_counts_comparison(self, results: Dict, output_dir: str):
        """
        Grouped bar chart comparing anomaly counts across methods.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating anomaly counts comparison...")

        individual_results = results['individual_results']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Create subplots (buggy and correct)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Anomalies in Buggy Code', 'Anomalies in Correct Code'),
            vertical_spacing=0.12
        )

        # Buggy code
        for method in methods:
            counts = [
                r['buggy'][method]['num_anomalies']
                for r in individual_results
                if 'error' not in r
            ]
            example_names = [r['example_name'] for r in individual_results if 'error' not in r]

            fig.add_trace(
                go.Bar(
                    name=self.method_names[method],
                    x=example_names,
                    y=counts,
                    marker_color=self.method_colors[method],
                    showlegend=True
                ),
                row=1, col=1
            )

        # Correct code
        for method in methods:
            counts = [
                r['correct'][method]['num_anomalies']
                for r in individual_results
                if 'error' not in r
            ]
            example_names = [r['example_name'] for r in individual_results if 'error' not in r]

            fig.add_trace(
                go.Bar(
                    name=self.method_names[method],
                    x=example_names,
                    y=counts,
                    marker_color=self.method_colors[method],
                    showlegend=False
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Anomaly Counts Comparison Across Methods",
            height=900,
            barmode='group'
        )

        fig.update_xaxes(title_text="Test Example", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Test Example", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="Number of Anomalies", row=1, col=1)
        fig.update_yaxes(title_text="Number of Anomalies", row=2, col=1)

        filename = os.path.join(output_dir, "anomaly_counts_comparison.html")
        fig.write_html(filename)
        print(f"    Created: anomaly_counts_comparison.html")

    def plot_method_agreement_matrix(self, results: Dict, output_dir: str):
        """
        Heatmap showing agreement between methods.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating method agreement matrix...")

        individual_results = results['individual_results']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Average agreement matrices across all examples
        all_matrices = []
        for result in individual_results:
            if 'error' not in result and 'agreement_metrics' in result:
                avg_matrix = np.array(result['agreement_metrics']['average_agreement_matrix'])
                all_matrices.append(avg_matrix)

        if all_matrices:
            # Global average
            global_avg_matrix = np.mean(all_matrices, axis=0)
        else:
            global_avg_matrix = np.eye(len(methods))

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=global_avg_matrix,
            x=[self.method_names[m] for m in methods],
            y=[self.method_names[m] for m in methods],
            colorscale='Blues',
            colorbar=dict(title="Agreement<br>Score"),
            text=[[f"{val:.2f}" for val in row] for row in global_avg_matrix],
            texttemplate='%{text}',
            textfont=dict(size=14),
            hovertemplate='%{x}<br>vs<br>%{y}<br>Agreement: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title="Method Agreement Matrix (Average Across All Examples)",
            xaxis_title="Method",
            yaxis_title="Method",
            height=600,
            width=700
        )

        filename = os.path.join(output_dir, "method_agreement_matrix.html")
        fig.write_html(filename)
        print(f"    Created: method_agreement_matrix.html")

    def plot_token_level_multimethod_views(self, results: Dict, output_dir: str):
        """
        Create token-level visualizations for each example showing all methods.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating token-level multi-method views...")

        individual_results = results['individual_results']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        for result in individual_results:
            if 'error' in result:
                continue

            example_name = result['example_name']
            self._create_token_view_for_example(result, methods, output_dir)

        print(f"    Created: token_level_multimethod_view_*.html (×{len([r for r in individual_results if 'error' not in r])})")

    def _create_token_view_for_example(self, result: Dict, methods: List[str], output_dir: str):
        """Create token-level view for a single example."""
        example_name = result['example_name']

        # Create bar chart showing anomaly counts per method
        buggy_counts = [result['buggy'][m]['num_anomalies'] for m in methods]
        correct_counts = [result['correct'][m]['num_anomalies'] for m in methods]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Buggy Code',
            x=[self.method_names[m] for m in methods],
            y=buggy_counts,
            marker_color='#d62728',
            text=buggy_counts,
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='Correct Code',
            x=[self.method_names[m] for m in methods],
            y=correct_counts,
            marker_color='#2ca02c',
            text=correct_counts,
            textposition='auto'
        ))

        # Add hypothesis confirmation markers
        for i, method in enumerate(methods):
            confirmed = result['hypothesis_confirmation'][method]
            symbol = '✓' if confirmed else '✗'
            color = 'green' if confirmed else 'red'

            fig.add_annotation(
                x=i,
                y=max(buggy_counts[i], correct_counts[i]) + 2,
                text=f"<b>{symbol}</b>",
                showarrow=False,
                font=dict(size=24, color=color)
            )

        fig.update_layout(
            title=f"Token-Level Comparison: {example_name}",
            xaxis_title="Detection Method",
            yaxis_title="Number of Anomalous Tokens",
            barmode='group',
            height=500,
            showlegend=True
        )

        filename = os.path.join(output_dir, f"token_level_multimethod_view_{example_name}.html")
        fig.write_html(filename)

    def plot_method_performance_radar(self, results: Dict, output_dir: str):
        """
        Radar chart showing multi-dimensional performance of each method.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating method performance radar...")

        aggregate_stats = results['aggregate_statistics']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Extract metrics for each method
        categories = [
            'Confirmation<br>Rate',
            'Anomaly<br>Differential',
            'Speed<br>(Inverse Time)',
            'Buggy<br>Detection',
            'Precision<br>(Low False Pos)'
        ]

        fig = go.Figure()

        for method in methods:
            stats = aggregate_stats[method]

            # Normalize metrics to [0, 1]
            conf_rate = stats['confirmation_rate']

            # Differential (normalized)
            diff = stats['avg_buggy_anomalies'] - stats['avg_correct_anomalies']
            max_diff = max([
                aggregate_stats[m]['avg_buggy_anomalies'] - aggregate_stats[m]['avg_correct_anomalies']
                for m in methods
            ])
            diff_norm = diff / max_diff if max_diff > 0 else 0.5

            # Speed (inverse time, normalized)
            times = [aggregate_stats[m]['avg_execution_time'] for m in methods]
            max_time = max(times)
            min_time = min(times)
            if max_time > min_time:
                speed_norm = 1.0 - (stats['avg_execution_time'] - min_time) / (max_time - min_time)
            else:
                speed_norm = 1.0

            # Buggy detection (normalized)
            max_buggy = max([aggregate_stats[m]['avg_buggy_anomalies'] for m in methods])
            buggy_norm = stats['avg_buggy_anomalies'] / max_buggy if max_buggy > 0 else 0.5

            # Precision (inverse of correct anomalies)
            max_correct = max([aggregate_stats[m]['avg_correct_anomalies'] for m in methods])
            precision_norm = 1.0 - (stats['avg_correct_anomalies'] / max_correct if max_correct > 0 else 0.0)

            values = [conf_rate, diff_norm, speed_norm, buggy_norm, precision_norm]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=self.method_names[method],
                line=dict(color=self.method_colors[method])
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Method Performance Radar (Normalized Metrics)",
            height=600,
            showlegend=True
        )

        filename = os.path.join(output_dir, "method_performance_radar.html")
        fig.write_html(filename)
        print(f"    Created: method_performance_radar.html")

    def plot_venn_diagram_overlap(self, results: Dict, output_dir: str):
        """
        Venn-style diagram showing overlap of anomalies detected by methods.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating Venn diagram overlap...")

        # For simplicity, create stacked bar showing consensus levels
        individual_results = results['individual_results']

        # Count consensus levels across all examples
        consensus_counts = {
            '4 methods': 0,
            '3 methods': 0,
            '2 methods': 0,
            '1 method': 0,
            '0 methods': 0
        }

        for result in individual_results:
            if 'error' in result:
                continue

            # Count how many methods confirmed hypothesis
            confirmations = sum(result['hypothesis_confirmation'].values())
            if confirmations == 4:
                consensus_counts['4 methods'] += 1
            elif confirmations == 3:
                consensus_counts['3 methods'] += 1
            elif confirmations == 2:
                consensus_counts['2 methods'] += 1
            elif confirmations == 1:
                consensus_counts['1 method'] += 1
            else:
                consensus_counts['0 methods'] += 1

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(consensus_counts.keys()),
                y=list(consensus_counts.values()),
                marker_color=['#2ca02c', '#7fbc41', '#fec44f', '#f16913', '#d62728'],
                text=list(consensus_counts.values()),
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Method Agreement: Consensus Levels<br>(How many methods agree on hypothesis confirmation)",
            xaxis_title="Number of Methods in Agreement",
            yaxis_title="Number of Test Examples",
            height=500
        )

        filename = os.path.join(output_dir, "venn_diagram_overlap.html")
        fig.write_html(filename)
        print(f"    Created: venn_diagram_overlap.html")

    def create_interactive_method_explorer(self, results: Dict, output_dir: str):
        """
        Create comprehensive interactive dashboard.

        Args:
            results: Complete comparison results
            output_dir: Output directory
        """
        print("\n  Creating interactive method explorer...")

        individual_results = results['individual_results']
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Filter valid results
        valid_results = [r for r in individual_results if 'error' not in r]
        example_names = [r['example_name'] for r in valid_results]

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Methods Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .controls {{
            margin: 20px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }}
        select {{
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            min-width: 300px;
            margin-right: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        .metric-card.lecprompt {{ background: linear-gradient(135deg, #1f77b4, #4a9fd8); }}
        .metric-card.semantic_energy {{ background: linear-gradient(135deg, #ff7f0e, #ffb366); }}
        .metric-card.conformal {{ background: linear-gradient(135deg, #2ca02c, #5cb85c); }}
        .metric-card.attention {{ background: linear-gradient(135deg, #d62728, #f44336); }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-status {{
            font-size: 14px;
            margin-top: 5px;
        }}
        #comparisonChart {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .confirmed {{
            color: green;
            font-weight: bold;
        }}
        .not-confirmed {{
            color: red;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Advanced Methods Explorer</h1>

        <div class="controls">
            <label for="exampleSelect"><strong>Select Test Example:</strong></label>
            <select id="exampleSelect" onchange="updateDashboard()">
                {"".join([f'<option value="{ex}">{ex}</option>' for ex in example_names])}
            </select>
        </div>

        <h2>📊 Method Performance Metrics</h2>
        <div class="metrics-grid" id="metricsGrid"></div>

        <h2>📈 Anomaly Counts Comparison</h2>
        <div id="comparisonChart"></div>

        <h2>📋 Detailed Method Comparison</h2>
        <div id="detailsTable"></div>
    </div>

    <script>
        const resultsData = {json.dumps(self._convert_to_serializable(valid_results))};
        const methodColors = {json.dumps(self.method_colors)};
        const methodNames = {json.dumps(self.method_names)};

        function updateDashboard() {{
            const exampleName = document.getElementById('exampleSelect').value;
            const result = resultsData.find(r => r.example_name === exampleName);

            if (!result) return;

            updateMetrics(result);
            updateChart(result);
            updateTable(result);
        }}

        function updateMetrics(result) {{
            const grid = document.getElementById('metricsGrid');
            const methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention'];

            let html = '';
            methods.forEach(method => {{
                const buggyAnom = result.buggy[method].num_anomalies;
                const correctAnom = result.correct[method].num_anomalies;
                const confirmed = result.hypothesis_confirmation[method];
                const status = confirmed ? '✓ Confirmed' : '✗ Not Confirmed';
                const statusClass = confirmed ? 'confirmed' : 'not-confirmed';

                html += `
                    <div class="metric-card ${{method}}">
                        <div class="metric-label">${{methodNames[method]}}</div>
                        <div class="metric-value">${{buggyAnom}}</div>
                        <div class="metric-status ${{statusClass}}">${{status}}</div>
                    </div>
                `;
            }});

            grid.innerHTML = html;
        }}

        function updateChart(result) {{
            const methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention'];

            const buggyData = {{
                x: methods.map(m => methodNames[m]),
                y: methods.map(m => result.buggy[m].num_anomalies),
                name: 'Buggy Code',
                type: 'bar',
                marker: {{color: '#d62728'}}
            }};

            const correctData = {{
                x: methods.map(m => methodNames[m]),
                y: methods.map(m => result.correct[m].num_anomalies),
                name: 'Correct Code',
                type: 'bar',
                marker: {{color: '#2ca02c'}}
            }};

            const layout = {{
                title: `Anomaly Counts: ${{result.example_name}}`,
                barmode: 'group',
                xaxis: {{title: 'Method'}},
                yaxis: {{title: 'Number of Anomalies'}},
                height: 400
            }};

            Plotly.newPlot('comparisonChart', [buggyData, correctData], layout);
        }}

        function updateTable(result) {{
            const methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention'];
            const table = document.getElementById('detailsTable');

            let html = `
                <table>
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Buggy Anomalies</th>
                            <th>Correct Anomalies</th>
                            <th>Differential</th>
                            <th>Hypothesis</th>
                            <th>Avg Exec Time</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            methods.forEach(method => {{
                const buggyAnom = result.buggy[method].num_anomalies;
                const correctAnom = result.correct[method].num_anomalies;
                const diff = buggyAnom - correctAnom;
                const confirmed = result.hypothesis_confirmation[method];
                const statusClass = confirmed ? 'confirmed' : 'not-confirmed';
                const statusText = confirmed ? '✓ Confirmed' : '✗ Not Confirmed';
                const avgTime = (
                    (result.execution_times.buggy[method] || 0) +
                    (result.execution_times.correct[method] || 0)
                ) / 2;

                html += `
                    <tr>
                        <td><strong>${{methodNames[method]}}</strong></td>
                        <td>${{buggyAnom}}</td>
                        <td>${{correctAnom}}</td>
                        <td>${{diff}}</td>
                        <td class="${{statusClass}}">${{statusText}}</td>
                        <td>${{avgTime.toFixed(2)}}s</td>
                    </tr>
                `;
            }});

            html += '</tbody></table>';
            table.innerHTML = html;
        }}

        // Initialize
        updateDashboard();
    </script>
</body>
</html>
"""

        filename = os.path.join(output_dir, "interactive_method_explorer.html")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"    Created: interactive_method_explorer.html")

    def _create_index_page(self, results: Dict, output_dir: str):
        """Create navigation index page."""
        print("\n  Creating index page...")

        metadata = results['metadata']
        individual_results = [r for r in results['individual_results'] if 'error' not in r]

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Methods Comparison - Visualizations</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 36px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 18px;
        }}
        .info-box {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            padding: 10px;
        }}
        .info-label {{
            font-weight: bold;
            color: #555;
            font-size: 14px;
        }}
        .info-value {{
            font-size: 18px;
            color: #333;
            margin-top: 5px;
        }}
        .section {{
            margin: 30px 0;
        }}
        h2 {{
            color: #444;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .viz-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            text-decoration: none;
            color: #333;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }}
        .viz-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .viz-description {{
            font-size: 14px;
            color: #666;
            line-height: 1.5;
        }}
        .icon {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .methods-legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .method-badge {{
            padding: 10px 20px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Advanced Methods Comparison</h1>
        <p class="subtitle">Comprehensive Visualization Dashboard</p>

        <div class="info-box">
            <h3 style="margin-top: 0;">Study Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Date</div>
                    <div class="info-value">{metadata.get('timestamp', 'N/A')[:10]}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Test Examples</div>
                    <div class="info-value">{metadata.get('num_examples', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Sensitivity (k)</div>
                    <div class="info-value">{metadata.get('sensitivity_factor', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Model</div>
                    <div class="info-value">{metadata.get('model_name', 'N/A').split('/')[-1]}</div>
                </div>
            </div>
        </div>"""

        # Add calibration information if available
        calibration_results = results.get('calibration_results')
        if calibration_results and metadata.get('calibration_performed'):
            cal_meta = calibration_results.get('conformal_metadata', {})
            cal_info = calibration_results.get('calibration_info', {})

            html_content += f"""
        <div class="info-box" style="border-left-color: #2ca02c;">
            <h3 style="margin-top: 0;">🎯 Conformal Prediction Calibration</h3>
            <p style="color: #666; margin-bottom: 15px;">
                Calibration provides formal statistical coverage guarantees for conformal prediction uncertainty quantification.
            </p>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Calibration Examples</div>
                    <div class="info-value">{cal_info.get('calibration_count', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Calibration Tokens</div>
                    <div class="info-value">{cal_meta.get('num_calibration_tokens', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Coverage Target</div>
                    <div class="info-value">{cal_meta.get('coverage_target', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Quantile Threshold</div>
                    <div class="info-value">{cal_meta.get('quantile_threshold', 0):.4f}</div>
                </div>
            </div>
            <details style="margin-top: 15px;">
                <summary style="cursor: pointer; font-weight: bold; color: #555;">📋 Calibration Details</summary>
                <div style="margin-top: 10px; padding: 10px; background: white; border-radius: 5px;">
                    <p><strong>Calibration Set:</strong> {', '.join(cal_info.get('calibration_examples', []))}</p>
                    <p><strong>Alpha:</strong> {cal_meta.get('alpha', 'N/A')}</p>
                    <p><strong>Mean Score:</strong> {cal_meta.get('mean_score', 0):.4f}</p>
                    <p><strong>Std Score:</strong> {cal_meta.get('std_score', 0):.4f}</p>
                    <p><strong>Score Range:</strong> [{cal_meta.get('min_score', 0):.4f}, {cal_meta.get('max_score', 0):.4f}]</p>
                </div>
            </details>
        </div>"""
        else:
            html_content += """
        <div class="info-box" style="border-left-color: #ffc107; background-color: #fff3cd;">
            <h3 style="margin-top: 0;">⚠️ Conformal Prediction Not Calibrated</h3>
            <p style="color: #856404; margin-bottom: 0;">
                Conformal prediction is using default thresholds. For formal coverage guarantees, run with calibration enabled.
            </p>
        </div>"""

        html_content += """
        <div class="highlight">
            <strong>💡 Quick Start:</strong> Begin with the <a href="interactive_method_explorer.html">Interactive Method Explorer</a>
            for a comprehensive overview, then explore individual visualizations below.
        </div>

        <div class="section">
            <h2>🧪 Methods Compared</h2>
            <div class="methods-legend">
                <div class="method-badge" style="background-color: #1f77b4;">LecPrompt (Baseline)</div>
                <div class="method-badge" style="background-color: #ff7f0e;">Semantic Energy</div>
                <div class="method-badge" style="background-color: #2ca02c;">Conformal Prediction</div>
                <div class="method-badge" style="background-color: #d62728;">Attention Anomaly</div>"""

        # Add semantic context badge if available
        if len(individual_results) > 0:
            first_result = individual_results[0]
            if 'buggy' in first_result and first_result['buggy'].get('semantic_context'):
                html_content += """
                <div class="method-badge" style="background-color: #9467bd;">Semantic Context</div>"""
            if 'buggy' in first_result and first_result['buggy'].get('masked_token_replacement'):
                html_content += """
                <div class="method-badge" style="background-color: #8c564b;">Masked Token Replacement ⭐ NEW</div>"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>📊 Main Visualizations</h2>
            <div class="viz-grid">
                <a href="interactive_method_explorer.html" class="viz-card">
                    <div class="icon">🎯</div>
                    <div class="viz-title">Interactive Method Explorer</div>
                    <div class="viz-description">
                        Comprehensive dashboard with dynamic example selection and real-time method comparison
                    </div>
                </a>

                <a href="methods_comparison_heatmap.html" class="viz-card">
                    <div class="icon">📋</div>
                    <div class="viz-title">Methods Comparison Heatmap</div>
                    <div class="viz-description">
                        Matrix view showing which methods confirmed the hypothesis for each example
                    </div>
                </a>

                <a href="anomaly_counts_comparison.html" class="viz-card">
                    <div class="icon">📊</div>
                    <div class="viz-title">Anomaly Counts Comparison</div>
                    <div class="viz-description">
                        Grouped bar charts comparing anomaly detection across all methods
                    </div>
                </a>

                <a href="method_agreement_matrix.html" class="viz-card">
                    <div class="icon">🤝</div>
                    <div class="viz-title">Method Agreement Matrix</div>
                    <div class="viz-description">
                        Heatmap showing correlation and agreement between different methods
                    </div>
                </a>

                <a href="method_performance_radar.html" class="viz-card">
                    <div class="icon">📡</div>
                    <div class="viz-title">Method Performance Radar</div>
                    <div class="viz-description">
                        Multi-dimensional radar chart comparing methods across 5 key metrics
                    </div>
                </a>

                <a href="venn_diagram_overlap.html" class="viz-card">
                    <div class="icon">🔵</div>
                    <div class="viz-title">Method Consensus Analysis</div>
                    <div class="viz-description">
                        Visual representation of agreement levels across different methods
                    </div>
                </a>
            </div>
        </div>

        <div class="section">
            <h2>🔬 Per-Example Analysis</h2>
            <p>Token-level comparisons for individual test examples:</p>
            <div class="viz-grid">
"""

        for result in individual_results:
            example_name = result['example_name']
            bug_type = result.get('bug_type', 'unknown')
            html_content += f"""
                <a href="token_level_multimethod_view_{example_name}.html" class="viz-card">
                    <div class="icon">🎯</div>
                    <div class="viz-title">{example_name}</div>
                    <div class="viz-description">
                        Bug type: {bug_type}<br>
                        Token-level comparison across all methods
                    </div>
                </a>
"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>📚 Additional Resources</h2>
            <ul style="line-height: 2;">
                <li><a href="../complete_comparison_results.json">💾 Complete Results (JSON)</a></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

        filename = os.path.join(output_dir, "index.html")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"    Created: index.html")


if __name__ == "__main__":
    print("Advanced Methods Visualizer")
    print("\nCreates 7 interactive visualizations:")
    print("  1. Methods comparison heatmap")
    print("  2. Anomaly counts comparison")
    print("  3. Method agreement matrix")
    print("  4. Token-level multi-method views")
    print("  5. Method performance radar")
    print("  6. Venn diagram overlap")
    print("  7. Interactive method explorer")
