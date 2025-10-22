#!/usr/bin/env python3
"""
Detailed Comparison Visualizer

Creates detailed token-level and per-example visualizations for comparing
multiple models' error detection performance.
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List
import numpy as np
import html


class DetailedComparisonVisualizer:
    """Create detailed visualizations for multi-model comparison."""

    def __init__(self):
        """Initialize the visualizer."""
        self.colors = {
            'starcoder2-7b': '#1f77b4',
            'codet5p-2b': '#ff7f0e',
            'deepseek-6.7b': '#2ca02c',
            'codebert': '#d62728',
            'qwen-7b': '#9467bd'
        }

    def create_all_detailed_visualizations(self, results: Dict, output_dir: str):
        """
        Create all detailed visualizations.

        Args:
            results: Complete benchmark results
            output_dir: Directory to save visualizations
        """
        print("\n" + "="*80)
        print("CREATING DETAILED VISUALIZATIONS")
        print("="*80)

        vis_dir = os.path.join(output_dir, "detailed_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # 1. Token-level comparison for each example
        self._create_token_comparisons(results, vis_dir)

        # 2. Performance matrix
        self.plot_performance_matrix(results, vis_dir)

        # 3. Anomaly count comparison
        self.plot_anomaly_count_comparison(results, vis_dir)

        # 4. Line-level detection comparison
        self.plot_line_level_detection(results, vis_dir)

        # 5. Probability distributions
        self.plot_probability_distributions(results, vis_dir)

        # 6. Interactive dashboard
        self.create_interactive_dashboard(results, vis_dir)

        # 7. Create index page
        self._create_index_page(results, vis_dir)

        print(f"\nDetailed visualizations saved to {vis_dir}/")

    def _create_token_comparisons(self, results: Dict, output_dir: str):
        """Create token-level comparison for each example."""
        print("\n  Creating token-level comparisons...")

        model_results = results['model_results']

        # Get all examples from first successful model
        examples = None
        for model_key, data in model_results.items():
            if 'error' not in data:
                examples = data['individual_results']
                break

        if not examples:
            print("    No examples found")
            return

        for example in examples:
            example_name = example.get('example_name', 'unknown')
            if 'error' in example:
                continue

            self.plot_token_level_comparison(results, example_name, output_dir)

    def plot_token_level_comparison(self, results: Dict, example_name: str, output_dir: str):
        """
        Create side-by-side token-level comparison for a specific example.

        Args:
            results: Complete benchmark results
            example_name: Name of example to visualize
            output_dir: Output directory
        """
        model_results = results['model_results']

        # Collect data for this example across all models
        example_data = {}
        for model_key, data in model_results.items():
            if 'error' in data:
                continue

            for result in data['individual_results']:
                if result.get('example_name') == example_name:
                    example_data[model_key] = result
                    break

        if not example_data:
            return

        # Create visualization
        fig = go.Figure()

        models = list(example_data.keys())

        # Create grouped bar chart
        buggy_anomalies = [example_data[m]['buggy_anomalies'] for m in models]
        correct_anomalies = [example_data[m]['correct_anomalies'] for m in models]

        fig.add_trace(go.Bar(
            name='Buggy Code',
            x=models,
            y=buggy_anomalies,
            marker_color='#d62728',
            text=buggy_anomalies,
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            name='Correct Code',
            x=models,
            y=correct_anomalies,
            marker_color='#2ca02c',
            text=correct_anomalies,
            textposition='auto'
        ))

        # Add hypothesis confirmation markers
        for i, model in enumerate(models):
            confirmed = example_data[model]['hypothesis_confirmed']
            symbol = '✓' if confirmed else '✗'
            color = 'green' if confirmed else 'red'

            fig.add_annotation(
                x=model,
                y=max(buggy_anomalies[i], correct_anomalies[i]) + 1,
                text=f"<b>{symbol}</b>",
                showarrow=False,
                font=dict(size=20, color=color)
            )

        fig.update_layout(
            title=f"Token-Level Comparison: {example_name}",
            xaxis_title="Model",
            yaxis_title="Number of Anomalous Tokens",
            barmode='group',
            height=500,
            showlegend=True
        )

        filename = os.path.join(output_dir, f"token_comparison_{example_name}.html")
        fig.write_html(filename)
        print(f"    Created: token_comparison_{example_name}.html")

    def plot_performance_matrix(self, results: Dict, output_dir: str):
        """
        Create interactive performance matrix (examples × models).

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        print("\n  Creating performance matrix...")

        model_results = results['model_results']

        # Collect all models and examples
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Get examples from first model
        examples = []
        example_names = []
        for model_key in models:
            for result in model_results[model_key]['individual_results']:
                if 'error' not in result:
                    name = result['example_name']
                    if name not in example_names:
                        example_names.append(name)
                        examples.append(result)

        # Create matrix
        matrix = []
        hover_text = []

        for example in examples:
            example_name = example['example_name']
            row = []
            hover_row = []

            for model_key in models:
                # Find result for this example in this model
                confirmed = False
                buggy_anom = 0
                correct_anom = 0

                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name:
                        if 'error' not in result:
                            confirmed = result['hypothesis_confirmed']
                            buggy_anom = result['buggy_anomalies']
                            correct_anom = result['correct_anomalies']
                        break

                row.append(1 if confirmed else 0)
                hover_row.append(
                    f"Model: {model_key}<br>"
                    f"Example: {example_name}<br>"
                    f"Confirmed: {'Yes' if confirmed else 'No'}<br>"
                    f"Buggy anomalies: {buggy_anom}<br>"
                    f"Correct anomalies: {correct_anom}"
                )

            matrix.append(row)
            hover_text.append(hover_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=models,
            y=example_names,
            colorscale=[[0, '#ffcccc'], [1, '#ccffcc']],
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showscale=False
        ))

        # Add checkmarks/crosses
        for i, example_name in enumerate(example_names):
            for j, model in enumerate(models):
                symbol = '✓' if matrix[i][j] == 1 else '✗'
                color = 'green' if matrix[i][j] == 1 else 'red'

                fig.add_annotation(
                    x=model,
                    y=example_name,
                    text=symbol,
                    showarrow=False,
                    font=dict(size=16, color=color)
                )

        fig.update_layout(
            title="Performance Matrix: Model vs Example",
            xaxis_title="Model",
            yaxis_title="Test Example",
            height=600
        )

        filename = os.path.join(output_dir, "performance_matrix.html")
        fig.write_html(filename)
        print(f"    Created: performance_matrix.html")

    def plot_anomaly_count_comparison(self, results: Dict, output_dir: str):
        """
        Create grouped bar chart of anomaly counts per example.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        print("\n  Creating anomaly count comparison...")

        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Get examples
        example_names = []
        for result in model_results[models[0]]['individual_results']:
            if 'error' not in result:
                example_names.append(result['example_name'])

        # Create subplot with two rows (buggy and correct)
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Anomalies in Buggy Code', 'Anomalies in Correct Code'),
            vertical_spacing=0.15
        )

        # Buggy code anomalies
        for model_key in models:
            buggy_counts = []
            for example_name in example_names:
                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name and 'error' not in result:
                        buggy_counts.append(result['buggy_anomalies'])
                        break

            fig.add_trace(
                go.Bar(
                    name=model_key,
                    x=example_names,
                    y=buggy_counts,
                    marker_color=self.colors.get(model_key, '#888888'),
                    showlegend=True
                ),
                row=1, col=1
            )

        # Correct code anomalies
        for model_key in models:
            correct_counts = []
            for example_name in example_names:
                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name and 'error' not in result:
                        correct_counts.append(result['correct_anomalies'])
                        break

            fig.add_trace(
                go.Bar(
                    name=model_key,
                    x=example_names,
                    y=correct_counts,
                    marker_color=self.colors.get(model_key, '#888888'),
                    showlegend=False
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Anomaly Count Comparison Across Models",
            height=800,
            barmode='group'
        )

        fig.update_xaxes(title_text="Test Example", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="Test Example", row=2, col=1, tickangle=45)
        fig.update_yaxes(title_text="Anomalous Tokens", row=1, col=1)
        fig.update_yaxes(title_text="Anomalous Tokens", row=2, col=1)

        filename = os.path.join(output_dir, "anomaly_counts.html")
        fig.write_html(filename)
        print(f"    Created: anomaly_counts.html")

    def plot_line_level_detection(self, results: Dict, output_dir: str):
        """
        Show which lines were identified as errors by each model.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        print("\n  Creating line-level detection comparison...")

        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Get examples
        example_names = []
        for result in model_results[models[0]]['individual_results']:
            if 'error' not in result:
                example_names.append(result['example_name'])

        # Create data for heatmap
        buggy_lines_data = []
        correct_lines_data = []
        hover_buggy = []
        hover_correct = []

        for example_name in example_names:
            buggy_row = []
            correct_row = []
            hover_b = []
            hover_c = []

            for model_key in models:
                buggy_lines = 0
                correct_lines = 0

                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name and 'error' not in result:
                        buggy_lines = result.get('buggy_error_lines', 0)
                        correct_lines = result.get('correct_error_lines', 0)
                        break

                buggy_row.append(buggy_lines)
                correct_row.append(correct_lines)
                hover_b.append(f"Model: {model_key}<br>Example: {example_name}<br>Error lines: {buggy_lines}")
                hover_c.append(f"Model: {model_key}<br>Example: {example_name}<br>Error lines: {correct_lines}")

            buggy_lines_data.append(buggy_row)
            correct_lines_data.append(correct_row)
            hover_buggy.append(hover_b)
            hover_correct.append(hover_c)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Error Lines in Buggy Code', 'Error Lines in Correct Code'),
            horizontal_spacing=0.1
        )

        # Buggy code heatmap
        fig.add_trace(
            go.Heatmap(
                z=buggy_lines_data,
                x=models,
                y=example_names,
                colorscale='Reds',
                text=hover_buggy,
                hovertemplate='%{text}<extra></extra>',
                colorbar=dict(title="Lines", x=0.45)
            ),
            row=1, col=1
        )

        # Correct code heatmap
        fig.add_trace(
            go.Heatmap(
                z=correct_lines_data,
                x=models,
                y=example_names,
                colorscale='Blues',
                text=hover_correct,
                hovertemplate='%{text}<extra></extra>',
                colorbar=dict(title="Lines", x=1.02)
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="Line-Level Error Detection Comparison",
            height=600
        )

        filename = os.path.join(output_dir, "line_detection.html")
        fig.write_html(filename)
        print(f"    Created: line_detection.html")

    def plot_probability_distributions(self, results: Dict, output_dir: str):
        """
        Create box plots of log probabilities for buggy vs correct code.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        print("\n  Creating probability distribution comparison...")

        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Collect mean log probabilities
        buggy_probs = {model: [] for model in models}
        correct_probs = {model: [] for model in models}

        for model_key in models:
            for result in model_results[model_key]['individual_results']:
                if 'error' not in result:
                    buggy_probs[model_key].append(result.get('buggy_mean_log_prob', 0))
                    correct_probs[model_key].append(result.get('correct_mean_log_prob', 0))

        # Create box plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Mean Log-Probability (Buggy Code)', 'Mean Log-Probability (Correct Code)')
        )

        # Buggy code
        for model_key in models:
            fig.add_trace(
                go.Box(
                    y=buggy_probs[model_key],
                    name=model_key,
                    marker_color=self.colors.get(model_key, '#888888'),
                    showlegend=True
                ),
                row=1, col=1
            )

        # Correct code
        for model_key in models:
            fig.add_trace(
                go.Box(
                    y=correct_probs[model_key],
                    name=model_key,
                    marker_color=self.colors.get(model_key, '#888888'),
                    showlegend=False
                ),
                row=1, col=2
            )

        fig.update_layout(
            title="Distribution of Mean Log-Probabilities",
            height=500
        )

        fig.update_yaxes(title_text="Mean Log-Probability", row=1, col=1)
        fig.update_yaxes(title_text="Mean Log-Probability", row=1, col=2)

        filename = os.path.join(output_dir, "probability_distributions.html")
        fig.write_html(filename)
        print(f"    Created: probability_distributions.html")

    def create_interactive_dashboard(self, results: Dict, output_dir: str):
        """
        Create comprehensive interactive dashboard.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        print("\n  Creating interactive dashboard...")

        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Get all examples
        examples = []
        for result in model_results[models[0]]['individual_results']:
            if 'error' not in result:
                examples.append(result['example_name'])

        # Build comprehensive HTML dashboard
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
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
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .selector {{
            margin: 20px 0;
        }}
        select {{
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            min-width: 300px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        #detailsPanel {{
            margin-top: 30px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
            border: 1px solid #ddd;
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
        <h1>🔍 Multi-Model Error Detection Dashboard</h1>

        <div class="selector">
            <label for="exampleSelect"><strong>Select Test Example:</strong></label>
            <select id="exampleSelect" onchange="updateDashboard()">
                {"".join([f'<option value="{ex}">{ex}</option>' for ex in examples])}
            </select>
        </div>

        <h2>📊 Overall Performance Metrics</h2>
        <div class="metric-grid" id="metricsGrid"></div>

        <h2>📈 Detailed Comparison</h2>
        <div id="comparisonChart"></div>

        <h2>📋 Model Details</h2>
        <div id="detailsPanel"></div>
    </div>

    <script>
        const resultsData = {json.dumps(model_results)};
        const models = {json.dumps(models)};

        function updateDashboard() {{
            const exampleName = document.getElementById('exampleSelect').value;

            // Update metrics
            updateMetrics(exampleName);

            // Update chart
            updateChart(exampleName);

            // Update details table
            updateDetails(exampleName);
        }}

        function updateMetrics(exampleName) {{
            const grid = document.getElementById('metricsGrid');

            let totalConfirmed = 0;
            let totalModels = 0;

            models.forEach(model => {{
                const result = findResult(model, exampleName);
                if (result && !result.error) {{
                    totalModels++;
                    if (result.hypothesis_confirmed) totalConfirmed++;
                }}
            }});

            const confirmationRate = totalModels > 0 ? (totalConfirmed / totalModels * 100).toFixed(1) : 0;

            grid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-label">Models Tested</div>
                    <div class="metric-value">${{totalModels}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Hypothesis Confirmed</div>
                    <div class="metric-value">${{totalConfirmed}}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confirmation Rate</div>
                    <div class="metric-value">${{confirmationRate}}%</div>
                </div>
            `;
        }}

        function updateChart(exampleName) {{
            const buggyAnomalies = [];
            const correctAnomalies = [];
            const modelNames = [];

            models.forEach(model => {{
                const result = findResult(model, exampleName);
                if (result && !result.error) {{
                    modelNames.push(model);
                    buggyAnomalies.push(result.buggy_anomalies);
                    correctAnomalies.push(result.correct_anomalies);
                }}
            }});

            const trace1 = {{
                x: modelNames,
                y: buggyAnomalies,
                name: 'Buggy Code',
                type: 'bar',
                marker: {{color: '#d62728'}}
            }};

            const trace2 = {{
                x: modelNames,
                y: correctAnomalies,
                name: 'Correct Code',
                type: 'bar',
                marker: {{color: '#2ca02c'}}
            }};

            const layout = {{
                title: `Anomalous Tokens: ${{exampleName}}`,
                barmode: 'group',
                xaxis: {{title: 'Model'}},
                yaxis: {{title: 'Number of Anomalous Tokens'}},
                height: 400
            }};

            Plotly.newPlot('comparisonChart', [trace1, trace2], layout);
        }}

        function updateDetails(exampleName) {{
            const panel = document.getElementById('detailsPanel');

            let tableHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Buggy Anomalies</th>
                            <th>Correct Anomalies</th>
                            <th>Buggy Error Lines</th>
                            <th>Correct Error Lines</th>
                            <th>Hypothesis Confirmed</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            models.forEach(model => {{
                const result = findResult(model, exampleName);
                if (result && !result.error) {{
                    const confirmed = result.hypothesis_confirmed;
                    const confirmedClass = confirmed ? 'confirmed' : 'not-confirmed';
                    const confirmedText = confirmed ? '✓ Yes' : '✗ No';

                    tableHTML += `
                        <tr>
                            <td><strong>${{model}}</strong></td>
                            <td>${{result.buggy_anomalies}}</td>
                            <td>${{result.correct_anomalies}}</td>
                            <td>${{result.buggy_error_lines}}</td>
                            <td>${{result.correct_error_lines}}</td>
                            <td class="${{confirmedClass}}">${{confirmedText}}</td>
                        </tr>
                    `;
                }}
            }});

            tableHTML += '</tbody></table>';
            panel.innerHTML = tableHTML;
        }}

        function findResult(model, exampleName) {{
            if (!resultsData[model] || !resultsData[model].individual_results) return null;
            return resultsData[model].individual_results.find(r => r.example_name === exampleName);
        }}

        // Initialize dashboard
        updateDashboard();
    </script>
</body>
</html>
"""

        filename = os.path.join(output_dir, "dashboard.html")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"    Created: dashboard.html")

    def _create_index_page(self, results: Dict, output_dir: str):
        """Create navigation index page."""
        print("\n  Creating index page...")

        metadata = results.get('metadata', {})

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model Comparison - Detailed Visualizations</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
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
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Multi-Model Error Detection Study</h1>
        <p class="subtitle">Detailed Visualization Dashboard</p>

        <div class="info-box">
            <h3 style="margin-top: 0;">Study Information</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Date</div>
                    <div class="info-value">{metadata.get('timestamp', 'N/A')[:10]}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Models Tested</div>
                    <div class="info-value">{metadata.get('num_models', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Test Examples</div>
                    <div class="info-value">{metadata.get('num_examples', 'N/A')}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Sensitivity (k)</div>
                    <div class="info-value">{metadata.get('sensitivity_factor', 'N/A')}</div>
                </div>
            </div>
        </div>

        <div class="highlight">
            <strong>💡 Quick Start:</strong> Begin with the <a href="dashboard.html">Interactive Dashboard</a>
            for an overview, then explore individual visualizations below for detailed analysis.
        </div>

        <div class="section">
            <h2>📊 Main Visualizations</h2>
            <div class="viz-grid">
                <a href="dashboard.html" class="viz-card">
                    <div class="icon">🎯</div>
                    <div class="viz-title">Interactive Dashboard</div>
                    <div class="viz-description">
                        Comprehensive dashboard with dynamic example selection and real-time metrics
                    </div>
                </a>

                <a href="performance_matrix.html" class="viz-card">
                    <div class="icon">📋</div>
                    <div class="viz-title">Performance Matrix</div>
                    <div class="viz-description">
                        Grid view showing which models confirmed the hypothesis for each example
                    </div>
                </a>

                <a href="anomaly_counts.html" class="viz-card">
                    <div class="icon">📊</div>
                    <div class="viz-title">Anomaly Counts</div>
                    <div class="viz-description">
                        Comparison of anomalous token counts across models for buggy vs correct code
                    </div>
                </a>

                <a href="line_detection.html" class="viz-card">
                    <div class="icon">🔍</div>
                    <div class="viz-title">Line-Level Detection</div>
                    <div class="viz-description">
                        Heatmap showing number of error lines detected by each model
                    </div>
                </a>

                <a href="probability_distributions.html" class="viz-card">
                    <div class="icon">📈</div>
                    <div class="viz-title">Probability Distributions</div>
                    <div class="viz-description">
                        Box plots of mean log-probabilities for buggy and correct code
                    </div>
                </a>
            </div>
        </div>

        <div class="section">
            <h2>🔬 Per-Example Analysis</h2>
            <p>Token-level comparisons for individual test examples:</p>
            <div class="viz-grid">
"""

        # Add links to per-example visualizations
        model_results = results['model_results']
        examples = []
        for model_key, data in model_results.items():
            if 'error' not in data:
                for result in data['individual_results']:
                    if 'error' not in result:
                        name = result['example_name']
                        if name not in examples:
                            examples.append(name)
                break

        for example_name in examples:
            html_content += f"""
                <a href="token_comparison_{example_name}.html" class="viz-card">
                    <div class="icon">🎯</div>
                    <div class="viz-title">{example_name}</div>
                    <div class="viz-description">
                        Token-level anomaly comparison for this example
                    </div>
                </a>
"""

        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>📚 Additional Resources</h2>
            <ul style="line-height: 2;">
                <li><a href="../comparison_report.md">📄 Full Comparison Report (Markdown)</a></li>
                <li><a href="../complete_benchmark_results.json">💾 Raw Results (JSON)</a></li>
                <li><a href="../statistical_analysis.json">📊 Statistical Analysis (JSON)</a></li>
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
    print("Detailed Comparison Visualizer")
    print("\nUsage:")
    print("  from comparison.detailed_comparison_visualizer import DetailedComparisonVisualizer")
    print("  visualizer = DetailedComparisonVisualizer()")
    print("  visualizer.create_all_detailed_visualizations(results, output_dir)")
