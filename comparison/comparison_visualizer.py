#!/usr/bin/env python3
"""
Comparison Visualizer

Creates visualizations for multi-model comparison study.
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any
import numpy as np


class ComparisonVisualizer:
    """Create visualizations for model comparison."""

    def create_all_visualizations(self, results: Dict, stats: Dict, output_dir: str):
        """Create all visualization charts."""
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        # 1. Confirmation rates bar chart
        self.plot_confirmation_rates(results, output_dir)

        # 2. Bug type performance heatmap
        self.plot_bug_type_heatmap(stats, output_dir)

        # 3. Model comparison radar
        self.plot_radar_comparison(results, output_dir)

        # 4. Statistical significance matrix
        self.plot_significance_matrix(stats, output_dir)

        # 5. Example breakdown
        self.plot_example_breakdown(results, output_dir)

        # 6. Anomaly heatmap
        self.plot_anomaly_heatmap(results, output_dir)

        print(f"\nVisualizations saved to {output_dir}/")

    def plot_confirmation_rates(self, results: Dict, output_dir: str):
        """Bar chart of confirmation rates."""
        model_results = results['model_results']
        models = []
        rates = []

        for model_key, data in model_results.items():
            if 'error' not in data:
                models.append(model_key)
                rates.append(data['aggregate_stats']['confirmation_rate'])

        fig = go.Figure(data=[
            go.Bar(x=models, y=rates, text=[f"{r:.1%}" for r in rates], textposition='auto')
        ])

        fig.update_layout(
            title="Hypothesis Confirmation Rate by Model",
            xaxis_title="Model",
            yaxis_title="Confirmation Rate",
            yaxis=dict(tickformat=".0%", range=[0, 1]),
            height=500
        )

        filename = os.path.join(output_dir, "confirmation_rates.html")
        fig.write_html(filename)
        print(f"  Created: confirmation_rates.html")

    def plot_bug_type_heatmap(self, stats: Dict, output_dir: str):
        """Heatmap of performance by bug type."""
        bug_type_data = stats.get('bug_type_analysis', {})
        if not bug_type_data:
            return

        bug_types = list(bug_type_data.keys())
        models = list(list(bug_type_data.values())[0].keys())

        # Create matrix
        z_data = []
        for bug_type in bug_types:
            row = [bug_type_data[bug_type].get(model, 0) for model in models]
            z_data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=models,
            y=bug_types,
            colorscale='RdYlGn',
            text=[[f"{val:.1%}" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Rate", tickformat=".0%")
        ))

        fig.update_layout(
            title="Performance by Bug Type",
            xaxis_title="Model",
            yaxis_title="Bug Type",
            height=400
        )

        filename = os.path.join(output_dir, "bug_type_heatmap.html")
        fig.write_html(filename)
        print(f"  Created: bug_type_heatmap.html")

    def plot_radar_comparison(self, results: Dict, output_dir: str):
        """Radar chart comparing models."""
        model_results = results['model_results']

        fig = go.Figure()

        for model_key, data in model_results.items():
            if 'error' in data:
                continue

            agg = data['aggregate_stats']
            bug_breakdown = agg.get('bug_type_breakdown', {})

            # Categories: overall + bug types
            categories = ['Overall'] + list(bug_breakdown.keys())
            values = [agg['confirmation_rate']] + [bug_breakdown[bt]['confirmation_rate']
                                                   for bt in bug_breakdown.keys()]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_key
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-Dimensional Model Comparison",
            height=600
        )

        filename = os.path.join(output_dir, "radar_comparison.html")
        fig.write_html(filename)
        print(f"  Created: radar_comparison.html")

    def plot_significance_matrix(self, stats: Dict, output_dir: str):
        """Matrix showing statistical significance."""
        pairwise = stats.get('pairwise_comparisons', {})
        if not pairwise:
            return

        # Extract unique models
        all_models = set()
        for key in pairwise.keys():
            m1, m2 = key.split('_vs_')
            all_models.add(m1)
            all_models.add(m2)
        models = sorted(list(all_models))

        # Create matrix
        z_data = np.zeros((len(models), len(models)))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i == j:
                    z_data[i][j] = 0.5  # Diagonal
                else:
                    key = f"{m1}_vs_{m2}" if f"{m1}_vs_{m2}" in pairwise else f"{m2}_vs_{m1}"
                    if key in pairwise and isinstance(pairwise[key], dict):
                        z_data[i][j] = 1 if pairwise[key].get('significant', False) else 0

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=models,
            y=models,
            colorscale=[[0, 'white'], [0.5, 'lightgray'], [1, 'red']],
            showscale=False
        ))

        fig.update_layout(
            title="Statistical Significance Matrix (McNemar Test, p<0.05)",
            xaxis_title="Model",
            yaxis_title="Model",
            height=500
        )

        filename = os.path.join(output_dir, "significance_matrix.html")
        fig.write_html(filename)
        print(f"  Created: significance_matrix.html")

    def plot_example_breakdown(self, results: Dict, output_dir: str):
        """
        Create stacked bar chart showing performance breakdown by example.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        if not models:
            return

        # Get all examples
        example_names = []
        for result in model_results[models[0]]['individual_results']:
            if 'error' not in result:
                example_names.append(result['example_name'])

        # Count confirmations per example
        confirmations = []
        total_models = []

        for example_name in example_names:
            confirmed_count = 0
            total_count = 0

            for model_key in models:
                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name and 'error' not in result:
                        total_count += 1
                        if result['hypothesis_confirmed']:
                            confirmed_count += 1
                        break

            confirmations.append(confirmed_count)
            total_models.append(total_count)

        # Calculate percentages
        percentages = [c / t * 100 if t > 0 else 0 for c, t in zip(confirmations, total_models)]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=example_names,
            y=confirmations,
            name='Confirmed',
            marker_color='#2ca02c',
            text=[f"{p:.0f}%" for p in percentages],
            textposition='auto'
        ))

        fig.add_trace(go.Bar(
            x=example_names,
            y=[t - c for t, c in zip(total_models, confirmations)],
            name='Not Confirmed',
            marker_color='#d62728'
        ))

        fig.update_layout(
            title="Hypothesis Confirmation by Test Example",
            xaxis_title="Test Example",
            yaxis_title="Number of Models",
            barmode='stack',
            height=500,
            xaxis_tickangle=45
        )

        filename = os.path.join(output_dir, "example_breakdown.html")
        fig.write_html(filename)
        print(f"  Created: example_breakdown.html")

    def plot_anomaly_heatmap(self, results: Dict, output_dir: str):
        """
        Create heatmap showing difference in anomalies (buggy - correct) for each model and example.

        Args:
            results: Complete benchmark results
            output_dir: Output directory
        """
        model_results = results['model_results']
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        if not models:
            return

        # Get all examples
        example_names = []
        for result in model_results[models[0]]['individual_results']:
            if 'error' not in result:
                example_names.append(result['example_name'])

        # Create matrix of anomaly differences
        diff_matrix = []
        hover_text = []

        for example_name in example_names:
            row = []
            hover_row = []

            for model_key in models:
                diff = 0
                buggy_anom = 0
                correct_anom = 0

                for result in model_results[model_key]['individual_results']:
                    if result.get('example_name') == example_name and 'error' not in result:
                        buggy_anom = result['buggy_anomalies']
                        correct_anom = result['correct_anomalies']
                        diff = buggy_anom - correct_anom
                        break

                row.append(diff)
                hover_row.append(
                    f"Model: {model_key}<br>"
                    f"Example: {example_name}<br>"
                    f"Buggy: {buggy_anom}<br>"
                    f"Correct: {correct_anom}<br>"
                    f"Difference: {diff}"
                )

            diff_matrix.append(row)
            hover_text.append(hover_row)

        fig = go.Figure(data=go.Heatmap(
            z=diff_matrix,
            x=models,
            y=example_names,
            colorscale='RdYlGn',
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorbar=dict(title="Anomaly<br>Difference")
        ))

        fig.update_layout(
            title="Anomaly Difference Heatmap (Buggy - Correct)",
            xaxis_title="Model",
            yaxis_title="Test Example",
            height=600
        )

        filename = os.path.join(output_dir, "anomaly_difference_heatmap.html")
        fig.write_html(filename)
        print(f"  Created: anomaly_difference_heatmap.html")
