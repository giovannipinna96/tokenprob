#!/usr/bin/env python3
"""
Token-Level Visualization Generator

Generates HTML visualizations showing individual tokens highlighted by uncertainty
for each detection method on each test example.

Outputs:
- token_visualizations/by_example/{example}/{method}_{buggy|correct}.html (80 files)
- token_visualizations/by_example/{example}/index.html (10 files)
- token_visualizations/by_example/index.html (1 file)
- index.html (main index)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import html as html_module

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizer import TokenVisualizer, TokenVisualizationMode
from LLM import TokenAnalysis


class TokenLevelVisualizer:
    """
    Generates token-level HTML visualizations for all methods on all examples.

    Creates individual HTML files showing code with tokens color-coded by
    uncertainty/anomaly scores for each detection method.
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.visualizer = TokenVisualizer()

        # Map method names to visualization modes
        self.method_modes = {
            'lecprompt': TokenVisualizationMode.LOGICAL_ERROR_DETECTION,
            'semantic_energy': TokenVisualizationMode.SEMANTIC_ENERGY,
            'conformal': TokenVisualizationMode.CONFORMAL_SCORE,
            'attention': TokenVisualizationMode.ATTENTION_ANOMALY_SCORE
        }

        # Display names for methods
        self.method_display_names = {
            'lecprompt': 'LecPrompt (Baseline)',
            'semantic_energy': 'Semantic Energy',
            'conformal': 'Conformal Prediction',
            'attention': 'Attention Anomaly'
        }

    def generate_all_token_visualizations(self,
                                         results: Dict,
                                         output_dir: str) -> None:
        """
        Generate all token-level visualizations.

        Creates:
        - 80 HTML files (10 examples × 4 methods × 2 versions)
        - 10 example index files
        - 1 main index file
        - 1 root index file

        Args:
            results: Complete comparison results from AdvancedMethodsComparisonRunner
            output_dir: Base output directory
        """
        print("\n" + "="*80)
        print("GENERATING TOKEN-LEVEL VISUALIZATIONS")
        print("="*80)

        # Create output directory structure
        token_viz_dir = os.path.join(output_dir, "token_visualizations")
        by_example_dir = os.path.join(token_viz_dir, "by_example")
        os.makedirs(by_example_dir, exist_ok=True)

        # Get individual results
        individual_results = results.get('individual_results', [])

        if not individual_results:
            print("⚠ No individual results found in results")
            return

        print(f"\nGenerating visualizations for {len(individual_results)} examples...")

        # Generate visualizations for each example
        example_names = []
        for i, result in enumerate(individual_results, 1):
            example_name = result.get('example_name', f'example_{i}')
            example_names.append(example_name)

            print(f"\n[{i}/{len(individual_results)}] {example_name}")

            try:
                self._generate_for_example(result, by_example_dir)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Create index files
        print("\nCreating index files...")
        self._create_by_example_index(individual_results, by_example_dir)
        self._create_root_index(output_dir)

        print("\n" + "="*80)
        print("TOKEN-LEVEL VISUALIZATIONS COMPLETE")
        print("="*80)
        print(f"\nGenerated files:")
        print(f"  - {len(individual_results) * 8} token visualization HTML files")
        print(f"  - {len(individual_results)} example index files")
        print(f"  - 2 navigation index files")
        print(f"\nOutput directory: {token_viz_dir}/")

    def _generate_for_example(self,
                             result: Dict,
                             by_example_dir: str) -> None:
        """
        Generate visualizations for a single example (8 HTML files).

        Args:
            result: Result dict for this example
            by_example_dir: Base directory for by_example organization
        """
        example_name = result['example_name']

        # Create example directory
        example_dir = os.path.join(by_example_dir, example_name)
        os.makedirs(example_dir, exist_ok=True)

        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']

        # Generate for each method
        for method in methods:
            # Buggy version
            print(f"  - {method} (buggy)")
            self._generate_method_html(
                result,
                method,
                'buggy',
                example_dir
            )

            # Correct version
            print(f"  - {method} (correct)")
            self._generate_method_html(
                result,
                method,
                'correct',
                example_dir
            )

        # Create example index
        self._create_example_index(result, methods, example_dir)

    def _generate_method_html(self,
                             result: Dict,
                             method: str,
                             code_type: str,  # 'buggy' or 'correct'
                             output_dir: str) -> None:
        """
        Generate HTML visualization for a single method on buggy or correct code.

        Args:
            result: Result dict for the example
            method: Method name ('lecprompt', 'semantic_energy', 'conformal', 'attention')
            code_type: 'buggy' or 'correct'
            output_dir: Directory to save HTML
        """
        example_name = result['example_name']

        # Get method result
        method_data = result.get(code_type, {}).get(method)

        if not method_data:
            print(f"    WARNING: No data for {method} {code_type}")
            return

        # Check for error
        if 'error' in method_data:
            print(f"    WARNING: Method error: {method_data['error']}")
            return

        # Get token_analyses and code
        token_analyses_dicts = method_data.get('token_analyses', [])
        code = method_data.get('code', '')

        if not token_analyses_dicts:
            print(f"    WARNING: No token_analyses for {method} {code_type}")
            return

        if not code:
            print(f"    WARNING: No code for {method} {code_type}")
            return

        # Convert dicts to TokenAnalysis objects
        token_analyses = self._convert_to_token_analyses(token_analyses_dicts)

        # Get visualization mode
        mode = self.method_modes.get(method, TokenVisualizationMode.PROBABILITY)

        # Create title
        title = f"{example_name} - {self.method_display_names.get(method, method)} - {code_type.capitalize()} Code"

        # Generate HTML
        html_content = self.visualizer.create_html_visualization(
            token_analyses,
            mode=mode,
            title=title
        )

        # Save to file
        filename = f"{method}_{code_type}.html"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _convert_to_token_analyses(self,
                                   dicts: List[Dict]) -> List[TokenAnalysis]:
        """
        Convert list of dicts to TokenAnalysis objects.

        Args:
            dicts: List of TokenAnalysis dicts

        Returns:
            List of TokenAnalysis objects
        """
        analyses = []
        for d in dicts:
            # Handle both dict and TokenAnalysis objects
            if isinstance(d, TokenAnalysis):
                analyses.append(d)
            else:
                # Convert dict to TokenAnalysis
                # Use **d to unpack dict as kwargs
                try:
                    analysis = TokenAnalysis(**d)
                    analyses.append(analysis)
                except Exception as e:
                    print(f"      Warning: Could not convert dict to TokenAnalysis: {e}")
                    # Fallback: create minimal TokenAnalysis
                    analysis = TokenAnalysis(
                        token=d.get('token', ''),
                        token_id=d.get('token_id', 0),
                        position=d.get('position', 0),
                        probability=d.get('probability', 0.0),
                        logit=d.get('logit', 0.0),
                        rank=d.get('rank', 1),
                        perplexity=d.get('perplexity', 1.0),
                        entropy=d.get('entropy', 0.0),
                        surprisal=d.get('surprisal', 0.0),
                        top_k_probs=d.get('top_k_probs', []),
                        max_probability=d.get('max_probability', 0.0),
                        probability_margin=d.get('probability_margin', 0.0),
                        shannon_entropy=d.get('shannon_entropy', 0.0),
                        local_perplexity=d.get('local_perplexity', 1.0),
                        sequence_improbability=d.get('sequence_improbability', 0.0),
                        confidence_score=d.get('confidence_score', 0.0),
                        semantic_energy=d.get('semantic_energy'),
                        conformal_score=d.get('conformal_score'),
                        attention_entropy=d.get('attention_entropy'),
                        attention_self_attention=d.get('attention_self_attention'),
                        attention_variance=d.get('attention_variance'),
                        attention_anomaly_score=d.get('attention_anomaly_score'),
                        is_anomalous=d.get('is_anomalous')
                    )
                    analyses.append(analysis)

        return analyses

    def _create_example_index(self,
                             result: Dict,
                             methods: List[str],
                             output_dir: str) -> None:
        """
        Create index.html for a single example showing all methods.

        Args:
            result: Result dict for this example
            methods: List of method names
            output_dir: Directory to save index
        """
        example_name = result['example_name']
        description = result.get('description', '')
        bug_type = result.get('bug_type', '')

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{example_name} - Token Visualizations</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .info {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        .method-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .method-section h2 {{
            color: #555;
            margin-top: 0;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 8px;
        }}
        .links {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }}
        .btn {{
            display: inline-block;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
            flex: 1;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .btn-buggy {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
            color: white;
        }}
        .btn-correct {{
            background: linear-gradient(135deg, #51cf66, #37b24d);
            color: white;
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        .back-link:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <a href="../index.html" class="back-link">← Back to All Examples</a>

    <h1>{example_name}</h1>

    <div class="info">
        <p><strong>Description:</strong> {html_module.escape(description)}</p>
        <p><strong>Bug Type:</strong> {html_module.escape(bug_type)}</p>
    </div>

    <div class="grid">
"""

        # Add section for each method
        for method in methods:
            method_display = self.method_display_names.get(method, method)

            html_content += f"""
        <div class="method-section">
            <h2>{method_display}</h2>
            <div class="links">
                <a href="{method}_buggy.html" class="btn btn-buggy">Buggy Code</a>
                <a href="{method}_correct.html" class="btn btn-correct">Correct Code</a>
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        # Save index
        filepath = os.path.join(output_dir, "index.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_by_example_index(self,
                                 individual_results: List[Dict],
                                 by_example_dir: str) -> None:
        """
        Create index.html for by_example directory listing all examples.

        Args:
            individual_results: List of all example results
            by_example_dir: Directory to save index
        """
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Visualizations - By Example</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .example-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .example-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .example-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .example-card h3 {
            color: #555;
            margin-top: 0;
        }
        .example-card p {
            color: #777;
            font-size: 0.9em;
        }
        .example-card a {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }
        .example-card a:hover {
            background: linear-gradient(135deg, #5568d3, #63408b);
        }
        .bug-badge {
            display: inline-block;
            padding: 4px 8px;
            background: #ff6b6b;
            color: white;
            border-radius: 4px;
            font-size: 0.8em;
            margin-top: 5px;
        }
        .back-link {
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .back-link:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <a href="../../index.html" class="back-link">← Back to Main Index</a>

    <h1>🔬 Token-Level Visualizations by Example</h1>

    <p>View detailed token-level analysis for each test example across all detection methods.</p>

    <div class="example-grid">
"""

        # Add card for each example
        for result in individual_results:
            example_name = result.get('example_name', 'unknown')
            description = result.get('description', 'No description')
            bug_type = result.get('bug_type', 'unknown')

            html_content += f"""
        <div class="example-card">
            <h3>{example_name}</h3>
            <p>{html_module.escape(description[:100])}{'...' if len(description) > 100 else ''}</p>
            <span class="bug-badge">{html_module.escape(bug_type)}</span>
            <br>
            <a href="{example_name}/index.html">View Analysis →</a>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        # Save index
        filepath = os.path.join(by_example_dir, "index.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_root_index(self, output_dir: str) -> None:
        """
        Create main index.html in output directory root.

        Args:
            output_dir: Base output directory
        """
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Error Detection Methods - Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 40px;
        }
        .section {
            margin: 30px 0;
            padding: 25px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: #fafafa;
        }
        .section h2 {
            color: #555;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 10px;
            transition: transform 0.3s, box-shadow 0.3s;
            display: block;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .card h3 {
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }
        .card p {
            margin: 0;
            opacity: 0.9;
        }
        .data-link {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 20px;
            background: #2196F3;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .data-link:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Advanced Error Detection Methods</h1>
        <p class="subtitle">Comprehensive Analysis Results</p>

        <div class="section">
            <h2>📊 Comparative Visualizations</h2>
            <p>High-level comparisons across all methods and examples</p>
            <div class="grid">
                <a href="advanced_visualizations/index.html" class="card">
                    <h3>🎯 All Comparative Views</h3>
                    <p>Interactive dashboards, heatmaps, and performance radars</p>
                </a>
                <a href="advanced_visualizations/interactive_method_explorer.html" class="card">
                    <h3>🔍 Quick Explorer</h3>
                    <p>Interactive method comparison dashboard</p>
                </a>
            </div>
        </div>

        <div class="section">
            <h2>🎨 Token-Level Visualizations</h2>
            <p>Detailed code highlighting showing exact token anomalies</p>
            <div class="grid">
                <a href="token_visualizations/by_example/index.html" class="card">
                    <h3>🎯 Browse by Example</h3>
                    <p>Compare all methods on each test example</p>
                </a>
            </div>
        </div>

        <div class="section">
            <h2>📁 Raw Data</h2>
            <a href="complete_comparison_results.json" class="data-link">💾 Complete Results (JSON)</a>
        </div>
    </div>
</body>
</html>
"""

        # Save index
        filepath = os.path.join(output_dir, "index.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)


if __name__ == "__main__":
    print("Token-Level Visualization Generator")
    print("\nThis module generates HTML visualizations showing individual tokens")
    print("highlighted by uncertainty/anomaly scores for each detection method.")
    print("\nUsage:")
    print("  from comparison.token_level_visualizer import TokenLevelVisualizer")
    print("  visualizer = TokenLevelVisualizer()")
    print("  visualizer.generate_all_token_visualizations(results, output_dir)")
