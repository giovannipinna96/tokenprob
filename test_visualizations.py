#!/usr/bin/env python3
"""
Multi-Model Visualization Testing Script

Questo script genera visualizzazioni per tutti i modelli supportati
e crea plot organizzati in cartelle per ogni modello.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LLM import QwenProbabilityAnalyzer
from visualizer import TokenVisualizer, TokenVisualizationMode

# Modelli da testare (ottimizzati per performance e compatibilità)
MODELS_TO_TEST = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "google/gemma-3-270m-it"
]

# Esempi da visualizzare
VISUALIZATION_EXAMPLES = [
    {
        "name": "fibonacci_bug",
        "prompt": "Write a Python function to calculate the nth Fibonacci number using recursion:",
        "expected_code": '''def fibonacci(n):
    if n <= 1:  # Missing validation for negative numbers
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''
    },
    {
        "name": "binary_search_correct",
        "prompt": "Implement binary search algorithm in Python:",
        "expected_code": '''def binary_search(arr, target):
    left = 0
    right = len(arr) - 1  # Correct bounds

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1'''
    },
    {
        "name": "factorial_with_validation",
        "prompt": "Create a factorial function with proper input validation:",
        "expected_code": '''def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)'''
    }
]

# Modalità di visualizzazione da testare (tutte le modalità disponibili)
VISUALIZATION_MODES = [
    # === Metriche Base ===
    ("PROBABILITY", TokenVisualizationMode.PROBABILITY),
    ("RANK", TokenVisualizationMode.RANK),
    ("LOGITS", TokenVisualizationMode.LOGITS),

    # === Teoria dell'Informazione ===
    ("ENTROPY", TokenVisualizationMode.ENTROPY),
    ("SURPRISAL", TokenVisualizationMode.SURPRISAL),
    ("PERPLEXITY", TokenVisualizationMode.PERPLEXITY),
    ("SHANNON_ENTROPY", TokenVisualizationMode.SHANNON_ENTROPY),

    # === Metriche Avanzate ===
    ("MAX_PROBABILITY", TokenVisualizationMode.MAX_PROBABILITY),
    ("PROBABILITY_MARGIN", TokenVisualizationMode.PROBABILITY_MARGIN),
    ("LOCAL_PERPLEXITY", TokenVisualizationMode.LOCAL_PERPLEXITY),
    ("SEQUENCE_IMPROBABILITY", TokenVisualizationMode.SEQUENCE_IMPROBABILITY),
    ("CONFIDENCE_SCORE", TokenVisualizationMode.CONFIDENCE_SCORE)
]

class VisualizationTester:
    """Testa le visualizzazioni per tutti i modelli."""

    def __init__(self, output_dir: str = "model_visualizations"):
        """Inizializza il tester."""
        self.output_dir = output_dir
        self.visualizer = TokenVisualizer()
        os.makedirs(output_dir, exist_ok=True)

    def test_model_visualizations(self, model_name: str) -> Dict[str, Any]:
        """
        Testa le visualizzazioni per un singolo modello.

        Args:
            model_name: Nome del modello da testare

        Returns:
            Risultati del test
        """
        print(f"\n{'='*60}")
        print(f"🎨 Testing Visualizations for: {model_name}")
        print(f"{'='*60}")

        # Crea directory per il modello
        model_safe_name = model_name.replace("/", "_").replace("-", "_")
        model_dir = os.path.join(self.output_dir, model_safe_name)
        os.makedirs(model_dir, exist_ok=True)

        # Inizializza analyzer
        start_time = time.time()
        try:
            print(f"⚡ Loading model...")
            analyzer = QwenProbabilityAnalyzer(model_name=model_name)
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return {
                "status": "error",
                "model": model_name,
                "error": str(e),
                "load_time": time.time() - start_time
            }

        results = {
            "status": "success",
            "model": model_name,
            "load_time": load_time,
            "examples": [],
            "visualizations_created": 0
        }

        # Testa ogni esempio
        for i, example in enumerate(VISUALIZATION_EXAMPLES, 1):
            print(f"\n📝 Example {i}/{len(VISUALIZATION_EXAMPLES)}: {example['name']}")

            try:
                # Genera analisi
                print("🔍 Generating analysis...")
                generated_text, token_analyses = analyzer.generate_with_analysis(
                    prompt=example['prompt'],
                    max_new_tokens=100,  # Ridotto per velocità
                    temperature=0.3
                )

                example_result = {
                    "name": example['name'],
                    "status": "success",
                    "prompt": example['prompt'],
                    "generated_text": generated_text,
                    "num_tokens": len(token_analyses),
                    "visualizations": []
                }

                # Genera visualizzazioni per ogni modalità
                for mode_name, mode_value in VISUALIZATION_MODES:
                    print(f"🎨 Creating {mode_name} visualization...")

                    try:
                        # Crea visualizzazione HTML
                        html_viz = self.visualizer.create_html_visualization(
                            token_analyses,
                            mode=mode_value,
                            title=f"{model_safe_name} - {example['name']} - {mode_name}"
                        )

                        # Salva file HTML
                        html_filename = f"{example['name']}_{mode_name.lower()}.html"
                        html_path = os.path.join(model_dir, html_filename)

                        with open(html_path, 'w', encoding='utf-8') as f:
                            f.write(html_viz)

                        # Crea anche plot matplotlib se possibile
                        try:
                            plot_filename = f"{example['name']}_{mode_name.lower()}.png"
                            plot_path = os.path.join(model_dir, plot_filename)

                            fig = self.visualizer.create_matplotlib_visualization(
                                token_analyses,
                                mode=mode_value
                            )
                            fig.suptitle(f"{model_name} - {example['name']} ({mode_name})")
                            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)

                            viz_result = {
                                "mode": mode_name,
                                "html_file": html_path,
                                "plot_file": plot_path,
                                "status": "success"
                            }

                        except Exception as plot_error:
                            print(f"⚠️ Plot creation failed: {plot_error}")
                            viz_result = {
                                "mode": mode_name,
                                "html_file": html_path,
                                "plot_file": None,
                                "status": "html_only",
                                "plot_error": str(plot_error)
                            }

                        example_result["visualizations"].append(viz_result)
                        results["visualizations_created"] += 1

                    except Exception as viz_error:
                        print(f"❌ Visualization failed for {mode_name}: {viz_error}")
                        example_result["visualizations"].append({
                            "mode": mode_name,
                            "status": "error",
                            "error": str(viz_error)
                        })

                results["examples"].append(example_result)

                # Salva anche l'analisi JSON
                analysis_file = os.path.join(model_dir, f"{example['name']}_analysis.json")
                analyzer.save_analysis(analysis_file)

                print(f"✅ Completed {example['name']} ({len(example_result['visualizations'])} visualizations)")

            except Exception as e:
                print(f"❌ Example failed: {e}")
                results["examples"].append({
                    "name": example['name'],
                    "status": "error",
                    "error": str(e)
                })

        # Crea indice HTML per il modello
        self.create_model_index(model_dir, model_name, results)

        return results

    def create_model_index(self, model_dir: str, model_name: str, results: Dict[str, Any]):
        """Crea un file indice HTML per visualizzare tutti i risultati del modello."""

        index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizations - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; border-left: 4px solid #2196F3; padding-left: 15px; }}
        .example-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .example-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa; }}
        .example-title {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
        .viz-links {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .viz-link {{ padding: 8px 12px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; font-size: 14px; }}
        .viz-link:hover {{ background: #45a049; }}
        .viz-link.plot {{ background: #2196F3; }}
        .viz-link.plot:hover {{ background: #1976D2; }}
        .stats {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .stats p {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Visualization Results - {model_name}</h1>

        <div class="stats">
            <h3>📊 Summary</h3>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Load Time:</strong> {results.get('load_time', 0):.1f}s</p>
            <p><strong>Examples Processed:</strong> {len([ex for ex in results.get('examples', []) if ex.get('status') == 'success'])}/{len(results.get('examples', []))}</p>
            <p><strong>Visualizations Created:</strong> {results.get('visualizations_created', 0)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <h2>📝 Examples and Visualizations</h2>

        <div class="example-grid">
"""

        # Aggiungi ogni esempio
        for example in results.get("examples", []):
            if example.get("status") == "success":
                index_html += f"""
            <div class="example-card">
                <div class="example-title">{example['name']}</div>
                <p><strong>Tokens:</strong> {example.get('num_tokens', 'N/A')}</p>
                <p><strong>Prompt:</strong> {example['prompt'][:100]}...</p>

                <div class="viz-links">
"""

                # Organizza le visualizzazioni in gruppi logici
                visualization_groups = {
                    "Metriche Base": ["PROBABILITY", "RANK", "LOGITS"],
                    "Teoria dell'Informazione": ["ENTROPY", "SURPRISAL", "PERPLEXITY", "SHANNON_ENTROPY"],
                    "Metriche Avanzate": ["MAX_PROBABILITY", "PROBABILITY_MARGIN", "LOCAL_PERPLEXITY", "SEQUENCE_IMPROBABILITY", "CONFIDENCE_SCORE"]
                }

                # Crea una mappa delle visualizzazioni per modalità
                viz_by_mode = {}
                for viz in example.get("visualizations", []):
                    if viz.get("status") in ["success", "html_only"]:
                        viz_by_mode[viz["mode"]] = viz

                # Genera link organizzati per gruppo
                for group_name, modes in visualization_groups.items():
                    index_html += f'<h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">{group_name}</h4>\n'
                    index_html += '<div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;">\n'

                    for mode in modes:
                        if mode in viz_by_mode:
                            viz = viz_by_mode[mode]
                            if viz.get("html_file"):
                                html_file = os.path.basename(viz["html_file"])
                                index_html += f'<a href="{html_file}" class="viz-link" target="_blank">{mode}</a>\n'

                            if viz.get("plot_file"):
                                plot_file = os.path.basename(viz["plot_file"])
                                index_html += f'<a href="{plot_file}" class="viz-link plot" target="_blank">{mode} (PNG)</a>\n'

                    index_html += '</div>\n'

                # Aggiungi link all'analisi JSON
                index_html += '<h4 style="margin: 15px 0 8px 0; color: #555; font-size: 14px;">Dati Raw</h4>\n'
                analysis_file = f"{example['name']}_analysis.json"
                index_html += f'<a href="{analysis_file}" class="viz-link" style="background: #FF9800;" target="_blank">Analisi JSON</a>\n'

                index_html += """
                </div>
            </div>
"""

        index_html += """
        </div>

        <h2>🔍 About Token Probability Analysis</h2>
        <p>These visualizations show the confidence levels of the language model for each generated token:</p>
        <ul>
            <li><strong>Probability:</strong> How confident the model was in selecting each token (green = high, red = low)</li>
            <li><strong>Entropy:</strong> How much uncertainty the model had (red = high uncertainty, blue = low uncertainty)</li>
            <li><strong>Surprisal:</strong> How unexpected each token was (red = surprising, blue = expected)</li>
            <li><strong>Rank:</strong> Position of selected token among all possibilities (red = high rank/unusual, green = low rank/common)</li>
        </ul>

        <p><em>Hypothesis: Low confidence tokens may correlate with areas prone to bugs or errors.</em></p>
    </div>
</body>
</html>
"""

        index_path = os.path.join(model_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)

        print(f"📄 Model index created: {index_path}")

    def run_all_visualizations(self) -> Dict[str, Any]:
        """Esegue i test di visualizzazione per tutti i modelli."""

        print("🎨 Starting Multi-Model Visualization Testing")
        print(f"📋 Models to test: {len(MODELS_TO_TEST)}")
        print(f"📝 Examples per model: {len(VISUALIZATION_EXAMPLES)}")
        print(f"🎯 Visualization modes: {len(VISUALIZATION_MODES)}")

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "models": [],
            "summary": {
                "total_models": len(MODELS_TO_TEST),
                "successful_models": 0,
                "failed_models": 0,
                "total_visualizations": 0
            }
        }

        for i, model in enumerate(MODELS_TO_TEST, 1):
            print(f"\n🤖 Progress: {i}/{len(MODELS_TO_TEST)} - {model}")

            model_result = self.test_model_visualizations(model)
            all_results["models"].append(model_result)

            if model_result["status"] == "success":
                all_results["summary"]["successful_models"] += 1
                all_results["summary"]["total_visualizations"] += model_result.get("visualizations_created", 0)
            else:
                all_results["summary"]["failed_models"] += 1

            # Pausa tra modelli per raffreddamento GPU
            if i < len(MODELS_TO_TEST):
                print("😴 Cooling down GPU (15s)...")
                time.sleep(15)

        # Crea indice generale
        self.create_main_index(all_results)

        return all_results

    def create_main_index(self, results: Dict[str, Any]):
        """Crea l'indice principale con link a tutti i modelli."""

        main_index = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Model Visualization Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f0f8ff; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 15px; }}
        .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 30px 0; }}
        .model-card {{ border: 2px solid #ecf0f1; border-radius: 10px; padding: 20px; background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); transition: transform 0.3s ease; }}
        .model-card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }}
        .model-name {{ font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .model-status {{ padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 15px; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-error {{ background: #f8d7da; color: #721c24; }}
        .model-link {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 25px; margin-top: 10px; transition: background 0.3s ease; }}
        .model-link:hover {{ background: #2980b9; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 14px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Multi-Model LLM Visualization Results</h1>

        <div class="summary">
            <h2 style="margin-top: 0; color: white;">📊 Summary</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{results['summary']['total_models']}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Models Tested</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{results['summary']['successful_models']}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Successful</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{results['summary']['total_visualizations']}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Visualizations</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: white;">{len(VISUALIZATION_EXAMPLES)}</div>
                    <div class="stat-label" style="color: #ecf0f1;">Examples Each</div>
                </div>
            </div>
            <p style="margin-bottom: 0; text-align: center; font-size: 14px; color: #ecf0f1;">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>

        <h2>🤖 Model Results</h2>

        <div class="model-grid">
"""

        for model_result in results["models"]:
            model_name = model_result["model"]
            model_safe_name = model_name.replace("/", "_").replace("-", "_")
            status = model_result["status"]

            status_class = "status-success" if status == "success" else "status-error"
            status_text = "✅ Success" if status == "success" else "❌ Failed"

            main_index += f"""
            <div class="model-card">
                <div class="model-name">{model_name}</div>
                <div class="model-status {status_class}">{status_text}</div>
"""

            if status == "success":
                successful_examples = len([ex for ex in model_result.get("examples", []) if ex.get("status") == "success"])
                main_index += f"""
                <p><strong>Load Time:</strong> {model_result.get('load_time', 0):.1f}s</p>
                <p><strong>Examples:</strong> {successful_examples}/{len(VISUALIZATION_EXAMPLES)}</p>
                <p><strong>Visualizations:</strong> {model_result.get('visualizations_created', 0)}</p>
                <a href="{model_safe_name}/index.html" class="model-link">View Visualizations</a>
"""
            else:
                error_msg = model_result.get("error", "Unknown error")[:100]
                main_index += f"""
                <p><strong>Error:</strong> {error_msg}...</p>
                <p><strong>Load Time:</strong> {model_result.get('load_time', 0):.1f}s</p>
"""

            main_index += """
            </div>
"""

        main_index += """
        </div>

        <h2>🔍 About This Analysis</h2>
        <p>This comprehensive multi-model visualization study compares how different language models generate code and the confidence levels they exhibit. Each model was tested with the same set of coding examples, and 12 different types of visualizations were created to analyze token generation patterns.</p>

        <h3>📊 Visualization Types</h3>

        <h4>🔹 Metriche Base</h4>
        <ul>
            <li><strong>Token Probability:</strong> How confident each model was in selecting specific tokens</li>
            <li><strong>Token Rank:</strong> Position of chosen tokens in the probability-sorted vocabulary</li>
            <li><strong>Logits:</strong> Raw model outputs before probability normalization</li>
        </ul>

        <h4>🔹 Teoria dell'Informazione</h4>
        <ul>
            <li><strong>Entropy:</strong> Uncertainty in the model's probability distribution</li>
            <li><strong>Surprisal:</strong> How unexpected each chosen token was</li>
            <li><strong>Perplexity:</strong> Model uncertainty measure (2^entropy)</li>
            <li><strong>Shannon Entropy:</strong> Information-theoretic measure of distribution spread</li>
        </ul>

        <h4>🔹 Metriche Avanzate</h4>
        <ul>
            <li><strong>Max Probability:</strong> Highest probability in the distribution for each position</li>
            <li><strong>Probability Margin:</strong> Difference between top-1 and top-2 probabilities</li>
            <li><strong>Local Perplexity:</strong> Per-token perplexity calculation</li>
            <li><strong>Sequence Improbability:</strong> Cumulative improbability of the generated sequence</li>
            <li><strong>Confidence Score:</strong> Composite metric combining multiple uncertainty factors</li>
        </ul>

        <p><strong>Research Hypothesis:</strong> Low-confidence tokens (identified through these various metrics) correlate with areas prone to bugs or errors in generated code. The comprehensive analysis using multiple uncertainty quantification methods provides deeper insights into model behavior and potential error-prone regions.</p>

        <h3>📁 Directory Structure</h3>
        <ul>
            <li><code>index.html</code> - This main overview page</li>
            <li><code>[model_name]/</code> - Individual directories for each model</li>
            <li><code>[model_name]/index.html</code> - Model-specific visualization overview</li>
            <li><code>[model_name]/[example]_[mode].html</code> - Interactive HTML visualizations</li>
            <li><code>[model_name]/[example]_[mode].png</code> - Static plot images</li>
            <li><code>[model_name]/[example]_analysis.json</code> - Raw analysis data with top-10 tokens</li>
        </ul>
    </div>
</body>
</html>
"""

        index_path = os.path.join(self.output_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(main_index)

        print(f"\n📄 Main index created: {index_path}")

def main():
    """Funzione principale."""
    print("🎨 Multi-Model Visualization Testing")
    print("="*60)

    tester = VisualizationTester()
    results = tester.run_all_visualizations()

    print(f"\n🎉 Visualization testing complete!")
    print(f"📁 Results saved in: {tester.output_dir}/")
    print(f"✅ Successful models: {results['summary']['successful_models']}/{results['summary']['total_models']}")
    print(f"🎨 Total visualizations created: {results['summary']['total_visualizations']}")
    print(f"🌐 Open: {tester.output_dir}/index.html")

if __name__ == "__main__":
    main()