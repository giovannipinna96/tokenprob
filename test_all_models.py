#!/usr/bin/env python3
"""
Multi-Model Analysis Script

Questo script esegue l'analisi con tutti i modelli richiesti e genera un report comparativo.
"""

import os
import sys
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Modelli da testare (Phi e Qwen 32B/14B rimossi per ottimizzazione performance)
MODELS_TO_TEST = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "google/gemma-3-270m-it"
]

# Esempi da testare (subset per velocità)
TEST_EXAMPLES = [
    "binary_search_missing_bounds",
    "factorial_recursion_base_case",
    "bubble_sort_inner_loop",
    "fibonacci_negative_input"
]

def run_single_model_analysis(model_name: str, example: str, output_base_dir: str) -> Dict[str, Any]:
    """
    Esegue l'analisi per un singolo modello e esempio.
    """
    print(f"\n{'='*60}")
    print(f"🤖 Testing Model: {model_name}")
    print(f"📝 Example: {example}")
    print(f"{'='*60}")

    # Crea directory per il modello
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    model_dir = os.path.join(output_base_dir, model_safe_name)
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()

    try:
        # Esegui analisi
        cmd = [
            "uv", "run", "python", "run_analysis.py",
            "--example", example,
            "--model", model_name,
            "--output-dir", model_dir
        ]

        print(f"📊 Running analysis...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ Success! Duration: {duration:.1f}s")

            # Carica il risultato
            result_file = os.path.join(model_dir, f"{example}_single_analysis.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    analysis_data = json.load(f)

                return {
                    "status": "success",
                    "model": model_name,
                    "example": example,
                    "duration": duration,
                    "result_file": result_file,
                    "buggy_stats": analysis_data["buggy_analysis"]["statistics"],
                    "correct_stats": analysis_data["correct_analysis"]["statistics"],
                    "comparison": analysis_data["comparison"],
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "status": "error",
                    "model": model_name,
                    "example": example,
                    "duration": duration,
                    "error": "Result file not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            print(f"❌ Failed! Return code: {result.returncode}")
            return {
                "status": "error",
                "model": model_name,
                "example": example,
                "duration": duration,
                "error": f"Process failed with code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after 30 minutes")
        return {
            "status": "timeout",
            "model": model_name,
            "example": example,
            "duration": 1800,
            "error": "Analysis timed out after 30 minutes"
        }
    except Exception as e:
        print(f"💥 Exception: {e}")
        return {
            "status": "exception",
            "model": model_name,
            "example": example,
            "duration": time.time() - start_time,
            "error": str(e)
        }

def generate_comparative_report(results: List[Dict[str, Any]], output_dir: str):
    """
    Genera un report comparativo tra tutti i modelli.
    """
    print(f"\n{'='*60}")
    print("📊 Generating Comparative Report")
    print(f"{'='*60}")

    # Organizza risultati per modello e esempio
    model_results = {}
    for result in results:
        model = result["model"]
        if model not in model_results:
            model_results[model] = {}
        model_results[model][result["example"]] = result

    # Genera report markdown
    report = f"""# 🤖 Multi-Model LLM Token Probability Analysis Report

**Generated:** {datetime.now().isoformat()}
**Models Tested:** {len(MODELS_TO_TEST)}
**Examples Tested:** {len(TEST_EXAMPLES)}

## 📋 Executive Summary

"""

    # Tabella riassuntiva successi
    report += """
### Success Rate by Model

| Model | Successful | Failed | Success Rate |
|-------|------------|--------|--------------|
"""

    for model in MODELS_TO_TEST:
        if model in model_results:
            successful = sum(1 for ex_result in model_results[model].values() if ex_result["status"] == "success")
            total = len(model_results[model])
            success_rate = successful / total * 100 if total > 0 else 0
            failed = total - successful
            report += f"| {model} | {successful} | {failed} | {success_rate:.1f}% |\n"
        else:
            report += f"| {model} | 0 | {len(TEST_EXAMPLES)} | 0.0% |\n"

    # Analisi ipotesi per modello
    report += """

## 🎯 Hypothesis Validation Results

The hypothesis: **Low probability tokens correlate with buggy code areas**

"""

    for model in MODELS_TO_TEST:
        if model in model_results:
            report += f"\n### {model}\n\n"

            confirmations = {"probability": 0, "regions": 0, "entropy": 0}
            total_examples = 0
            avg_durations = []

            for example, result in model_results[model].items():
                if result["status"] == "success":
                    total_examples += 1
                    avg_durations.append(result["duration"])

                    comp = result["comparison"]
                    if comp["probability_difference"]["hypothesis_confirmed"]:
                        confirmations["probability"] += 1
                    if comp["low_confidence_comparison"]["hypothesis_confirmed"]:
                        confirmations["regions"] += 1
                    if comp["entropy_comparison"]["hypothesis_confirmed"]:
                        confirmations["entropy"] += 1

            if total_examples > 0:
                avg_duration = sum(avg_durations) / len(avg_durations)

                report += f"""
**Performance:**
- Average Analysis Time: {avg_duration:.1f}s
- Successfully Analyzed: {total_examples}/{len(TEST_EXAMPLES)} examples

**Hypothesis Validation:**
- Probability Difference: {confirmations['probability']}/{total_examples} ({confirmations['probability']/total_examples*100:.1f}%) ✅
- Low Confidence Regions: {confirmations['regions']}/{total_examples} ({confirmations['regions']/total_examples*100:.1f}%) ✅
- Entropy Difference: {confirmations['entropy']}/{total_examples} ({confirmations['entropy']/total_examples*100:.1f}%) ✅

**Overall Hypothesis Status:** {'✅ CONFIRMED' if (sum(confirmations.values()) / (total_examples * 3)) > 0.6 else '❌ REJECTED'}

"""
            else:
                report += "**Status:** ❌ No successful analyses\n\n"

    # Dettagli per esempio
    report += """
## 📝 Detailed Results by Example

"""

    for example in TEST_EXAMPLES:
        report += f"\n### {example}\n\n"
        report += "| Model | Status | Avg Prob (Buggy) | Avg Prob (Correct) | Diff | Hypothesis |\n"
        report += "|-------|--------|-------------------|---------------------|------|------------|\n"

        for model in MODELS_TO_TEST:
            if model in model_results and example in model_results[model]:
                result = model_results[model][example]
                if result["status"] == "success":
                    buggy_prob = result["buggy_stats"]["avg_probability"]
                    correct_prob = result["correct_stats"]["avg_probability"]
                    diff = correct_prob - buggy_prob
                    hypothesis = "✅" if result["comparison"]["probability_difference"]["hypothesis_confirmed"] else "❌"

                    report += f"| {model} | ✅ Success | {buggy_prob:.3f} | {correct_prob:.3f} | {diff:+.3f} | {hypothesis} |\n"
                else:
                    report += f"| {model} | ❌ {result['status'].title()} | - | - | - | - |\n"
            else:
                report += f"| {model} | ❌ Not Run | - | - | - | - |\n"

    # Errori e problemi
    report += """

## 🚨 Errors and Issues

"""

    errors_found = False
    for model in MODELS_TO_TEST:
        if model in model_results:
            model_errors = [result for result in model_results[model].values() if result["status"] != "success"]
            if model_errors:
                errors_found = True
                report += f"\n### {model}\n\n"
                for error_result in model_errors:
                    report += f"**{error_result['example']}:** {error_result['status']} - {error_result.get('error', 'Unknown error')}\n\n"
                    if 'stderr' in error_result and error_result['stderr']:
                        report += f"```\n{error_result['stderr'][:500]}...\n```\n\n"

    if not errors_found:
        report += "🎉 No errors encountered!\n\n"

    # Raccomandazioni
    report += """
## 💡 Recommendations

Based on the multi-model analysis:

1. **Best Performing Models:** Models with highest success rates and hypothesis confirmation
2. **Speed vs Accuracy:** Consider model size vs analysis time trade-offs
3. **Reliability:** Models with consistent results across examples
4. **Resource Usage:** Monitor GPU memory usage for larger models

## 📊 Data Files

All detailed results are available in the following directories:
"""

    for model in MODELS_TO_TEST:
        model_safe_name = model.replace("/", "_").replace("-", "_")
        report += f"- `{model_safe_name}/` - Results for {model}\n"

    # Salva report
    report_file = os.path.join(output_dir, "multi_model_comparative_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Salva anche JSON dei risultati
    results_file = os.path.join(output_dir, "multi_model_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📄 Report saved to: {report_file}")
    print(f"📊 Raw results saved to: {results_file}")

def main():
    """
    Funzione principale per eseguire l'analisi multi-modello.
    """
    print("🚀 Starting Multi-Model LLM Token Probability Analysis")
    print(f"📋 Models to test: {len(MODELS_TO_TEST)}")
    print(f"📝 Examples per model: {len(TEST_EXAMPLES)}")
    print(f"🎯 Total analyses: {len(MODELS_TO_TEST) * len(TEST_EXAMPLES)}")

    output_dir = "multi_model_analysis"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_analyses = len(MODELS_TO_TEST) * len(TEST_EXAMPLES)
    current_analysis = 0

    for model in MODELS_TO_TEST:
        print(f"\n🤖 Starting analysis for model: {model}")

        for example in TEST_EXAMPLES:
            current_analysis += 1
            print(f"\n📊 Progress: {current_analysis}/{total_analyses}")

            result = run_single_model_analysis(model, example, output_dir)
            all_results.append(result)

            # Pausa breve tra analisi per GPU cooling
            if result["status"] == "success":
                print("😴 Cooling down GPU (10s)...")
                time.sleep(10)

    # Genera report comparativo
    generate_comparative_report(all_results, output_dir)

    print(f"\n🎉 Multi-model analysis complete!")
    print(f"📁 Results saved in: {output_dir}/")

    # Riassunto finale
    successful = sum(1 for r in all_results if r["status"] == "success")
    print(f"✅ Successful analyses: {successful}/{total_analyses}")
    print(f"❌ Failed analyses: {total_analyses - successful}/{total_analyses}")

if __name__ == "__main__":
    main()