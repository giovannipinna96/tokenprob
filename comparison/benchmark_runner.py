#!/usr/bin/env python3
"""
Multi-Model Benchmark Runner

Runs error detection benchmarks across multiple models sequentially and
collects comparative results.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.starcoder2_detector import StarCoder2ErrorDetector
from detectors.codet5_detector import CodeT5ErrorDetector
from detectors.deepseek_detector import DeepSeekErrorDetector
from codebert_error_detector import CodeBERTErrorDetector
from logical_error_detector import LogicalErrorDetector
from test_examples import TestExamplesDataset, TestExample


class MultiModelBenchmark:
    """
    Benchmark runner for comparing multiple error detection models.

    Runs all models sequentially (not in parallel) and collects results
    for statistical comparison.
    """

    def __init__(self, sensitivity_factor: float = 1.5):
        """
        Initialize the benchmark with detectors.

        Args:
            sensitivity_factor: k parameter for all detectors
        """
        self.k = sensitivity_factor
        self.dataset = TestExamplesDataset()

        # Initialize detectors (will be loaded on demand to save memory)
        self.detector_configs = {
            'starcoder2-7b': {
                'class': StarCoder2ErrorDetector,
                'name': 'StarCoder2-7B',
                'enabled': True
            },
            'codet5p-2b': {
                'class': CodeT5ErrorDetector,
                'name': 'CodeT5+ 2B',
                'enabled': True
            },
            'deepseek-6.7b': {
                'class': DeepSeekErrorDetector,
                'name': 'DeepSeek-Coder 6.7B',
                'enabled': True
            },
            'codebert': {
                'class': CodeBERTErrorDetector,
                'name': 'CodeBERT',
                'enabled': True
            },
            'qwen-7b': {
                'class': LogicalErrorDetector,
                'name': 'Qwen 2.5 Coder 7B',
                'kwargs': {'model_name': "Qwen/Qwen2.5-Coder-7B-Instruct"},
                'enabled': True
            }
        }

    def run_full_benchmark(self,
                          models: List[str] = None,
                          output_dir: str = "comparison_study") -> Dict[str, Any]:
        """
        Run benchmark on all (or selected) models.

        Executes sequentially to avoid memory issues.

        Args:
            models: List of model names to test (None = all)
            output_dir: Directory to save results

        Returns:
            Complete benchmark results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Determine which models to test
        if models is None or 'all' in models:
            models_to_test = [k for k, v in self.detector_configs.items() if v['enabled']]
        else:
            models_to_test = [m for m in models if m in self.detector_configs]

        print("="*80)
        print("MULTI-MODEL ERROR DETECTION BENCHMARK")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Models to test: {len(models_to_test)}")
        print(f"Test examples: {len(self.dataset.examples)}")
        print(f"Sensitivity factor (k): {self.k}")
        print(f"Output directory: {output_dir}")
        print("="*80)

        # Initialize results structure
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_examples': len(self.dataset.examples),
                'num_models': len(models_to_test),
                'sensitivity_factor': self.k,
                'models_tested': models_to_test
            },
            'model_results': {},
            'timing': {}
        }

        # Test each model sequentially
        for model_idx, model_key in enumerate(models_to_test, 1):
            print(f"\n{'='*80}")
            print(f"[{model_idx}/{len(models_to_test)}] Testing: {self.detector_configs[model_key]['name']}")
            print(f"{'='*80}")

            start_time = time.time()

            try:
                # Load detector
                detector = self._load_detector(model_key)

                # Run tests on all examples
                model_results = self._test_model_on_all_examples(detector, model_key)

                # Save results for this model
                results['model_results'][model_key] = model_results

                # Record timing
                elapsed = time.time() - start_time
                results['timing'][model_key] = {
                    'total_seconds': elapsed,
                    'avg_per_example': elapsed / len(self.dataset.examples)
                }

                print(f"\n  Completed in {elapsed:.1f} seconds")
                print(f"  Average per example: {elapsed/len(self.dataset.examples):.1f}s")

                # Save intermediate results (in case of crash)
                self._save_intermediate_results(results, output_dir, model_key)

                # Free memory (delete detector)
                del detector
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n  ERROR testing {model_key}: {e}")
                results['model_results'][model_key] = {
                    'error': str(e),
                    'status': 'failed'
                }
                continue

        # Compute aggregate statistics
        results['aggregate_stats'] = self._compute_aggregate_statistics(results['model_results'])

        # Save final results
        self._save_final_results(results, output_dir)

        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"Successful models: {len([k for k, v in results['model_results'].items() if 'error' not in v])}")

        return results

    def _load_detector(self, model_key: str):
        """Load a detector on demand."""
        config = self.detector_configs[model_key]
        detector_class = config['class']
        kwargs = config.get('kwargs', {})

        print(f"  Loading {config['name']}...")
        detector = detector_class(sensitivity_factor=self.k, **kwargs)

        return detector

    def _test_model_on_all_examples(self, detector, model_key: str) -> Dict[str, Any]:
        """Test a single model on all examples."""
        individual_results = []

        for i, example in enumerate(self.dataset.examples, 1):
            print(f"\n  [{i}/{len(self.dataset.examples)}] {example.name} ({example.bug_type})")

            try:
                # Analyze buggy and correct code
                comparison = detector.compare_buggy_vs_correct(
                    example.buggy_code,
                    example.correct_code,
                    k=self.k
                )

                # Extract key metrics
                buggy_stats = comparison['buggy_analysis']['statistics']
                correct_stats = comparison['correct_analysis']['statistics']

                result = {
                    'example_name': example.name,
                    'bug_type': example.bug_type,
                    'buggy_anomalies': buggy_stats['anomalous_tokens'],
                    'correct_anomalies': correct_stats['anomalous_tokens'],
                    'buggy_error_lines': buggy_stats['error_lines'],
                    'correct_error_lines': correct_stats['error_lines'],
                    'buggy_mean_log_prob': buggy_stats['mean_log_prob'],
                    'correct_mean_log_prob': correct_stats['mean_log_prob'],
                    'hypothesis_confirmed': comparison['comparison']['hypothesis_confirmed']
                }

                individual_results.append(result)

                # Print result
                status = "✓" if result['hypothesis_confirmed'] else "✗"
                print(f"     Buggy: {result['buggy_anomalies']} anomalies | "
                      f"Correct: {result['correct_anomalies']} anomalies | {status}")

            except Exception as e:
                print(f"     ERROR: {e}")
                individual_results.append({
                    'example_name': example.name,
                    'error': str(e),
                    'status': 'failed'
                })
                continue

        # Compute aggregate statistics for this model
        aggregate_stats = self._compute_model_aggregate_stats(individual_results)

        return {
            'individual_results': individual_results,
            'aggregate_stats': aggregate_stats,
            'metadata': detector.get_metadata().__dict__
        }

    def _compute_model_aggregate_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics for a single model."""
        # Filter out failed results
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {'error': 'No valid results'}

        # Count confirmations
        confirmations = sum(1 for r in valid_results if r['hypothesis_confirmed'])
        total = len(valid_results)

        # Group by bug type
        bug_type_stats = {}
        for result in valid_results:
            bug_type = result['bug_type']
            if bug_type not in bug_type_stats:
                bug_type_stats[bug_type] = {'total': 0, 'confirmed': 0}
            bug_type_stats[bug_type]['total'] += 1
            if result['hypothesis_confirmed']:
                bug_type_stats[bug_type]['confirmed'] += 1

        # Compute confirmation rates by bug type
        for bug_type in bug_type_stats:
            stats = bug_type_stats[bug_type]
            stats['confirmation_rate'] = stats['confirmed'] / stats['total'] if stats['total'] > 0 else 0.0

        return {
            'total_examples': total,
            'successful_examples': len(valid_results),
            'failed_examples': len(results) - len(valid_results),
            'confirmations': confirmations,
            'confirmation_rate': confirmations / total if total > 0 else 0.0,
            'bug_type_breakdown': bug_type_stats,
            'avg_buggy_anomalies': sum(r['buggy_anomalies'] for r in valid_results) / len(valid_results),
            'avg_correct_anomalies': sum(r['correct_anomalies'] for r in valid_results) / len(valid_results)
        }

    def _compute_aggregate_statistics(self, model_results: Dict) -> Dict[str, Any]:
        """Compute cross-model aggregate statistics."""
        stats = {}

        for model_key, results in model_results.items():
            if 'error' in results:
                continue

            agg = results.get('aggregate_stats', {})
            stats[model_key] = {
                'confirmation_rate': agg.get('confirmation_rate', 0.0),
                'confirmations': agg.get('confirmations', 0),
                'total': agg.get('total_examples', 0)
            }

        # Rank models by confirmation rate
        ranked = sorted(stats.items(), key=lambda x: x[1]['confirmation_rate'], reverse=True)
        stats['ranking'] = [model for model, _ in ranked]

        return stats

    def _save_intermediate_results(self, results: Dict, output_dir: str, model_key: str):
        """Save intermediate results after each model."""
        intermediate_file = os.path.join(output_dir, f"results_{model_key}.json")

        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results['model_results'][model_key], f, indent=2, ensure_ascii=False)

        print(f"  Saved intermediate results to {intermediate_file}")

    def _save_final_results(self, results: Dict, output_dir: str):
        """Save final complete results."""
        final_file = os.path.join(output_dir, "complete_benchmark_results.json")

        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n  Final results saved to {final_file}")


if __name__ == "__main__":
    # Example usage (not executed automatically)
    print("Multi-Model Benchmark Runner")
    print("\nTo run benchmark:")
    print("  benchmark = MultiModelBenchmark(sensitivity_factor=1.5)")
    print("  results = benchmark.run_full_benchmark()")
    print("\nOr use test_multi_model_comparison.py script")
