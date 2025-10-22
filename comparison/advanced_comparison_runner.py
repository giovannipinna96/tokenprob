#!/usr/bin/env python3
"""
Advanced Methods Comparison Runner

Runs comprehensive comparison between 4 error detection methods:
1. LecPrompt (baseline log-probability)
2. Semantic Energy
3. Conformal Prediction
4. Attention Anomaly

Collects detailed metrics for performance analysis and visualization.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.advanced_methods import (
    SemanticEnergyDetector,
    ConformalPredictionDetector,
    AttentionAnomalyDetector,
    SemanticContextDetector,
    MaskedTokenReplacementDetector,
    AdvancedMethodsComparator,
    MethodComparisonResult
)
from test_examples import TestExamplesDataset, TestExample


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AdvancedMethodsComparisonRunner:
    """
    Runner for comparing advanced error detection methods.

    Executes all 4 methods on test examples and collects comprehensive
    comparison metrics.
    """

    def __init__(self,
                 model,
                 tokenizer,
                 baseline_detector,
                 sensitivity_factor: float = 1.5,
                 conformal_alpha: float = 0.1):
        """
        Initialize comparison runner.

        Args:
            model: Language model
            tokenizer: Tokenizer
            baseline_detector: LecPrompt baseline detector
            sensitivity_factor: k parameter for all detectors
            conformal_alpha: Significance level for conformal prediction
        """
        self.model = model
        self.tokenizer = tokenizer
        self.baseline_detector = baseline_detector
        self.k = sensitivity_factor

        # Initialize advanced detectors
        self.semantic_energy = SemanticEnergyDetector(sensitivity_factor)
        self.conformal = ConformalPredictionDetector(conformal_alpha, sensitivity_factor)
        self.attention = AttentionAnomalyDetector(sensitivity_factor)

        # Initialize semantic context detector (5th method - optional)
        print("\nInitializing Semantic Context Detector...")
        self.semantic_context = SemanticContextDetector(
            context_window=3,
            sensitivity_factor=sensitivity_factor
        )

        # Initialize masked token replacement detector (6th method - optional)
        print("\nInitializing Masked Token Replacement Detector...")
        self.masked_token_replacement = MaskedTokenReplacementDetector(
            model_name="microsoft/codebert-base",
            sensitivity_threshold=0.7
        )

        # Initialize comparator with all detectors (up to 6 methods)
        self.comparator = AdvancedMethodsComparator(
            self.semantic_energy,
            self.conformal,
            self.attention,
            self.semantic_context if self.semantic_context.is_available() else None,
            self.masked_token_replacement if self.masked_token_replacement.is_available() else None
        )

        # Load test dataset
        self.dataset = TestExamplesDataset()

        # Calibration state
        self.calibration_performed = False

    def _run_calibration_phase(self, exclude_examples: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run calibration phase for conformal prediction using calibration set.

        This method extracts logits and token IDs from the CORRECT code of
        calibration examples and uses them to compute quantile threshold for
        formal coverage guarantees.

        Args:
            exclude_examples: Examples to exclude from calibration (e.g., requested test examples)

        Returns:
            Dictionary with calibration results and metadata
        """
        print("\n" + "="*80)
        print("CONFORMAL PREDICTION CALIBRATION PHASE")
        print("="*80)

        # Get calibration examples (dynamic: excludes requested test examples)
        calibration_examples = self.dataset.get_calibration_set(exclude_examples=exclude_examples)
        calibration_info = self.dataset.get_calibration_info()

        print(f"\nCalibration set: {len(calibration_examples)} examples")
        for ex in calibration_examples:
            print(f"  - {ex.name} ({ex.bug_type})")

        if exclude_examples:
            print(f"\nℹ️  Dynamically chosen to exclude requested test examples: {exclude_examples}")

        # Collect calibration data from CORRECT code
        calibration_data = []
        total_tokens = 0

        print("\nExtracting calibration data from correct code...")
        for ex in calibration_examples:
            print(f"  Processing: {ex.name}")

            # Tokenize correct code
            encoding = self.tokenizer(ex.correct_code, return_tensors="pt")
            input_ids = encoding.input_ids.to(self.model.device)

            # Get logits from model
            with torch.no_grad():
                outputs = self.model(input_ids)
                # For causal models: logits at position i-1 predict token i
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # For each token position (skip first token)
            for i in range(1, len(input_ids[0])):
                token_id = input_ids[0][i]
                # Use logits from position i-1 to predict token i
                token_logits = logits[i-1].unsqueeze(0)  # [1, vocab_size]
                token_ids = token_id.unsqueeze(0)  # [1]

                calibration_data.append((token_logits, token_ids))
                total_tokens += 1

        print(f"\nCollected {total_tokens} tokens from {len(calibration_examples)} examples")

        # Calibrate conformal predictor
        print("\nCalibrating conformal predictor...")
        metadata = {
            'calibration_examples': [ex.name for ex in calibration_examples],
            'calibration_bug_types': [ex.bug_type for ex in calibration_examples]
        }
        self.conformal.calibrate(calibration_data, metadata)

        self.calibration_performed = True

        print("\n" + "="*80)

        return {
            'calibration_info': calibration_info,
            'conformal_metadata': self.conformal.get_calibration_info()
        }

    def run_comparison_on_example(self,
                                 example: TestExample) -> Dict[str, Any]:
        """
        Run all 4 methods on a single test example (buggy and correct versions).

        Args:
            example: TestExample to analyze

        Returns:
            Comprehensive comparison results
        """
        print(f"\n  Analyzing: {example.name}")

        # Analyze buggy code
        print("    - Buggy code...")
        buggy_comparison = self.comparator.compare_all_methods(
            example.buggy_code,
            self.model,
            self.tokenizer,
            self.baseline_detector,
            f"{example.name}_buggy"
        )

        # Analyze correct code
        print("    - Correct code...")
        correct_comparison = self.comparator.compare_all_methods(
            example.correct_code,
            self.model,
            self.tokenizer,
            self.baseline_detector,
            f"{example.name}_correct"
        )

        # Compute hypothesis confirmation for each method
        hypothesis_results = self._compute_hypothesis_confirmations(
            buggy_comparison,
            correct_comparison
        )

        # Compute inter-method agreement
        agreement_metrics = self._compute_inter_method_agreement(
            buggy_comparison,
            correct_comparison
        )

        # Aggregate results
        result = {
            'example_name': example.name,
            'bug_type': example.bug_type,
            'description': example.description,

            # Buggy code results
            'buggy': {
                'lecprompt': self._extract_method_summary(buggy_comparison.lecprompt_result),
                'semantic_energy': self._extract_method_summary(buggy_comparison.semantic_energy_result),
                'conformal': self._extract_method_summary(buggy_comparison.conformal_result),
                'attention': self._extract_method_summary(buggy_comparison.attention_result),
                'semantic_context': self._extract_method_summary(buggy_comparison.semantic_context_result) if buggy_comparison.semantic_context_result else None,
                'masked_token_replacement': self._extract_method_summary(buggy_comparison.masked_token_replacement_result) if buggy_comparison.masked_token_replacement_result else None,
                'consensus_anomalies': buggy_comparison.consensus_anomalies,
                'best_method': buggy_comparison.best_method,
                'agreement_matrix': buggy_comparison.method_agreement_matrix.tolist()
            },

            # Correct code results
            'correct': {
                'lecprompt': self._extract_method_summary(correct_comparison.lecprompt_result),
                'semantic_energy': self._extract_method_summary(correct_comparison.semantic_energy_result),
                'conformal': self._extract_method_summary(correct_comparison.conformal_result),
                'attention': self._extract_method_summary(correct_comparison.attention_result),
                'semantic_context': self._extract_method_summary(correct_comparison.semantic_context_result) if correct_comparison.semantic_context_result else None,
                'masked_token_replacement': self._extract_method_summary(correct_comparison.masked_token_replacement_result) if correct_comparison.masked_token_replacement_result else None,
                'consensus_anomalies': correct_comparison.consensus_anomalies,
                'best_method': correct_comparison.best_method,
                'agreement_matrix': correct_comparison.method_agreement_matrix.tolist()
            },

            # Hypothesis confirmation per method
            'hypothesis_confirmation': hypothesis_results,

            # Inter-method agreement
            'agreement_metrics': agreement_metrics,

            # Execution times
            'execution_times': {
                'buggy': buggy_comparison.execution_times,
                'correct': correct_comparison.execution_times
            }
        }

        return result

    def run_full_comparison(self,
                           output_dir: str = "advanced_methods_comparison",
                           examples: Optional[List[str]] = None,
                           skip_calibration: bool = False) -> Dict[str, Any]:
        """
        Run comparison on all (or selected) test examples.

        Args:
            output_dir: Directory to save results
            examples: List of example names to test (None = all from test set)
            skip_calibration: If True, skip calibration phase (use for debugging)

        Returns:
            Complete comparison results
        """
        os.makedirs(output_dir, exist_ok=True)

        print("="*80)
        print("ADVANCED METHODS COMPARISON")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Sensitivity factor (k): {self.k}")
        print(f"  Conformal alpha: {self.conformal.alpha}")
        print(f"  Output directory: {output_dir}")

        # Phase 1: Calibration (unless skipped)
        # Use dynamic calibration that excludes requested test examples
        calibration_results = None
        if not skip_calibration and not self.calibration_performed:
            calibration_results = self._run_calibration_phase(exclude_examples=examples)
        elif self.calibration_performed:
            print("\n✓ Conformal prediction already calibrated")
            calibration_results = {
                'calibration_info': self.dataset.get_calibration_info(),
                'conformal_metadata': self.conformal.get_calibration_info()
            }
        else:
            print("\n⚠ Skipping calibration phase (skip_calibration=True)")

        # Phase 2: Get test examples
        # Use dynamic test set that includes requested examples
        test_examples = self.dataset.get_test_set(requested_examples=examples)

        print(f"\n" + "="*80)
        print("TESTING PHASE")
        print("="*80)
        print(f"\nTesting {len(test_examples)} examples (from test set):")
        for ex in test_examples:
            print(f"  - {ex.name} ({ex.bug_type})")

        # Run comparison on each example
        individual_results = []
        for i, example in enumerate(test_examples, 1):
            print(f"\n[{i}/{len(test_examples)}] {example.name}")
            try:
                result = self.run_comparison_on_example(example)
                individual_results.append(result)

                # Save intermediate result
                intermediate_file = os.path.join(
                    output_dir,
                    f"result_{example.name}.json"
                )
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                individual_results.append({
                    'example_name': example.name,
                    'error': str(e)
                })

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(individual_results)

        # Compile complete results
        complete_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_examples': len(test_examples),
                'num_calibration_examples': len(self.dataset.get_calibration_set()),
                'num_test_examples': len(self.dataset.get_test_set()),
                'sensitivity_factor': self.k,
                'conformal_alpha': self.conformal.alpha,
                'model_name': getattr(self.model, 'name_or_path', 'unknown'),
                'calibration_performed': self.calibration_performed
            },
            'calibration_results': calibration_results,
            'individual_results': individual_results,
            'aggregate_statistics': aggregate_stats,
            'method_ranking': self._rank_methods(aggregate_stats)
        }

        # Save complete results
        complete_file = os.path.join(output_dir, "complete_comparison_results.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - complete_comparison_results.json")
        print(f"  - result_<example>.json (×{len(individual_results)})")

        return complete_results

    def _extract_method_summary(self, method_result: Dict) -> Dict[str, Any]:
        """
        Extract summary statistics from method result.

        NEW: Also preserves token_analyses and code for visualization.
        """
        if 'error' in method_result:
            return {
                'error': method_result['error'],
                'num_anomalies': 0,
                'anomaly_rate': 0.0
            }

        # FIX: Handle different result structures (LecPrompt vs Advanced Methods)
        # LecPrompt returns: {'statistics': {'total_tokens': X, 'anomalous_tokens': Y}}
        # Advanced methods return: {'num_tokens': X, 'num_anomalies': Y}

        if 'statistics' in method_result:
            # LecPrompt/BaseDetector format
            stats = method_result['statistics']
            summary = {
                'num_tokens': stats.get('total_tokens', 0),
                'num_anomalies': stats.get('anomalous_tokens', 0)
            }
        else:
            # Advanced methods format
            summary = {
                'num_tokens': method_result.get('num_tokens', 0),
                'num_anomalies': method_result.get('num_anomalies', 0)
            }

        if summary['num_tokens'] > 0:
            summary['anomaly_rate'] = summary['num_anomalies'] / summary['num_tokens']
        else:
            summary['anomaly_rate'] = 0.0

        # Add method-specific metrics
        if 'statistics' in method_result:
            stats = method_result['statistics']
            summary['statistics'] = {
                k: v for k, v in stats.items()
                if isinstance(v, (int, float, bool, str))
            }

        # NEW: Preserve token_analyses for visualization
        if 'token_analyses' in method_result:
            # Convert TokenAnalysis objects to dicts for JSON serialization
            token_analyses = method_result['token_analyses']
            if token_analyses and hasattr(token_analyses[0], '__dict__'):
                # TokenAnalysis dataclass objects - convert to dicts
                summary['token_analyses'] = [vars(t) for t in token_analyses]
            else:
                # Already dicts
                summary['token_analyses'] = token_analyses

        # NEW: Preserve code for visualization
        if 'code' in method_result:
            summary['code'] = method_result['code']

        return summary

    def _compute_hypothesis_confirmations(self,
                                         buggy_comparison: MethodComparisonResult,
                                         correct_comparison: MethodComparisonResult) -> Dict[str, bool]:
        """
        Check if each method confirms the hypothesis:
        "Buggy code has more anomalies than correct code"
        """
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']
        confirmations = {}

        for method in methods:
            buggy_anom = getattr(buggy_comparison, f'{method}_result', {}).get('num_anomalies', 0)
            correct_anom = getattr(correct_comparison, f'{method}_result', {}).get('num_anomalies', 0)

            confirmations[method] = buggy_anom > correct_anom

        return confirmations

    def _compute_inter_method_agreement(self,
                                       buggy_comparison: MethodComparisonResult,
                                       correct_comparison: MethodComparisonResult) -> Dict[str, Any]:
        """Compute agreement metrics between methods."""
        buggy_matrix = buggy_comparison.method_agreement_matrix
        correct_matrix = correct_comparison.method_agreement_matrix

        # Average agreement across buggy and correct
        avg_matrix = (buggy_matrix + correct_matrix) / 2

        # Compute overall agreement score (off-diagonal mean)
        n = len(avg_matrix)
        off_diag_sum = 0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_sum += avg_matrix[i][j]
                    count += 1

        overall_agreement = off_diag_sum / count if count > 0 else 0.0

        return {
            'buggy_agreement_matrix': buggy_matrix.tolist(),
            'correct_agreement_matrix': correct_matrix.tolist(),
            'average_agreement_matrix': avg_matrix.tolist(),
            'overall_agreement_score': float(overall_agreement)
        }

    def _compute_aggregate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across all examples."""
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']
        stats = {}

        for method in methods:
            # Confirmation rate
            confirmations = [
                r['hypothesis_confirmation'][method]
                for r in results
                if 'hypothesis_confirmation' in r and method in r['hypothesis_confirmation']
            ]
            confirmation_rate = sum(confirmations) / len(confirmations) if confirmations else 0.0

            # Average anomaly counts
            buggy_counts = [
                r['buggy'][method]['num_anomalies']
                for r in results
                if 'buggy' in r and method in r['buggy']
            ]
            correct_counts = [
                r['correct'][method]['num_anomalies']
                for r in results
                if 'correct' in r and method in r['correct']
            ]

            # Average execution times
            exec_times = []
            for r in results:
                if 'execution_times' in r:
                    buggy_time = r['execution_times'].get('buggy', {}).get(method, 0)
                    correct_time = r['execution_times'].get('correct', {}).get(method, 0)
                    exec_times.append((buggy_time + correct_time) / 2)

            stats[method] = {
                'confirmation_rate': confirmation_rate,
                'avg_buggy_anomalies': np.mean(buggy_counts) if buggy_counts else 0.0,
                'avg_correct_anomalies': np.mean(correct_counts) if correct_counts else 0.0,
                'avg_execution_time': np.mean(exec_times) if exec_times else 0.0,
                'std_execution_time': np.std(exec_times) if exec_times else 0.0
            }

        # Compute inter-method agreement
        all_agreements = [
            r['agreement_metrics']['overall_agreement_score']
            for r in results
            if 'agreement_metrics' in r
        ]
        stats['overall_inter_method_agreement'] = float(np.mean(all_agreements)) if all_agreements else 0.0

        return stats

    def _rank_methods(self, aggregate_stats: Dict) -> List[Dict[str, Any]]:
        """
        Rank methods based on multiple criteria.

        Ranking factors:
        1. Confirmation rate (40%)
        2. Anomaly differential (buggy - correct) (30%)
        3. Execution speed (20%)
        4. Agreement with others (10%)
        """
        methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']
        scores = {}

        # Get max/min values for normalization
        max_time = max([aggregate_stats[m]['avg_execution_time'] for m in methods])
        min_time = min([aggregate_stats[m]['avg_execution_time'] for m in methods])

        for method in methods:
            stats = aggregate_stats[method]

            # 1. Confirmation rate (higher is better)
            conf_score = stats['confirmation_rate']

            # 2. Anomaly differential (higher differential is better)
            diff = stats['avg_buggy_anomalies'] - stats['avg_correct_anomalies']
            max_diff = max([
                aggregate_stats[m]['avg_buggy_anomalies'] - aggregate_stats[m]['avg_correct_anomalies']
                for m in methods
            ])
            diff_score = diff / max_diff if max_diff > 0 else 0.0

            # 3. Speed (lower time is better, invert)
            if max_time > min_time:
                speed_score = 1.0 - (stats['avg_execution_time'] - min_time) / (max_time - min_time)
            else:
                speed_score = 1.0

            # 4. Agreement score (use global agreement)
            agreement_score = aggregate_stats.get('overall_inter_method_agreement', 0.5)

            # Weighted sum
            total_score = (
                0.40 * conf_score +
                0.30 * diff_score +
                0.20 * speed_score +
                0.10 * agreement_score
            )

            scores[method] = {
                'method': method,
                'total_score': float(total_score),
                'confirmation_rate': float(conf_score),
                'anomaly_differential': float(diff),
                'avg_execution_time': float(stats['avg_execution_time']),
                'components': {
                    'confirmation_score': float(conf_score),
                    'differential_score': float(diff_score),
                    'speed_score': float(speed_score),
                    'agreement_score': float(agreement_score)
                }
            }

        # Sort by total score (descending)
        ranked = sorted(scores.values(), key=lambda x: x['total_score'], reverse=True)

        return ranked


if __name__ == "__main__":
    print("Advanced Methods Comparison Runner")
    print("\nThis module compares 4 error detection methods:")
    print("  1. LecPrompt (baseline)")
    print("  2. Semantic Energy")
    print("  3. Conformal Prediction")
    print("  4. Attention Anomaly")
    print("\nUsage:")
    print("  from comparison.advanced_comparison_runner import AdvancedMethodsComparisonRunner")
    print("  runner = AdvancedMethodsComparisonRunner(model, tokenizer, baseline_detector)")
    print("  results = runner.run_full_comparison()")
