#!/usr/bin/env python3
"""
Statistical Analyzer for Multi-Model Comparison

Performs statistical tests and analysis on benchmark results.
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats
from scipy.stats import mcnemar, friedmanchisquare


class StatisticalAnalyzer:
    """Perform statistical analysis on model comparison results."""

    def analyze(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform complete statistical analysis.

        Args:
            model_results: Results from benchmark_runner

        Returns:
            Dictionary with statistical analysis results
        """
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)

        analysis = {
            'confirmation_rates': self._extract_confirmation_rates(model_results),
            'pairwise_comparisons': self._pairwise_mcnemar_tests(model_results),
            'overall_friedman_test': self._friedman_test(model_results),
            'effect_sizes': self._compute_effect_sizes(model_results),
            'bug_type_analysis': self._analyze_by_bug_type(model_results),
            'ranking': self._rank_models(model_results)
        }

        self._print_summary(analysis)

        return analysis

    def _extract_confirmation_rates(self, model_results: Dict) -> Dict[str, float]:
        """Extract confirmation rates for each model."""
        rates = {}
        for model_key, results in model_results.items():
            if 'error' in results:
                rates[model_key] = 0.0
            else:
                rates[model_key] = results['aggregate_stats']['confirmation_rate']
        return rates

    def _pairwise_mcnemar_tests(self, model_results: Dict) -> Dict[str, Any]:
        """Perform pairwise McNemar tests between models."""
        print("\nPerforming pairwise McNemar's tests...")

        models = [k for k in model_results.keys() if 'error' not in model_results[k]]
        comparisons = {}

        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Create contingency table
                table = self._create_contingency_table(
                    model_results[model1]['individual_results'],
                    model_results[model2]['individual_results']
                )

                # Perform McNemar's test
                try:
                    result = mcnemar(table, exact=True)
                    comparisons[f"{model1}_vs_{model2}"] = {
                        'statistic': float(result.statistic),
                        'p_value': float(result.pvalue),
                        'significant': result.pvalue < 0.05,
                        'significance_level': 'p<0.05' if result.pvalue < 0.05 else 'ns'
                    }
                except Exception as e:
                    comparisons[f"{model1}_vs_{model2}"] = {'error': str(e)}

        return comparisons

    def _create_contingency_table(self, results1: List, results2: List) -> np.ndarray:
        """Create 2x2 contingency table for McNemar's test."""
        # both_correct, only1_correct, only2_correct, both_wrong
        both_correct = 0
        only1_correct = 0
        only2_correct = 0
        both_wrong = 0

        for r1, r2 in zip(results1, results2):
            if 'error' in r1 or 'error' in r2:
                continue

            confirmed1 = r1.get('hypothesis_confirmed', False)
            confirmed2 = r2.get('hypothesis_confirmed', False)

            if confirmed1 and confirmed2:
                both_correct += 1
            elif confirmed1 and not confirmed2:
                only1_correct += 1
            elif not confirmed1 and confirmed2:
                only2_correct += 1
            else:
                both_wrong += 1

        # McNemar's test uses: [[both_correct, only1_correct], [only2_correct, both_wrong]]
        return np.array([[both_correct, only1_correct], [only2_correct, both_wrong]])

    def _friedman_test(self, model_results: Dict) -> Dict[str, Any]:
        """Perform Friedman test for overall ranking."""
        print("\nPerforming Friedman test...")

        models = [k for k in model_results.keys() if 'error' not in model_results[k]]

        # Get confirmation results for each model on each example
        model_scores = []
        for model_key in models:
            scores = [1 if r.get('hypothesis_confirmed', False) else 0
                     for r in model_results[model_key]['individual_results']
                     if 'error' not in r]
            model_scores.append(scores)

        try:
            statistic, p_value = friedmanchisquare(*model_scores)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Significant difference exists between models' if p_value < 0.05
                                 else 'No significant difference between models'
            }
        except Exception as e:
            return {'error': str(e)}

    def _compute_effect_sizes(self, model_results: Dict) -> Dict[str, float]:
        """Compute Cohen's d effect sizes between models."""
        models = [k for k in model_results.keys() if 'error' not in model_results[k]]
        effect_sizes = {}

        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                scores1 = [1 if r.get('hypothesis_confirmed', False) else 0
                          for r in model_results[model1]['individual_results']
                          if 'error' not in r]
                scores2 = [1 if r.get('hypothesis_confirmed', False) else 0
                          for r in model_results[model2]['individual_results']
                          if 'error' not in r]

                # Cohen's d
                mean1, mean2 = np.mean(scores1), np.mean(scores2)
                std_pooled = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                d = (mean1 - mean2) / std_pooled if std_pooled > 0 else 0.0

                effect_sizes[f"{model1}_vs_{model2}"] = float(d)

        return effect_sizes

    def _analyze_by_bug_type(self, model_results: Dict) -> Dict[str, Any]:
        """Analyze performance by bug type."""
        bug_type_performance = {}

        for model_key, results in model_results.items():
            if 'error' in results:
                continue

            bug_type_stats = results['aggregate_stats'].get('bug_type_breakdown', {})
            for bug_type, stats in bug_type_stats.items():
                if bug_type not in bug_type_performance:
                    bug_type_performance[bug_type] = {}
                bug_type_performance[bug_type][model_key] = stats['confirmation_rate']

        return bug_type_performance

    def _rank_models(self, model_results: Dict) -> List[str]:
        """Rank models by confirmation rate."""
        rates = self._extract_confirmation_rates(model_results)
        ranked = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in ranked]

    def _print_summary(self, analysis: Dict):
        """Print summary of statistical analysis."""
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)

        print("\nModel Ranking:")
        for i, model in enumerate(analysis['ranking'], 1):
            rate = analysis['confirmation_rates'][model]
            print(f"  {i}. {model:20s} - {rate:.1%}")

        if 'overall_friedman_test' in analysis and 'p_value' in analysis['overall_friedman_test']:
            friedman = analysis['overall_friedman_test']
            print(f"\nFriedman Test: χ²={friedman['statistic']:.3f}, p={friedman['p_value']:.4f}")
            print(f"  {friedman['interpretation']}")

        print("\nPairwise Comparisons (McNemar):")
        sig_count = sum(1 for v in analysis['pairwise_comparisons'].values()
                       if isinstance(v, dict) and v.get('significant', False))
        print(f"  Significant differences: {sig_count}/{len(analysis['pairwise_comparisons'])}")
