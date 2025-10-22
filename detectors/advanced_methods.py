#!/usr/bin/env python3
"""
Advanced Error Detection Methods

Implements three state-of-the-art techniques for code error detection:
1. Semantic Energy - Uses logits instead of probabilities (2024)
2. Conformal Prediction - Statistical guarantees for uncertainty (2024)
3. Attention Anomaly - Analyzes attention patterns for anomalies (2024)

These methods complement the baseline LecPrompt log-probability approach.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import TokenAnalysis for visualization support
from LLM import TokenAnalysis


@dataclass
class AdvancedTokenMetrics:
    """
    Comprehensive metrics for a single token from all advanced methods.
    """
    token: str
    token_id: int
    position: int
    line_number: int

    # Baseline (LecPrompt)
    log_probability: float
    lecprompt_anomaly: bool
    lecprompt_deviation: float

    # Semantic Energy
    semantic_energy: float
    energy_anomaly: bool
    energy_score: float  # Normalized 0-1

    # Conformal Prediction
    prediction_set_size: int
    conformal_uncertainty: float  # 0-1
    conformal_anomaly: bool

    # Attention Anomaly
    attention_entropy: float
    attention_anomaly_score: float  # 0-1
    attention_anomaly: bool

    # Combined metrics
    combined_anomaly_score: float  # Weighted average of all methods
    is_highly_suspicious: bool  # True if multiple methods agree
    agreement_count: int  # How many methods flag as anomalous


@dataclass
class MethodComparisonResult:
    """
    Comparison results between different detection methods.
    """
    example_name: str
    buggy_code: str
    correct_code: str

    # Per-method results
    lecprompt_result: Dict[str, Any]
    semantic_energy_result: Dict[str, Any]
    conformal_result: Dict[str, Any]
    attention_result: Dict[str, Any]
    semantic_context_result: Optional[Dict[str, Any]] = None  # 5th method (optional)
    masked_token_replacement_result: Optional[Dict[str, Any]] = None  # 6th method (optional)

    # Agreement metrics
    method_agreement_matrix: np.ndarray = None  # Up to 6x6 correlation matrix
    token_overlap: Dict[str, int] = None  # Overlap between method pairs

    # Overall assessment
    best_method: str = ""  # Method with best performance on this example
    consensus_anomalies: List[int] = None  # Token positions flagged by majority
    execution_times: Dict[str, float] = None


class SemanticEnergyDetector:
    """
    Semantic Energy-based error detection.

    Based on: "Semantic Energy: Detecting LLM Hallucination Beyond Entropy" (2024)

    Key idea: Use logits (pre-softmax) instead of probabilities to better capture
    model uncertainty. Lower energy = higher confidence.

    Formula: U(x) = -(1/nT) * Σ z_θ(x_t)
    where z_θ are the logits for chosen tokens.
    """

    def __init__(self, sensitivity_factor: float = 1.5):
        """
        Initialize semantic energy detector.

        Args:
            sensitivity_factor: k for threshold (τ = μ + k×σ)
        """
        self.k = sensitivity_factor

    def compute_semantic_energy(self,
                               logits: torch.Tensor,
                               token_ids: torch.Tensor) -> List[float]:
        """
        Compute semantic energy for each token from logits.

        Args:
            logits: Model logits [seq_len, vocab_size]
            token_ids: Chosen token IDs [seq_len]

        Returns:
            List of energy values (one per token)
        """
        energies = []

        for i in range(len(token_ids)):
            if i >= len(logits):
                break

            token_id = token_ids[i].item()

            # Energy = negative logit of chosen token
            # Lower energy = higher confidence = less likely to be error
            energy = -logits[i][token_id].item()
            energies.append(energy)

        return energies

    def detect_anomalies(self,
                        energies: List[float],
                        k: Optional[float] = None) -> Tuple[List[bool], Dict[str, float]]:
        """
        Detect anomalous tokens based on energy threshold.

        High energy = low confidence = potential error.

        Args:
            energies: List of energy values
            k: Sensitivity factor (uses self.k if None)

        Returns:
            Tuple of (anomaly_flags, statistics_dict)
        """
        if k is None:
            k = self.k

        energies_array = np.array(energies)

        # Calculate statistics
        mean_energy = np.mean(energies_array)
        std_energy = np.std(energies_array)

        # Threshold: high energy is anomalous
        threshold = mean_energy + k * std_energy

        # Flag anomalies
        anomalies = [e > threshold for e in energies]

        # Calculate normalized scores (0-1, higher = more anomalous)
        if std_energy > 0:
            scores = [(e - mean_energy) / (3 * std_energy) for e in energies]
            scores = [min(1.0, max(0.0, s + 0.5)) for s in scores]
        else:
            scores = [0.5] * len(energies)

        stats = {
            'mean_energy': float(mean_energy),
            'std_energy': float(std_energy),
            'threshold': float(threshold),
            'num_anomalies': sum(anomalies),
            'anomaly_rate': sum(anomalies) / len(anomalies) if anomalies else 0.0,
            'scores': scores
        }

        return anomalies, stats

    def analyze_code(self,
                    code: str,
                    model,
                    tokenizer,
                    baseline_log_probs: List[float] = None) -> Dict[str, Any]:
        """
        Complete analysis of code using semantic energy.

        Args:
            code: Source code to analyze
            model: Language model
            tokenizer: Tokenizer
            baseline_log_probs: Optional log-probs from baseline for comparison

        Returns:
            Dictionary with analysis results including TokenAnalysis objects
        """
        # Tokenize
        encoding = tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(model.device)

        # Get logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Compute energies
        energies = self.compute_semantic_energy(logits, input_ids[0][1:])

        # Detect anomalies
        anomalies, stats = self.detect_anomalies(energies)

        # Map to tokens
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0][1:]]

        # NEW: Create TokenAnalysis objects for visualization
        token_analyses = []
        for i in range(len(tokens)):
            if i >= len(logits):
                break

            token_id = input_ids[0][i + 1].item()
            token_logits = logits[i]

            # Compute probabilities
            probs = F.softmax(token_logits, dim=-1)
            token_prob = probs[token_id].item()
            token_logit = token_logits[token_id].item()

            # Compute rank
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Compute surprisal
            surprisal = -np.log2(token_prob + 1e-10)

            # Compute perplexity
            perplexity = 2 ** entropy

            # Get top-10 probabilities
            top_k_probs = [(sorted_indices[j].item(), sorted_probs[j].item())
                          for j in range(min(10, len(sorted_probs)))]

            # Create TokenAnalysis object
            analysis = TokenAnalysis(
                token=tokens[i],
                token_id=token_id,
                position=i,
                probability=token_prob,
                logit=token_logit,
                rank=rank,
                perplexity=perplexity,
                entropy=entropy,
                surprisal=surprisal,
                top_k_probs=top_k_probs,
                max_probability=sorted_probs[0].item(),
                probability_margin=sorted_probs[0].item() - sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0,
                shannon_entropy=entropy,
                local_perplexity=perplexity,
                sequence_improbability=0.0,  # Not computed in this method
                confidence_score=token_prob,
                semantic_energy=energies[i],  # ← Key metric for this method
                is_anomalous=anomalies[i] if i < len(anomalies) else False
            )
            token_analyses.append(analysis)

        result = {
            'method': 'semantic_energy',
            'energies': energies,
            'anomalies': anomalies,
            'tokens': tokens,
            'statistics': stats,
            'num_tokens': len(tokens),
            'num_anomalies': sum(anomalies),
            'token_analyses': token_analyses,  # ← NEW: For visualization
            'code': code  # ← NEW: For visualization
        }

        # Compare with baseline if provided
        if baseline_log_probs is not None:
            result['correlation_with_baseline'] = float(
                np.corrcoef(energies[:len(baseline_log_probs)],
                           baseline_log_probs[:len(energies)])[0, 1]
            )

        return result


class ConformalPredictionDetector:
    """
    Conformal Prediction-based uncertainty quantification.

    Based on: "API Is Enough: Conformal Prediction for LLMs" (2024)

    Key idea: Create prediction sets with statistical coverage guarantees.
    Larger prediction set = higher uncertainty = potential error.

    Provides formal guarantees: P(true_token in prediction_set) >= 1 - alpha
    """

    def __init__(self,
                 alpha: float = 0.1,
                 sensitivity_factor: float = 1.5):
        """
        Initialize conformal prediction detector.

        Args:
            alpha: Significance level (default 0.1 = 90% coverage)
            sensitivity_factor: k for anomaly threshold
        """
        self.alpha = alpha
        self.k = sensitivity_factor
        self.calibration_scores = None
        self.quantile_threshold = None
        self.calibration_metadata = None  # Store calibration info

    def calibrate(self, calibration_data: List[Tuple[torch.Tensor, torch.Tensor]],
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Calibrate the conformal predictor on a calibration set.

        This method computes the quantile threshold from calibration data to provide
        formal statistical coverage guarantees: P(true_token in prediction_set) >= 1 - alpha

        Args:
            calibration_data: List of (logits, true_token_ids) pairs from calibration examples
            metadata: Optional metadata about calibration (example names, etc.)
        """
        scores = []

        for logits, token_ids in calibration_data:
            # Compute conformal scores (inverse probability)
            probs = F.softmax(logits, dim=-1)
            for i, tid in enumerate(token_ids):
                if i < len(probs):
                    score = 1.0 - probs[i][tid].item()
                    scores.append(score)

        self.calibration_scores = np.array(scores)

        # Compute quantile threshold following Shafer & Vovk (2008)
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile_threshold = np.quantile(self.calibration_scores, q)

        # Store calibration metadata
        self.calibration_metadata = {
            'num_calibration_tokens': n,
            'num_calibration_examples': len(calibration_data),
            'alpha': self.alpha,
            'coverage_target': f"{(1-self.alpha)*100:.1f}%",
            'quantile': float(q),
            'quantile_threshold': float(self.quantile_threshold),
            'mean_score': float(np.mean(self.calibration_scores)),
            'std_score': float(np.std(self.calibration_scores)),
            'min_score': float(np.min(self.calibration_scores)),
            'max_score': float(np.max(self.calibration_scores))
        }

        if metadata:
            self.calibration_metadata.update(metadata)

        print(f"✓ Conformal calibration completed:")
        print(f"  - Calibration tokens: {n}")
        print(f"  - Quantile threshold (q={q:.3f}): {self.quantile_threshold:.4f}")
        print(f"  - Coverage guarantee: {(1-self.alpha)*100:.1f}%")

    def is_calibrated(self) -> bool:
        """
        Check if the detector has been calibrated.

        Returns:
            True if calibration has been performed, False otherwise
        """
        return self.quantile_threshold is not None

    def get_calibration_info(self) -> Dict[str, Any]:
        """
        Get information about the calibration.

        Returns:
            Dictionary with calibration metadata, or warning if not calibrated
        """
        if not self.is_calibrated():
            return {
                'calibrated': False,
                'warning': 'Detector has not been calibrated. Using default threshold.'
            }

        return {
            'calibrated': True,
            **self.calibration_metadata
        }

    def compute_prediction_sets(self,
                               logits: torch.Tensor,
                               return_sizes_only: bool = True) -> List[int]:
        """
        Compute prediction set sizes for each position.

        Larger set = higher uncertainty.

        Args:
            logits: Model logits [seq_len, vocab_size]
            return_sizes_only: If True, return only set sizes (not full sets)

        Returns:
            List of prediction set sizes
        """
        # FIX: Warn if not calibrated instead of silently using default
        if self.quantile_threshold is None:
            import warnings
            warnings.warn(
                "⚠️  Conformal predictor has NOT been calibrated! "
                "Using default threshold 0.9 WITHOUT coverage guarantees. "
                "Call calibrate() first for formal statistical guarantees.",
                UserWarning,
                stacklevel=2
            )
            self.quantile_threshold = 0.9  # Default fallback

        probs = F.softmax(logits, dim=-1)
        set_sizes = []

        for i in range(len(probs)):
            # Include tokens with score <= threshold
            scores = 1.0 - probs[i]
            prediction_set_mask = scores <= self.quantile_threshold
            set_size = prediction_set_mask.sum().item()
            set_sizes.append(set_size)

        return set_sizes

    def detect_anomalies(self,
                        set_sizes: List[int],
                        vocab_size: int = 50000) -> Tuple[List[bool], Dict[str, float]]:
        """
        Detect anomalies based on prediction set sizes.

        Large prediction set = high uncertainty = potential error.

        Args:
            set_sizes: Prediction set sizes
            vocab_size: Vocabulary size for normalization

        Returns:
            Tuple of (anomaly_flags, statistics_dict)
        """
        sizes_array = np.array(set_sizes)

        # Normalize to [0, 1]
        uncertainties = sizes_array / vocab_size

        # Calculate statistics
        mean_uncertainty = np.mean(uncertainties)
        std_uncertainty = np.std(uncertainties)

        # Threshold: high uncertainty is anomalous
        threshold = mean_uncertainty + self.k * std_uncertainty

        # Flag anomalies
        anomalies = [u > threshold for u in uncertainties]

        stats = {
            'mean_set_size': float(np.mean(sizes_array)),
            'std_set_size': float(np.std(sizes_array)),
            'mean_uncertainty': float(mean_uncertainty),
            'threshold': float(threshold),
            'num_anomalies': sum(anomalies),
            'anomaly_rate': sum(anomalies) / len(anomalies) if anomalies else 0.0,
            'uncertainties': uncertainties.tolist()
        }

        return anomalies, stats

    def analyze_code(self,
                    code: str,
                    model,
                    tokenizer) -> Dict[str, Any]:
        """
        Complete analysis using conformal prediction.

        Args:
            code: Source code to analyze
            model: Language model
            tokenizer: Tokenizer

        Returns:
            Dictionary with analysis results including TokenAnalysis objects
        """
        # Tokenize
        encoding = tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(model.device)

        # Get logits
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Compute prediction sets
        set_sizes = self.compute_prediction_sets(logits)

        # Detect anomalies
        vocab_size = logits.shape[-1]
        anomalies, stats = self.detect_anomalies(set_sizes, vocab_size)

        # Map to tokens
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0][1:]]

        # NEW: Create TokenAnalysis objects for visualization
        token_analyses = []
        for i in range(len(tokens)):
            if i >= len(logits):
                break

            token_id = input_ids[0][i + 1].item()
            token_logits = logits[i]

            # Compute probabilities
            probs = F.softmax(token_logits, dim=-1)
            token_prob = probs[token_id].item()
            token_logit = token_logits[token_id].item()

            # Compute conformal score (1 - P(token))
            conformal_score = 1.0 - token_prob

            # Compute rank
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Compute surprisal
            surprisal = -np.log2(token_prob + 1e-10)

            # Compute perplexity
            perplexity = 2 ** entropy

            # Get top-10 probabilities
            top_k_probs = [(sorted_indices[j].item(), sorted_probs[j].item())
                          for j in range(min(10, len(sorted_probs)))]

            # Get uncertainty from stats
            uncertainty = stats['uncertainties'][i] if i < len(stats['uncertainties']) else 0.0

            # Create TokenAnalysis object
            analysis = TokenAnalysis(
                token=tokens[i],
                token_id=token_id,
                position=i,
                probability=token_prob,
                logit=token_logit,
                rank=rank,
                perplexity=perplexity,
                entropy=entropy,
                surprisal=surprisal,
                top_k_probs=top_k_probs,
                max_probability=sorted_probs[0].item(),
                probability_margin=sorted_probs[0].item() - sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0,
                shannon_entropy=entropy,
                local_perplexity=perplexity,
                sequence_improbability=0.0,  # Not computed in this method
                confidence_score=token_prob,
                conformal_score=conformal_score,  # ← Key metric for this method
                is_anomalous=anomalies[i] if i < len(anomalies) else False
            )
            token_analyses.append(analysis)

        result = {
            'method': 'conformal_prediction',
            'prediction_set_sizes': set_sizes,
            'anomalies': anomalies,
            'tokens': tokens,
            'statistics': stats,
            'num_tokens': len(tokens),
            'num_anomalies': sum(anomalies),
            'coverage_guarantee': f"{(1-self.alpha)*100:.0f}%",
            'calibration_info': self.get_calibration_info(),  # Include calibration metadata
            'token_analyses': token_analyses,  # ← NEW: For visualization
            'code': code  # ← NEW: For visualization
        }

        return result


class AttentionAnomalyDetector:
    """
    Attention Pattern-based anomaly detection.

    Based on: ACL 2024 research on attention analysis in code models

    Key idea: Analyze attention distribution patterns. Uniform attention
    (high entropy) indicates uncertainty. Focused attention (low entropy)
    indicates confidence.

    Uses attention weights that are already computed by transformer models.
    """

    def __init__(self, sensitivity_factor: float = 1.5):
        """
        Initialize attention anomaly detector.

        Args:
            sensitivity_factor: k for threshold
        """
        self.k = sensitivity_factor

    def compute_attention_entropy(self,
                                 attention_weights: torch.Tensor) -> List[float]:
        """
        Compute entropy of attention distribution for each token.

        High entropy = attention is spread out = uncertainty.
        Low entropy = attention is focused = confidence.

        Args:
            attention_weights: [num_layers, num_heads, seq_len, seq_len]

        Returns:
            List of entropy values (one per token)
        """
        # Average across layers and heads
        # Shape: [seq_len, seq_len]
        avg_attention = attention_weights.mean(dim=[0, 1])

        entropies = []
        for i in range(avg_attention.shape[0]):
            # Attention distribution for token i
            attn_dist = avg_attention[i]

            # Compute Shannon entropy
            # Add small epsilon to avoid log(0)
            entropy = -torch.sum(
                attn_dist * torch.log(attn_dist + 1e-10)
            ).item()

            entropies.append(entropy)

        return entropies

    def compute_attention_anomaly_score(self,
                                       attention_weights: torch.Tensor) -> List[float]:
        """
        Compute comprehensive anomaly score from attention patterns.

        Considers:
        1. Entropy (uniformity of attention)
        2. Self-attention strength (how much token attends to itself)
        3. Attention spread (variance of attention distribution)

        Args:
            attention_weights: [num_layers, num_heads, seq_len, seq_len]

        Returns:
            List of anomaly scores (0-1, higher = more anomalous)
        """
        avg_attention = attention_weights.mean(dim=[0, 1])
        scores = []

        for i in range(avg_attention.shape[0]):
            attn_dist = avg_attention[i]

            # 1. Entropy score (normalized)
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-10))
            max_entropy = np.log(len(attn_dist))
            entropy_score = (entropy / max_entropy).item()

            # 2. Self-attention score (inverted - low self-attention is suspicious)
            self_attn = attn_dist[i].item()
            self_attn_score = 1.0 - self_attn

            # 3. Spread score (variance)
            variance = torch.var(attn_dist).item()
            spread_score = min(1.0, variance * 10)  # Scale to [0, 1]

            # Combined score (weighted average)
            combined_score = (
                0.5 * entropy_score +
                0.3 * self_attn_score +
                0.2 * spread_score
            )

            scores.append(combined_score)

        return scores

    def detect_anomalies(self,
                        attention_entropies: List[float],
                        anomaly_scores: List[float]) -> Tuple[List[bool], Dict[str, float]]:
        """
        Detect anomalies based on attention patterns.

        Args:
            attention_entropies: Entropy values
            anomaly_scores: Anomaly scores (0-1)

        Returns:
            Tuple of (anomaly_flags, statistics_dict)
        """
        entropies_array = np.array(attention_entropies)
        scores_array = np.array(anomaly_scores)

        # Calculate statistics
        mean_entropy = np.mean(entropies_array)
        std_entropy = np.std(entropies_array)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)

        # Threshold based on entropy (high entropy = anomalous)
        entropy_threshold = mean_entropy + self.k * std_entropy

        # Flag anomalies (using both entropy and score)
        anomalies = [
            (e > entropy_threshold) or (s > (mean_score + self.k * std_score))
            for e, s in zip(attention_entropies, anomaly_scores)
        ]

        stats = {
            'mean_entropy': float(mean_entropy),
            'std_entropy': float(std_entropy),
            'entropy_threshold': float(entropy_threshold),
            'mean_anomaly_score': float(mean_score),
            'num_anomalies': sum(anomalies),
            'anomaly_rate': sum(anomalies) / len(anomalies) if anomalies else 0.0
        }

        return anomalies, stats

    def analyze_code(self,
                    code: str,
                    model,
                    tokenizer) -> Dict[str, Any]:
        """
        Complete analysis using attention patterns.

        Args:
            code: Source code to analyze
            model: Language model (must support output_attentions=True)
            tokenizer: Tokenizer

        Returns:
            Dictionary with analysis results including TokenAnalysis objects
        """
        # Tokenize
        encoding = tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(model.device)

        # Get attention weights and logits
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                return {
                    'method': 'attention_anomaly',
                    'error': 'Model does not support attention output',
                    'num_tokens': 0,
                    'num_anomalies': 0
                }

            # FIX: Stack attention weights with robust dimension handling
            # Expected shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
            attention_weights = torch.stack(outputs.attentions)

            # Robust batch dimension removal
            if attention_weights.dim() == 5:
                # Has batch dimension at position 1
                attention_weights = attention_weights.squeeze(1)
            elif attention_weights.dim() == 4:
                # Already has shape [layers, heads, seq, seq] - no batch dim
                pass
            else:
                # Unexpected shape
                return {
                    'method': 'attention_anomaly',
                    'error': f'Unexpected attention shape: {attention_weights.shape}',
                    'num_tokens': 0,
                    'num_anomalies': 0
                }

            # Get logits for TokenAnalysis
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Compute metrics
        entropies = self.compute_attention_entropy(attention_weights)
        anomaly_scores = self.compute_attention_anomaly_score(attention_weights)

        # Detect anomalies
        anomalies, stats = self.detect_anomalies(entropies, anomaly_scores)

        # Map to tokens
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0][1:]]

        # NEW: Create TokenAnalysis objects for visualization
        # Compute average attention across layers and heads for detailed metrics
        avg_attention = attention_weights.mean(dim=[0, 1])  # [seq_len, seq_len]

        token_analyses = []
        for i in range(len(tokens)):
            if i >= len(logits):
                break

            token_id = input_ids[0][i + 1].item()
            token_logits = logits[i]

            # Compute probabilities
            probs = F.softmax(token_logits, dim=-1)
            token_prob = probs[token_id].item()
            token_logit = token_logits[token_id].item()

            # Compute rank
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1

            # Compute entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            # Compute surprisal
            surprisal = -np.log2(token_prob + 1e-10)

            # Compute perplexity
            perplexity = 2 ** entropy

            # Get top-10 probabilities
            top_k_probs = [(sorted_indices[j].item(), sorted_probs[j].item())
                          for j in range(min(10, len(sorted_probs)))]

            # Attention metrics for this token
            if i < len(avg_attention):
                attn_dist = avg_attention[i]

                # Self-attention (diagonal element)
                self_attention = attn_dist[i].item()

                # Variance of attention distribution
                attn_variance = torch.var(attn_dist).item()
            else:
                self_attention = 0.0
                attn_variance = 0.0

            # Create TokenAnalysis object
            analysis = TokenAnalysis(
                token=tokens[i],
                token_id=token_id,
                position=i,
                probability=token_prob,
                logit=token_logit,
                rank=rank,
                perplexity=perplexity,
                entropy=entropy,
                surprisal=surprisal,
                top_k_probs=top_k_probs,
                max_probability=sorted_probs[0].item(),
                probability_margin=sorted_probs[0].item() - sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0,
                shannon_entropy=entropy,
                local_perplexity=perplexity,
                sequence_improbability=0.0,  # Not computed in this method
                confidence_score=token_prob,
                attention_entropy=entropies[i] if i < len(entropies) else 0.0,  # ← Key metric
                attention_self_attention=self_attention,  # ← Key metric
                attention_variance=attn_variance,  # ← Key metric
                attention_anomaly_score=anomaly_scores[i] if i < len(anomaly_scores) else 0.0,  # ← Key metric
                is_anomalous=anomalies[i] if i < len(anomalies) else False
            )
            token_analyses.append(analysis)

        result = {
            'method': 'attention_anomaly',
            'attention_entropies': entropies[:len(tokens)],
            'anomaly_scores': anomaly_scores[:len(tokens)],
            'anomalies': anomalies[:len(tokens)],
            'tokens': tokens,
            'statistics': stats,
            'num_tokens': len(tokens),
            'num_anomalies': sum(anomalies[:len(tokens)]),
            'token_analyses': token_analyses,  # ← NEW: For visualization
            'code': code  # ← NEW: For visualization
        }

        return result


class AdvancedMethodsComparator:
    """
    Compares all methods (baseline + up to 5 advanced) on the same code.
    Supports 4-6 methods depending on which optional detectors are available.
    """

    def __init__(self,
                 semantic_energy_detector: SemanticEnergyDetector,
                 conformal_detector: ConformalPredictionDetector,
                 attention_detector: AttentionAnomalyDetector,
                 semantic_context_detector: Optional[SemanticContextDetector] = None,
                 masked_token_replacement_detector: Optional[MaskedTokenReplacementDetector] = None):
        """
        Initialize comparator with all detectors.

        Args:
            semantic_energy_detector: Semantic energy detector
            conformal_detector: Conformal prediction detector
            attention_detector: Attention anomaly detector
            semantic_context_detector: Optional semantic context detector (5th method)
            masked_token_replacement_detector: Optional MTR detector (6th method)
        """
        self.semantic_energy = semantic_energy_detector
        self.conformal = conformal_detector
        self.attention = attention_detector
        self.semantic_context = semantic_context_detector
        self.masked_token_replacement = masked_token_replacement_detector

    def compare_all_methods(self,
                           code: str,
                           model,
                           tokenizer,
                           baseline_detector,
                           example_name: str = "unknown") -> MethodComparisonResult:
        """
        Run all 4 methods and compare results.

        Args:
            code: Source code to analyze
            model: Language model
            tokenizer: Tokenizer
            baseline_detector: LecPrompt baseline detector
            example_name: Name of the example

        Returns:
            MethodComparisonResult with comprehensive comparison
        """
        # FIX: Use perf_counter for better time resolution (nanosecond precision)
        import time

        # 1. Baseline (LecPrompt)
        start = time.perf_counter()
        baseline_result = baseline_detector.localize_errors(code)
        lecprompt_time = time.perf_counter() - start

        # NEW: Ensure baseline result includes 'code' for visualization
        if 'code' not in baseline_result:
            baseline_result['code'] = code

        # NEW: Ensure baseline result includes 'token_analyses' if available
        # Check if baseline_detector provides token_analyses (newer detectors do)
        if 'token_analyses' not in baseline_result and hasattr(baseline_detector, 'get_token_analyses'):
            try:
                baseline_result['token_analyses'] = baseline_detector.get_token_analyses(code)
            except:
                pass  # Older detectors may not support this

        # 2. Semantic Energy
        start = time.perf_counter()
        baseline_log_probs = [t[2] for t in baseline_detector.compute_token_log_probabilities(code)]
        energy_result = self.semantic_energy.analyze_code(
            code, model, tokenizer, baseline_log_probs
        )
        energy_time = time.perf_counter() - start

        # 3. Conformal Prediction
        start = time.perf_counter()
        conformal_result = self.conformal.analyze_code(code, model, tokenizer)
        conformal_time = time.perf_counter() - start

        # 4. Attention Anomaly
        start = time.perf_counter()
        attention_result = self.attention.analyze_code(code, model, tokenizer)
        attention_time = time.perf_counter() - start

        # 5. Semantic Context (optional)
        semantic_context_result = None
        semantic_context_time = 0.0
        if self.semantic_context and self.semantic_context.is_available():
            start = time.perf_counter()
            semantic_context_result = self.semantic_context.analyze_code(code)
            semantic_context_time = time.perf_counter() - start

        # 6. Masked Token Replacement (optional)
        mtr_result = None
        mtr_time = 0.0
        if self.masked_token_replacement and self.masked_token_replacement.is_available():
            start = time.perf_counter()
            mtr_result = self.masked_token_replacement.analyze_code(code)
            mtr_time = time.perf_counter() - start

        # Prepare results list for agreement computation
        results_for_agreement = [baseline_result, energy_result, conformal_result, attention_result]
        if semantic_context_result and semantic_context_result.get('available'):
            results_for_agreement.append(semantic_context_result)
        if mtr_result and mtr_result.get('available'):
            results_for_agreement.append(mtr_result)

        # Compute agreement matrix (4x4 or 5x5 depending on semantic context availability)
        agreement_matrix = self._compute_agreement_matrix(*results_for_agreement)

        # Compute token overlap
        token_overlap = self._compute_token_overlap(*results_for_agreement)

        # Find consensus anomalies (flagged by majority)
        num_methods = len(results_for_agreement)
        consensus_threshold = (num_methods + 1) // 2  # Majority = ceil(n/2)
        consensus = self._find_consensus_anomalies(*results_for_agreement, threshold=consensus_threshold)

        # Determine best method (most anomalies detected)
        method_scores = {
            'lecprompt': baseline_result.get('num_anomalies', baseline_result['statistics'].get('anomalous_tokens', 0)),
            'semantic_energy': energy_result.get('num_anomalies', energy_result['statistics'].get('num_anomalies', 0)),
            'conformal': conformal_result.get('num_anomalies', 0),
            'attention': attention_result.get('num_anomalies', 0)
        }
        if semantic_context_result and semantic_context_result.get('available'):
            method_scores['semantic_context'] = semantic_context_result.get('num_anomalous_lines', 0)
        if mtr_result and mtr_result.get('available'):
            method_scores['masked_token_replacement'] = mtr_result.get('num_anomalies', 0)

        best_method = max(method_scores, key=method_scores.get)

        exec_times = {
            'lecprompt': lecprompt_time,
            'semantic_energy': energy_time,
            'conformal': conformal_time,
            'attention': attention_time
        }
        if semantic_context_result:
            exec_times['semantic_context'] = semantic_context_time
        if mtr_result:
            exec_times['masked_token_replacement'] = mtr_time

        return MethodComparisonResult(
            example_name=example_name,
            buggy_code=code,
            correct_code="",  # To be filled by caller
            lecprompt_result=baseline_result,
            semantic_energy_result=energy_result,
            conformal_result=conformal_result,
            attention_result=attention_result,
            semantic_context_result=semantic_context_result,
            masked_token_replacement_result=mtr_result,
            method_agreement_matrix=agreement_matrix,
            token_overlap=token_overlap,
            best_method=best_method,
            consensus_anomalies=consensus,
            execution_times=exec_times
        )

    def _compute_agreement_matrix(self, *results) -> np.ndarray:
        """Compute pairwise agreement between methods."""
        n_methods = len(results)
        matrix = np.zeros((n_methods, n_methods))

        # Extract anomaly flags
        anomaly_lists = []
        for result in results:
            if 'anomalies' in result:
                anomaly_lists.append(result['anomalies'])
            elif 'token_errors' in result:
                anomaly_lists.append([t.is_anomalous for t in result['token_errors']])
            else:
                anomaly_lists.append([])

        # Compute pairwise Jaccard similarity
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    # Jaccard similarity
                    set_i = set([k for k, v in enumerate(anomaly_lists[i]) if v])
                    set_j = set([k for k, v in enumerate(anomaly_lists[j]) if v])

                    if len(set_i) == 0 and len(set_j) == 0:
                        matrix[i][j] = 1.0
                    elif len(set_i.union(set_j)) == 0:
                        matrix[i][j] = 0.0
                    else:
                        matrix[i][j] = len(set_i.intersection(set_j)) / len(set_i.union(set_j))

        return matrix

    def _compute_token_overlap(self, *results) -> Dict[str, int]:
        """Compute token overlap between method pairs."""
        overlap = {}

        anomaly_sets = []
        method_names = []

        # Build method names and anomaly sets dynamically
        base_methods = ['lecprompt', 'semantic_energy', 'conformal', 'attention']
        for i, result in enumerate(results):
            # Determine method name
            if i < len(base_methods):
                method_name = base_methods[i]
            else:
                # Extra methods (semantic_context, etc.)
                method_name = result.get('method', f'method_{i}')

            method_names.append(method_name)

            # Extract anomaly set
            if 'anomalies' in result:
                anomaly_set = set([k for k, v in enumerate(result['anomalies']) if v])
            elif 'token_errors' in result:
                anomaly_set = set([t.position for t in result['token_errors'] if t.is_anomalous])
            elif 'line_anomalies' in result:
                # For line-level methods, use line numbers as "positions"
                anomaly_set = set([line_num for line_num, _, is_anom in result['line_anomalies'] if is_anom])
            else:
                anomaly_set = set()
            anomaly_sets.append(anomaly_set)

        # Compute all pairwise overlaps
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                key = f"{method_names[i]}_vs_{method_names[j]}"
                overlap[key] = len(anomaly_sets[i].intersection(anomaly_sets[j]))

        return overlap

    def _find_consensus_anomalies(self, *results, threshold: int = 3) -> List[int]:
        """
        Find tokens flagged as anomalous by majority of methods.

        Args:
            *results: Variable number of method results
            threshold: Minimum number of methods that must agree (default: 3)

        Returns:
            List of token positions flagged by at least threshold methods
        """
        # Count votes for each token position
        vote_counts = {}

        for result in results:
            if 'anomalies' in result:
                for k, is_anom in enumerate(result['anomalies']):
                    if is_anom:
                        vote_counts[k] = vote_counts.get(k, 0) + 1
            elif 'token_errors' in result:
                for token_error in result['token_errors']:
                    if token_error.is_anomalous:
                        pos = token_error.position
                        vote_counts[pos] = vote_counts.get(pos, 0) + 1
            elif 'line_anomalies' in result:
                # Semantic context detector returns line-level anomalies
                # Include in consensus but note these are line-level
                for line_num, dissim, is_anom in result['line_anomalies']:
                    if is_anom:
                        # Use negative indices to distinguish line-level from token-level
                        vote_counts[f"line_{line_num}"] = vote_counts.get(f"line_{line_num}", 0) + 1

        # Return positions with threshold+ votes
        consensus = [pos for pos, votes in vote_counts.items() if votes >= threshold]
        return sorted([p for p in consensus if isinstance(p, int)])  # Return only token positions


class MaskedTokenReplacementDetector:
    """
    Masked Token Replacement (MTR) error detection using direct prediction matching.

    Based on: Masked Language Modeling (Devlin et al., BERT 2019)

    Key idea: For each token in the code, mask it and ask a BERT-like model to predict
    what token should be there. If the top-1 prediction differs from the original token,
    it may indicate an error. This is a more direct and interpretable approach than
    probability-based methods.

    Unlike probability-based approaches, this method makes a binary decision:
    - Token matches prediction → OK
    - Token differs from prediction → Potential error

    This is particularly effective for detecting:
    - Variable name typos
    - Wrong operator usage
    - Incorrect keywords
    - Copy-paste errors
    """

    def __init__(self,
                 model_name: str = "microsoft/codebert-base",
                 sensitivity_threshold: float = 0.7,
                 device: str = "auto"):
        """
        Initialize masked token replacement detector.

        Args:
            model_name: HuggingFace model identifier for MLM model
            sensitivity_threshold: Minimum confidence for accepting a mismatch as error (0-1)
            device: Device to run model on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.confidence_threshold = sensitivity_threshold

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Try to load model
        try:
            from transformers import RobertaTokenizer, RobertaForMaskedLM
            print(f"✓ Loading MLM model: {model_name}")
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.available = True
            print(f"  Model loaded on {self.device}")
        except Exception as e:
            print(f"⚠ Could not load MLM model: {e}")
            self.tokenizer = None
            self.model = None
            self.available = False

    def is_available(self) -> bool:
        """Check if the detector is available (model loaded)."""
        return self.available

    def predict_masked_token(self,
                             input_ids: torch.Tensor,
                             mask_position: int) -> Tuple[int, str, float]:
        """
        Predict token at masked position.

        Args:
            input_ids: Token IDs with one position masked
            mask_position: Position of the mask

        Returns:
            Tuple of (predicted_token_id, predicted_token_text, confidence)
        """
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0).to(self.device))
            logits = outputs.logits[0, mask_position]  # Logits for masked position

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Get top-1 prediction
            top_prob, top_id = torch.max(probs, dim=-1)
            top_token = self.tokenizer.decode([top_id.item()])

            return top_id.item(), top_token, top_prob.item()

    def detect_token_mismatches(self, code: str) -> Tuple[List[Tuple[int, str, str, str, float, bool]], Dict[str, Any]]:
        """
        Detect tokens that don't match model predictions.

        Args:
            code: Source code to analyze

        Returns:
            Tuple of (token_mismatches, statistics)
            - token_mismatches: List of (position, original_token, predicted_token,
                                         decoded_predicted, confidence, is_mismatch)
            - statistics: Dictionary with detection stats
        """
        if not self.available:
            return [], {'error': 'MTR detector not available'}

        # Tokenize code
        encoding = self.tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids[0]

        # Get tokens (for display)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        mismatches = []
        num_mismatches = 0
        total_confidence = 0.0

        # For each token (skip special tokens)
        for i in range(len(input_ids)):
            # Skip special tokens
            if tokens[i] in ['<s>', '</s>', '<pad>', '<unk>']:
                continue

            original_token_id = input_ids[i].item()
            original_token = tokens[i]

            # Create masked version
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            # Predict token
            predicted_id, predicted_token, confidence = self.predict_masked_token(
                masked_input_ids, i
            )

            # Check for mismatch
            is_mismatch = (predicted_id != original_token_id) and (confidence >= self.confidence_threshold)

            if is_mismatch:
                num_mismatches += 1

            total_confidence += confidence

            mismatches.append((
                i,
                original_token,
                predicted_token,
                self.tokenizer.decode([predicted_id]),
                confidence,
                is_mismatch
            ))

        # Statistics
        stats = {
            'method': 'masked_token_replacement',
            'num_tokens': len(mismatches),
            'num_mismatches': num_mismatches,
            'mismatch_rate': num_mismatches / len(mismatches) if mismatches else 0.0,
            'avg_prediction_confidence': total_confidence / len(mismatches) if mismatches else 0.0,
            'confidence_threshold': self.confidence_threshold
        }

        return mismatches, stats

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Complete analysis using masked token replacement.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary with analysis results
        """
        if not self.available:
            return {
                'method': 'masked_token_replacement',
                'available': False,
                'error': 'transformers library or model not available'
            }

        mismatches, stats = self.detect_token_mismatches(code)

        # Extract anomalous tokens (mismatches)
        anomalies = [is_mismatch for _, _, _, _, _, is_mismatch in mismatches]

        # Detailed mismatch information
        mismatch_details = []
        for pos, orig, pred, pred_decoded, conf, is_mismatch in mismatches:
            if is_mismatch:
                mismatch_details.append({
                    'position': pos,
                    'original_token': orig,
                    'predicted_token': pred_decoded,
                    'prediction_confidence': conf
                })

        result = {
            'method': 'masked_token_replacement',
            'available': True,
            'anomalies': anomalies,  # Boolean list for compatibility
            'mismatches': mismatch_details,
            'statistics': stats,
            'num_tokens': stats['num_tokens'],
            'num_anomalies': stats['num_mismatches']
        }

        return result


class SemanticContextDetector:
    """
    Semantic Context-based error detection using sentence embeddings.

    Based on: "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (Gao et al., EMNLP 2021)

    Key idea: Analyze whether each line of code is semantically coherent with its
    surrounding context. Lines that are semantically distant from their context
    may indicate logical errors or out-of-place operations.

    Unlike token-level methods, this operates at line granularity and captures
    high-level semantic inconsistencies.
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 context_window: int = 3,
                 sensitivity_factor: float = 1.5):
        """
        Initialize semantic context detector.

        Args:
            embedding_model: HuggingFace sentence-transformer model name
            context_window: Number of lines before/after to use as context
            sensitivity_factor: k for anomaly threshold (τ = μ + k×σ)
        """
        self.context_window = context_window
        self.k = sensitivity_factor

        # Try to load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embedding_model)
            self.available = True
            print(f"✓ Loaded semantic embedding model: {embedding_model}")
        except ImportError:
            print("⚠ sentence-transformers not available. Install with: pip install sentence-transformers")
            self.encoder = None
            self.available = False
        except Exception as e:
            print(f"⚠ Could not load embedding model: {e}")
            self.encoder = None
            self.available = False

    def is_available(self) -> bool:
        """Check if the detector is available (dependencies installed)."""
        return self.available

    def compute_line_embeddings(self, code: str) -> List[np.ndarray]:
        """
        Compute embedding for each non-empty line of code.

        Args:
            code: Source code to analyze

        Returns:
            List of embedding vectors, one per line
        """
        if not self.available:
            return []

        lines = code.split('\n')
        # Filter out empty lines but keep track of original indices
        non_empty_lines = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]

        if not non_empty_lines:
            return []

        # Encode all lines at once (efficient batching)
        texts = [line for _, line in non_empty_lines]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)

        return embeddings

    def compute_context_embedding(self,
                                  line_idx: int,
                                  lines: List[str]) -> np.ndarray:
        """
        Compute embedding for the context around a line.

        Context includes ±context_window lines (excluding the line itself).

        Args:
            line_idx: Index of the current line
            lines: All lines of code

        Returns:
            Context embedding vector
        """
        if not self.available:
            return np.array([])

        # Get context lines (before and after, excluding current line)
        start_idx = max(0, line_idx - self.context_window)
        end_idx = min(len(lines), line_idx + self.context_window + 1)

        context_lines = []
        for i in range(start_idx, end_idx):
            if i != line_idx and lines[i].strip():
                context_lines.append(lines[i].strip())

        if not context_lines:
            # No context available, return zero vector
            return np.zeros(self.encoder.get_sentence_embedding_dimension())

        # Concatenate context lines
        context_text = " ".join(context_lines)

        # Encode context
        context_embedding = self.encoder.encode(context_text, convert_to_numpy=True)

        return context_embedding

    def compute_semantic_similarity(self,
                                   line_embedding: np.ndarray,
                                   context_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between line and context embeddings.

        Args:
            line_embedding: Embedding of the line
            context_embedding: Embedding of the context

        Returns:
            Cosine similarity (0-1, higher = more similar)
        """
        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        similarity = cosine_similarity(
            line_embedding.reshape(1, -1),
            context_embedding.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def detect_semantic_anomalies(self, code: str) -> Tuple[List[Tuple[int, float, bool]], Dict[str, Any]]:
        """
        Detect semantically anomalous lines in code.

        A line is anomalous if it has low semantic similarity with its context,
        indicating it may be logically out of place.

        Args:
            code: Source code to analyze

        Returns:
            Tuple of (line_anomalies, statistics)
            - line_anomalies: List of (line_number, dissimilarity_score, is_anomalous)
            - statistics: Dictionary with detection stats
        """
        if not self.available:
            return [], {'error': 'Semantic context detector not available'}

        lines = code.split('\n')
        line_embeddings = self.compute_line_embeddings(code)

        if len(line_embeddings) == 0:
            return [], {'error': 'No non-empty lines to analyze'}

        # Compute dissimilarity score for each line
        dissimilarity_scores = []
        line_numbers = []

        non_empty_lines = [(i, line.strip()) for i, line in enumerate(lines) if line.strip()]

        for idx, (line_num, line_text) in enumerate(non_empty_lines):
            line_emb = line_embeddings[idx]
            context_emb = self.compute_context_embedding(line_num, lines)

            if context_emb.size == 0:
                # No context, mark as neutral
                dissimilarity_scores.append(0.5)
            else:
                similarity = self.compute_semantic_similarity(line_emb, context_emb)
                dissimilarity = 1.0 - similarity  # Convert to dissimilarity
                dissimilarity_scores.append(dissimilarity)

            line_numbers.append(line_num + 1)  # 1-indexed

        # Compute statistical threshold
        dissim_array = np.array(dissimilarity_scores)
        mean_dissim = np.mean(dissim_array)
        std_dissim = np.std(dissim_array)

        # High dissimilarity is anomalous
        threshold = mean_dissim + self.k * std_dissim

        # Flag anomalies
        line_anomalies = []
        for line_num, dissim in zip(line_numbers, dissimilarity_scores):
            is_anomalous = dissim > threshold
            line_anomalies.append((line_num, dissim, is_anomalous))

        # Statistics
        stats = {
            'method': 'semantic_context',
            'num_lines': len(line_anomalies),
            'num_anomalous_lines': sum(1 for _, _, anom in line_anomalies if anom),
            'mean_dissimilarity': float(mean_dissim),
            'std_dissimilarity': float(std_dissim),
            'threshold': float(threshold),
            'context_window': self.context_window
        }

        return line_anomalies, stats

    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Complete semantic context analysis of code.

        Args:
            code: Source code to analyze

        Returns:
            Dictionary with analysis results
        """
        if not self.available:
            return {
                'method': 'semantic_context',
                'available': False,
                'error': 'sentence-transformers not installed'
            }

        line_anomalies, stats = self.detect_semantic_anomalies(code)

        # Extract lines from code
        lines = code.split('\n')

        # Build detailed results
        anomalous_lines = []
        for line_num, dissim_score, is_anom in line_anomalies:
            if is_anom:
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                anomalous_lines.append({
                    'line_number': line_num,
                    'line_content': line_content,
                    'dissimilarity_score': dissim_score
                })

        result = {
            'method': 'semantic_context',
            'available': True,
            'line_anomalies': line_anomalies,
            'anomalous_lines': anomalous_lines,
            'statistics': stats,
            'num_anomalous_lines': stats['num_anomalous_lines']
        }

        return result


if __name__ == "__main__":
    print("Advanced Error Detection Methods")
    print("\nAvailable detectors:")
    print("  1. SemanticEnergyDetector - Uses logits for better uncertainty")
    print("  2. ConformalPredictionDetector - Statistical guarantees")
    print("  3. AttentionAnomalyDetector - Analyzes attention patterns")
    print("  4. SemanticContextDetector - Line-level semantic coherence analysis")
    print("  5. MaskedTokenReplacementDetector - Direct prediction matching (NEW)")
    print("\nUsage:")
    print("  from detectors.advanced_methods import SemanticEnergyDetector")
    print("  detector = SemanticEnergyDetector()")
    print("  result = detector.analyze_code(code, model, tokenizer)")
    print("\n  # For MTR detector:")
    print("  from detectors.advanced_methods import MaskedTokenReplacementDetector")
    print("  mtr = MaskedTokenReplacementDetector()")
    print("  result = mtr.analyze_code(code)")
