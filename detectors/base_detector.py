#!/usr/bin/env python3
"""
Base Error Detector - Abstract Base Class

Defines the interface that all error detection implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectorMetadata:
    """Metadata about the error detector model"""
    model_name: str
    model_type: str  # "causal", "mlm", "encoder-decoder"
    parameters: str  # e.g., "7B", "2B"
    approach: str  # e.g., "Autoregressive", "MLM"
    year: int
    sequence_length: int = 8192
    special_features: Optional[str] = None


@dataclass
class TokenErrorInfo:
    """Information about a potentially erroneous token"""
    token: str
    token_id: int
    position: int
    line_number: int
    log_probability: float
    is_anomalous: bool
    deviation_score: float  # in standard deviations
    error_likelihood: float  # 0-1


@dataclass
class LineErrorInfo:
    """Information about a potentially erroneous line"""
    line_number: int
    line_content: str
    num_tokens: int
    num_anomalous_tokens: int
    avg_log_prob: float
    min_log_prob: float
    error_score: float  # 0-1
    is_error_line: bool


class BaseErrorDetector(ABC):
    """
    Abstract base class for all error detection implementations.

    All concrete detector classes must inherit from this and implement
    the abstract methods.
    """

    def __init__(self, sensitivity_factor: float = 1.5):
        """
        Initialize the detector.

        Args:
            sensitivity_factor: k parameter for threshold (τ = μ - k×σ)
        """
        self.k = sensitivity_factor
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def compute_token_log_probabilities(self, code: str) -> List[Tuple[str, int, float, int]]:
        """
        Compute log probabilities for each token in the code.

        Args:
            code: Source code to analyze

        Returns:
            List of (token_text, token_id, log_probability, char_position) tuples
        """
        pass

    @abstractmethod
    def get_metadata(self) -> DetectorMetadata:
        """
        Return metadata about this detector.

        Returns:
            DetectorMetadata object with model information
        """
        pass

    def compute_statistical_threshold(self,
                                     log_probs: List[float],
                                     k: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Compute the statistical threshold for anomaly detection.

        Uses the formula: τ = μ - k×σ

        Args:
            log_probs: List of log probabilities
            k: Sensitivity factor (uses self.k if not specified)

        Returns:
            Tuple of (mean, std_dev, threshold)
        """
        if k is None:
            k = self.k

        log_probs_array = np.array(log_probs)

        # Calculate mean and standard deviation
        mean = np.mean(log_probs_array)
        std_dev = np.std(log_probs_array)

        # Calculate threshold
        threshold = mean - k * std_dev

        return float(mean), float(std_dev), float(threshold)

    def identify_anomalous_tokens(self,
                                  token_data: List[Tuple[str, int, float, int]],
                                  code: str,
                                  k: Optional[float] = None) -> List[TokenErrorInfo]:
        """
        Identify tokens that are anomalous based on statistical analysis.

        Args:
            token_data: List of (token_text, token_id, log_prob, char_pos)
            code: Original source code
            k: Sensitivity factor

        Returns:
            List of TokenErrorInfo objects
        """
        # Extract log probabilities
        log_probs = [data[2] for data in token_data]

        # Compute statistical threshold
        mean, std_dev, threshold = self.compute_statistical_threshold(log_probs, k)

        # Map tokens to line numbers
        token_to_line_map = self._map_tokens_to_lines(code, [data[3] for data in token_data])

        # Identify anomalous tokens
        token_errors = []
        for i, (token_text, token_id, log_prob, char_pos) in enumerate(token_data):
            # Calculate deviation score (in standard deviations)
            deviation_score = (log_prob - mean) / std_dev if std_dev > 0 else 0.0

            # Check if anomalous
            is_anomalous = log_prob < threshold

            # Compute error likelihood (0-1, higher = more likely error)
            error_likelihood = min(1.0, max(0.0, -deviation_score / 3.0))

            # Get line number
            line_num = token_to_line_map.get(i, 1)

            token_error = TokenErrorInfo(
                token=token_text,
                token_id=token_id,
                position=i,
                line_number=line_num,
                log_probability=log_prob,
                is_anomalous=is_anomalous,
                deviation_score=deviation_score,
                error_likelihood=error_likelihood
            )

            token_errors.append(token_error)

        return token_errors

    def aggregate_to_line_level(self,
                               token_errors: List[TokenErrorInfo],
                               code: str) -> List[LineErrorInfo]:
        """
        Aggregate token-level errors to line-level error scores.

        Args:
            token_errors: List of TokenErrorInfo objects
            code: Original source code

        Returns:
            List of LineErrorInfo objects
        """
        lines = code.split('\n')

        # Group tokens by line
        line_to_tokens: Dict[int, List[TokenErrorInfo]] = {}
        for token_error in token_errors:
            line_num = token_error.line_number
            if line_num not in line_to_tokens:
                line_to_tokens[line_num] = []
            line_to_tokens[line_num].append(token_error)

        # Compute line-level errors
        line_errors = []
        for line_num in sorted(line_to_tokens.keys()):
            # Get line content
            line_content = lines[line_num - 1] if line_num <= len(lines) else ""

            tokens = line_to_tokens[line_num]

            # Compute statistics
            log_probs = [t.log_probability for t in tokens]
            avg_log_prob = np.mean(log_probs)
            min_log_prob = np.min(log_probs)

            # Count anomalous tokens
            anomalous_tokens = [t for t in tokens if t.is_anomalous]
            num_anomalous = len(anomalous_tokens)

            # Compute line error score
            anomaly_ratio = num_anomalous / len(tokens) if tokens else 0.0
            avg_error_likelihood = np.mean([t.error_likelihood for t in tokens]) if tokens else 0.0
            error_score = (anomaly_ratio + avg_error_likelihood) / 2.0

            # Determine if this is an error line
            is_error_line = num_anomalous > 0

            line_error = LineErrorInfo(
                line_number=line_num,
                line_content=line_content,
                num_tokens=len(tokens),
                num_anomalous_tokens=num_anomalous,
                avg_log_prob=float(avg_log_prob),
                min_log_prob=float(min_log_prob),
                error_score=float(error_score),
                is_error_line=is_error_line
            )

            line_errors.append(line_error)

        return line_errors

    def localize_errors(self,
                       code: str,
                       k: Optional[float] = None) -> Dict[str, Any]:
        """
        Perform complete error localization on code.

        Args:
            code: Source code to analyze
            k: Sensitivity factor (uses self.k if not specified)

        Returns:
            Dictionary with complete error analysis results
        """
        print(f"Analyzing code with {self.get_metadata().model_name} (k={k or self.k})...")

        # Step 1: Compute token log probabilities
        token_data = self.compute_token_log_probabilities(code)
        print(f"  Computed probabilities for {len(token_data)} tokens")

        # Step 2: Identify anomalous tokens
        token_errors = self.identify_anomalous_tokens(token_data, code, k)
        anomalous_count = sum(1 for t in token_errors if t.is_anomalous)
        print(f"  Found {anomalous_count} anomalous tokens")

        # Step 3: Aggregate to line level
        line_errors = self.aggregate_to_line_level(token_errors, code)
        error_lines_count = sum(1 for l in line_errors if l.is_error_line)
        print(f"  Detected {error_lines_count} error lines")

        # Compute overall statistics
        all_log_probs = [t.log_probability for t in token_errors]
        mean, std_dev, threshold = self.compute_statistical_threshold(all_log_probs, k)

        metadata = self.get_metadata()

        return {
            "model_name": metadata.model_name,
            "model_type": metadata.model_type,
            "code": code,
            "sensitivity_factor": k or self.k,
            "statistics": {
                "total_tokens": len(token_errors),
                "anomalous_tokens": anomalous_count,
                "total_lines": len(line_errors),
                "error_lines": error_lines_count,
                "mean_log_prob": float(mean),
                "std_dev": float(std_dev),
                "threshold": float(threshold)
            },
            "token_errors": token_errors,
            "line_errors": line_errors
        }

    def compare_buggy_vs_correct(self,
                                buggy_code: str,
                                correct_code: str,
                                k: Optional[float] = None) -> Dict[str, Any]:
        """
        Compare error detection between buggy and correct code.

        Args:
            buggy_code: Code with bugs
            correct_code: Correct code
            k: Sensitivity factor

        Returns:
            Comparison results
        """
        print(f"\n{'='*60}")
        print(f"Comparing buggy vs correct code with {self.get_metadata().model_name}")
        print(f"{'='*60}")

        print("\nAnalyzing buggy code...")
        buggy_results = self.localize_errors(buggy_code, k)

        print("\nAnalyzing correct code...")
        correct_results = self.localize_errors(correct_code, k)

        # Compare statistics
        buggy_stats = buggy_results["statistics"]
        correct_stats = correct_results["statistics"]

        comparison = {
            "buggy_analysis": buggy_results,
            "correct_analysis": correct_results,
            "comparison": {
                "anomalous_tokens_diff": (
                    buggy_stats["anomalous_tokens"] -
                    correct_stats["anomalous_tokens"]
                ),
                "error_lines_diff": (
                    buggy_stats["error_lines"] -
                    correct_stats["error_lines"]
                ),
                "mean_log_prob_diff": (
                    buggy_stats["mean_log_prob"] -
                    correct_stats["mean_log_prob"]
                ),
                "hypothesis_confirmed": (
                    buggy_stats["anomalous_tokens"] >
                    correct_stats["anomalous_tokens"]
                )
            }
        }

        print(f"\nComparison Results:")
        print(f"  Buggy anomalies: {buggy_stats['anomalous_tokens']}")
        print(f"  Correct anomalies: {correct_stats['anomalous_tokens']}")
        print(f"  Hypothesis: {'✓ CONFIRMED' if comparison['comparison']['hypothesis_confirmed'] else '✗ REJECTED'}")

        return comparison

    def _map_tokens_to_lines(self, code: str, char_positions: List[int]) -> Dict[int, int]:
        """
        Map token positions to line numbers.

        Args:
            code: Source code
            char_positions: List of character positions for each token

        Returns:
            Dictionary mapping token index to line number
        """
        lines = code.split('\n')
        char_to_line = {}
        current_char = 0

        for line_idx, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            for i in range(line_len):
                char_to_line[current_char + i] = line_idx + 1
            current_char += line_len

        # Map tokens to lines
        token_to_line = {}
        for token_idx, char_pos in enumerate(char_positions):
            if char_pos in char_to_line:
                token_to_line[token_idx] = char_to_line[char_pos]
            else:
                # Default to first line if mapping fails
                token_to_line[token_idx] = 1

        return token_to_line

    # ========================================================================
    # OPTIONAL METHODS FOR ADVANCED DETECTION
    # ========================================================================
    # Subclasses can optionally implement these methods to support
    # advanced error detection techniques (Semantic Energy, Conformal
    # Prediction, Attention Anomaly).

    def compute_semantic_energy(self, code: str) -> Optional[List[float]]:
        """
        Optional: Compute semantic energy from logits.

        Semantic energy uses pre-softmax logits to capture model uncertainty.
        Lower energy = higher confidence.

        Args:
            code: Source code to analyze

        Returns:
            List of energy values (one per token), or None if not supported
        """
        return None  # Override in subclass to enable

    def get_attention_weights(self, code: str) -> Optional[Any]:
        """
        Optional: Return attention weights for analysis.

        Attention patterns can reveal model uncertainty. Uniform attention
        (high entropy) indicates uncertainty.

        Args:
            code: Source code to analyze

        Returns:
            Attention weights tensor, or None if not supported
        """
        return None  # Override in subclass to enable

    def compute_conformal_scores(self,
                                 code: str,
                                 calibration_data: Optional[List] = None) -> Optional[List[float]]:
        """
        Optional: Compute conformal prediction scores.

        Conformal prediction provides statistical guarantees for uncertainty
        quantification. Larger prediction sets indicate higher uncertainty.

        Args:
            code: Source code to analyze
            calibration_data: Optional calibration data for conformal prediction

        Returns:
            List of conformal scores, or None if not supported
        """
        return None  # Override in subclass to enable
