#!/usr/bin/env python3
"""
CodeBERT-based Logical Error Detector (Original LecPrompt Implementation)

This module implements the error localization approach using CodeBERT
as described in the original LecPrompt paper, using masked language modeling
to compute token probabilities.

Key difference from logical_error_detector.py:
- Uses CodeBERT (microsoft/codebert-base) - Encoder model with MLM
- Masks each token and predicts probability (BERT-style)
- More faithful to the original paper implementation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import RobertaTokenizer, RobertaForMaskedLM
import re


@dataclass
class TokenError:
    """Data class for a detected token-level error"""
    token: str
    token_id: int
    position: int  # Position in token sequence
    line_number: int  # Line number in source code
    column: int  # Column position in line
    log_probability: float
    mean_log_prob: float
    std_dev: float
    threshold: float
    deviation_score: float  # Number of std devs from mean
    is_anomalous: bool
    error_likelihood: float  # Normalized score 0-1


@dataclass
class LineError:
    """Data class for a detected line-level error"""
    line_number: int
    line_content: str
    avg_log_prob: float
    min_log_prob: float
    max_log_prob: float
    num_tokens: int
    num_anomalous_tokens: int
    anomalous_tokens: List[TokenError]
    error_score: float  # Normalized aggregate score 0-1
    is_error_line: bool


class CodeBERTErrorDetector:
    """
    Logical error detector using CodeBERT with masked language modeling.

    This implementation follows the original LecPrompt paper more closely
    by using CodeBERT (a BERT-style encoder model) and computing token
    probabilities via masking.
    """

    def __init__(self,
                 model_name: str = "microsoft/codebert-base",
                 device: str = "auto",
                 sensitivity_factor: float = 1.5):
        """
        Initialize the CodeBERT error detector.

        Args:
            model_name: HuggingFace model identifier (CodeBERT variant)
            device: Device to load model on ("auto", "cpu", "cuda")
            sensitivity_factor: k parameter for threshold (τ = μ - k×σ)
        """
        print(f"Loading CodeBERT model: {model_name}")
        self.model_name = model_name
        self.k = sensitivity_factor

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load CodeBERT tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on device: {self.device}")

        # Patterns to exclude from analysis (per LecPrompt)
        self.exclude_patterns = [
            r'^def\s+\w+',  # Function definitions
            r'^class\s+\w+',  # Class definitions
            r'^import\s+',  # Import statements
            r'^from\s+\w+\s+import',  # From imports
        ]

    def _should_exclude_line(self, line: str) -> bool:
        """Check if a line should be excluded from error analysis."""
        line_stripped = line.strip()
        for pattern in self.exclude_patterns:
            if re.match(pattern, line_stripped):
                return True
        return False

    def _map_token_to_line(self, code: str, token_positions: List[int]) -> Dict[int, Tuple[int, int]]:
        """
        Map token positions to (line_number, column) in source code.

        Args:
            code: Source code string
            token_positions: List of character positions for each token

        Returns:
            Dictionary mapping token index to (line_number, column)
        """
        lines = code.split('\n')
        char_to_line = {}
        current_char = 0

        for line_idx, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            for i in range(line_len):
                char_to_line[current_char + i] = (line_idx + 1, i)
            current_char += line_len

        # Map tokens to lines
        token_to_line = {}
        for token_idx, char_pos in enumerate(token_positions):
            if char_pos in char_to_line:
                token_to_line[token_idx] = char_to_line[char_pos]
            else:
                # Default to first line if mapping fails
                token_to_line[token_idx] = (1, 0)

        return token_to_line

    def compute_token_log_probabilities(self, code: str) -> List[Tuple[str, int, float, int]]:
        """
        Compute log probabilities (wrapper for MLM method to match BaseDetector interface).

        Args:
            code: Source code to analyze

        Returns:
            List of (token_text, token_id, log_probability, char_position) tuples
        """
        return self.compute_token_log_probabilities_mlm(code)

    def compute_token_log_probabilities_mlm(self, code: str) -> List[Tuple[str, int, float, int]]:
        """
        Compute log probabilities using Masked Language Modeling (CodeBERT approach).

        For each token, mask it and compute the probability that CodeBERT
        assigns to the original token. This is the BERT-style approach.

        Args:
            code: Source code to analyze

        Returns:
            List of (token_text, token_id, log_probability, char_position) tuples
        """
        print(f"Computing token probabilities using MLM (masking approach)...")

        # Tokenize the code
        # Note: offset_mapping not available with Python tokenizers, only Fast tokenizers
        try:
            encoding = self.tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
            offset_mapping = encoding.offset_mapping[0] if "offset_mapping" in encoding else None
        except:
            # Fallback for Python tokenizers
            encoding = self.tokenizer(code, return_tensors="pt")
            offset_mapping = None

        input_ids = encoding.input_ids[0].to(self.device)

        # Get tokens (excluding special tokens)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        token_data = []

        # For each token (skip special tokens <s> and </s>)
        for i in range(len(input_ids)):
            # Skip special tokens
            if tokens[i] in ['<s>', '</s>', '<pad>']:
                continue

            # Create masked version
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                # Get model predictions for the masked position
                outputs = self.model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits[0, i]  # Logits for position i

                # Compute probabilities
                log_probs = F.log_softmax(logits, dim=-1)

                # Get log probability of the ORIGINAL token
                original_token_id = input_ids[i].item()
                token_log_prob = log_probs[original_token_id].item()

            # Get character position
            char_pos = offset_mapping[i][0].item() if offset_mapping is not None else 0

            # Get token text
            token_text = tokens[i]

            token_data.append((token_text, original_token_id, token_log_prob, char_pos))

        print(f"  Computed probabilities for {len(token_data)} tokens")
        return token_data

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
                                  k: Optional[float] = None) -> List[TokenError]:
        """
        Identify tokens that are anomalous based on statistical analysis.

        Args:
            token_data: List of (token_text, token_id, log_prob, char_pos)
            code: Original source code
            k: Sensitivity factor

        Returns:
            List of TokenError objects
        """
        # Extract log probabilities
        log_probs = [data[2] for data in token_data]

        # Compute statistical threshold
        mean, std_dev, threshold = self.compute_statistical_threshold(log_probs, k)

        # Map tokens to line numbers
        char_positions = [data[3] for data in token_data]
        token_to_line_map = self._map_token_to_line(code, char_positions)

        # Identify anomalous tokens
        token_errors = []
        for i, (token_text, token_id, log_prob, char_pos) in enumerate(token_data):
            # Calculate deviation score (in standard deviations)
            deviation_score = (log_prob - mean) / std_dev if std_dev > 0 else 0.0

            # Check if anomalous
            is_anomalous = log_prob < threshold

            # Compute error likelihood (0-1, higher = more likely error)
            # Map deviation to [0, 1] where -3 std devs = 1.0
            error_likelihood = min(1.0, max(0.0, -deviation_score / 3.0))

            # Get line number and column
            line_num, column = token_to_line_map.get(i, (1, 0))

            token_error = TokenError(
                token=token_text,
                token_id=token_id,
                position=i,
                line_number=line_num,
                column=column,
                log_probability=log_prob,
                mean_log_prob=mean,
                std_dev=std_dev,
                threshold=threshold,
                deviation_score=deviation_score,
                is_anomalous=is_anomalous,
                error_likelihood=error_likelihood
            )

            token_errors.append(token_error)

        return token_errors

    def aggregate_to_line_level(self,
                               token_errors: List[TokenError],
                               code: str,
                               exclude_patterns: bool = True) -> List[LineError]:
        """
        Aggregate token-level errors to line-level error scores.

        Args:
            token_errors: List of TokenError objects
            code: Original source code
            exclude_patterns: Whether to exclude certain patterns (imports, defs, etc.)

        Returns:
            List of LineError objects
        """
        lines = code.split('\n')

        # Group tokens by line
        line_to_tokens: Dict[int, List[TokenError]] = {}
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

            # Check if should exclude
            if exclude_patterns and self._should_exclude_line(line_content):
                continue

            tokens = line_to_tokens[line_num]

            # Compute statistics
            log_probs = [t.log_probability for t in tokens]
            avg_log_prob = np.mean(log_probs)
            min_log_prob = np.min(log_probs)
            max_log_prob = np.max(log_probs)

            # Count anomalous tokens
            anomalous_tokens = [t for t in tokens if t.is_anomalous]
            num_anomalous = len(anomalous_tokens)

            # Compute line error score
            # Based on: ratio of anomalous tokens + severity of deviations
            anomaly_ratio = num_anomalous / len(tokens) if tokens else 0.0
            avg_error_likelihood = np.mean([t.error_likelihood for t in tokens]) if tokens else 0.0
            error_score = (anomaly_ratio + avg_error_likelihood) / 2.0

            # Determine if this is an error line (at least 1 anomalous token)
            is_error_line = num_anomalous > 0

            line_error = LineError(
                line_number=line_num,
                line_content=line_content,
                avg_log_prob=float(avg_log_prob),
                min_log_prob=float(min_log_prob),
                max_log_prob=float(max_log_prob),
                num_tokens=len(tokens),
                num_anomalous_tokens=num_anomalous,
                anomalous_tokens=anomalous_tokens,
                error_score=float(error_score),
                is_error_line=is_error_line
            )

            line_errors.append(line_error)

        return line_errors

    def localize_errors(self,
                       code: str,
                       k: Optional[float] = None,
                       exclude_patterns: bool = True) -> Dict:
        """
        Perform complete error localization on code using CodeBERT.

        Args:
            code: Source code to analyze
            k: Sensitivity factor (uses self.k if not specified)
            exclude_patterns: Whether to exclude certain patterns

        Returns:
            Dictionary with complete error analysis results
        """
        print(f"Analyzing code for logical errors with CodeBERT (k={k or self.k})...")

        # Step 1: Compute token log probabilities using MLM
        token_data = self.compute_token_log_probabilities_mlm(code)

        # Step 2: Identify anomalous tokens
        token_errors = self.identify_anomalous_tokens(token_data, code, k)
        anomalous_count = sum(1 for t in token_errors if t.is_anomalous)
        print(f"  Found {anomalous_count} anomalous tokens")

        # Step 3: Aggregate to line level
        line_errors = self.aggregate_to_line_level(token_errors, code, exclude_patterns)
        error_lines_count = sum(1 for l in line_errors if l.is_error_line)
        print(f"  Detected {error_lines_count} error lines")

        # Compute overall statistics
        all_log_probs = [t.log_probability for t in token_errors]
        mean, std_dev, threshold = self.compute_statistical_threshold(all_log_probs, k)

        return {
            "model_name": self.model_name,
            "model_type": "CodeBERT (MLM)",
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
                                k: Optional[float] = None) -> Dict:
        """
        Compare error detection between buggy and correct code using CodeBERT.

        Args:
            buggy_code: Code with bugs
            correct_code: Correct code
            k: Sensitivity factor

        Returns:
            Comparison results
        """
        print("Analyzing buggy code with CodeBERT...")
        buggy_results = self.localize_errors(buggy_code, k)

        print("\nAnalyzing correct code with CodeBERT...")
        correct_results = self.localize_errors(correct_code, k)

        # Compare statistics
        comparison = {
            "buggy_analysis": buggy_results,
            "correct_analysis": correct_results,
            "comparison": {
                "anomalous_tokens_diff": (
                    buggy_results["statistics"]["anomalous_tokens"] -
                    correct_results["statistics"]["anomalous_tokens"]
                ),
                "error_lines_diff": (
                    buggy_results["statistics"]["error_lines"] -
                    correct_results["statistics"]["error_lines"]
                ),
                "mean_log_prob_diff": (
                    buggy_results["statistics"]["mean_log_prob"] -
                    correct_results["statistics"]["mean_log_prob"]
                ),
                "hypothesis_confirmed": (
                    buggy_results["statistics"]["anomalous_tokens"] >
                    correct_results["statistics"]["anomalous_tokens"]
                )
            }
        }

        return comparison

    def compute_semantic_energy(self, code: str) -> List[float]:
        """
        Compute semantic energy from logits for CodeBERT.

        For MLM models like CodeBERT, we compute energy by masking each token
        and extracting the logit value for the actual token.

        Args:
            code: Source code to analyze

        Returns:
            List of energy values (one per token)
        """
        encoding = self.tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encoding.input_ids[0].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        energies = []

        for i in range(len(input_ids)):
            # Skip special tokens
            if tokens[i] in ['<s>', '</s>', '<pad>']:
                continue

            # Create masked version
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits[0, i]

                # Get logit for the actual token
                original_token_id = input_ids[i].item()
                energy = -logits[original_token_id].item()  # Energy = -logit

                energies.append(energy)

        return energies

    def get_attention_weights(self, code: str) -> Optional[torch.Tensor]:
        """
        Get attention weights from CodeBERT.

        CodeBERT is based on RoBERTa, which has multi-head self-attention.

        Args:
            code: Source code to analyze

        Returns:
            Attention tensor of shape (num_layers, num_heads, seq_len, seq_len)
        """
        encoding = self.tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.roberta(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True
            )

            # Stack all attention layers
            # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(outputs.attentions)

            # Remove batch dimension
            # Shape: (num_layers, num_heads, seq_len, seq_len)
            attention_weights = attention_weights.squeeze(1)

        return attention_weights

    def compute_conformal_scores(self, code: str, calibration_data: Optional[List] = None) -> List[float]:
        """
        Compute conformal prediction scores for each token.

        For MLM models, we mask each token and compute score = 1 - P(token).

        Args:
            code: Source code to analyze
            calibration_data: Optional calibration set (not used in basic implementation)

        Returns:
            List of conformal scores (one per token)
        """
        encoding = self.tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids[0].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        scores = []

        for i in range(len(input_ids)):
            # Skip special tokens
            if tokens[i] in ['<s>', '</s>', '<pad>']:
                continue

            # Create masked version
            masked_input_ids = input_ids.clone()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.model(masked_input_ids.unsqueeze(0))
                logits = outputs.logits[0, i]
                probs = F.softmax(logits, dim=-1)

                # Conformal score = 1 - P(actual_token)
                original_token_id = input_ids[i].item()
                score = 1.0 - probs[original_token_id].item()

                scores.append(score)

        return scores


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("CodeBERT-based Logical Error Detection")
    print("="*60)

    detector = CodeBERTErrorDetector(sensitivity_factor=1.5)

    # Example buggy code
    buggy_code = """def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)"""

    # Example correct code
    correct_code = """def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)"""

    print("\n" + "="*60)
    print("BUGGY CODE ANALYSIS")
    print("="*60)
    buggy_results = detector.localize_errors(buggy_code)

    print("\nError Lines:")
    for line_error in buggy_results["line_errors"]:
        if line_error.is_error_line:
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    Error score: {line_error.error_score:.3f}")
            print(f"    Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")

    print("\n" + "="*60)
    print("CORRECT CODE ANALYSIS")
    print("="*60)
    correct_results = detector.localize_errors(correct_code)

    print("\nError Lines:")
    for line_error in correct_results["line_errors"]:
        if line_error.is_error_line:
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    Error score: {line_error.error_score:.3f}")
            print(f"    Anomalous tokens: {line_error.num_anomalous_tokens}/{line_error.num_tokens}")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Anomalous tokens - Buggy: {buggy_results['statistics']['anomalous_tokens']}, "
          f"Correct: {correct_results['statistics']['anomalous_tokens']}")
    print(f"Error lines - Buggy: {buggy_results['statistics']['error_lines']}, "
          f"Correct: {correct_results['statistics']['error_lines']}")
