#!/usr/bin/env python3
"""
StarCoder2-7B Error Detector

Implementation using BigCode's StarCoder2-7B model with causal language modeling
for logical error detection.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base_detector import BaseErrorDetector, DetectorMetadata


class StarCoder2ErrorDetector(BaseErrorDetector):
    """
    Error detector using StarCoder2-7B (Causal LM approach).

    StarCoder2 is a state-of-the-art code generation model from BigCode.
    It uses autoregressive prediction to compute token probabilities.
    """

    def __init__(self,
                 model_name: str = "bigcode/starcoder2-7b",
                 device: str = "auto",
                 sensitivity_factor: float = 1.5):
        """
        Initialize the StarCoder2 error detector.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cpu", "cuda")
            sensitivity_factor: k parameter for threshold (τ = μ - k×σ)
        """
        super().__init__(sensitivity_factor)

        print(f"Loading StarCoder2 model: {model_name}")
        self.model_name = model_name
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (full precision BF16, no quantization)
        print("  Loading model in BF16 (full precision)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Full BF16
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

        print(f"  Model loaded successfully on device: {self.model.device}")

    def compute_token_log_probabilities(self, code: str) -> List[Tuple[str, int, float, int]]:
        """
        Compute log probabilities using causal (autoregressive) approach.

        For each token at position i, compute:
        P(token[i] | token[0], token[1], ..., token[i-1])

        Args:
            code: Source code to analyze

        Returns:
            List of (token_text, token_id, log_probability, char_position) tuples
        """
        # Tokenize the code
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding.input_ids.to(self.model.device)
        offset_mapping = encoding.offset_mapping[0] if "offset_mapping" in encoding else None

        token_data = []

        with torch.no_grad():
            # Single forward pass to get all logits
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute log probabilities for each token (skip first token)
            for i in range(1, len(input_ids[0])):
                token_id = input_ids[0][i].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                # Get logits from PREVIOUS position (autoregressive)
                prev_logits = logits[i-1]
                log_probs = F.log_softmax(prev_logits, dim=-1)
                token_log_prob = log_probs[token_id].item()

                # Get character position
                char_pos = offset_mapping[i][0].item() if offset_mapping is not None else 0

                token_data.append((token_text, token_id, token_log_prob, char_pos))

        return token_data

    def get_metadata(self) -> DetectorMetadata:
        """Return metadata about this detector."""
        return DetectorMetadata(
            model_name="StarCoder2-7B",
            model_type="causal",
            parameters="7B",
            approach="Autoregressive",
            year=2024,
            sequence_length=16384,
            special_features="State-of-the-art code generation, trained on The Stack"
        )

    # ========================================================================
    # ADVANCED METHODS IMPLEMENTATION
    # ========================================================================

    def compute_semantic_energy(self, code: str) -> List[float]:
        """
        Compute semantic energy from logits (pre-softmax values).

        Semantic energy provides better uncertainty estimation than probabilities.
        Energy = -logit of chosen token. Lower energy = higher confidence.

        Args:
            code: Source code to analyze

        Returns:
            List of energy values (one per token)
        """
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids = encoding.input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        energies = []
        for i in range(1, len(input_ids[0])):
            token_id = input_ids[0][i].item()
            # Energy = negative logit of chosen token
            energy = -logits[i-1][token_id].item()
            energies.append(energy)

        return energies

    def get_attention_weights(self, code: str) -> torch.Tensor:
        """
        Get attention weights from the model.

        Attention patterns reveal where the model focuses. Uniform attention
        (high entropy) indicates uncertainty.

        Args:
            code: Source code to analyze

        Returns:
            Attention weights: [num_layers, num_heads, seq_len, seq_len]
        """
        encoding = self.tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)

            if outputs.attentions is None:
                raise ValueError("Model does not support attention output")

            # Stack attention weights from all layers
            attention_weights = torch.stack(outputs.attentions)
            # Remove batch dimension: [layers, heads, seq, seq]
            attention_weights = attention_weights.squeeze(1)

        return attention_weights

    def compute_conformal_scores(self,
                                 code: str,
                                 calibration_data: Optional[List] = None) -> List[float]:
        """
        Compute conformal prediction scores.

        Returns inverse probabilities which can be used for conformal prediction
        with coverage guarantees.

        Args:
            code: Source code to analyze
            calibration_data: Not used for now (would be used for calibration)

        Returns:
            List of conformal scores (1 - probability)
        """
        encoding = self.tokenizer(code, return_tensors="pt")
        input_ids = encoding.input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0]

        scores = []
        for i in range(1, len(input_ids[0])):
            token_id = input_ids[0][i].item()
            probs = F.softmax(logits[i-1], dim=-1)
            # Conformal score = 1 - probability (higher = more uncertain)
            score = 1.0 - probs[token_id].item()
            scores.append(score)

        return scores


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("StarCoder2-7B Error Detection Test")
    print("="*60)

    detector = StarCoder2ErrorDetector(sensitivity_factor=1.5)

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
    found = False
    for line_error in correct_results["line_errors"]:
        if line_error.is_error_line:
            found = True
            print(f"  Line {line_error.line_number}: {line_error.line_content}")
            print(f"    Error score: {line_error.error_score:.3f}")

    if not found:
        print("  No error lines detected ✓")

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Anomalous tokens - Buggy: {buggy_results['statistics']['anomalous_tokens']}, "
          f"Correct: {correct_results['statistics']['anomalous_tokens']}")
    print(f"Error lines - Buggy: {buggy_results['statistics']['error_lines']}, "
          f"Correct: {correct_results['statistics']['error_lines']}")
