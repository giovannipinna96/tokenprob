#!/usr/bin/env python3
"""
CodeT5+ 2B Error Detector

Implementation using Salesforce's CodeT5+ 2B model with encoder-decoder
architecture for logical error detection.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .base_detector import BaseErrorDetector, DetectorMetadata


class CodeT5ErrorDetector(BaseErrorDetector):
    """
    Error detector using CodeT5+ 2B (Encoder-Decoder with MLM capability).

    CodeT5+ is a powerful code understanding and generation model from Salesforce.
    We use it in a masked prediction mode similar to CodeBERT.
    """

    def __init__(self,
                 model_name: str = "Salesforce/codet5p-2b",
                 device: str = "auto",
                 sensitivity_factor: float = 1.5):
        """
        Initialize the CodeT5+ error detector.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cpu", "cuda")
            sensitivity_factor: k parameter for threshold (τ = μ - k×σ)
        """
        super().__init__(sensitivity_factor)

        print(f"Loading CodeT5+ model: {model_name}")
        self.model_name = model_name

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model (full precision FP16, no quantization)
        print("  Loading model in FP16 (full precision)...")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Full FP16
            device_map=device
        )
        self.model.eval()

        print(f"  Model loaded successfully on device: {self.device}")

    def compute_token_log_probabilities(self, code: str) -> List[Tuple[str, int, float, int]]:
        """
        Compute log probabilities using encoder-decoder architecture.

        OPTIMIZED VERSION: Single forward pass with teacher forcing.

        For CodeT5+ (encoder-decoder):
        - Encoder processes the entire code once (full context)
        - Decoder processes all tokens in one pass (teacher forcing)
        - Extract log probabilities for each position efficiently

        Args:
            code: Source code to analyze

        Returns:
            List of (token_text, token_id, log_probability, char_position) tuples
        """
        # Tokenize the code
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=512,
            truncation=True
        )

        input_ids = encoding.input_ids[0]
        offset_mapping = encoding.offset_mapping[0] if "offset_mapping" in encoding else None

        # Get individual tokens
        tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

        token_data = []

        # FIX: Single forward pass with teacher forcing
        # Encoder sees full code, decoder sees shifted code (for autoregressive prediction)
        encoder_input = input_ids.unsqueeze(0).to(self.model.device)

        # Decoder input: shifted right by 1 (prepend PAD/BOS token)
        decoder_input_ids = input_ids[:-1].unsqueeze(0).to(self.model.device)
        # Prepend PAD token
        pad_token = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.model.device)
        decoder_input_ids = torch.cat([pad_token, decoder_input_ids], dim=1)

        with torch.no_grad():
            # Single forward pass for all tokens!
            outputs = self.model(
                input_ids=encoder_input,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

            # Get logits for all positions: [batch=1, seq_len, vocab_size]
            all_logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute log probabilities for all tokens at once
            all_log_probs = F.log_softmax(all_logits, dim=-1)

        # Extract log probability for each actual token
        for i in range(len(input_ids)):
            # Skip special tokens
            if input_ids[i].item() in [self.tokenizer.pad_token_id,
                                       self.tokenizer.eos_token_id,
                                       self.tokenizer.bos_token_id]:
                continue

            original_token_id = input_ids[i].item()

            # Position i in decoder predicts token i (due to right-shift)
            if i < len(all_log_probs):
                token_log_prob = all_log_probs[i, original_token_id].item()
            else:
                # Safety check
                continue

            # Get character position
            char_pos = offset_mapping[i][0].item() if offset_mapping is not None else 0

            token_data.append((tokens[i], original_token_id, token_log_prob, char_pos))

        return token_data

    def compute_semantic_energy(self, code: str) -> List[float]:
        """
        Compute semantic energy from logits for CodeT5+.

        OPTIMIZED VERSION: Single forward pass.

        Energy = -logit(token), following Farquhar et al., NeurIPS 2024.
        Higher energy indicates higher model uncertainty.

        For encoder-decoder:
        - Encoder processes full code (complete context)
        - Decoder processes all tokens in one pass (teacher forcing)
        - Energy computed from pre-softmax logits

        Args:
            code: Source code to analyze

        Returns:
            List of energy values (one per token)
        """
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        input_ids = encoding.input_ids[0]
        energies = []

        # FIX: Single forward pass with teacher forcing
        encoder_input = input_ids.unsqueeze(0).to(self.model.device)

        # Decoder input: shifted right by 1
        decoder_input_ids = input_ids[:-1].unsqueeze(0).to(self.model.device)
        pad_token = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.model.device)
        decoder_input_ids = torch.cat([pad_token, decoder_input_ids], dim=1)

        with torch.no_grad():
            # Single forward pass
            outputs = self.model(
                input_ids=encoder_input,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

            # Get logits for all positions
            all_logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Extract energy for each token
        for i in range(len(input_ids)):
            # Skip special tokens
            if input_ids[i].item() in [self.tokenizer.pad_token_id,
                                       self.tokenizer.eos_token_id,
                                       self.tokenizer.bos_token_id]:
                continue

            token_id = input_ids[i].item()

            # Get the logit value for the actual token i
            if i < len(all_logits):
                energy = -all_logits[i, token_id].item()  # Energy = -logit
                energies.append(energy)

        return energies

    def get_attention_weights(self, code: str) -> Optional[torch.Tensor]:
        """
        Get attention weights from CodeT5+ encoder.

        For encoder-decoder models, we extract encoder self-attention weights.

        Args:
            code: Source code to analyze

        Returns:
            Attention tensor of shape (num_layers, num_heads, seq_len, seq_len)
        """
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        input_ids = encoding.input_ids.to(self.model.device)

        with torch.no_grad():
            # Get encoder outputs with attention
            outputs = self.model.encoder(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True
            )

            # Stack all encoder attention layers
            # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attention_weights = torch.stack(outputs.attentions)

            # Remove batch dimension
            # Shape: (num_layers, num_heads, seq_len, seq_len)
            attention_weights = attention_weights.squeeze(1)

        return attention_weights

    def compute_conformal_scores(self, code: str, calibration_data: Optional[List] = None) -> List[float]:
        """
        Compute conformal prediction scores for each token.

        OPTIMIZED VERSION: Single forward pass.

        Score = 1 - P(token), following Quach et al., ICLR 2024.
        Higher scores indicate higher uncertainty (larger prediction set).

        For encoder-decoder:
        - Encoder processes full code (complete context)
        - Decoder processes all tokens in one pass (teacher forcing)
        - Score computed from softmax probabilities

        Args:
            code: Source code to analyze
            calibration_data: Optional calibration set (not used in basic implementation)

        Returns:
            List of conformal scores (one per token)
        """
        encoding = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        input_ids = encoding.input_ids[0]
        scores = []

        # FIX: Single forward pass with teacher forcing
        encoder_input = input_ids.unsqueeze(0).to(self.model.device)

        # Decoder input: shifted right by 1
        decoder_input_ids = input_ids[:-1].unsqueeze(0).to(self.model.device)
        pad_token = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.model.device)
        decoder_input_ids = torch.cat([pad_token, decoder_input_ids], dim=1)

        with torch.no_grad():
            # Single forward pass
            outputs = self.model(
                input_ids=encoder_input,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )

            # Get logits and probabilities for all positions
            all_logits = outputs.logits[0]  # [seq_len, vocab_size]
            all_probs = F.softmax(all_logits, dim=-1)

        # Extract conformal score for each token
        for i in range(len(input_ids)):
            # Skip special tokens
            if input_ids[i].item() in [self.tokenizer.pad_token_id,
                                       self.tokenizer.eos_token_id,
                                       self.tokenizer.bos_token_id]:
                continue

            token_id = input_ids[i].item()

            # Conformal score = 1 - P(actual_token)
            if i < len(all_probs):
                score = 1.0 - all_probs[i, token_id].item()
                scores.append(score)

        return scores

    def get_metadata(self) -> DetectorMetadata:
        """Return metadata about this detector."""
        return DetectorMetadata(
            model_name="CodeT5+ 2B",
            model_type="encoder-decoder",
            parameters="2B",
            approach="Encoder-Decoder with Seq2Seq",
            year=2023,
            sequence_length=512,
            special_features="Unified encoder-decoder for code understanding and generation"
        )


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("CodeT5+ 2B Error Detection Test")
    print("="*60)

    detector = CodeT5ErrorDetector(sensitivity_factor=1.5)

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
