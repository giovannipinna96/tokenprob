#!/usr/bin/env python3
"""
CodeT5 Token Validator

This module uses CodeT5 (Salesforce) to validate individual tokens in generated code.
The key idea is to mask each token and ask CodeT5 to predict what should be there,
then compare the probability of the original token vs alternatives.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Dict, Optional
import re


class CodeT5Validator:
    """
    Validator that uses CodeT5 to check if tokens are "correct" by masking
    and asking the model to predict alternatives.
    """

    def __init__(self, model_name: str = "Salesforce/codet5-base", device: str = "auto"):
        """
        Initialize CodeT5 validator.

        Args:
            model_name: HuggingFace model identifier (codet5-base or codet5p-220m)
            device: Device to load model on
        """
        print(f"Loading CodeT5 validator: {model_name}")
        self.model_name = model_name
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map=device
        )
        self.model.eval()

    def _mask_token_at_position(self, code: str, token: str, position: int) -> str:
        """
        Create a masked version of the code by replacing the token at given position
        with CodeT5's mask token <extra_id_0>.

        Args:
            code: Original code string
            token: Token to mask
            position: Position of token in the token sequence

        Returns:
            Masked code string
        """
        # Tokenize the code to get tokens
        tokens = self.tokenizer.tokenize(code)

        # Check if position is valid
        if position >= len(tokens):
            # If position is out of bounds, try to find the token in the code
            # and replace it directly
            return code.replace(token, "<extra_id_0>", 1)

        # Replace token at position with mask
        tokens[position] = "<extra_id_0>"

        # Reconstruct code
        masked_code = self.tokenizer.convert_tokens_to_string(tokens)

        return masked_code

    def validate_token(self, code: str, token: str, position: int) -> Dict[str, any]:
        """
        Validate a single token by masking it and asking CodeT5 what should be there.

        Args:
            code: Full code context
            token: Token to validate
            position: Position of token in sequence

        Returns:
            Dictionary with:
                - validation_score: Probability that CodeT5 assigns to this token (0-1)
                - alternatives: List of (token, probability) tuples for top alternatives
                - predicted_token: Top prediction from CodeT5
        """
        # Mask the token
        masked_code = self._mask_token_at_position(code, token, position)

        # Encode masked code
        inputs = self.tokenizer(
            masked_code,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.model.device)

        # Generate with scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=5,
                num_return_sequences=5,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Get predictions
        predictions = []
        for i in range(min(5, len(outputs.sequences))):
            pred_tokens = outputs.sequences[i]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)

            # Calculate approximate probability
            # Note: This is a simplified approach; exact probability would require
            # computing log likelihood of the specific token
            predictions.append(pred_text.strip())

        # Try to calculate actual probability of the original token
        # by computing loss when predicting it
        token_id = self.tokenizer.encode(token, add_special_tokens=False)
        if len(token_id) > 0:
            labels = torch.tensor([token_id]).to(self.model.device)

            with torch.no_grad():
                loss_output = self.model(
                    input_ids=inputs["input_ids"],
                    labels=labels
                )
                loss = loss_output.loss.item()
                # Convert loss to approximate probability
                validation_score = torch.exp(-torch.tensor(loss)).item()
        else:
            validation_score = 0.0

        # Get top prediction
        predicted_token = predictions[0] if predictions else ""

        # Create alternatives list
        # For simplicity, assign decreasing probabilities to predictions
        alternatives = []
        for i, pred in enumerate(predictions[:5]):
            prob = validation_score * (0.9 ** i) if i == 0 else (validation_score * 0.5 * (0.8 ** (i-1)))
            alternatives.append((pred, prob))

        return {
            "validation_score": validation_score,
            "alternatives": alternatives,
            "predicted_token": predicted_token,
            "matches_prediction": token.strip() == predicted_token
        }

    def validate_code_sequence(self, code: str, tokens: List[str]) -> List[Dict[str, any]]:
        """
        Validate a sequence of tokens in code.

        Args:
            code: Full code string
            tokens: List of tokens to validate

        Returns:
            List of validation results, one per token
        """
        results = []

        for i, token in enumerate(tokens):
            result = self.validate_token(code, token, i)
            result["position"] = i
            result["token"] = token
            results.append(result)

        return results


if __name__ == "__main__":
    # Quick test
    validator = CodeT5Validator()

    code = """def somma(n):
    s = 0
    for i in range(n):
        s += i
    return s"""

    # Test validating the "range" token
    result = validator.validate_token(code, "range", 10)

    print(f"Validation score for 'range': {result['validation_score']:.3f}")
    print(f"Top prediction: {result['predicted_token']}")
    print(f"Matches: {result['matches_prediction']}")
    print(f"Alternatives: {result['alternatives']}")
