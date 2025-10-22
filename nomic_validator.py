#!/usr/bin/env python3
"""
Nomic-embed-code Token Validator

This module uses nomic-embed-code (Nomic AI) to validate individual tokens
in generated code by measuring semantic coherence through embedding similarity.

Unlike CodeT5 (which predicts tokens), Nomic-embed-code measures how semantically
coherent a token is within its context using code embeddings.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
import numpy as np


class NomicCodeValidator:
    """
    Validator that uses nomic-embed-code to check token semantic coherence
    by computing embedding similarity.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-code", device: str = "auto"):
        """
        Initialize Nomic-embed-code validator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        print(f"Loading Nomic validator: {model_name}")
        self.model_name = model_name
        self.device = device

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()

    def _last_token_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Last token pooling specific to nomic-embed-code.

        Args:
            hidden_states: Model hidden states
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        sequence_lengths = attention_mask.sum(dim=-1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def _compute_embedding(self, code: str) -> torch.Tensor:
        """
        Compute embedding for a code snippet.

        Args:
            code: Code string to embed

        Returns:
            Normalized embedding tensor
        """
        # Tokenize
        encoded = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.model.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use last token pooling (specific to nomic-embed-code)
            embeddings = self._last_token_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def _compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        return similarity.item()

    def _remove_token_at_position(self, code: str, position: int) -> str:
        """
        Remove token at given position to create context without that token.

        Args:
            code: Original code
            position: Token position to remove

        Returns:
            Code with token removed (replaced with placeholder)
        """
        # Tokenize to get tokens
        tokens = self.tokenizer.tokenize(code)

        if position >= len(tokens):
            return code

        # Remove token at position
        tokens_without = tokens[:position] + ['[MASK]'] + tokens[position+1:]

        # Reconstruct
        code_without = self.tokenizer.convert_tokens_to_string(tokens_without)

        return code_without

    def validate_token(self, code: str, token: str, position: int) -> Dict[str, any]:
        """
        Validate a token by measuring its semantic coherence in context.

        Args:
            code: Full code context
            token: Token to validate
            position: Position of token in sequence

        Returns:
            Dictionary with:
                - coherence_score: How semantically coherent the token is (0-1)
                - similarity_drop: Similarity drop when removing token
                - context_similarity: Similarity between full code and context
        """
        try:
            # 1. Compute embedding of full code (with token)
            full_embedding = self._compute_embedding(code)

            # 2. Compute embedding of code without this token
            code_without_token = self._remove_token_at_position(code, position)
            context_embedding = self._compute_embedding(code_without_token)

            # 3. Compute similarity between full code and context
            context_similarity = self._compute_similarity(full_embedding, context_embedding)

            # 4. Coherence score: high similarity = token doesn't add much (potentially wrong)
            #    low similarity = token is important (potentially correct)
            # We invert it: coherence = 1 - similarity_to_context_without_token
            # Higher coherence = token is more necessary/correct
            coherence_score = 1.0 - abs(context_similarity)

            # 5. Similarity drop when removing token
            similarity_drop = 1.0 - context_similarity

            # 6. For alternative suggestions, we could embed variations
            # but for simplicity, we focus on coherence
            alternatives = []  # Could be extended with actual alternatives

            return {
                "coherence_score": coherence_score,
                "similarity_drop": similarity_drop,
                "context_similarity": context_similarity,
                "alternatives": alternatives
            }

        except Exception as e:
            print(f"Warning: Nomic validation failed for token at position {position}: {e}")
            return {
                "coherence_score": 0.5,  # Neutral score on error
                "similarity_drop": 0.0,
                "context_similarity": 0.5,
                "alternatives": []
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
    validator = NomicCodeValidator()

    code = """def factorial(n):
    if n < 0:
        raise ValueError
    return 1 if n == 0 else n * factorial(n - 1)"""

    # Test validating the "factorial" token
    result = validator.validate_token(code, "factorial", 5)

    print(f"Coherence score for 'factorial': {result['coherence_score']:.3f}")
    print(f"Similarity drop: {result['similarity_drop']:.3f}")
    print(f"Context similarity: {result['context_similarity']:.3f}")
