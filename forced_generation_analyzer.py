#!/usr/bin/env python3
"""
Forced Generation Analyzer for LLM Token Probability Analysis

This module implements a system that forces an LLM to generate specific target code
token by token while capturing the logits and uncertainty metrics for each forced token.

The key idea is:
1. Give LLM only the programming problem (not the solution)
2. Force the LLM to generate a pre-defined target code token by token
3. Capture logits/probabilities for each forced token to measure model confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass
from pathlib import Path

from LLM import QwenProbabilityAnalyzer, TokenAnalysis


@dataclass
class ForcedTokenAnalysis:
    """Data class to store analysis results for a single forced token"""
    token: str
    token_id: int
    position: int
    probability: float  # Probability the model assigned to this forced token
    logit: float       # Raw logit value for this forced token
    rank: int          # Rank among all possible tokens (1 = most likely)
    entropy: float     # Entropy of the full distribution at this position
    surprisal: float   # Surprisal (-log2(probability)) of the forced token
    perplexity: float  # Perplexity of the distribution
    confidence_score: float  # Combined confidence metric

    # Top alternatives the model considered
    top_k_alternatives: List[Tuple[int, float, str]]  # (token_id, prob, token_text)

    # Distribution statistics
    max_probability: float     # Highest probability in distribution
    probability_margin: float  # Difference between top-1 and our forced token
    distribution_entropy: float # Shannon entropy of the full distribution

    # CodeT5 validation metrics
    codet5_validation_score: Optional[float] = None  # Probability CodeT5 assigns to this token
    codet5_alternatives: Optional[List[Tuple[str, float]]] = None  # Top alternatives from CodeT5
    codet5_predicted_token: Optional[str] = None  # Top prediction from CodeT5
    codet5_matches: Optional[bool] = None  # Whether token matches CodeT5's top prediction
    # Nomic-embed-code validation metrics
    nomic_coherence_score: Optional[float] = None  # Semantic coherence score (0-1)
    nomic_similarity_drop: Optional[float] = None  # Similarity drop when removing token
    nomic_context_similarity: Optional[float] = None  # Similarity to context without token


@dataclass
class ForcedGenerationResult:
    """Complete result of forced generation analysis"""
    original_prompt: str
    target_code: str
    reconstructed_code: str
    model_name: str

    # Per-token analysis
    token_analyses: List[ForcedTokenAnalysis]

    # Overall statistics
    average_probability: float
    average_rank: float
    average_surprisal: float
    average_confidence: float
    total_tokens: int

    # Difficulty metrics
    high_uncertainty_tokens: int  # Number of tokens with prob < 0.1
    model_preferred_alternative: str  # What model would have generated instead


class ForcedGenerationAnalyzer:
    """
    Analyzer that forces LLM to generate specific target code while capturing
    the model's confidence/uncertainty about each forced token choice.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct", device: str = "auto"):
        """
        Initialize the forced generation analyzer.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cpu", "cuda", etc.)
        """
        print(f"Loading model for forced generation analysis: {model_name}")
        self.model_name = model_name
        self.device = device

        # Load tokenizer and model (reuse from base analyzer)
        self.base_analyzer = QwenProbabilityAnalyzer(model_name=model_name, device=device)
        self.tokenizer = self.base_analyzer.tokenizer
        self.model = self.base_analyzer.model

    def _prepare_prompt(self, problem_description: str) -> str:
        """
        Prepare the prompt for the model (without showing the target solution).
        Enhanced to explicitly request COMPLETE code implementation.

        Args:
            problem_description: Natural language description of the programming task

        Returns:
            Formatted prompt ready for the model
        """
        # Enhance the prompt to explicitly request complete code
        enhanced_problem = f"""{problem_description}

IMPORTANT: Write ONLY the Python code. No explanations, no comments, no markdown. Just pure code."""

        messages = [
            {"role": "system", "content": "You are a code generation assistant. Generate ONLY the requested Python code without any explanations, comments, or markdown formatting. Output pure Python code only."},
            {"role": "user", "content": enhanced_problem}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _tokenize_target_code(self, target_code: str) -> List[int]:
        """
        Tokenize the target code to get the sequence of token IDs we need to force.

        Args:
            target_code: The code we want to force the model to generate

        Returns:
            List of token IDs representing the target code
        """
        # Tokenize just the code (not including any chat template)
        tokens = self.tokenizer.encode(target_code, add_special_tokens=False)
        return tokens

    def _get_token_logits_and_metrics(self, input_ids: torch.Tensor, position: int) -> Dict[str, Any]:
        """
        Get model logits and calculate various metrics for a specific position.

        Args:
            input_ids: Current input sequence
            position: Position in sequence to analyze

        Returns:
            Dictionary with logits and computed metrics
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, position, :].float()  # Convert to float32 for compatibility

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate distribution metrics
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        max_prob = torch.max(probs).item()

        # Get top alternatives
        top_k = 10
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

        return {
            'logits': logits,
            'probabilities': probs,
            'entropy': entropy,
            'max_probability': max_prob,
            'top_alternatives': list(zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()))
        }

    def _analyze_forced_token(self,
                            token_id: int,
                            position: int,
                            logits: torch.Tensor,
                            probabilities: torch.Tensor,
                            distribution_metrics: Dict[str, Any]) -> ForcedTokenAnalysis:
        """
        Analyze a specific forced token given the model's distribution.

        Args:
            token_id: The token ID that was forced
            position: Position in the sequence
            logits: Raw model logits
            probabilities: Softmax probabilities
            distribution_metrics: Pre-computed distribution statistics

        Returns:
            Complete analysis for this forced token
        """
        # Token information
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

        # Metrics for the forced token
        token_prob = probabilities[token_id].item()
        token_logit = logits[token_id].item()
        surprisal = -np.log2(token_prob + 1e-10)

        # Get rank of forced token
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        rank = (sorted_indices == token_id).nonzero().item() + 1

        # Probability margin (difference between top-1 and our forced token)
        top1_prob = sorted_probs[0].item()
        prob_margin = top1_prob - token_prob

        # Confidence score (higher is better)
        # Combines probability and rank information
        confidence_score = token_prob * (1.0 / rank)

        # Get top alternatives with decoded text
        top_alternatives = []
        for alt_id, alt_prob in distribution_metrics['top_alternatives']:
            alt_text = self.tokenizer.decode([alt_id], skip_special_tokens=False)
            top_alternatives.append((int(alt_id), float(alt_prob), alt_text))

        return ForcedTokenAnalysis(
            token=token_text,
            token_id=token_id,
            position=position,
            probability=token_prob,
            logit=token_logit,
            rank=rank,
            entropy=distribution_metrics['entropy'],
            surprisal=surprisal,
            perplexity=2 ** distribution_metrics['entropy'],
            confidence_score=confidence_score,
            top_k_alternatives=top_alternatives,
            max_probability=distribution_metrics['max_probability'],
            probability_margin=prob_margin,
            distribution_entropy=distribution_metrics['entropy']
        )

    def force_generation_with_logits(self,
                                   problem_description: str,
                                   target_code: str,
                                   verbose: bool = True) -> ForcedGenerationResult:
        """
        Force the model to generate target code while capturing logits for each token.

        Args:
            problem_description: Natural language description of the programming problem
            target_code: The code we want to force the model to generate
            verbose: Whether to print progress information

        Returns:
            Complete analysis results with per-token logits and metrics
        """
        if verbose:
            print(f"Starting forced generation analysis...")
            print(f"Problem: {problem_description}")
            print(f"Target code length: {len(target_code)} chars")

        # Prepare inputs
        prompt = self._prepare_prompt(problem_description)
        target_tokens = self._tokenize_target_code(target_code)

        if verbose:
            print(f"Target tokenized into {len(target_tokens)} tokens")

        # Initialize generation
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        current_ids = prompt_ids.clone()

        token_analyses = []
        reconstructed_tokens = []

        # Process each target token
        for i, target_token_id in enumerate(target_tokens):
            if verbose and i % 10 == 0:
                print(f"Processing token {i+1}/{len(target_tokens)}")

            # Get model's distribution for current position
            distribution_metrics = self._get_token_logits_and_metrics(
                current_ids, position=-1  # Always analyze the last position
            )

            # Analyze the forced token
            analysis = self._analyze_forced_token(
                token_id=target_token_id,
                position=i,
                logits=distribution_metrics['logits'],
                probabilities=distribution_metrics['probabilities'],
                distribution_metrics=distribution_metrics
            )

            token_analyses.append(analysis)
            reconstructed_tokens.append(target_token_id)

            # Add the forced token to the sequence for next iteration
            next_token_tensor = torch.tensor([[target_token_id]], device=current_ids.device)
            current_ids = torch.cat([current_ids, next_token_tensor], dim=-1)

        # Reconstruct the code from forced tokens
        reconstructed_code = self.tokenizer.decode(reconstructed_tokens, skip_special_tokens=True)

        # Calculate overall statistics
        probabilities = [analysis.probability for analysis in token_analyses]
        ranks = [analysis.rank for analysis in token_analyses]
        surprisals = [analysis.surprisal for analysis in token_analyses]
        confidences = [analysis.confidence_score for analysis in token_analyses]

        # Count high uncertainty tokens (probability < 0.1)
        high_uncertainty_count = sum(1 for p in probabilities if p < 0.1)

        # Generate what the model would have preferred
        preferred_tokens = []
        for analysis in token_analyses:
            if analysis.top_k_alternatives:
                # Get the most likely alternative (top-1)
                preferred_token_id = analysis.top_k_alternatives[0][0]
                preferred_tokens.append(preferred_token_id)

        model_preferred = self.tokenizer.decode(preferred_tokens, skip_special_tokens=True) if preferred_tokens else ""

        result = ForcedGenerationResult(
            original_prompt=problem_description,
            target_code=target_code,
            reconstructed_code=reconstructed_code,
            model_name=self.model_name,
            token_analyses=token_analyses,
            average_probability=np.mean(probabilities),
            average_rank=np.mean(ranks),
            average_surprisal=np.mean(surprisals),
            average_confidence=np.mean(confidences),
            total_tokens=len(token_analyses),
            high_uncertainty_tokens=high_uncertainty_count,
            model_preferred_alternative=model_preferred
        )

        if verbose:
            print(f"\nForced generation completed!")
            print(f"Average probability: {result.average_probability:.3f}")
            print(f"Average rank: {result.average_rank:.1f}")
            print(f"High uncertainty tokens: {result.high_uncertainty_tokens}/{result.total_tokens}")
            print(f"Code reconstruction match: {target_code == reconstructed_code}")

        return result

    def save_analysis(self, result: ForcedGenerationResult, filepath: str):
        """
        Save the forced generation analysis to a JSON file.

        Args:
            result: Analysis result to save
            filepath: Path where to save the JSON file
        """
        # Convert result to dictionary for JSON serialization
        data = {
            "metadata": {
                "model_name": result.model_name,
                "original_prompt": result.original_prompt,
                "target_code": result.target_code,
                "reconstructed_code": result.reconstructed_code,
                "reconstruction_match": bool(result.target_code == result.reconstructed_code)
            },
            "summary_statistics": {
                "total_tokens": result.total_tokens,
                "average_probability": float(result.average_probability),
                "average_rank": float(result.average_rank),
                "average_surprisal": float(result.average_surprisal),
                "average_confidence": float(result.average_confidence),
                "high_uncertainty_tokens": result.high_uncertainty_tokens,
                "high_uncertainty_percentage": (result.high_uncertainty_tokens / result.total_tokens) * 100
            },
            "model_alternatives": {
                "what_model_would_generate": result.model_preferred_alternative
            },
            "per_token_analysis": []
        }

        # Add detailed per-token analysis
        for analysis in result.token_analyses:
            token_data = {
                "position": analysis.position,
                "token": analysis.token,
                "token_id": analysis.token_id,
                "probability": float(analysis.probability),
                "logit": float(analysis.logit),
                "rank": analysis.rank,
                "surprisal": float(analysis.surprisal),
                "confidence_score": float(analysis.confidence_score),
                "distribution_metrics": {
                    "entropy": float(analysis.entropy),
                    "perplexity": float(analysis.perplexity),
                    "max_probability": float(analysis.max_probability),
                    "probability_margin": float(analysis.probability_margin)
                },
                "top_alternatives": [
                    {
                        "token_id": int(token_id),
                        "probability": float(prob),
                        "token_text": token_text
                    }
                    for token_id, prob, token_text in analysis.top_k_alternatives[:5]  # Save top 5
                ],
                "codet5_validation": {
                    "validation_score": float(analysis.codet5_validation_score) if analysis.codet5_validation_score is not None else None,
                    "predicted_token": analysis.codet5_predicted_token,
                    "matches": bool(analysis.codet5_matches) if analysis.codet5_matches is not None else None,
                    "alternatives": [
                        {"token": alt_token, "probability": float(alt_prob)}
                        for alt_token, alt_prob in (analysis.codet5_alternatives[:3] if analysis.codet5_alternatives else [])
                    ]
                } if analysis.codet5_validation_score is not None else None,
                "nomic_validation": {
                    "coherence_score": float(analysis.nomic_coherence_score) if analysis.nomic_coherence_score is not None else None,
                    "similarity_drop": float(analysis.nomic_similarity_drop) if analysis.nomic_similarity_drop is not None else None,
                    "context_similarity": float(analysis.nomic_context_similarity) if analysis.nomic_context_similarity is not None else None
                } if analysis.nomic_coherence_score is not None else None
            }
            data["per_token_analysis"].append(token_data)

        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Forced generation analysis saved to: {filepath}")

    def compare_with_natural_generation(self,
                                      problem_description: str,
                                      target_code: str,
                                      max_tokens: int = 150) -> Dict[str, Any]:
        """
        Compare forced generation with natural generation to understand differences.

        Args:
            problem_description: The programming problem
            target_code: Code to force
            max_tokens: Max tokens for natural generation

        Returns:
            Comparison analysis
        """
        print("Running forced generation...")
        forced_result = self.force_generation_with_logits(problem_description, target_code, verbose=False)

        print("Running natural generation...")
        natural_text, natural_analyses = self.base_analyzer.generate_with_analysis(
            prompt=problem_description,
            max_new_tokens=max_tokens,
            temperature=0.1  # Low temperature for more deterministic comparison
        )

        return {
            "forced_generation": {
                "code": forced_result.reconstructed_code,
                "avg_probability": forced_result.average_probability,
                "avg_rank": forced_result.average_rank,
                "high_uncertainty_tokens": forced_result.high_uncertainty_tokens
            },
            "natural_generation": {
                "code": natural_text,
                "avg_probability": np.mean([a.probability for a in natural_analyses]),
                "avg_rank": np.mean([a.rank for a in natural_analyses]),
                "total_tokens": len(natural_analyses)
            },
            "comparison": {
                "codes_match": forced_result.reconstructed_code.strip() == natural_text.strip(),
                "probability_difference": forced_result.average_probability - np.mean([a.probability for a in natural_analyses]),
                "rank_difference": forced_result.average_rank - np.mean([a.rank for a in natural_analyses])
            }
        }


if __name__ == "__main__":
    # Quick test
    analyzer = ForcedGenerationAnalyzer()

    problem = "Write a Python function to calculate factorial using recursion."
    target = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

    result = analyzer.force_generation_with_logits(problem, target)
    analyzer.save_analysis(result, "test_forced_generation.json")