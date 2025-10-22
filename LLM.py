import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
import math
from dataclasses import dataclass

# LM-Polygraph integration (optional)
try:
    from lm_polygraph.estimators import *
    from lm_polygraph.utils.model import WhiteboxModel
    LM_POLYGRAPH_AVAILABLE = True
except ImportError:
    LM_POLYGRAPH_AVAILABLE = False
    print("LM-Polygraph not available. Advanced uncertainty methods will be skipped.")


@dataclass
class TokenAnalysis:
    """Data class to store analysis results for a single token"""
    token: str
    token_id: int
    position: int
    probability: float
    logit: float
    rank: int  # Rank among all possible tokens (1 = most likely)
    perplexity: float
    entropy: float
    surprisal: float
    top_k_probs: List[Tuple[int, float]]  # Top K token probabilities
    # New advanced metrics
    max_probability: float  # Maximum probability in the distribution
    probability_margin: float  # Difference between top-1 and top-2 probabilities
    shannon_entropy: float  # Shannon entropy of the distribution
    local_perplexity: float  # Local perplexity for this token
    sequence_improbability: float  # Cumulative sequence improbability
    confidence_score: float  # Overall confidence score
    # LM-Polygraph metrics (optional)
    lm_polygraph_metrics: Optional[Dict[str, float]] = None
    # CodeT5 validation metrics
    codet5_validation_score: Optional[float] = None  # Probability CodeT5 assigns to this token
    codet5_alternatives: Optional[List[Tuple[str, float]]] = None  # Top alternatives from CodeT5
    codet5_predicted_token: Optional[str] = None  # Top prediction from CodeT5
    codet5_matches: Optional[bool] = None  # Whether token matches CodeT5's top prediction
    # Nomic-embed-code validation metrics
    nomic_coherence_score: Optional[float] = None  # Semantic coherence score (0-1)
    nomic_similarity_drop: Optional[float] = None  # Similarity drop when removing token
    nomic_context_similarity: Optional[float] = None  # Similarity to context without token
    # LecPrompt logical error detection metrics
    is_anomalous: Optional[bool] = None  # Whether token is flagged as potential error (LecPrompt)
    statistical_score: Optional[float] = None  # Deviation from mean in std devs (LecPrompt)
    error_likelihood: Optional[float] = None  # Normalized error probability 0-1 (LecPrompt)
    # Semantic Energy (pre-softmax logits) - Method 2 from METHODS_OVERVIEW.md
    semantic_energy: Optional[float] = None  # Energy = -logit(token), higher = more uncertain
    # Conformal Prediction - Method 3 from METHODS_OVERVIEW.md
    conformal_score: Optional[float] = None  # Conformal score = 1 - P(token), higher = more uncertain
    # Attention Anomaly Detection - Method 4 from METHODS_OVERVIEW.md
    attention_entropy: Optional[float] = None  # Entropy of attention distribution for this token
    attention_self_attention: Optional[float] = None  # Self-attention weight (attention to itself)
    attention_variance: Optional[float] = None  # Variance of attention weights
    attention_anomaly_score: Optional[float] = None  # Combined attention anomaly score (0-1)
    # Aggregated suspicion score - Combined metric for error detection (0-100)
    suspicion_score: Optional[float] = None  # Higher score = more likely to be erroneous


class QwenProbabilityAnalyzer:
    """
    Analyzer for Qwen 2.5 Coder 7B Instruct model that captures token generation
    probabilities and computes various information theory metrics.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct", device: str = "auto"):
        """
        Initialize the analyzer with the specified model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cpu", "cuda", etc.)
        """
        print(f"Loading model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model for token probability analysis
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.generation_history: List[TokenAnalysis] = []
        self.sequence_log_prob: float = 0.0  # Track cumulative log probability

        # Initialize LM-Polygraph uncertainty estimators if available
        self.lm_polygraph_estimators = {}
        if LM_POLYGRAPH_AVAILABLE:
            try:
                # Create whitebox model wrapper for LM-Polygraph
                self.whitebox_model = WhiteboxModel(self.model, self.tokenizer)

                # Initialize various uncertainty estimators
                self.lm_polygraph_estimators = {
                    'max_probability': MaximumSequenceProbability(),
                    'perplexity': Perplexity(),
                    'mean_entropy': MeanTokenEntropy(),
                    'mean_log_probability': MeanTokenLogProbability(),
                    'semantic_entropy': SemanticEntropy(),
                }
                print(f"LM-Polygraph estimators initialized: {list(self.lm_polygraph_estimators.keys())}")
            except Exception as e:
                print(f"Warning: Could not initialize LM-Polygraph estimators: {e}")
                self.lm_polygraph_estimators = {}
        
    def _calculate_information_metrics(self, logits: torch.Tensor, selected_token_id: int) -> Dict[str, float]:
        """
        Calculate information theory metrics for the given logits.
        
        Args:
            logits: Raw model logits for vocabulary
            selected_token_id: The actually selected token ID
            
        Returns:
            Dictionary containing various metrics
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of selected token
        selected_prob = probs[selected_token_id].item()
        
        # Calculate entropy (information content of the distribution)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        
        # Calculate surprisal (negative log probability of selected token)
        surprisal = -math.log2(selected_prob + 1e-10)
        
        # Calculate perplexity (2^entropy, measure of uncertainty)
        perplexity = 2 ** entropy
        
        # Get rank of selected token
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        rank = (sorted_indices == selected_token_id).nonzero().item() + 1
        
        return {
            "probability": selected_prob,
            "entropy": entropy,
            "surprisal": surprisal,
            "perplexity": perplexity,
            "rank": rank
        }
    
    def _get_top_k_probabilities(self, logits: torch.Tensor, k: int = 1000) -> List[Tuple[int, float]]:
        """
        Get top K token probabilities from logits.
        
        Args:
            logits: Raw model logits
            k: Number of top tokens to return
            
        Returns:
            List of (token_id, probability) tuples sorted by probability (descending)
        """
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(k, len(probs)))
        
        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    def _calculate_advanced_metrics(self, logits: torch.Tensor, selected_token_id: int, position: int) -> Dict[str, float]:
        """
        Calculate advanced uncertainty quantification metrics.

        Args:
            logits: Raw model logits for vocabulary
            selected_token_id: The actually selected token ID
            position: Position in the sequence

        Returns:
            Dictionary containing advanced metrics
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # 1. Maximum predicted probability (max(p))
        max_prob = torch.max(probs).item()

        # 2. Get top-2 probabilities for margin calculation
        top2_probs, top2_indices = torch.topk(probs, k=2)
        top1_prob = top2_probs[0].item()
        top2_prob = top2_probs[1].item() if len(top2_probs) > 1 else 0.0

        # 3. Probability margin (top1-top2)
        prob_margin = top1_prob - top2_prob

        # 4. Shannon entropy of the distribution
        shannon_entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()

        # 5. Local perplexity (2^entropy)
        local_perplexity = 2 ** shannon_entropy

        # 6. Selected token probability
        selected_prob = probs[selected_token_id].item()

        # 7. Update sequence log probability and calculate improbability
        self.sequence_log_prob += math.log(selected_prob + 1e-10)
        sequence_prob = math.exp(self.sequence_log_prob)
        sequence_improbability = 1.0 - sequence_prob

        # 8. Confidence score (composite metric)
        # Combine multiple factors: high probability, low entropy, high margin
        confidence_score = (selected_prob * prob_margin) / (shannon_entropy + 1e-10)

        # 9. Surprisal of selected token
        surprisal = -math.log2(selected_prob + 1e-10)

        # 10. Rank of selected token
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        rank = (sorted_indices == selected_token_id).nonzero().item() + 1

        # 11. Semantic Energy (Method 2 from METHODS_OVERVIEW.md)
        # Energy = -logit(token), higher energy = more uncertain
        # Based on: Farquhar et al., "Detecting Hallucinations in LLMs via Semantic Entropy", NeurIPS 2024
        selected_logit = logits[selected_token_id].item()
        semantic_energy = -selected_logit

        # 12. Conformal Prediction Score (Method 3 from METHODS_OVERVIEW.md)
        # Conformal score = 1 - P(token), higher = larger prediction set needed
        # Based on: Quach et al., "Conformal Language Modeling", ICLR 2024
        conformal_score = 1.0 - selected_prob

        return {
            "max_probability": max_prob,
            "probability_margin": prob_margin,
            "shannon_entropy": shannon_entropy,
            "local_perplexity": local_perplexity,
            "sequence_improbability": sequence_improbability,
            "confidence_score": confidence_score,
            "selected_probability": selected_prob,
            "surprisal": surprisal,
            "rank": rank,
            "semantic_energy": semantic_energy,
            "conformal_score": conformal_score
        }

    def _calculate_suspicion_score(self,
                                   rank: int,
                                   surprisal: float,
                                   entropy: float,
                                   probability_margin: float,
                                   codet5_matches: Optional[bool] = None,
                                   nomic_coherence_score: Optional[float] = None) -> float:
        """
        Calculate an aggregated suspicion score for token error detection (0-100).
        Higher score indicates higher likelihood of being erroneous.

        Based on empirical analysis of buggy vs correct code:
        - RANK: Most discriminative metric (buggy: 2-60, correct: ~1)
        - SURPRISAL: Strong indicator (buggy: 1.5-2.2, correct: 0.3-0.9)
        - ENTROPY: Uncertainty measure (buggy: higher, correct: lower)
        - PROBABILITY_MARGIN: Small margin indicates uncertainty
        - CODET5_VALIDATION: External validator disagreement
        - NOMIC_COHERENCE: Semantic coherence check

        Args:
            rank: Token rank in probability distribution
            surprisal: Token surprisal (-log2(probability))
            entropy: Distribution entropy
            probability_margin: Difference between top-1 and top-2 probabilities
            codet5_matches: Whether CodeT5 validator agrees (optional)
            nomic_coherence_score: Semantic coherence score 0-1 (optional)

        Returns:
            Suspicion score (0-100), where higher = more suspicious
        """
        score = 0.0

        # 1. RANK (weight 30%) - Most discriminative metric
        if rank > 10:
            score += 30.0
        elif rank > 3:
            score += 20.0
        elif rank > 1:
            score += 10.0

        # 2. SURPRISAL (weight 25%) - Strong uncertainty indicator
        if surprisal > 2.0:
            score += 25.0
        elif surprisal > 1.0:
            score += 15.0
        elif surprisal > 0.5:
            score += 8.0

        # 3. ENTROPY (weight 15%) - Distribution uncertainty
        if entropy > 0.8:
            score += 15.0
        elif entropy > 0.5:
            score += 10.0

        # 4. PROBABILITY MARGIN (weight 15%) - Competition between alternatives
        if probability_margin < 0.2:
            score += 15.0
        elif probability_margin < 0.4:
            score += 10.0
        elif probability_margin < 0.6:
            score += 5.0

        # 5. CODET5 VALIDATION (weight 10%) - External validator
        if codet5_matches is not None and not codet5_matches:
            score += 10.0

        # 6. NOMIC COHERENCE (weight 5%) - Semantic coherence
        if nomic_coherence_score is not None:
            if nomic_coherence_score < 0.5:
                score += 5.0
            elif nomic_coherence_score < 0.7:
                score += 3.0

        return min(score, 100.0)  # Cap at 100

    def _calculate_lm_polygraph_metrics(self, input_text: str, generated_text: str) -> Dict[str, float]:
        """
        Calculate LM-Polygraph uncertainty metrics for the generated sequence.

        Args:
            input_text: Original input prompt
            generated_text: Generated text to analyze

        Returns:
            Dictionary with LM-Polygraph uncertainty estimates
        """
        if not self.lm_polygraph_estimators:
            return {}

        try:
            # Prepare input for LM-Polygraph
            full_text = input_text + generated_text

            # Calculate various uncertainty estimates
            polygraph_metrics = {}

            for estimator_name, estimator in self.lm_polygraph_estimators.items():
                try:
                    # Note: The exact API might differ based on LM-Polygraph version
                    # This is a general implementation that may need adjustment
                    uncertainty_score = estimator.estimate(self.whitebox_model, [full_text])
                    polygraph_metrics[f"polygraph_{estimator_name}"] = float(uncertainty_score[0])
                except Exception as e:
                    print(f"Warning: Could not calculate {estimator_name}: {e}")
                    polygraph_metrics[f"polygraph_{estimator_name}"] = 0.0

            return polygraph_metrics

        except Exception as e:
            print(f"Error in LM-Polygraph calculation: {e}")
            return {}
    
    def generate_with_analysis(self, 
                             prompt: str, 
                             max_new_tokens: int = 100,
                             temperature: float = 0.7,
                             top_p: float = 0.9,
                             do_sample: bool = True) -> Tuple[str, List[TokenAnalysis]]:
        """
        Generate text while capturing detailed probability analysis for each token.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Tuple of (generated_text, list_of_token_analyses)
        """
        self.generation_history = []
        self.sequence_log_prob = 0.0  # Reset for new generation
        
        # Prepare input
        messages = [
            {"role": "system", "content": "You are a code generation assistant. Generate ONLY the requested Python code without any explanations, comments, or markdown formatting. Output pure Python code only."},
            {"role": "user", "content": f"{prompt}\n\nIMPORTANT: Write ONLY the Python code. No explanations, no comments, no markdown. Just pure code."}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Initialize generation
        current_input_ids = inputs.input_ids
        
        print("Starting generation with probability analysis...")
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(current_input_ids)
                logits = outputs.logits[0, -1, :]  # Get logits for next token
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from distribution
                    next_token_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                else:
                    # Greedy decoding
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                
                next_token_id = next_token_id.item()
                
                # Calculate basic metrics for this token
                metrics = self._calculate_information_metrics(logits, next_token_id)
                # Calculate advanced metrics
                advanced_metrics = self._calculate_advanced_metrics(logits, next_token_id, step)
                top_k_probs = self._get_top_k_probabilities(logits, k=1000)
                
                # Decode token
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)

                # Calculate suspicion score
                suspicion_score = self._calculate_suspicion_score(
                    rank=metrics["rank"],
                    surprisal=metrics["surprisal"],
                    entropy=metrics["entropy"],
                    probability_margin=advanced_metrics["probability_margin"]
                )

                # Create analysis object with all metrics
                analysis = TokenAnalysis(
                    token=token_text,
                    token_id=next_token_id,
                    position=step,
                    probability=metrics["probability"],
                    logit=logits[next_token_id].item(),
                    rank=metrics["rank"],
                    perplexity=metrics["perplexity"],
                    entropy=metrics["entropy"],
                    surprisal=metrics["surprisal"],
                    top_k_probs=top_k_probs,
                    # Advanced metrics
                    max_probability=advanced_metrics["max_probability"],
                    probability_margin=advanced_metrics["probability_margin"],
                    shannon_entropy=advanced_metrics["shannon_entropy"],
                    local_perplexity=advanced_metrics["local_perplexity"],
                    sequence_improbability=advanced_metrics["sequence_improbability"],
                    confidence_score=advanced_metrics["confidence_score"],
                    # LM-Polygraph metrics will be calculated at the end
                    lm_polygraph_metrics=None,
                    # Advanced error detection methods
                    semantic_energy=advanced_metrics["semantic_energy"],
                    conformal_score=advanced_metrics["conformal_score"],
                    # Aggregated suspicion score
                    suspicion_score=suspicion_score
                )
                
                self.generation_history.append(analysis)
                
                # Update input for next iteration
                current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token_id]]).to(self.model.device)], dim=-1)
                
                # Check for end of generation
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                    
                # Print progress
                if step % 10 == 0:
                    print(f"Generated {step} tokens...")
        
        # Generate the complete output text
        generated_ids = current_input_ids[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Generation complete! Generated {len(self.generation_history)} tokens.")

        # Calculate LM-Polygraph metrics for the complete sequence if available
        if self.lm_polygraph_estimators and generated_text:
            print("Calculating LM-Polygraph uncertainty metrics...")
            polygraph_metrics = self._calculate_lm_polygraph_metrics(text, generated_text)

            # Add LM-Polygraph metrics to the last token (as sequence-level metrics)
            if self.generation_history and polygraph_metrics:
                self.generation_history[-1].lm_polygraph_metrics = polygraph_metrics

        return generated_text, self.generation_history

    def generate_with_output_scores(self,
                                   prompt: str,
                                   max_new_tokens: int = 100,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9,
                                   do_sample: bool = True) -> Tuple[str, List[TokenAnalysis]]:
        """
        Generate text using HuggingFace's native .generate() with output_scores=True.
        This is an alternative to generate_with_analysis() that uses the built-in
        output_scores parameter instead of a custom generation loop.

        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding

        Returns:
            Tuple of (generated_text, list_of_token_analyses)
        """
        self.generation_history = []
        self.sequence_log_prob = 0.0  # Reset for new generation

        # Prepare input
        messages = [
            {"role": "system", "content": "You are a code generation assistant. Generate ONLY the requested Python code without any explanations, comments, or markdown formatting. Output pure Python code only."},
            {"role": "user", "content": f"{prompt}\n\nIMPORTANT: Write ONLY the Python code. No explanations, no comments, no markdown. Just pure code."}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        print("Starting generation with output_scores=True...")

        # Generate using HuggingFace's native method with output_scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract generated tokens
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Process scores (logits) for each generated token
        # outputs.scores is a tuple of tensors, one per generation step
        # Each tensor has shape [batch_size, vocab_size]
        for step, (token_id, score_tensor) in enumerate(zip(generated_ids, outputs.scores)):
            token_id = token_id.item()
            # Extract logits for first (and only) batch element
            logits = score_tensor[0].float()  # Convert to float32 for compatibility

            # Calculate basic metrics for this token
            metrics = self._calculate_information_metrics(logits, token_id)
            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(logits, token_id, step)
            top_k_probs = self._get_top_k_probabilities(logits, k=1000)

            # Decode token
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

            # Calculate suspicion score
            suspicion_score = self._calculate_suspicion_score(
                rank=metrics["rank"],
                surprisal=metrics["surprisal"],
                entropy=metrics["entropy"],
                probability_margin=advanced_metrics["probability_margin"]
            )

            # Create analysis object with all metrics
            analysis = TokenAnalysis(
                token=token_text,
                token_id=token_id,
                position=step,
                probability=metrics["probability"],
                logit=logits[token_id].item(),
                rank=metrics["rank"],
                perplexity=metrics["perplexity"],
                entropy=metrics["entropy"],
                surprisal=metrics["surprisal"],
                top_k_probs=top_k_probs,
                # Advanced metrics
                max_probability=advanced_metrics["max_probability"],
                probability_margin=advanced_metrics["probability_margin"],
                shannon_entropy=advanced_metrics["shannon_entropy"],
                local_perplexity=advanced_metrics["local_perplexity"],
                sequence_improbability=advanced_metrics["sequence_improbability"],
                confidence_score=advanced_metrics["confidence_score"],
                lm_polygraph_metrics=None,
                # Advanced error detection methods
                semantic_energy=advanced_metrics["semantic_energy"],
                conformal_score=advanced_metrics["conformal_score"],
                # Aggregated suspicion score
                suspicion_score=suspicion_score
            )

            self.generation_history.append(analysis)

            # Print progress
            if step % 10 == 0:
                print(f"Processed {step} tokens...")

        print(f"Generation complete! Generated {len(self.generation_history)} tokens.")

        # Calculate LM-Polygraph metrics for the complete sequence if available
        if self.lm_polygraph_estimators and generated_text:
            print("Calculating LM-Polygraph uncertainty metrics...")
            polygraph_metrics = self._calculate_lm_polygraph_metrics(text, generated_text)

            # Add LM-Polygraph metrics to the last token (as sequence-level metrics)
            if self.generation_history and polygraph_metrics:
                self.generation_history[-1].lm_polygraph_metrics = polygraph_metrics

        return generated_text, self.generation_history

    def analyze_code_for_errors(self,
                               code: str,
                               sensitivity_factor: float = 1.5) -> Dict[str, Any]:
        """
        Analyze existing code for logical errors using LecPrompt technique.

        This method computes log probabilities for tokens in existing code
        and identifies anomalous tokens/lines using statistical analysis.

        Args:
            code: Source code to analyze
            sensitivity_factor: k parameter for threshold (τ = μ - k×σ)

        Returns:
            Dictionary with error analysis results including token-level and
            line-level error detection
        """
        print(f"Analyzing code for logical errors (sensitivity={sensitivity_factor})...")

        # Tokenize the code
        inputs = self.tokenizer(code, return_tensors="pt", return_offsets_mapping=True)
        input_ids = inputs.input_ids.to(self.model.device)
        offset_mapping = inputs.offset_mapping[0] if "offset_mapping" in inputs else None

        token_analyses = []
        log_probs_list = []

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute log probabilities for each token
            for i in range(1, len(input_ids[0])):  # Start from 1
                token_id = input_ids[0][i].item()
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                # Get logits for predicting this token from previous context
                prev_logits = logits[i-1]
                log_probs = F.log_softmax(prev_logits, dim=-1)
                token_log_prob = log_probs[token_id].item()
                log_probs_list.append(token_log_prob)

                # Get character position
                char_pos = offset_mapping[i][0].item() if offset_mapping is not None else 0

                # Store token data temporarily
                token_analyses.append({
                    'token': token_text,
                    'token_id': token_id,
                    'position': i-1,
                    'log_prob': token_log_prob,
                    'char_pos': char_pos
                })

        # Compute statistical threshold (LecPrompt method)
        log_probs_array = np.array(log_probs_list)
        mean = np.mean(log_probs_array)
        std_dev = np.std(log_probs_array)
        threshold = mean - sensitivity_factor * std_dev

        print(f"  Mean log prob: {mean:.4f}")
        print(f"  Std dev: {std_dev:.4f}")
        print(f"  Threshold: {threshold:.4f}")

        # Map tokens to lines
        lines = code.split('\n')
        char_to_line = {}
        current_char = 0
        for line_idx, line in enumerate(lines):
            line_len = len(line) + 1  # +1 for newline
            for i in range(line_len):
                char_to_line[current_char + i] = line_idx + 1
            current_char += line_len

        # Annotate tokens with error detection metrics
        annotated_tokens = []
        anomalous_count = 0

        for token_data in token_analyses:
            log_prob = token_data['log_prob']

            # Calculate deviation score
            deviation_score = (log_prob - mean) / std_dev if std_dev > 0 else 0.0

            # Check if anomalous
            is_anomalous = log_prob < threshold
            if is_anomalous:
                anomalous_count += 1

            # Compute error likelihood (0-1)
            error_likelihood = min(1.0, max(0.0, -deviation_score / 3.0))

            # Get line number
            char_pos = token_data['char_pos']
            line_num = char_to_line.get(char_pos, 1)

            # Create TokenAnalysis-like object with error metrics
            annotated_tokens.append({
                'token': token_data['token'],
                'token_id': token_data['token_id'],
                'position': token_data['position'],
                'line_number': line_num,
                'log_probability': log_prob,
                'is_anomalous': is_anomalous,
                'statistical_score': deviation_score,
                'error_likelihood': error_likelihood
            })

        # Aggregate to line-level errors
        line_errors = {}
        for token in annotated_tokens:
            line_num = token['line_number']
            if line_num not in line_errors:
                line_errors[line_num] = {
                    'line_number': line_num,
                    'line_content': lines[line_num - 1] if line_num <= len(lines) else "",
                    'tokens': [],
                    'anomalous_tokens': []
                }

            line_errors[line_num]['tokens'].append(token)
            if token['is_anomalous']:
                line_errors[line_num]['anomalous_tokens'].append(token)

        # Compute line-level statistics
        for line_num, line_data in line_errors.items():
            tokens = line_data['tokens']
            anomalous = line_data['anomalous_tokens']

            log_probs_line = [t['log_probability'] for t in tokens]
            line_data['avg_log_prob'] = float(np.mean(log_probs_line))
            line_data['min_log_prob'] = float(np.min(log_probs_line))
            line_data['num_tokens'] = len(tokens)
            line_data['num_anomalous'] = len(anomalous)
            line_data['anomaly_ratio'] = len(anomalous) / len(tokens) if tokens else 0.0
            line_data['is_error_line'] = len(anomalous) > 0

        print(f"  Found {anomalous_count} anomalous tokens")
        print(f"  Identified {sum(1 for l in line_errors.values() if l['is_error_line'])} error lines")

        return {
            'code': code,
            'model_name': self.model_name,
            'sensitivity_factor': sensitivity_factor,
            'statistics': {
                'total_tokens': len(annotated_tokens),
                'anomalous_tokens': anomalous_count,
                'mean_log_prob': float(mean),
                'std_dev': float(std_dev),
                'threshold': float(threshold),
                'total_lines': len(line_errors),
                'error_lines': sum(1 for l in line_errors.values() if l['is_error_line'])
            },
            'tokens': annotated_tokens,
            'lines': list(line_errors.values())
        }

    def get_generation_stats(self) -> Dict[str, float]:
        """
        Get statistics about the generation process.
        
        Returns:
            Dictionary with various statistics
        """
        if not self.generation_history:
            return {}
        
        probs = [analysis.probability for analysis in self.generation_history]
        ranks = [analysis.rank for analysis in self.generation_history]
        entropies = [analysis.entropy for analysis in self.generation_history]
        surprisals = [analysis.surprisal for analysis in self.generation_history]

        # New advanced metrics
        max_probs = [analysis.max_probability for analysis in self.generation_history]
        prob_margins = [analysis.probability_margin for analysis in self.generation_history]
        shannon_entropies = [analysis.shannon_entropy for analysis in self.generation_history]
        local_perplexities = [analysis.local_perplexity for analysis in self.generation_history]
        confidence_scores = [analysis.confidence_score for analysis in self.generation_history]
        sequence_improbabilities = [analysis.sequence_improbability for analysis in self.generation_history]

        return {
            # Original metrics
            "avg_probability": np.mean(probs),
            "min_probability": np.min(probs),
            "max_probability": np.max(probs),
            "avg_rank": np.mean(ranks),
            "min_rank": np.min(ranks),
            "max_rank": np.max(ranks),
            "avg_entropy": np.mean(entropies),
            "avg_surprisal": np.mean(surprisals),
            "total_tokens": len(self.generation_history),
            # New advanced metrics
            "avg_max_probability": np.mean(max_probs),
            "avg_probability_margin": np.mean(prob_margins),
            "min_probability_margin": np.min(prob_margins),
            "avg_shannon_entropy": np.mean(shannon_entropies),
            "avg_local_perplexity": np.mean(local_perplexities),
            "avg_confidence_score": np.mean(confidence_scores),
            "final_sequence_improbability": sequence_improbabilities[-1] if sequence_improbabilities else 0.0,
            "max_sequence_improbability": np.max(sequence_improbabilities) if sequence_improbabilities else 0.0
        }
    
    def save_analysis(self, filename: str):
        """
        Save the generation analysis to a file.
        
        Args:
            filename: Path to save the analysis
        """
        import json
        import numpy as np

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        stats = self.get_generation_stats()
        data = {
            "model_name": self.model_name,
            "generation_stats": {k: convert_numpy_types(v) for k, v in stats.items()},
            "tokens": [
                {
                    "token": analysis.token,
                    "token_id": convert_numpy_types(analysis.token_id),
                    "position": convert_numpy_types(analysis.position),
                    "probability": convert_numpy_types(analysis.probability),
                    "logit": convert_numpy_types(analysis.logit),
                    "rank": convert_numpy_types(analysis.rank),
                    "perplexity": convert_numpy_types(analysis.perplexity),
                    "entropy": convert_numpy_types(analysis.entropy),
                    "surprisal": convert_numpy_types(analysis.surprisal),
                    # New advanced metrics
                    "max_probability": convert_numpy_types(analysis.max_probability),
                    "probability_margin": convert_numpy_types(analysis.probability_margin),
                    "shannon_entropy": convert_numpy_types(analysis.shannon_entropy),
                    "local_perplexity": convert_numpy_types(analysis.local_perplexity),
                    "sequence_improbability": convert_numpy_types(analysis.sequence_improbability),
                    "confidence_score": convert_numpy_types(analysis.confidence_score),
                    # LM-Polygraph metrics (if available)
                    "lm_polygraph_metrics": {k: convert_numpy_types(v) for k, v in analysis.lm_polygraph_metrics.items()} if analysis.lm_polygraph_metrics else None,
                    # Token probability data
                    "top_10_probs": [(convert_numpy_types(tid), convert_numpy_types(prob)) for tid, prob in analysis.top_k_probs[:10]],
                    "top_10_tokens": [
                        {
                            "token_id": convert_numpy_types(token_id),
                            "probability": convert_numpy_types(prob),
                            "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False)
                        }
                        for token_id, prob in analysis.top_k_probs[:10]
                    ],
                    # CodeT5 validation metrics
                    "codet5_validation": {
                        "validation_score": convert_numpy_types(analysis.codet5_validation_score) if analysis.codet5_validation_score is not None else None,
                        "predicted_token": analysis.codet5_predicted_token,
                        "matches": analysis.codet5_matches,
                        "alternatives": [
                            {"token": alt_token, "probability": convert_numpy_types(alt_prob)}
                            for alt_token, alt_prob in (analysis.codet5_alternatives[:3] if analysis.codet5_alternatives else [])
                        ]
                    } if analysis.codet5_validation_score is not None else None,
                    # Nomic-embed-code validation metrics
                    "nomic_validation": {
                        "coherence_score": convert_numpy_types(analysis.nomic_coherence_score) if analysis.nomic_coherence_score is not None else None,
                        "similarity_drop": convert_numpy_types(analysis.nomic_similarity_drop) if analysis.nomic_similarity_drop is not None else None,
                        "context_similarity": convert_numpy_types(analysis.nomic_context_similarity) if analysis.nomic_context_similarity is not None else None
                    } if analysis.nomic_coherence_score is not None else None
                }
                for analysis in self.generation_history
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis saved to {filename}")


if __name__ == "__main__":
    # Example usage
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Coder-7B-Instruct"
    analyzer = QwenProbabilityAnalyzer(model_name=model_name)
    
    prompt = "Write a Python function to calculate the factorial of a number using recursion."
    
    generated_text, analysis = analyzer.generate_with_analysis(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.7
    )
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    
    print("\n" + "="*50)
    print("GENERATION STATISTICS:")
    print("="*50)
    stats = analyzer.get_generation_stats()
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    
    # Save analysis
    analyzer.save_analysis("generation_analysis.json")