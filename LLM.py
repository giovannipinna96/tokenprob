import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import HTML, display
import math
from dataclasses import dataclass


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
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.generation_history: List[TokenAnalysis] = []
        
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
        
        # Prepare input
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
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
                
                # Calculate metrics for this token
                metrics = self._calculate_information_metrics(logits, next_token_id)
                top_k_probs = self._get_top_k_probabilities(logits, k=1000)
                
                # Decode token
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                
                # Create analysis object
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
                    top_k_probs=top_k_probs
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
        
        return generated_text, self.generation_history
    
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
        
        return {
            "avg_probability": np.mean(probs),
            "min_probability": np.min(probs),
            "max_probability": np.max(probs),
            "avg_rank": np.mean(ranks),
            "min_rank": np.min(ranks),
            "max_rank": np.max(ranks),
            "avg_entropy": np.mean(entropies),
            "avg_surprisal": np.mean(surprisals),
            "total_tokens": len(self.generation_history)
        }
    
    def save_analysis(self, filename: str):
        """
        Save the generation analysis to a file.
        
        Args:
            filename: Path to save the analysis
        """
        import json
        
        data = {
            "model_name": self.model_name,
            "generation_stats": self.get_generation_stats(),
            "tokens": [
                {
                    "token": analysis.token,
                    "token_id": analysis.token_id,
                    "position": analysis.position,
                    "probability": analysis.probability,
                    "logit": analysis.logit,
                    "rank": analysis.rank,
                    "perplexity": analysis.perplexity,
                    "entropy": analysis.entropy,
                    "surprisal": analysis.surprisal,
                    "top_10_probs": analysis.top_k_probs[:10]  # Save only top 10 for space
                }
                for analysis in self.generation_history
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis saved to {filename}")


if __name__ == "__main__":
    # Example usage
    analyzer = QwenProbabilityAnalyzer()
    
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