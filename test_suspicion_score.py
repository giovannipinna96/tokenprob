#!/usr/bin/env python3
"""
Quick test of the suspicion score feature.

This script tests the newly implemented suspicion score aggregation
across different token generation metrics.
"""

from LLM import QwenProbabilityAnalyzer
from visualizer import TokenVisualizer, TokenVisualizationMode
import json

def main():
    print("🧪 Testing Suspicion Score Implementation")
    print("=" * 60)

    # Initialize analyzer
    print("\n1. Loading model...")
    analyzer = QwenProbabilityAnalyzer(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

    # Simple test prompt
    prompt = "Write a Python function to calculate factorial recursively."

    print(f"\n2. Generating code with analysis...")
    print(f"   Prompt: {prompt}")

    # Generate with analysis
    generated_text, token_analyses = analyzer.generate_with_analysis(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=True
    )

    print(f"\n3. Generated text:")
    print(f"   {generated_text[:200]}...")

    # Check suspicion scores
    print(f"\n4. Analyzing Suspicion Scores:")
    print(f"   Total tokens: {len(token_analyses)}")

    # Extract suspicion scores
    scores = [t.suspicion_score for t in token_analyses if t.suspicion_score is not None]

    if scores:
        print(f"   Suspicion scores available: {len(scores)}/{len(token_analyses)}")
        print(f"   Min score: {min(scores):.1f}")
        print(f"   Max score: {max(scores):.1f}")
        print(f"   Avg score: {sum(scores)/len(scores):.1f}")

        # Count by risk level
        high_risk = sum(1 for s in scores if s >= 60)
        medium_risk = sum(1 for s in scores if 40 <= s < 60)
        low_risk = sum(1 for s in scores if 20 <= s < 40)
        safe = sum(1 for s in scores if s < 20)

        print(f"\n5. Risk Level Distribution:")
        print(f"   🔴 HIGH (≥60):     {high_risk} tokens ({high_risk/len(scores)*100:.1f}%)")
        print(f"   🟡 MEDIUM (40-59): {medium_risk} tokens ({medium_risk/len(scores)*100:.1f}%)")
        print(f"   🟠 LOW (20-39):    {low_risk} tokens ({low_risk/len(scores)*100:.1f}%)")
        print(f"   🟢 SAFE (<20):     {safe} tokens ({safe/len(scores)*100:.1f}%)")

        # Show top 5 most suspicious tokens
        print(f"\n6. Top 5 Most Suspicious Tokens:")
        sorted_analyses = sorted(token_analyses, key=lambda t: t.suspicion_score or 0, reverse=True)[:5]
        for i, analysis in enumerate(sorted_analyses, 1):
            print(f"   {i}. '{analysis.token}' - Score: {analysis.suspicion_score:.1f}")
            print(f"      Rank: {analysis.rank}, Surprisal: {analysis.surprisal:.2f}, ")
            print(f"      Entropy: {analysis.entropy:.3f}, Margin: {analysis.probability_margin:.3f}")

        # Test visualization
        print(f"\n7. Generating visualizations...")
        visualizer = TokenVisualizer()

        # Create suspicion score visualization
        html_output = visualizer.create_interactive_html(
            analyses=token_analyses,
            title="Suspicion Score Visualization Test",
            mode=TokenVisualizationMode.SUSPICION_SCORE
        )

        # Save HTML visualization
        output_file = "test_suspicion_score_visualization.html"
        with open(output_file, 'w') as f:
            f.write(html_output)
        print(f"   ✅ Saved visualization to: {output_file}")

        # Save analysis data
        analysis_data = {
            "prompt": prompt,
            "generated_text": generated_text,
            "total_tokens": len(token_analyses),
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk,
                "safe": safe
            },
            "score_stats": {
                "min": min(scores),
                "max": max(scores),
                "avg": sum(scores) / len(scores)
            },
            "tokens": [
                {
                    "token": t.token,
                    "position": t.position,
                    "suspicion_score": t.suspicion_score,
                    "rank": t.rank,
                    "surprisal": t.surprisal,
                    "entropy": t.entropy,
                    "probability_margin": t.probability_margin
                }
                for t in token_analyses
            ]
        }

        analysis_file = "test_suspicion_score_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"   ✅ Saved analysis data to: {analysis_file}")

        print(f"\n{'=' * 60}")
        print("✅ Test completed successfully!")
        print(f"\nView results:")
        print(f"  - Open {output_file} in a browser for interactive visualization")
        print(f"  - Check {analysis_file} for detailed analysis data")

    else:
        print("   ❌ ERROR: No suspicion scores found!")
        print("   Check that suspicion_score is being calculated correctly.")
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
