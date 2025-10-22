#!/usr/bin/env python3
"""
Quick test of the forced generation analyzer with a simple example.
"""

from forced_generation_analyzer import ForcedGenerationAnalyzer

def main():
    print("🧪 Quick test of Forced Generation Analyzer")

    # Initialize analyzer
    analyzer = ForcedGenerationAnalyzer()

    # Simple test case
    problem = "Write a Python function to add two numbers."
    target_code = "def add(a, b):\n    return a + b"

    print(f"Problem: {problem}")
    print(f"Target code: {repr(target_code)}")

    try:
        # Run forced generation
        result = analyzer.force_generation_with_logits(
            problem_description=problem,
            target_code=target_code,
            verbose=True
        )

        print(f"\n✅ SUCCESS!")
        print(f"Reconstructed code: {repr(result.reconstructed_code)}")
        print(f"Match: {result.target_code == result.reconstructed_code}")
        print(f"Average probability: {result.average_probability:.3f}")
        print(f"Average rank: {result.average_rank:.1f}")
        print(f"High uncertainty tokens: {result.high_uncertainty_tokens}")

        # Save results
        analyzer.save_analysis(result, "quick_test_result.json")

        # Show some token details
        print(f"\nFirst 5 token analysis:")
        for i, analysis in enumerate(result.token_analyses[:5]):
            print(f"  {i}: '{analysis.token}' prob={analysis.probability:.3f} rank={analysis.rank}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()