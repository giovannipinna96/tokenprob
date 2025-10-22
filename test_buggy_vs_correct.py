#!/usr/bin/env python3
"""
Test buggy vs correct code with forced generation to validate hypothesis.
"""

from forced_generation_analyzer import ForcedGenerationAnalyzer

def main():
    print("🔬 Testing Buggy vs Correct Code with Forced Generation")

    # Initialize analyzer
    analyzer = ForcedGenerationAnalyzer()

    # Test case: factorial function
    problem = "Write a Python function to calculate factorial using recursion."

    correct_code = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""

    buggy_code = """def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)"""

    print(f"Problem: {problem}")
    print(f"Correct code:\n{correct_code}")
    print(f"Buggy code:\n{buggy_code}")

    try:
        print(f"\n🟢 Testing CORRECT code...")
        correct_result = analyzer.force_generation_with_logits(
            problem_description=problem,
            target_code=correct_code,
            verbose=False
        )

        print(f"\n🔴 Testing BUGGY code...")
        buggy_result = analyzer.force_generation_with_logits(
            problem_description=problem,
            target_code=buggy_code,
            verbose=False
        )

        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"{'Metric':<20} {'Correct':<10} {'Buggy':<10} {'Difference':<12} {'Hypothesis'}")
        print(f"{'-'*70}")

        prob_diff = buggy_result.average_probability - correct_result.average_probability
        rank_diff = buggy_result.average_rank - correct_result.average_rank
        uncert_diff = buggy_result.high_uncertainty_tokens - correct_result.high_uncertainty_tokens

        print(f"{'Avg Probability':<20} {correct_result.average_probability:<10.3f} {buggy_result.average_probability:<10.3f} {prob_diff:<12.3f} {'✓' if prob_diff < 0 else '✗'}")
        print(f"{'Avg Rank':<20} {correct_result.average_rank:<10.1f} {buggy_result.average_rank:<10.1f} {rank_diff:<12.1f} {'✓' if rank_diff > 0 else '✗'}")
        print(f"{'High Uncertainty':<20} {correct_result.high_uncertainty_tokens:<10} {buggy_result.high_uncertainty_tokens:<10} {uncert_diff:<12} {'✓' if uncert_diff > 0 else '✗'}")

        # Find the specific different tokens
        correct_tokens = [analysis.token for analysis in correct_result.token_analyses]
        buggy_tokens = [analysis.token for analysis in buggy_result.token_analyses]

        print(f"\n🎯 TOKEN-LEVEL DIFFERENCES:")
        min_len = min(len(correct_tokens), len(buggy_tokens))
        differences_found = False

        for i in range(min_len):
            if correct_tokens[i] != buggy_tokens[i]:
                correct_analysis = correct_result.token_analyses[i]
                buggy_analysis = buggy_result.token_analyses[i]

                print(f"Position {i}:")
                print(f"  Correct: '{correct_analysis.token}' (prob={correct_analysis.probability:.3f}, rank={correct_analysis.rank})")
                print(f"  Buggy:   '{buggy_analysis.token}' (prob={buggy_analysis.probability:.3f}, rank={buggy_analysis.rank})")
                differences_found = True

        if not differences_found:
            print("  No token differences found in overlapping positions")

        # Save detailed results
        analyzer.save_analysis(correct_result, "factorial_correct_forced.json")
        analyzer.save_analysis(buggy_result, "factorial_buggy_forced.json")

        print(f"\n💾 Results saved to:")
        print(f"  • factorial_correct_forced.json")
        print(f"  • factorial_buggy_forced.json")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()