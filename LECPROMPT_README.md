# LecPrompt Logical Error Detection

Implementation of the logical error detection technique from the paper:
**"LecPrompt: A Prompt-based Approach for Logical Error Correction with CodeBERT"** (arXiv:2410.08241)

## Overview

This implementation provides automated logical error detection in Python code using statistical analysis of token probabilities. The technique identifies anomalous tokens and lines that likely contain logical errors without requiring test cases or external oracles.

## Key Features

- **Token-level error detection** using log probability analysis
- **Line-level error aggregation** to identify problematic code sections
- **Statistical anomaly detection** with configurable sensitivity (τ = μ - k×σ)
- **Comparison analysis** between buggy and correct code
- **Integration with existing analysis pipeline** (LLM.py, visualizer.py)
- **Comprehensive test suite** on 10 buggy/correct code pairs

## Core Algorithm

### Mathematical Foundation

```
1. Compute log probability pᵢ for each token i in the code
2. Calculate mean: μ = (1/n) × Σpᵢ
3. Calculate standard deviation: σ = √[(1/n) × Σ(pᵢ - μ)²]
4. Set threshold: τ = μ - k×σ
   where k is the sensitivity factor (default: 1.5)
5. Flag tokens where pᵢ < τ as anomalous (potential errors)
6. Aggregate token-level anomalies to line-level error scores
```

### Statistical Interpretation

- **Anomalous tokens**: Tokens with log probabilities below the threshold
- **Error likelihood**: Normalized score (0-1) indicating error probability
- **Deviation score**: How many standard deviations a token is from the mean
- **Line error score**: Combination of anomaly ratio and average error likelihood

## Installation

No additional dependencies beyond the existing project requirements:

```bash
# The implementation uses existing dependencies
# torch, transformers, numpy are already required
```

## Quick Start

### 1. Analyze a Test Example

```bash
# Run error detection on a specific example from the test dataset
python run_error_detection.py --example factorial_recursion_base_case

# Use custom sensitivity factor
python run_error_detection.py --example binary_search_missing_bounds --sensitivity 2.0
```

### 2. Analyze Custom Code

```python
from logical_error_detector import LogicalErrorDetector

# Initialize detector
detector = LogicalErrorDetector(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    sensitivity_factor=1.5
)

# Analyze code
code = """
def factorial(n):
    if n == 1:  # Bug: missing base case for n=0
        return 1
    return n * factorial(n - 1)
"""

results = detector.localize_errors(code, k=1.5)

# Show detected errors
for line_error in results['line_errors']:
    if line_error.is_error_line:
        print(f"Line {line_error.line_number}: {line_error.line_content}")
        print(f"  Error score: {line_error.error_score:.3f}")
        print(f"  Anomalous tokens: {line_error.num_anomalous_tokens}")
```

### 3. Compare Buggy vs Correct Code

```python
from logical_error_detector import LogicalErrorDetector

detector = LogicalErrorDetector()

buggy_code = """
def binary_search(arr, target):
    left = 0
    right = len(arr)  # Bug: should be len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

correct_code = """
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1  # Correct
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

comparison = detector.compare_buggy_vs_correct(buggy_code, correct_code)

print(f"Buggy code: {comparison['buggy_analysis']['statistics']['anomalous_tokens']} anomalous tokens")
print(f"Correct code: {comparison['correct_analysis']['statistics']['anomalous_tokens']} anomalous tokens")
print(f"Hypothesis confirmed: {comparison['comparison']['hypothesis_confirmed']}")
```

### 4. Integration with Existing LLM.py

```python
from LLM import QwenProbabilityAnalyzer

# Use the integrated method in QwenProbabilityAnalyzer
analyzer = QwenProbabilityAnalyzer()

code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):  # Bug: inefficient, should be sqrt(n)
        if n % i == 0:
            return False
    return True
"""

# Analyze code for errors
error_analysis = analyzer.analyze_code_for_errors(code, sensitivity_factor=1.5)

print(f"Anomalous tokens: {error_analysis['statistics']['anomalous_tokens']}")
print(f"Error lines: {error_analysis['statistics']['error_lines']}")

# Show error lines
for line in error_analysis['lines']:
    if line['is_error_line']:
        print(f"Line {line['line_number']}: {line['line_content']}")
        print(f"  Anomaly ratio: {line['anomaly_ratio']:.2%}")
```

## Command-Line Tools

### run_error_detection.py

Main script for running error detection:

```bash
# Run on a test example
python run_error_detection.py --example factorial_recursion_base_case

# Run on a code file
python run_error_detection.py --code-file mycode.py

# Specify model and sensitivity
python run_error_detection.py \
    --example binary_search_missing_bounds \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --sensitivity 2.0 \
    --output-dir results/
```

### test_logical_error_detection.py

Comprehensive test suite:

```bash
# Test all examples from test_examples.py
python test_logical_error_detection.py

# Test with custom model
python test_logical_error_detection.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Test single example
python test_logical_error_detection.py --example factorial_recursion_base_case

# Run sensitivity analysis
python test_logical_error_detection.py --sensitivity-analysis

# Custom sensitivity
python test_logical_error_detection.py --sensitivity 2.0
```

## Configuration Parameters

### Sensitivity Factor (k)

The sensitivity factor controls the threshold for anomaly detection:

- **k = 1.0**: More sensitive, detects more anomalies (may have false positives)
- **k = 1.5**: Balanced (default, recommended)
- **k = 2.0**: Less sensitive, detects only strong anomalies
- **k = 2.5**: Very conservative, minimal false positives

Formula: `threshold = mean - k × std_dev`

### Model Selection

Any HuggingFace causal language model can be used:

```python
# Code-specific models (recommended)
detector = LogicalErrorDetector(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")
detector = LogicalErrorDetector(model_name="meta-llama/Llama-3.2-3B-Instruct")

# General language models
detector = LogicalErrorDetector(model_name="microsoft/phi-2")
```

## Output Format

### Token-Level Results

```json
{
  "token": "n",
  "token_id": 1234,
  "position": 15,
  "line_number": 3,
  "log_probability": -8.42,
  "is_anomalous": true,
  "statistical_score": -2.45,
  "error_likelihood": 0.82
}
```

### Line-Level Results

```json
{
  "line_number": 3,
  "line_content": "if n == 1:",
  "avg_log_prob": -7.83,
  "min_log_prob": -8.42,
  "num_tokens": 4,
  "num_anomalous_tokens": 2,
  "error_score": 0.65,
  "is_error_line": true
}
```

### Statistics Summary

```json
{
  "total_tokens": 45,
  "anomalous_tokens": 8,
  "total_lines": 6,
  "error_lines": 2,
  "mean_log_prob": -3.21,
  "std_dev": 2.14,
  "threshold": -6.42
}
```

## Visualization

The visualizer.py module includes three new visualization modes:

```python
from visualizer import TokenVisualizer, TokenVisualizationMode
from LLM import TokenAnalysis

visualizer = TokenVisualizer()

# Mode 1: Logical error detection (red = error, green = normal)
html = visualizer.create_html_visualization(
    token_analyses,
    mode=TokenVisualizationMode.LOGICAL_ERROR_DETECTION
)

# Mode 2: Statistical deviation (shows deviation in std devs)
html = visualizer.create_html_visualization(
    token_analyses,
    mode=TokenVisualizationMode.STATISTICAL_DEVIATION
)

# Mode 3: Error likelihood (0 = safe, 1 = error)
html = visualizer.create_html_visualization(
    token_analyses,
    mode=TokenVisualizationMode.ERROR_LIKELIHOOD
)
```

## Test Dataset

The implementation includes 10 test examples from `test_examples.py`:

1. **binary_search_missing_bounds** (logic error)
2. **factorial_recursion_base_case** (edge case)
3. **bubble_sort_inner_loop** (logic error)
4. **list_max_empty_check** (edge case)
5. **fibonacci_negative_input** (edge case)
6. **string_reverse_indexing** (logic error)
7. **prime_check_optimization** (logic error)
8. **merge_arrays_index_bounds** (logic error)
9. **count_vowels_case_sensitivity** (logic error)
10. **division_zero_check** (edge case)

Each example has both buggy and correct versions for comparison.

## Expected Results

Based on the LecPrompt paper and our implementation:

- **Token-level precision**: Anomalous tokens in buggy code should be higher than in correct code
- **Line-level detection**: Error lines should correlate with actual bug locations
- **Confirmation rate**: 50-80% of examples should confirm the hypothesis (buggy > correct anomalies)
- **Bug type variations**: Logic errors and edge cases may show different detection patterns

## Advanced Usage

### Custom Token Analysis

```python
from logical_error_detector import LogicalErrorDetector

detector = LogicalErrorDetector()

# Get raw token data
token_data = detector.compute_token_log_probabilities(code)

# Compute custom threshold
mean, std_dev, threshold = detector.compute_statistical_threshold(
    [t[2] for t in token_data],
    k=2.0  # Custom sensitivity
)

# Identify anomalies with custom threshold
token_errors = detector.identify_anomalous_tokens(
    token_data,
    code,
    k=2.0
)
```

### Batch Analysis

```python
from test_examples import TestExamplesDataset
from logical_error_detector import LogicalErrorDetector

detector = LogicalErrorDetector()
dataset = TestExamplesDataset()

results = []
for example in dataset.get_all_examples():
    buggy_analysis = detector.localize_errors(example.buggy_code)
    correct_analysis = detector.localize_errors(example.correct_code)

    results.append({
        'name': example.name,
        'buggy_anomalies': buggy_analysis['statistics']['anomalous_tokens'],
        'correct_anomalies': correct_analysis['statistics']['anomalous_tokens']
    })

# Analyze results
import pandas as pd
df = pd.DataFrame(results)
print(df)
```

## Limitations

1. **Model-dependent**: Results vary based on the language model used
2. **Code-specific**: Trained primarily on Python code patterns
3. **Statistical nature**: May produce false positives/negatives
4. **No semantic understanding**: Pure statistical analysis without deep semantic reasoning
5. **Single-file analysis**: Does not consider cross-file dependencies

## Citation

If you use this implementation, please cite the original LecPrompt paper:

```bibtex
@article{lecprompt2024,
  title={LecPrompt: A Prompt-based Approach for Logical Error Correction with CodeBERT},
  author={[Authors]},
  journal={arXiv preprint arXiv:2410.08241},
  year={2024}
}
```

## License

This implementation follows the same license as the parent project.

## Contributing

Contributions are welcome! Areas for improvement:

- Support for additional programming languages
- Integration with more sophisticated error correction
- Ensemble methods combining multiple models
- Active learning for threshold optimization
- Integration with IDE tools

## Support

For questions or issues:
1. Check existing test examples in `test_examples.py`
2. Run `test_logical_error_detection.py` to verify setup
3. Review the LecPrompt paper for theoretical details
4. Open an issue with reproducible example

---

**Last Updated**: 2025-10-02
**Implementation Version**: 1.0
