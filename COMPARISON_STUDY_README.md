# Multi-Model Error Detection Comparison Study

Comprehensive comparison framework for evaluating state-of-the-art code language models on logical error detection using the LecPrompt approach.

## Overview

This framework implements a rigorous comparative study across multiple LLM architectures to determine which models are most effective at detecting logical errors in code through statistical anomaly detection.

### Research Question

**Which state-of-the-art code language model performs best at detecting logical errors using log-probability-based statistical anomaly detection?**

### Methodology

- **Technique:** LecPrompt (Log-probability Error Correction Prompting)
- **Threshold:** τ = μ - k×σ (default k=1.5)
- **Hypothesis:** Buggy code exhibits more anomalous tokens (below threshold) than correct code
- **Statistical Tests:** McNemar's test (pairwise), Friedman test (overall ranking)

## Models Tested

### 1. StarCoder2-7B (BigCode)
- **Architecture:** Causal Language Model
- **Parameters:** 7 billion
- **Context Length:** 16,384 tokens
- **Precision:** BF16 (full precision, no quantization)
- **Approach:** Autoregressive prediction P(token[i] | token[0:i-1])
- **Model ID:** `bigcode/starcoder2-7b`

### 2. CodeT5+ 2B (Salesforce)
- **Architecture:** Encoder-Decoder (T5-based)
- **Parameters:** 2 billion
- **Context Length:** 512 tokens
- **Precision:** FP16 (full precision, no quantization)
- **Approach:** Masked Language Modeling (bidirectional context)
- **Model ID:** `Salesforce/codet5p-2b`

### 3. DeepSeek-Coder 6.7B (DeepSeek AI)
- **Architecture:** Causal Language Model
- **Parameters:** 6.7 billion
- **Context Length:** 16,384 tokens
- **Precision:** BF16 (full precision, no quantization)
- **Approach:** Autoregressive prediction
- **Training:** 2 trillion tokens of code
- **Model ID:** `deepseek-ai/deepseek-coder-6.7b-base`

### 4. CodeBERT (Microsoft)
- **Architecture:** Masked Language Model (BERT-based)
- **Parameters:** 125 million
- **Context Length:** 512 tokens
- **Precision:** FP32
- **Approach:** Masked Language Modeling
- **Model ID:** `microsoft/codebert-base`

### 5. Qwen 2.5 Coder 7B (Alibaba)
- **Architecture:** Causal Language Model
- **Parameters:** 7 billion
- **Context Length:** 32,768 tokens
- **Precision:** BF16 (full precision, no quantization)
- **Approach:** Autoregressive prediction
- **Model ID:** `Qwen/Qwen2.5-Coder-7B-Instruct`

## Installation

```bash
# Install dependencies
pip install torch transformers scipy plotly numpy

# Or use uv for dependency management
uv sync
```

## Usage

### Basic Comparison (All Models)

```bash
python test_multi_model_comparison.py
```

This will:
1. Run all enabled models sequentially
2. Perform statistical analysis
3. Generate visualizations
4. Create comprehensive markdown report

### Advanced Options

```bash
# Test specific models only
python test_multi_model_comparison.py --models starcoder2-7b codet5p-2b deepseek-6.7b

# Adjust sensitivity factor
python test_multi_model_comparison.py --sensitivity 2.0

# Custom output directory
python test_multi_model_comparison.py --output my_comparison_study

# Skip visualizations (faster)
python test_multi_model_comparison.py --skip-visualizations

# Skip report generation
python test_multi_model_comparison.py --skip-report
```

### Available Model Keys

- `starcoder2-7b` - StarCoder2 7B
- `codet5p-2b` - CodeT5+ 2B
- `deepseek-6.7b` - DeepSeek-Coder 6.7B
- `codebert` - CodeBERT
- `qwen-7b` - Qwen 2.5 Coder 7B

## Output Files

After running the comparison study, you'll find:

### 1. `complete_benchmark_results.json`
Complete results including:
- Individual results for each test example
- Per-model aggregate statistics
- Timing information
- Metadata

### 2. `statistical_analysis.json`
Statistical analysis including:
- Confirmation rates for each model
- McNemar's test results (pairwise comparisons)
- Friedman test results (overall ranking)
- Cohen's d effect sizes
- Performance by bug type
- Model ranking

### 3. `comparison_report.md`
Comprehensive markdown report with:
- Executive summary
- Model ranking
- Detailed performance tables
- Statistical significance analysis
- Conclusions and recommendations

### 4. Interactive Visualizations

- **`confirmation_rates.html`** - Bar chart comparing confirmation rates
- **`bug_type_heatmap.html`** - Performance by bug type (heatmap)
- **`radar_comparison.html`** - Multi-dimensional radar chart
- **`significance_matrix.html`** - Statistical significance matrix

### 5. Intermediate Results
- `results_<model_key>.json` - Individual model results (saved after each model)

## Architecture

### Core Components

```
tokenprob/
├── detectors/
│   ├── base_detector.py          # Abstract base class
│   ├── starcoder2_detector.py    # StarCoder2 implementation
│   ├── codet5_detector.py        # CodeT5+ implementation
│   └── deepseek_detector.py      # DeepSeek implementation
├── comparison/
│   ├── benchmark_runner.py       # Sequential benchmark execution
│   ├── statistical_analyzer.py   # Statistical tests
│   ├── comparison_visualizer.py  # Plotly visualizations
│   └── report_generator.py       # Markdown report generation
└── test_multi_model_comparison.py # Main script
```

### Base Detector Interface

All detectors inherit from `BaseErrorDetector` which provides:

```python
# Core methods
compute_token_log_probabilities(code: str) -> List[Tuple[str, int, float, int]]
localize_errors(code: str, k: float = None) -> Dict
compare_buggy_vs_correct(buggy: str, correct: str, k: float = None) -> Dict

# Statistical analysis
compute_statistical_threshold(log_probs: List[float], k: float) -> Tuple[float, float, float]
identify_anomalous_tokens(...) -> List[TokenErrorInfo]
aggregate_to_line_level(...) -> List[LineErrorInfo]
```

### Statistical Tests

#### McNemar's Test (Pairwise)
- Tests if two models perform significantly differently
- Creates 2x2 contingency table for each pair
- Reports p-value and significance (p < 0.05)

#### Friedman Test (Overall)
- Non-parametric test for overall ranking
- Tests if any model performs significantly better
- Provides interpretation of results

#### Cohen's d (Effect Size)
- Measures practical significance between models
- Values: small (0.2), medium (0.5), large (0.8)

## Test Dataset

Uses `TestExamplesDataset` with 10 examples covering:

### Bug Types
- **Logic Errors** - Off-by-one, incorrect conditionals
- **Edge Cases** - Missing boundary checks, null handling
- **Algorithm Errors** - Wrong implementation logic

### Examples
1. Binary search (missing bounds check)
2. Factorial (missing n=0 case)
3. Fibonacci (wrong base case)
4. List reversal (off-by-one error)
5. String palindrome (edge case)
6. Prime checker (missing n=2 case)
7. Max element (empty array)
8. Remove duplicates (logic error)
9. Is sorted (edge case)
10. Merge sorted lists (index error)

## Hardware Requirements

### Recommended
- **GPU:** NVIDIA A100 80GB (tested and verified)
- **RAM:** 32GB+ system memory
- **CUDA:** 12.4+ compatible

### Minimum
- **GPU:** RTX 3070/4060 with 8GB+ VRAM
- **RAM:** 16GB+ system memory
- **CUDA:** 11.8+

### Model Memory Requirements
- StarCoder2-7B: ~14GB GPU memory (BF16)
- CodeT5+ 2B: ~4GB GPU memory (FP16)
- DeepSeek-Coder 6.7B: ~14GB GPU memory (BF16)
- CodeBERT: ~500MB GPU memory (FP32)
- Qwen 7B: ~14GB GPU memory (BF16)

**Note:** Models are tested sequentially to avoid memory issues. GPU cache is cleared between models.

## Technical Details

### Full Precision Loading (No Quantization)

All models are loaded in full precision as specified:

```python
# Causal LM models (StarCoder2, DeepSeek, Qwen)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Full BF16
    device_map="auto",
    trust_remote_code=True
)

# Encoder-Decoder models (CodeT5+)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Full FP16
    device_map="auto"
)
```

### Sequential Execution

Models are tested one at a time to avoid memory issues:

```python
for model_key in models_to_test:
    detector = load_detector(model_key)
    results = test_model(detector)
    save_results(results)

    # Free memory before next model
    del detector
    torch.cuda.empty_cache()
```

### Probability Computation Approaches

#### Causal LM (Autoregressive)
- Used by: StarCoder2, DeepSeek, Qwen
- P(token[i] | token[0], ..., token[i-1])
- Single forward pass for all tokens

#### Masked LM (Bidirectional)
- Used by: CodeBERT, CodeT5+
- P(token[i] | all other tokens)
- Separate forward pass per token (slower)

## Expected Results

Based on initial testing:

### Typical Confirmation Rates
- StarCoder2-7B: 50-60%
- DeepSeek-Coder 6.7B: 55-65%
- CodeT5+ 2B: 45-55%
- CodeBERT: 40-50%
- Qwen 7B: 45-55%

**Note:** Results may vary based on:
- Test dataset composition
- Sensitivity factor (k)
- Bug complexity

### Statistical Significance
- Friedman test typically shows p < 0.05 (significant differences exist)
- Pairwise McNemar tests identify which specific pairs differ significantly

## Customization

### Adding New Models

1. Create detector class inheriting from `BaseErrorDetector`:

```python
from detectors.base_detector import BaseErrorDetector, DetectorMetadata

class MyModelDetector(BaseErrorDetector):
    def __init__(self, sensitivity_factor: float = 1.5):
        super().__init__(sensitivity_factor)
        # Load your model

    def compute_token_log_probabilities(self, code: str):
        # Implement probability computation
        pass

    def get_metadata(self) -> DetectorMetadata:
        return DetectorMetadata(
            model_name="My Model",
            model_type="causal",
            parameters="X billion",
            # ...
        )
```

2. Register in `benchmark_runner.py`:

```python
self.detector_configs = {
    'my-model': {
        'class': MyModelDetector,
        'name': 'My Model',
        'enabled': True
    },
    # ... existing models
}
```

### Adjusting Test Dataset

Modify `test_examples.py` to add/remove examples:

```python
dataset = TestExamplesDataset()
dataset.add_example(TestExample(
    name="my_test",
    prompt="Write a function to...",
    buggy_code="...",
    correct_code="...",
    description="...",
    bug_type="logic_error"
))
```

### Changing Sensitivity Factor

Higher k = less sensitive (fewer anomalies):
```bash
python test_multi_model_comparison.py --sensitivity 2.0
```

Lower k = more sensitive (more anomalies):
```bash
python test_multi_model_comparison.py --sensitivity 1.0
```

## Troubleshooting

### Out of Memory Errors

1. Test fewer models:
```bash
python test_multi_model_comparison.py --models codebert qwen-7b
```

2. Reduce test dataset size in `benchmark_runner.py`

3. Use smaller models (CodeBERT, CodeT5+)

### Model Loading Failures

- Ensure HuggingFace transformers is up to date: `pip install -U transformers`
- Check CUDA compatibility: `torch.cuda.is_available()`
- Verify internet connection for model downloads

### Slow Execution

- Expected: Each model takes 5-15 minutes for 10 examples
- CodeT5+ is slowest (MLM requires multiple forward passes per token)
- StarCoder2/DeepSeek are faster (single forward pass)

## Performance Benchmarks

On NVIDIA A100 80GB:

| Model | Load Time | Per Example | Total (10 examples) |
|-------|-----------|-------------|---------------------|
| StarCoder2-7B | ~30s | ~30s | ~5min |
| CodeT5+ 2B | ~20s | ~60s | ~10min |
| DeepSeek-6.7B | ~30s | ~30s | ~5min |
| CodeBERT | ~5s | ~45s | ~8min |
| Qwen 7B | ~30s | ~30s | ~5min |

**Total study runtime:** ~35-45 minutes for all 5 models

## Citation

If you use this comparison framework, please cite the original LecPrompt paper:

```
@article{lecprompt2023,
  title={LecPrompt: A Prompt-based Approach for Logical Error Correction with CodeBERT},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```

## Related Work

- **LecPrompt** - Statistical anomaly detection for error localization
- **CodeBERT** - Pre-trained model for programming and natural languages
- **StarCoder** - Open-source code generation model
- **DeepSeek-Coder** - Recent state-of-the-art code model

## Future Work

Potential extensions:

1. **More models:** GPT-4, Claude, Gemini, Codex
2. **Larger dataset:** HumanEval, MBPP benchmarks
3. **Fine-tuning:** Train models specifically for error detection
4. **Ensemble methods:** Combine predictions from multiple models
5. **Semantic analysis:** Incorporate AST-based features

## License

See project root LICENSE file.

## Contact

For questions or issues, please refer to the main project documentation.
