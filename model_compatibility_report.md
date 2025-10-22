# 🔧 Model Compatibility Report - LLM Token Probability Analysis

## ✅ Successfully Tested Models

### 1. **meta-llama/Llama-3.2-3B-Instruct**
- **Status**: ✅ Fully Compatible
- **Performance**: ~27.6s per analysis (150 tokens)
- **Memory Usage**: Moderate (~4GB VRAM)
- **Hypothesis Validation**: Mixed results (75% probability confirmation)
- **Notes**: Fast, reliable, good for quick analysis

### 2. **Qwen/Qwen2.5-Coder-32B-Instruct**
- **Status**: ✅ Fully Compatible
- **Performance**: ~189.9s per analysis (150 tokens)
- **Memory Usage**: High (~30GB VRAM)
- **Hypothesis Validation**: Mixed results (75% probability confirmation)
- **Notes**: Slower but high-capacity model, requires significant GPU memory

### 3. **Qwen/Qwen2.5-Coder-7B-Instruct**
- **Status**: ✅ Fully Compatible
- **Performance**: ~30.4s per analysis (150 tokens)
- **Memory Usage**: Moderate (~8GB VRAM)
- **Hypothesis Validation**: Moderate results (50% probability confirmation)
- **Notes**: Good balance of speed and performance

### 4. **google/gemma-3-270m-it**
- **Status**: ✅ Fully Compatible
- **Performance**: ~32.9s per analysis (150 tokens)
- **Memory Usage**: Low (~1GB VRAM)
- **Hypothesis Validation**: Lower results (42% probability confirmation)
- **Notes**: Fastest model, good for resource-constrained environments

## ❌ Compatibility Issues

### 1. **microsoft/Phi-4-mini-instruct**
- **Status**: ❌ Import Error
- **Error**: `ImportError: cannot import name 'LossKwargs' from 'transformers.utils'`
- **Cause**: Newer model architecture incompatible with current transformers version
- **Tested Versions**: transformers 4.56.2
- **Potential Solution**: Wait for transformers library update or use older Phi models

### 2. **microsoft/Phi-3-mini-4k-instruct**
- **Status**: ❌ Runtime Error
- **Error**: `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`
- **Cause**: Cache implementation incompatibility
- **Impact**: Model loads but fails during generation
- **Potential Solution**: Use different attention implementation or wait for fix

## 📊 Performance Summary

| Model | Size | Avg Time | Memory | Compatibility | Hypothesis Support |
|-------|------|----------|--------|---------------|-------------------|
| **Llama 3.2 3B** | 3B | 27.6s | ~4GB | ✅ Perfect | 🟡 Moderate (58%) |
| **Qwen 32B** | 32B | 189.9s | ~30GB | ✅ Perfect | 🟡 Moderate (58%) |
| **Qwen 7B** | 7B | 30.4s | ~8GB | ✅ Perfect | 🟡 Moderate (50%) |
| **Gemma 270M** | 270M | 32.9s | ~1GB | ✅ Perfect | 🟡 Lower (42%) |
| **Phi-4 Mini** | ~14B | - | - | ❌ Import Error | - |
| **Phi-3 Mini** | ~4B | - | - | ❌ Runtime Error | - |

## 🎯 Analysis Results Summary

### Hypothesis Testing Results
**Hypothesis**: Low probability tokens correlate with buggy code areas

#### Results by Model:
- **Llama 3.2 3B**: 58% overall confirmation (moderate support)
- **Qwen 32B**: 58% overall confirmation (moderate support)
- **Qwen 7B**: 50% overall confirmation (weak support)
- **Gemma 270M**: 42% overall confirmation (weak support)

#### Key Findings:
1. **Model Size ≠ Hypothesis Support**: Larger models don't necessarily show stronger correlation
2. **Consistent Patterns**: All models show similar confidence levels (>0.97 avg probability)
3. **Limited Differentiation**: Small differences between buggy and correct code probabilities
4. **Low Confidence Regions**: All models showed 0 low-confidence regions, suggesting high overall confidence

## 🚨 Technical Issues Encountered

### Transformers Library Compatibility
- **Phi-4**: Requires newer transformers features not yet available
- **Phi-3**: Cache implementation conflicts with current transformers version
- **Recommendation**: Monitor transformers releases for Phi model support

### Memory Requirements
- **32B Model**: Requires A100 80GB for comfortable operation
- **GPU Utilization**: Models efficiently use available VRAM
- **Recommendation**: Use smaller models for resource-constrained environments

## 💡 Recommendations

### For Production Use:
1. **Best Overall**: `Qwen/Qwen2.5-Coder-7B-Instruct` - Good balance of speed/performance
2. **High Performance**: `Qwen/Qwen2.5-Coder-32B-Instruct` - For detailed analysis with ample resources
3. **Fast Analysis**: `meta-llama/Llama-3.2-3B-Instruct` - Quick results with moderate accuracy
4. **Resource Constrained**: `google/gemma-3-270m-it` - Minimal memory usage

### For Research:
- Test with additional examples to validate hypothesis
- Experiment with different temperature settings
- Investigate why low-confidence regions are rare
- Compare with human-annotated bug locations

### For Development:
- Monitor Phi model compatibility in future transformers releases
- Consider implementing alternative attention mechanisms
- Test with larger diverse code examples dataset

## 📈 Future Work

1. **Extended Testing**: More examples and bug types
2. **Alternative Models**: CodeT5, StarCoder, CodeGen variants
3. **Hyperparameter Tuning**: Temperature, top-p optimization
4. **Human Validation**: Compare with expert bug identification
5. **Real-world Datasets**: Test on actual codebases with known bugs

---

**Generated**: 2025-09-21
**GPU**: NVIDIA A100 80GB PCIe
**CUDA**: 12.4
**Environment**: Python 3.13, transformers 4.56.2, torch 2.8.0