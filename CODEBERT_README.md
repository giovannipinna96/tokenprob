# CodeBERT-based Logical Error Detection

Implementazione fedele al paper originale **LecPrompt** utilizzando **CodeBERT** con Masked Language Modeling (MLM).

## 🎯 Differenza Fondamentale

### CodeBERT (Questa Implementazione)
```
Modello: microsoft/codebert-base
Tipo: RobertaForMaskedLM (Encoder, BERT-style)
Approccio: Masked Language Modeling
```

**Come funziona:**
```python
Per ogni token nella posizione i:
1. Maschera il token: code[i] → [MASK]
2. Il modello predice: P(token_originale | contesto con maschera)
3. Estrae log-probability del token originale
4. Token con bassa probabilità = anomalo
```

### Causal LM (Implementazione Alternativa)
```
Modello: Qwen/Llama (configurabile)
Tipo: AutoModelForCausalLM (Decoder, GPT-style)
Approccio: Autoregressive prediction
```

**Come funziona:**
```python
Per ogni token nella posizione i:
1. Usa contesto precedente: code[0]...code[i-1]
2. Il modello predice: P(code[i] | code[0]...code[i-1])
3. Estrae log-probability autoregressiva
4. Token con bassa probabilità = anomalo
```

## 📊 Confronto Visivo

### CodeBERT (MLM)
```
Codice: "def factorial(n):"

Token "n":
  Input:  "def factorial([MASK]):"
          CodeBERT → predice token mascherato
          P("n" | "def factorial([MASK]):") = 0.85
```

### Causal LM
```
Codice: "def factorial(n):"

Token "n":
  Input:  "def factorial("
          Qwen → predice token successivo
          P("n" | "def factorial(") = 0.82
```

## 🚀 Utilizzo

### 1. Analisi con CodeBERT

```bash
# Usa CodeBERT invece di Causal LM
python run_error_detection.py --example factorial_recursion_base_case --use-codebert

# Su file custom
python run_error_detection.py --code-file mycode.py --use-codebert
```

### 2. Test Suite Completa

```bash
# Test su tutti gli esempi con CodeBERT
python test_codebert_detector.py --all

# Test su esempio specifico
python test_codebert_detector.py --example binary_search_missing_bounds

# Confronto CodeBERT vs Causal LM
python test_codebert_detector.py --compare
```

### 3. Uso Programmatico

```python
from codebert_error_detector import CodeBERTErrorDetector

# Inizializza detector
detector = CodeBERTErrorDetector(
    model_name="microsoft/codebert-base",
    sensitivity_factor=1.5
)

# Analizza codice
code = """
def factorial(n):
    if n == 1:  # Bug: manca caso n=0
        return 1
    return n * factorial(n - 1)
"""

results = detector.localize_errors(code, k=1.5)

# Mostra errori rilevati
for line_error in results['line_errors']:
    if line_error.is_error_line:
        print(f"Line {line_error.line_number}: {line_error.line_content}")
        print(f"  Error score: {line_error.error_score:.3f}")
        print(f"  Anomalous tokens: {line_error.num_anomalous_tokens}")
```

## 🔬 Dettagli Tecnici

### Algoritmo MLM di CodeBERT

```python
def compute_token_log_probabilities_mlm(code):
    """
    Per ogni token:
    1. Tokenizza il codice
    2. Crea versione con token mascherato
    3. Fa forward pass con CodeBERT
    4. Estrae log-prob del token originale dalla previsione
    """
    tokens = tokenizer.tokenize(code)
    log_probs = []

    for i, token in enumerate(tokens):
        # Maschera token i
        masked_tokens = tokens.copy()
        masked_tokens[i] = '[MASK]'

        # Predici token mascherato
        outputs = model(masked_tokens)
        logits = outputs.logits[i]  # Logits per posizione i

        # Log-prob del token originale
        log_prob = F.log_softmax(logits)[token_id]
        log_probs.append(log_prob)

    return log_probs
```

### Soglia Statistica

Identico per entrambi gli approcci:

```
μ = (1/n) × Σpᵢ                    # Media
σ = √[(1/n) × Σ(pᵢ - μ)²]         # Deviazione standard
τ = μ - k×σ                        # Threshold (k=1.5 default)
anomalo = (pᵢ < τ)                 # Token anomalo
```

## 📈 Vantaggi e Svantaggi

### CodeBERT (MLM)

**✅ Vantaggi:**
- Fedele al paper originale LecPrompt
- Contesto bidirezionale (vede prima e dopo)
- Specializzato per code (pre-trained su codice)
- Buono per pattern recognition

**❌ Svantaggi:**
- Più lento (N forward pass, uno per token)
- Modello più piccolo (125M parametri)
- Richiede masking per ogni token

### Causal LM (Qwen/Llama)

**✅ Vantaggi:**
- Veloce (1 solo forward pass)
- Modelli più grandi e recenti (3B-32B parametri)
- Meglio per sequenze lunghe
- Integrato con resto del progetto

**❌ Svantaggi:**
- Contesto unidirezionale (solo prima)
- Meno fedele al paper originale

## 🎯 Quale Usare?

### Usa CodeBERT se:
- Vuoi replicare esattamente il paper LecPrompt
- Hai codice corto (< 512 token)
- Preferisci contesto bidirezionale
- Vuoi comparare con risultati del paper

### Usa Causal LM se:
- Vuoi velocità e scalabilità
- Hai codice lungo (> 512 token)
- Vuoi modelli più potenti (Qwen 32B)
- Integrazione con generazione di codice

## 📊 Risultati Attesi

Entrambi gli approcci dovrebbero dare risultati simili:

```
Esempio: factorial_recursion_base_case

CodeBERT:
  Buggy:   3 anomalous tokens
  Correct: 1 anomalous token
  ✓ Hypothesis confirmed

Causal LM (Qwen):
  Buggy:   4 anomalous tokens
  Correct: 1 anomalous token
  ✓ Hypothesis confirmed
```

## 🔧 Configurazione

### CodeBERT Models

```python
# Base (default)
detector = CodeBERTErrorDetector(model_name="microsoft/codebert-base")

# MLM variant
detector = CodeBERTErrorDetector(model_name="microsoft/codebert-base-mlm")
```

### Parametri

```python
detector = CodeBERTErrorDetector(
    model_name="microsoft/codebert-base",  # Modello
    device="auto",                          # "cuda", "cpu", "auto"
    sensitivity_factor=1.5                  # k per threshold
)
```

## 🧪 Test e Validazione

### Test Completo

```bash
# Test tutti gli esempi
python test_codebert_detector.py --all --output codebert_results.json

# Risultati attesi:
# - Total examples: 10
# - Confirmation rate: 50-70%
# - Logic errors: Alta detection rate
# - Edge cases: Media detection rate
```

### Confronto Diretto

```bash
# Confronta CodeBERT vs Qwen sullo stesso esempio
python test_codebert_detector.py --compare

# Output:
# Metric              CodeBERT    Causal LM
# Anomalous tokens    3           4
# Error lines         1           1
# Mean log prob       -2.45       -3.12
```

## 📝 Output Format

### Token-Level

```json
{
  "token": "1",
  "token_id": 112,
  "position": 8,
  "line_number": 2,
  "log_probability": -8.42,
  "is_anomalous": true,
  "deviation_score": -2.45,
  "error_likelihood": 0.82
}
```

### Line-Level

```json
{
  "line_number": 2,
  "line_content": "if n == 1:",
  "num_anomalous_tokens": 2,
  "num_tokens": 4,
  "error_score": 0.65,
  "is_error_line": true
}
```

## 🔍 Debug e Troubleshooting

### Issue: Model Loading Failed

```bash
# Soluzione: Installa transformers aggiornato
pip install transformers>=4.30.0
```

### Issue: CUDA Out of Memory

```bash
# Soluzione: Usa CPU
detector = CodeBERTErrorDetector(device="cpu")
```

### Issue: Slow Processing

CodeBERT richiede N forward passes (uno per token):
- Token < 100: ~5 secondi
- Token 100-500: ~20-60 secondi
- Token > 500: Considera Causal LM

## 🆚 Comparison Table

| Feature | CodeBERT | Causal LM |
|---------|----------|-----------|
| Approccio | Masked LM | Autoregressive |
| Velocità | Lento (N passes) | Veloce (1 pass) |
| Parametri | 125M | 3B-32B |
| Contesto | Bidirezionale | Unidirezionale |
| Fedeltà paper | ✓ Alta | ○ Media |
| Scalabilità | ○ Limitata | ✓ Alta |
| Code generation | ✗ No | ✓ Si |

## 📚 Riferimenti

**Paper Originale:**
```
LecPrompt: A Prompt-based Approach for Logical Error Correction with CodeBERT
arXiv:2410.08241
```

**CodeBERT:**
```
microsoft/codebert-base
Pre-trained on: Python, Java, JavaScript, PHP, Ruby, Go
Training: 2.1M code-comment pairs
```

## 🎓 Conclusioni

### Quando Usare CodeBERT:
- ✅ Ricerca accademica (fedeltà al paper)
- ✅ Benchmark e comparison
- ✅ Codice corto e specifico
- ✅ Validazione bidirezionale

### Quando Usare Causal LM:
- ✅ Produzione (velocità)
- ✅ Codice lungo
- ✅ Integrazione con generazione
- ✅ Modelli state-of-the-art

**Entrambi implementano la stessa idea fondamentale di LecPrompt:**
> Token con bassa probabilità = potenziali errori logici

---

**Implementato**: 2025-10-02
**Versione**: 1.0
**Modello**: microsoft/codebert-base
