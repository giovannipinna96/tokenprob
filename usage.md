# 🧠 LLM Token Probability Analysis System - Usage Guide

Questo documento fornisce tutte le informazioni necessarie per utilizzare il sistema di analisi delle probabilità dei token LLM per identificare aree di codice potenzialmente problematiche.

## 📋 Indice

- [Requisiti Hardware](#-requisiti-hardware)
- [Setup dell'Ambiente](#-setup-dellambiente)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Utilizzo Base](#-utilizzo-base)
- [Esempi di Test](#-esempi-di-test)
- [Analisi Completa](#-analisi-completa)
- [Interpretazione dei Risultati](#-interpretazione-dei-risultati)
- [Struttura Output JSON](#-struttura-output-json)
- [Esempi Pratici](#-esempi-pratici)
- [Risoluzione Problemi](#-risoluzione-problemi)

## 🖥️ Requisiti Hardware

### Hardware Verificato
- **GPU**: NVIDIA A100 80GB PCIe (consigliata)
- **CUDA**: 12.4+
- **RAM**: 16GB+ (32GB consigliati)
- **Storage**: 20GB+ spazio libero

### Hardware Minimo
- **GPU**: NVIDIA RTX 3070/4060 (8GB+ VRAM)
- **RAM**: 8GB+
- **Storage**: 10GB+ spazio libero

## 🚀 Setup dell'Ambiente

### 1. Verificare GPU e CUDA

```bash
# Verificare la GPU
nvidia-smi

# Output atteso:
# NVIDIA A100 80GB PCIe
# CUDA Version: 12.4+
```

### 2. Configurazione Ambiente Python

```bash
# Clona/naviga nella directory del progetto
cd tokenprob

# Il progetto usa uv per la gestione delle dipendenze
# Installa le dipendenze principali (già fatto se hai seguito setup)
uv sync

# Verifica installazione PyTorch con CUDA
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Struttura del Progetto

```
tokenprob/
├── LLM.py                    # Engine di analisi del modello
├── visualizer.py             # Sistema di visualizzazione
├── test_examples.py          # Dataset di esempi test
├── run_analysis.py           # Runner per analisi complete
├── example_usage.py          # Script di esempio
├── usage.md                  # Questa guida
├── CLAUDE.md                 # Configurazione per Claude Code
├── pyproject.toml            # Configurazione progetto
├── requirements.txt          # Dipendenze (riferimento)
└── analysis_results/         # Directory risultati analisi
```

## 🎯 Utilizzo Base

### 1. Test Rapido del Sistema

```bash
# Test con il modello di default (Qwen 2.5 Coder 7B)
uv run python LLM.py

# Test con modello personalizzato
uv run python LLM.py "microsoft/CodeT5-large"
```

### 2. Eseguire Esempi Dimostrativi

```bash
# Tutti gli esempi
uv run python example_usage.py

# Esempio specifico
uv run python example_usage.py --example simple

# Con browser automatico
uv run python example_usage.py --example simple --open-browser

# Modello personalizzato
uv run python example_usage.py --model "Qwen/Qwen2.5-Coder-7B-Instruct"
```

## 🧪 Esempi di Test

### Generare Dataset di Test

```bash
# Crea il dataset con esempi di codice buggato/corretto
uv run python test_examples.py

# Output:
# - test_examples.json (10 esempi strutturati)
# - test_prompts.json (20 prompt per test)
```

### Tipi di Esempi Disponibili

Il sistema include 10 esempi categorizzati per tipo di bug:

#### Logic Errors (6 esempi)
- **binary_search_missing_bounds**: Bounds incorretti nell'array
- **bubble_sort_inner_loop**: Loop interno inefficiente
- **string_reverse_indexing**: Direzione iterazione sbagliata
- **prime_check_optimization**: Algoritmo non ottimizzato
- **merge_arrays_index_bounds**: Implementazione incompleta
- **count_vowels_case_sensitivity**: Gestione case mancante

#### Edge Cases (4 esempi)
- **factorial_recursion_base_case**: Base case mancante per n=0
- **list_max_empty_check**: Validazione lista vuota assente
- **fibonacci_negative_input**: Validazione input negativo mancante
- **division_zero_check**: Protezione divisione per zero assente

## 📊 Analisi Completa

### Analisi Singolo Esempio

```bash
# Analizza specifico esempio
uv run python run_analysis.py --example binary_search_missing_bounds

# Con modello personalizzato
uv run python run_analysis.py --example factorial_recursion_base_case --model "microsoft/CodeT5-large"

# Output directory personalizzata
uv run python run_analysis.py --example prime_check_optimization --output-dir my_results
```

### Analisi Completa di Tutti gli Esempi

```bash
# Analisi completa (può richiedere 2-3 ore)
uv run python run_analysis.py

# Con parametri personalizzati
uv run python run_analysis.py --model "Qwen/Qwen2.5-Coder-7B-Instruct" --output-dir full_analysis
```

### Struttura Output Analisi

```
analysis_results/
├── complete_analysis.json           # Risultati completi
├── analysis_summary.md             # Report riassuntivo
├── binary_search_missing_bounds_analysis.json
├── factorial_recursion_base_case_analysis.json
└── ...                             # Un file per ogni esempio
```

## 🔍 Interpretazione dei Risultati

### Metriche Chiave

#### **Probability (Probabilità)**
- **Range**: 0.0 - 1.0
- **Significato**: Confidenza del modello nel token selezionato
- **❌ Problematico**: < 0.3 (bassa confidenza)
- **✅ Buono**: > 0.7 (alta confidenza)

#### **Rank (Classifica)**
- **Range**: 1 - vocab_size
- **Significato**: Posizione del token quando ordinati per probabilità
- **❌ Problematico**: > 100 (scelta inusuale)
- **✅ Buono**: 1-10 (scelta ovvia)

#### **Entropy (Entropia)**
- **Range**: 0.0 - log₂(vocab_size)
- **Significato**: Incertezza nella distribuzione di probabilità
- **❌ Problematico**: > 5.0 (molte alternative considerate)
- **✅ Buono**: < 1.0 (decisione chiara)

#### **Surprisal**
- **Range**: 0.0 - ∞
- **Significato**: Quanto "sorprendente" è stato il token selezionato
- **❌ Problematico**: > 5.0 (token inaspettato)
- **✅ Buono**: < 1.0 (token previsto)

### Indicatori di Codice Problematico

#### 🚨 Segnali di Alto Rischio
- **Average Probability < 0.4**: Bassa confidenza generale
- **Multiple Low-Confidence Regions**: Zone di incertezza multiple
- **High Entropy (> 5.0)**: Molte alternative considerate
- **Consecutive Low-Probability Tokens**: Possibili zone di errore

#### ✅ Segnali Positivi
- **Average Probability > 0.6**: Alta confidenza
- **Few Low-Confidence Regions**: Poche zone problematiche
- **Low Entropy (< 1.0)**: Decisioni chiare
- **Consistent High-Probability Patterns**: Generazione stabile

### Schema Colori Visualizzazioni

- 🟢 **Verde**: Alta confidenza (prob > 0.7) - Probabile codice corretto
- 🟡 **Giallo**: Confidenza media (0.3 ≤ prob ≤ 0.7) - Da monitorare
- 🔴 **Rosso**: Bassa confidenza (prob < 0.3) - Possibili problemi

## 📝 Struttura Output JSON

### File di Analisi Singola

```json
{
  "example": {
    "name": "binary_search_missing_bounds",
    "description": "Binary search with incorrect array bounds",
    "bug_type": "logic",
    "prompt": "Write a Python function...",
    "buggy_code": "def binary_search(arr, target):\n    left = 0\n    right = len(arr)  # Bug",
    "correct_code": "def binary_search(arr, target):\n    left = 0\n    right = len(arr) - 1  # Correct"
  },
  "buggy_analysis": {
    "statistics": {
      "avg_probability": 0.65,
      "min_probability": 0.15,
      "max_probability": 1.0,
      "avg_rank": 2.3,
      "avg_entropy": 1.8,
      "total_tokens": 150
    },
    "low_confidence_regions": 3,
    "low_confidence_tokens": [...],
    "token_analyses": [
      {
        "token": "def",
        "position": 0,
        "probability": 0.95,
        "rank": 1,
        "entropy": 0.2,
        "surprisal": 0.07,
        "top_10_tokens": [
          {
            "token_id": 1234,
            "probability": 0.95,
            "token_text": "def"
          },
          {
            "token_id": 5678,
            "probability": 0.03,
            "token_text": "function"
          },
          ...
        ]
      },
      ...
    ]
  },
  "correct_analysis": { ... },
  "comparison": {
    "probability_difference": {
      "buggy_avg": 0.65,
      "correct_avg": 0.85,
      "difference": -0.20,
      "hypothesis_confirmed": true
    },
    "low_confidence_comparison": {
      "buggy_regions": 3,
      "correct_regions": 1,
      "difference": 2,
      "hypothesis_confirmed": true
    }
  }
}
```

### Campi Importanti per l'Analisi

#### **top_10_tokens**: Lista dei 10 token più probabili per ogni posizione
- `token_id`: ID numerico del token
- `probability`: Probabilità del token (0.0-1.0)
- `token_text`: Testo decodificato del token

#### **low_confidence_tokens**: Token con probabilità nel percentile più basso (default: 20%)
- `position`: Posizione del token nella sequenza
- `tokens`: Lista dei token nella regione a bassa confidenza

#### **comparison**: Confronto tra codice buggato e corretto
- `hypothesis_confirmed`: true/false se l'ipotesi è confermata

## 🎯 Esempi Pratici

### Esempio 1: Analisi Rapida

```bash
# Analizza un esempio di ricerca binaria
uv run python run_analysis.py --example binary_search_missing_bounds

# Controlla risultati
cat analysis_results/binary_search_missing_bounds_single_analysis.json | jq '.comparison'
```

### Esempio 2: Confronto Modelli

```bash
# Testa con Qwen
uv run python run_analysis.py --example factorial_recursion_base_case --model "Qwen/Qwen2.5-Coder-7B-Instruct" --output-dir qwen_results

# Testa con altro modello (se disponibile)
uv run python run_analysis.py --example factorial_recursion_base_case --model "microsoft/CodeT5-large" --output-dir codet5_results
```

### Esempio 3: Analisi Personalizzata

```python
# Script personalizzato
from LLM import QwenProbabilityAnalyzer
from visualizer import TokenVisualizer

# Inizializza
analyzer = QwenProbabilityAnalyzer(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

# Analizza codice personalizzato
prompt = "Write a function to sort an array"
text, analyses = analyzer.generate_with_analysis(prompt, max_new_tokens=100)

# Salva risultati con top-10 token
analyzer.save_analysis("my_analysis.json")

# Crea visualizzazione
visualizer = TokenVisualizer()
html = visualizer.create_html_visualization(analyses)
with open("my_visualization.html", "w") as f:
    f.write(html)
```

## 🔧 Risoluzione Problemi

### Problemi Comuni

#### **OutOfMemoryError (CUDA)**
```bash
# Riduci parametri generazione
# In LLM.py, modifica max_new_tokens (default: 150 → 50)
uv run python run_analysis.py --example simple_example
```

#### **Modello Non Trovato**
```bash
# Verifica modelli disponibili su HuggingFace
# Usa modello di default
uv run python run_analysis.py --example binary_search_missing_bounds
```

#### **JSON Serialization Error**
```bash
# Il sistema dovrebbe gestire automaticamente i tipi numpy
# Se persiste, verifica versione numpy
uv run python -c "import numpy; print(numpy.__version__)"
```

#### **Lentezza Eccessiva**
```bash
# Per testing rapido, usa esempi singoli
uv run python run_analysis.py --example factorial_recursion_base_case

# Monitora GPU usage
watch -n 1 nvidia-smi
```

### Debug e Logging

```bash
# Controlla log di sistema
uv run python run_analysis.py --example binary_search_missing_bounds 2>&1 | tee analysis.log

# Verifica file generati
ls -la analysis_results/
du -sh analysis_results/*
```

### Performance Ottimizzazione

#### **Impostazioni Consigliate per A100**
- `max_new_tokens`: 150-200 (default ottimale)
- `temperature`: 0.1 (più deterministic)
- `batch_size`: 1 (per analisi dettagliata)

#### **Per GPU più Piccole**
- `max_new_tokens`: 50-100
- Analizza esempi singoli invece di batch completi
- Considera modelli più piccoli se disponibili

## 📈 Interpretazione Avanzata

### Correlazioni Significative

#### **Ipotesi Confermata (Strong Signal)**
- Buggy avg_probability < 0.4 AND Correct avg_probability > 0.7
- Buggy low_confidence_regions > 3 AND Correct low_confidence_regions < 2
- Buggy avg_entropy > 3.0 AND Correct avg_entropy < 1.0

#### **Ipotesi Supportata (Moderate Signal)**
- Differenza probabilità > 0.2 a favore del codice corretto
- Differenza regioni a bassa confidenza > 1
- Pattern consistenti attraverso multipli esempi

#### **Risultati Inconclusivi**
- Differenze minime (< 0.1) nelle metriche
- Pattern inconsistenti tra esempi
- Alta variabilità nella generazione

### Utilizzo dei Top-10 Token

I top-10 token per ogni posizione forniscono insight preziosi:

1. **Diversità Alternative**: Molte alternative high-probability indicano incertezza
2. **Semantic Similarity**: Token simili nell'elenco suggeriscono coerenza
3. **Ranking Patterns**: Grandi gap di probabilità indicano scelte chiare

---

## 🎯 Conclusioni

Questo sistema fornisce un framework completo per analizzare la relazione tra incertezza del modello e qualità del codice. L'utilizzo sistematico di queste metriche può aiutare a:

1. **Identificare aree di codice potenzialmente problematiche**
2. **Sviluppare metriche di qualità basate su AI**
3. **Migliorare i processi di code review**
4. **Guidare lo sviluppo di strumenti di assistenza alla programmazione**

Per domande o problemi specifici, riferirsi alla documentazione tecnica in `CLAUDE.md` o ai commenti nel codice sorgente.