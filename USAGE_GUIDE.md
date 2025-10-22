# 🚀 Token Probability Analysis - Usage Guide

Guida completa per eseguire l'analisi con tutti i 6 metodi implementati e visualizzare i risultati.

---

## 📋 Metodi Implementati

Il sistema ora supporta **6 metodi complementari** di rilevamento errori:

### Metodi Base (1-4)
1. **LecPrompt** - Baseline con log-probability
2. **Semantic Energy** - Analisi pre-softmax logits
3. **Conformal Prediction** - Garanzie statistiche formali (90% coverage)
4. **Attention Anomaly** - Analisi pattern di attenzione

### Metodi Avanzati (5-6) ⭐ NUOVI
5. **Semantic Context** - Coerenza semantica a livello di riga
6. **Masked Token Replacement (MTR)** - Confronto diretto con predizioni

---

## 🎯 Esecuzione su SLURM (Raccomandato)

### 1. Esecuzione Completa (Tutti i 6 Metodi)

```bash
# Submit job
cd /u/gpinna/phd_projects/tokenprob/tokenprob
sbatch run_advanced_methods_full.slurm

# Monitor job
squeue -u $USER

# Check logs in real-time
tail -f logs/tokenprob_full_<JOB_ID>.out
```

**Risorse allocate:**
- GPU: 1x A100 (80GB)
- RAM: 64GB
- CPU: 16 cores
- Tempo: 24h (tipicamente completa in 4-6h)

**Output generato:**
```
advanced_methods_comparison_full/
├── complete_comparison_results.json          # Risultati raw JSON
├── calibration_results/                      # Info calibrazione
├── result_*.json                             # Risultati per esempio
└── advanced_visualizations/                  # HTML interattivi
    ├── index.html                            # 🌟 START HERE
    ├── interactive_method_explorer.html      # Dashboard principale
    ├── methods_comparison_heatmap.html       # Matrice 6×8
    ├── anomaly_counts_comparison.html        # Bar chart
    ├── method_agreement_matrix.html          # 6×6 agreement
    ├── method_performance_radar.html         # 6-dimensional radar
    ├── venn_diagram_overlap.html             # Consensus
    └── token_level_multimethod_view_*.html   # Per-example details
```

---

## 💻 Esecuzione Locale (Interattiva)

### Opzione A: Tutti i metodi su tutti gli esempi

```bash
# Con GPU
uv run python test_advanced_methods.py --model starcoder2-7b

# Con modello più piccolo (se memoria limitata)
uv run python test_advanced_methods.py --model codebert
```

### Opzione B: Test su singolo esempio

```bash
# Test specifico
uv run python test_advanced_methods.py \
    --model starcoder2-7b \
    --example binary_search_missing_bounds
```

### Opzione C: Parametri personalizzati

```bash
# Sensitivity più alta (più sensibile agli errori)
uv run python test_advanced_methods.py \
    --model qwen-7b \
    --sensitivity 2.0 \
    --conformal-alpha 0.05 \
    --output my_custom_results
```

### Opzione D: Solo visualizzazioni (da risultati esistenti)

```bash
# Rigenera solo le visualizzazioni HTML
uv run python test_advanced_methods.py \
    --visualize-only \
    --input advanced_methods_comparison_full
```

---

## 📊 Come Visualizzare i Risultati

### 1. **Dashboard Principale** (Raccomandato)

Apri nel browser:
```
advanced_methods_comparison_full/advanced_visualizations/index.html
```

**Contenuto:**
- ✅ Informazioni calibrazione Conformal Prediction
- ✅ Badge dei 6 metodi (con "⭐ NEW" per MTR)
- ✅ Link a tutte le visualizzazioni
- ✅ Quick start guide

### 2. **Interactive Method Explorer**

Il più completo - confronto dinamico:
```
advanced_methods_comparison_full/advanced_visualizations/interactive_method_explorer.html
```

**Features:**
- Dropdown per selezionare esempi
- Confronto real-time tra metodi
- Token-level highlighting
- Agreement visualization

### 3. **Visualizzazioni Specifiche**

#### Heatmap (Hypothesis Confirmation)
```
methods_comparison_heatmap.html
```
- Matrice 6 metodi × 8 test examples
- Verde = Hypothesis confermata (buggy_anomalies > correct_anomalies)
- Rosso = Hypothesis non confermata

#### Agreement Matrix
```
method_agreement_matrix.html
```
- Matrice 6×6 di Jaccard similarity
- Mostra quanto i metodi concordano
- Valori 0-1 (1 = accordo perfetto)

#### Radar Chart
```
method_performance_radar.html
```
- 6 dimensioni di performance
- Confronto visivo tra metodi
- Identifica punti di forza/debolezza

#### Venn Diagram
```
venn_diagram_overlap.html
```
- Consensus tra metodi
- Quante anomalie rilevate da 1, 2, 3+ metodi

---

## 📈 Interpretazione Risultati

### Metriche Chiave

#### 1. **Confirmation Rate**
```
buggy_anomalies > correct_anomalies ✅
```
- Alta (>70%): Metodo affidabile
- Media (50-70%): Buono ma con falsi positivi
- Bassa (<50%): Serve tuning

#### 2. **Agreement Matrix**
```
Jaccard Similarity = |A ∩ B| / |A ∪ B|
```
- Alta (>0.7): Metodi complementari
- Media (0.4-0.7): Diversi approcci
- Bassa (<0.4): Rilevano errori diversi (ottimo per ensemble!)

#### 3. **Calibration Info** (Conformal Prediction)
```
Quantile Threshold: ~0.85-0.95
Coverage Target: 90%
Calibration Tokens: ~200-300
```
- Threshold alto = modello sicuro
- Copertura garantita formalmente

#### 4. **Mismatch Details** (MTR)
```json
{
  "original_token": "lenght",
  "predicted_token": "length",
  "prediction_confidence": 0.96
}
```
- **Interpretabile**: Mostra correzione suggerita
- Confidence alta (>0.8) = typo evidente

---

## 🔧 Troubleshooting

### Problema 1: Out of Memory (OOM)

**Soluzione:**
```bash
# Usa modello più piccolo
uv run python test_advanced_methods.py --model codebert

# Oppure testa un esempio alla volta
uv run python test_advanced_methods.py --example factorial_recursion_base_case
```

### Problema 2: Semantic Context non disponibile

**Causa:** sentence-transformers non installato

**Soluzione:**
```bash
pip install sentence-transformers
```

### Problema 3: MTR non disponibile

**Causa:** transformers non installato o CodeBERT non scaricato

**Soluzione:**
```bash
pip install transformers
# Il modello verrà scaricato automaticamente al primo run
```

### Problema 4: Calibrazione fallisce

**Causa:** Esempi di calibrazione insufficienti

**Soluzione:**
```bash
# Usa skip-calibration solo per debug
# La calibrazione è essenziale per garanzie formali
```

---

## 📁 Struttura File Principali

```
tokenprob/
├── test_advanced_methods.py          # 🎯 MAIN ENTRY POINT
├── run_advanced_methods_full.slurm   # SLURM job script
│
├── detectors/
│   ├── advanced_methods.py           # 6 metodi implementati
│   ├── starcoder2_detector.py        # Baseline detector
│   ├── deepseek_detector.py
│   ├── codet5_detector.py
│   └── ...
│
├── comparison/
│   ├── advanced_comparison_runner.py # Runner principale
│   └── advanced_visualizer.py        # Generatore HTML
│
├── test_examples.py                  # Dataset (10 esempi)
│   ├── Calibration set: 2 esempi
│   └── Test set: 8 esempi
│
└── advanced_methods_comparison_full/ # OUTPUT DIRECTORY
    ├── complete_comparison_results.json
    └── advanced_visualizations/
        └── *.html
```

---

## 🎓 Best Practices

### 1. **Sempre eseguire calibrazione**
La calibrazione Conformal Prediction fornisce garanzie statistiche formali. Non saltarla!

### 2. **Usa modelli appropriati**
- **GPU grande (A100)**: qwen-7b, starcoder2-7b, deepseek-6.7b
- **GPU media (RTX 3090)**: codebert, codet5p-2b
- **CPU/GPU piccola**: codebert (125M params)

### 3. **Interpreta i risultati in ensemble**
- Nessun metodo è perfetto
- Alta agreement = errore probabile
- Bassa agreement = falso positivo possibile
- MTR mostra **cosa suggerisce il modello** (interpretabile!)

### 4. **Salva i risultati**
```bash
# Backup results
cp -r advanced_methods_comparison_full/ results_backup_$(date +%Y%m%d)/
```

---

## 📞 Comandi Quick Reference

```bash
# Full run (SLURM)
sbatch run_advanced_methods_full.slurm

# Local quick test
uv run python test_advanced_methods.py --model codebert --example binary_search_missing_bounds

# Regenerate visualizations only
uv run python test_advanced_methods.py --visualize-only --input advanced_methods_comparison_full

# Check available models
uv run python test_advanced_methods.py --help

# Monitor SLURM job
squeue -u $USER
tail -f logs/tokenprob_full_*.out

# Cancel job
scancel <JOB_ID>
```

---

## 🎉 Expected Output Summary

Dopo l'esecuzione completa vedrai:

```
✅ Analysis completed successfully!

Generated Output:
  📁 Results Directory: advanced_methods_comparison_full/
  📊 Comparison Results: complete_comparison_results.json
  📈 Visualizations: advanced_visualizations/

View Results:
  1. Open: advanced_methods_comparison_full/advanced_visualizations/index.html
  2. Interactive Explorer: interactive_method_explorer.html
  3. Calibration Info: Displayed in index.html

Key Features:
  ✓ Conformal Prediction with formal calibration
  ✓ Semantic Context analysis (line-level coherence)
  ✓ Masked Token Replacement (direct prediction matching)
  ✓ 6×6 agreement matrix between all methods
  ✓ Method ranking with weighted scoring
```

---

## 📚 Documentazione Aggiuntiva

- **Metodi Overview**: `METHODS_OVERVIEW.md`
- **Project Overview**: `PROJECT_OVERVIEW.md`
- **Advanced Methods**: `ADVANCED_METHODS_README.md`
- **Conformal Prediction**: Vedi calibration info nell'HTML
- **MTR Details**: Commenti in `detectors/advanced_methods.py`

---

**Buona analisi!** 🚀

Per domande o problemi, controlla i log in `logs/` o rivedi questa guida.
