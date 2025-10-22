# 🎨 Multi-Model Visualization Report - LLM Token Probability Analysis

## ✅ Summary

Ho completato con successo la generazione delle visualizzazioni per tutti i modelli richiesti (escluso Qwen 32B per motivi di performance).

### 📊 Risultati Generati

**Modelli Testati**: 3/3 ✅
- ✅ **meta-llama/Llama-3.2-3B-Instruct**
- ✅ **Qwen/Qwen2.5-Coder-7B-Instruct**
- ✅ **google/gemma-3-270m-it**

**File Generati**:
- **40 file HTML** interattivi con visualizzazioni
- **9 file JSON** con analisi complete e top-10 token
- **4 file index.html** per navigazione

### 🗂️ Struttura Directory

```
model_visualizations/
├── index.html                           # 🌐 Main overview page
├── meta_llama_Llama_3.2_3B_Instruct/   # 📁 Llama 3.2 3B results
│   ├── index.html                       # Model-specific overview
│   ├── fibonacci_bug_probability.html   # 🎨 Interactive visualizations
│   ├── fibonacci_bug_entropy.html
│   ├── fibonacci_bug_surprisal.html
│   ├── fibonacci_bug_rank.html
│   ├── fibonacci_bug_analysis.json      # 📊 Raw data with top-10 tokens
│   └── ... (12 visualizations + 3 JSON files)
├── Qwen_Qwen2.5_Coder_7B_Instruct/     # 📁 Qwen 7B results
│   └── ... (same structure)
└── google_gemma_3_270m_it/              # 📁 Gemma 270M results
    └── ... (same structure)
```

## 🎯 Test Examples per Model

Ogni modello è stato testato con **3 esempi di codice**:

### 1. **fibonacci_bug**
- Prompt: "Write a Python function to calculate the nth Fibonacci number using recursion"
- Focus: Algoritmo ricorsivo con potenziali edge case

### 2. **binary_search_correct**
- Prompt: "Implement binary search algorithm in Python"
- Focus: Algoritmo di ricerca con gestione bounds

### 3. **factorial_with_validation**
- Prompt: "Create a factorial function with proper input validation"
- Focus: Funzione con validazione input e gestione errori

## 🎨 Modalità di Visualizzazione

Ogni esempio include **4 modalità di visualizzazione**:

### 1. **PROBABILITY** 🟢🟡🔴
- **Colori**: Verde (alta confidenza) → Giallo → Rosso (bassa confidenza)
- **Utilità**: Identifica token a bassa probabilità

### 2. **ENTROPY** 🔴🔵
- **Colori**: Rosso (alta incertezza) → Blu (bassa incertezza)
- **Utilità**: Mostra punti decisionali difficili

### 3. **SURPRISAL** 🔴🔵
- **Colori**: Rosso (sorprendente) → Blu (previsto)
- **Utilità**: Evidenzia scelte inaspettate

### 4. **RANK** 🔴🟢
- **Colori**: Rosso (alto rank/inusuale) → Verde (basso rank/comune)
- **Utilità**: Mostra quanto erano ovvie le scelte

## 📈 Performance dei Modelli

| Modello | Load Time | Tokens/Esempio | Visualizzazioni | Note |
|---------|-----------|----------------|-----------------|------|
| **Llama 3.2 3B** | 3.7s | 100 | 12 | ⚡ Veloce, efficiente |
| **Qwen 7B** | 5.6s | 100 | 12 | 🎯 Bilanciato |
| **Gemma 270M** | 2.7s | 100 | 12 | 🏃 Più veloce, leggero |

## 📊 Dati JSON con Top-10 Token

Ogni analisi JSON include:

```json
{
  "model_name": "google/gemma-3-270m-it",
  "generation_stats": {
    "avg_probability": 0.993,
    "avg_entropy": 0.018,
    "total_tokens": 100
  },
  "tokens": [
    {
      "token": "```",
      "position": 0,
      "probability": 1.0,
      "rank": 1,
      "entropy": 0.0,
      "surprisal": 0.0,
      "top_10_tokens": [
        {
          "token_id": 2717,
          "probability": 1.0,
          "token_text": "```"
        },
        {
          "token_id": 0,
          "probability": 0.0,
          "token_text": "<pad>"
        },
        ...
      ]
    }
  ]
}
```

### 🔍 Informazioni sui Top-10 Token

Per ogni token generato, il sistema salva:
- **token_id**: ID numerico nel vocabolario
- **probability**: Probabilità assegnata dal modello (0.0-1.0)
- **token_text**: Testo decodificato del token

Questo permette analisi dettagliate di:
- **Alternative considerate** dal modello
- **Distribuzione di probabilità** completa
- **Diversità semantica** delle opzioni
- **Pattern di incertezza** del modello

## 🌐 Come Visualizzare i Risultati

### 1. **Overview Principale**
```bash
# Apri nel browser
open model_visualizations/index.html
```

### 2. **Visualizzazioni Specifiche per Modello**
```bash
# Llama 3.2 3B
open model_visualizations/meta_llama_Llama_3.2_3B_Instruct/index.html

# Qwen 7B
open model_visualizations/Qwen_Qwen2.5_Coder_7B_Instruct/index.html

# Gemma 270M
open model_visualizations/google_gemma_3_270m_it/index.html
```

### 3. **Visualizzazioni Interattive Dirette**
```bash
# Esempio: Fibonacci con Llama, modalità probabilità
open model_visualizations/meta_llama_Llama_3.2_3B_Instruct/fibonacci_bug_probability.html
```

## 🎯 Insights dalle Visualizzazioni

### Patterns Osservati

#### **Alta Confidenza Generale**
- Tutti i modelli mostrano probabilità medie > 0.99
- Pochi token con bassa confidenza identificati
- Rank medio vicino a 1 (scelte ovvie)

#### **Differenze tra Modelli**
- **Gemma 270M**: Più variabile nelle scelte
- **Llama 3.2 3B**: Confidenza costante
- **Qwen 7B**: Performance bilanciata

#### **Aree di Incertezza**
- Sintassi Python: alta confidenza
- Nomi variabili: moderata variabilità
- Logica algoritmica: generalmente stabile

## 💡 Utilizzo Pratico

### Per Ricercatori
1. **Analizza pattern** di incertezza nei modelli
2. **Confronta comportamenti** tra architetture diverse
3. **Identifica punti critici** nella generazione

### Per Sviluppatori
1. **Monitora qualità** del codice generato
2. **Flagga aree problematiche** per review
3. **Ottimizza prompt** basandosi su confidenza

### Per Analisi Qualitativa
1. **Visualizza decisioni** del modello
2. **Understand uncertainty** patterns
3. **Valida ipotesi** su qualità codice

## 🔄 Prossimi Passi

### Possibili Estensioni
1. **Più esempi di test**: Dataset più ampio
2. **Modelli aggiuntivi**: Quando compatibili
3. **Analisi temporale**: Evoluzione confidenza
4. **Correlazione umana**: Validazione con esperti

### Miglioramenti Tecnici
1. **Plot matplotlib**: Aggiungere grafici statici
2. **Analisi batch**: Processing multiplo
3. **Metriche avanzate**: Nuovi indicatori
4. **Dashboard dinamico**: Interface web

---

## 📁 File Generati - Riepilogo

- **📊 Total**: 49 files
- **🎨 HTML Visualizations**: 40 files (~28KB each)
- **📋 JSON Data**: 9 files (~180KB each, with top-10 tokens)
- **🏠 Index Pages**: 4 files (navigation)

### Storage Used
- **Directory size**: ~2.8MB
- **Per model**: ~950KB
- **Per visualization**: ~28KB HTML + ~180KB JSON

**✅ Tutti i file sono pronti per la visualizzazione e l'analisi!**

Puoi aprire `model_visualizations/index.html` nel browser per iniziare l'esplorazione delle visualizzazioni interattive.