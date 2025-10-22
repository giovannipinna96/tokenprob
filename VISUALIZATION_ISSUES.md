# 🎨 Visualization Issues Report

Questo documento descrive i problemi critici identificati nel sistema di visualizzazione dei risultati di rilevamento errori.

**Data**: 2025-01-19
**Analisi**: Revisione completa del codice senza esecuzione
**Impatto**: CRITICO - Impossibile visualizzare i token individuali evidenziati per ogni metodo

---

## 📋 Indice dei Problemi

1. [PROBLEMA #1: Manca visualizzazione token-level per ogni metodo](#problema-1-manca-visualizzazione-token-level-per-ogni-metodo-critico) ⭐⭐⭐ **PRIORITÀ MASSIMA**
2. [PROBLEMA #2: Manca index.html principale](#problema-2-manca-indexhtml-principale-alto) ⭐⭐ **PRIORITÀ ALTA**
3. [PROBLEMA #3: Nessuna integrazione tra visualizer.py e advanced methods](#problema-3-nessuna-integrazione-tra-visualizerpy-e-advanced-methods-alto) ⭐⭐ **PRIORITÀ ALTA**
4. [✅ OK: Logica di evidenziazione token è corretta](#ok-logica-di-evidenziazione-token-è-corretta) ✅

---

## PROBLEMA #1: Manca visualizzazione token-level per ogni metodo (CRITICO)

### 🔴 Problema Identificato

**Gravità**: 🔴 **CRITICO**

Non esiste una visualizzazione HTML che mostra **i singoli token evidenziati** per ogni metodo avanzato su ogni programma di test.

### ❌ Cosa Manca

**Visualizzazioni Token-Level Richieste**:

Per ogni programma (es. `binary_search_missing_bounds`), servono visualizzazioni che mostrano il **CODICE con TOKEN EVIDENZIATI** per:

1. **LecPrompt** (baseline) - Buggy version
2. **LecPrompt** (baseline) - Correct version
3. **Semantic Energy** - Buggy version
4. **Semantic Energy** - Correct version
5. **Conformal Prediction** - Buggy version
6. **Conformal Prediction** - Correct version
7. **Attention Anomaly** - Buggy version
8. **Attention Anomaly** - Correct version

**Totale per 10 programmi**: 10 × 4 metodi × 2 versioni = **80 visualizzazioni HTML**

### 📁 Cosa Esiste Attualmente

**File**: `comparison/advanced_visualizer.py`

Genera solo **7 visualizzazioni comparative**:

| # | Tipo | File | Contenuto |
|---|------|------|-----------|
| 1 | Heatmap | `methods_comparison_heatmap.html` | Matrice metodi × esempi (conferma ipotesi) |
| 2 | Bar Chart | `anomaly_counts_comparison.html` | Conteggi anomalie (buggy vs correct) |
| 3 | Agreement | `method_agreement_matrix.html` | Jaccard similarity tra metodi |
| 4 | Token View | `token_level_multimethod_view_{example}.html` | **Solo bar chart, NON token evidenziati** |
| 5 | Radar | `method_performance_radar.html` | Performance multi-dimensionale |
| 6 | Venn | `venn_diagram_overlap.html` | Livelli di consenso |
| 7 | Explorer | `interactive_method_explorer.html` | Dashboard interattivo |

**Problema**: Nessuna di queste mostra i **token individuali evidenziati**!

### 🔍 Analisi Dettagliata: `_create_token_view_for_example`

**File**: `comparison/advanced_visualizer.py`, linee 324-376

```python
def _create_token_view_for_example(self, result: Dict, methods: List[str], output_dir: str):
    """Create token-level view for a single example."""
    example_name = result['example_name']

    # Create bar chart showing anomaly counts per method
    buggy_counts = [result['buggy'][m]['num_anomalies'] for m in methods]
    correct_counts = [result['correct'][m]['num_anomalies'] for m in methods]

    fig = go.Figure()

    # ❌ SOLO BAR CHART - Nessuna visualizzazione token!
    fig.add_trace(go.Bar(
        name='Buggy Code',
        x=[self.method_names[m] for m in methods],
        y=buggy_counts,
        marker_color='#d62728',
        text=buggy_counts,
        textposition='auto'
    ))

    # ... altro codice bar chart
```

**Mancanza**: Questo crea solo un bar chart con conteggi. **NON mostra quali token specifici sono stati identificati come anomali**.

### 📊 Esempio di Cosa Serve

#### Visualizzazione Corretta (MANCANTE):

```html
<!-- binary_search_missing_bounds_lecprompt_buggy.html -->
<h2>Binary Search - LecPrompt - Buggy Code</h2>

<pre>
def binary_search(arr, target):
    <span style="background-color: #90ee90;">left</span> = <span style="background-color: #90ee90;">0</span>
    <span style="background-color: #ffcccc;">right</span> = <span style="background-color: #ffcccc;">len</span>(<span style="background-color: #90ee90;">arr</span>)  ← ANOMALY: missing -1

    while <span style="background-color: #90ee90;">left</span> <= <span style="background-color: #ffcccc;">right</span>:
        <span style="background-color: #90ee90;">mid</span> = ...
</pre>

<div class="legend">
    <span style="background-color: #ffcccc;">■</span> High uncertainty (potential error)
    <span style="background-color: #90ee90;">■</span> Low uncertainty (confident)
</div>
```

**Caratteristiche**:
- Codice completo con syntax preservation
- Token evidenziati con colori basati su incertezza
- Tooltip con metriche dettagliate al hover
- Legenda chiara con interpretazione

### 🛠️ Come Implementare

**Soluzione**: Usare `visualizer.py` che **GIÀ HA** questa funzionalità!

**File**: `visualizer.py`, linee 253-480

```python
def create_html_visualization(self,
                            analyses: List[TokenAnalysis],
                            mode: str = TokenVisualizationMode.PROBABILITY,
                            title: str = "Token Analysis Visualization") -> str:
    """
    Create HTML visualization of tokens with color coding.
    """
    # ... codice che:
    # 1. Estrae valori per il mode selezionato
    # 2. Normalizza valori
    # 3. Crea HTML con token evidenziati
    # 4. Aggiunge tooltip con metriche
    # 5. Aggiunge legenda
```

**Modes supportati** (linee 159-200):
- `SEMANTIC_ENERGY` - Energia semantica (valori alti = ROSSO)
- `CONFORMAL_SCORE` - Score conformal (valori alti = ROSSO)
- `ATTENTION_ENTROPY` - Entropia attention (valori alti = ROSSO)
- `ATTENTION_ANOMALY_SCORE` - Score anomalia attention
- `LOGICAL_ERROR_DETECTION` - Rilevamento errori LecPrompt

**Ma questo non viene chiamato per generare le visualizzazioni per ogni metodo!**

### 📝 Implementazione Necessaria

**Nuovo file**: `comparison/token_level_visualizer.py`

```python
class TokenLevelVisualizer:
    """
    Generate token-level HTML visualizations for each method on each example.
    """

    def generate_all_token_visualizations(self, results: Dict, output_dir: str):
        """
        Generate token-level visualizations for all methods on all examples.

        Genera:
        - {example}_lecprompt_buggy.html
        - {example}_lecprompt_correct.html
        - {example}_semantic_energy_buggy.html
        - {example}_semantic_energy_correct.html
        - {example}_conformal_buggy.html
        - {example}_conformal_correct.html
        - {example}_attention_buggy.html
        - {example}_attention_correct.html

        Per ogni esempio (10 esempi × 4 metodi × 2 versioni = 80 file HTML)
        """

        visualizer = TokenVisualizer()

        for result in results['individual_results']:
            example_name = result['example_name']

            # 1. LecPrompt
            self._generate_method_html(
                visualizer,
                result,
                'lecprompt',
                TokenVisualizationMode.LOGICAL_ERROR_DETECTION,
                output_dir
            )

            # 2. Semantic Energy
            self._generate_method_html(
                visualizer,
                result,
                'semantic_energy',
                TokenVisualizationMode.SEMANTIC_ENERGY,
                output_dir
            )

            # 3. Conformal Prediction
            self._generate_method_html(
                visualizer,
                result,
                'conformal',
                TokenVisualizationMode.CONFORMAL_SCORE,
                output_dir
            )

            # 4. Attention Anomaly
            self._generate_method_html(
                visualizer,
                result,
                'attention',
                TokenVisualizationMode.ATTENTION_ANOMALY_SCORE,
                output_dir
            )

    def _generate_method_html(self, visualizer, result, method, mode, output_dir):
        """Generate HTML for a single method on buggy and correct code."""
        example_name = result['example_name']

        # Buggy version
        buggy_analyses = self._get_token_analyses(result, method, 'buggy')
        buggy_html = visualizer.create_html_visualization(
            buggy_analyses,
            mode=mode,
            title=f"{example_name} - {method} - Buggy Code"
        )

        filename = f"{example_name}_{method}_buggy.html"
        self._save_html(buggy_html, os.path.join(output_dir, filename))

        # Correct version
        correct_analyses = self._get_token_analyses(result, method, 'correct')
        correct_html = visualizer.create_html_visualization(
            correct_analyses,
            mode=mode,
            title=f"{example_name} - {method} - Correct Code"
        )

        filename = f"{example_name}_{method}_correct.html"
        self._save_html(correct_html, os.path.join(output_dir, filename))

    def _get_token_analyses(self, result, method, code_type):
        """
        Estrae i TokenAnalysis dal result per il metodo e tipo di codice specificati.

        PROBLEMA: Attualmente i risultati NON contengono TokenAnalysis objects!
        Contengono solo conteggi aggregati.

        Serve modificare AdvancedMethodsComparator per salvare anche gli analyses.
        """
        # TODO: Implementare estrazione da result
        pass
```

### 🔄 Modifiche Necessarie

**1. Modificare `detectors/advanced_methods.py`**

Linee 721-850 (`AdvancedMethodsComparator.compare_all_methods`)

```python
# ATTUALMENTE ritorna solo conteggi:
result = {
    'method': 'semantic_energy',
    'num_tokens': 50,
    'num_anomalies': 7,
    'statistics': {...}
}

# DOVREBBE ritornare ANCHE:
result = {
    'method': 'semantic_energy',
    'num_tokens': 50,
    'num_anomalies': 7,
    'statistics': {...},
    'token_analyses': [...],  # ← AGGIUNGERE: Lista di TokenAnalysis
    'anomalous_positions': [3, 7, 12, ...],  # ← AGGIUNGERE: Posizioni anomale
    'code': "def binary_search(...)..."  # ← AGGIUNGERE: Codice originale
}
```

**2. Modificare `comparison/advanced_comparison_runner.py`**

Linee 192-282 (`run_comparison_on_example`)

```python
# Salvare gli analyses completi:
result = {
    'example_name': example.name,
    'buggy': {
        'lecprompt': {
            **self._extract_method_summary(buggy_comparison.lecprompt_result),
            'analyses': buggy_lecprompt_analyses,  # ← NUOVO
            'code': example.buggy_code  # ← NUOVO
        },
        # ... altri metodi
    },
    'correct': {
        # ... stessa struttura
    }
}
```

**3. Chiamare TokenLevelVisualizer in `test_advanced_methods.py`**

```python
# Dopo advanced_visualizer
if not args.skip_visualizations:
    # Visualizzazioni comparative (esistenti)
    visualizer = AdvancedMethodsVisualizer()
    visualizer.create_all_visualizations(results, args.output)

    # NUOVO: Visualizzazioni token-level
    token_visualizer = TokenLevelVisualizer()
    token_visualizer.generate_all_token_visualizations(results, args.output)
```

### 📊 Impatto

| Aspetto | Stato Attuale | Dopo Fix |
|---------|---------------|----------|
| **Visualizzazione token** | ❌ Assente | ✅ 80 HTML per tutte le combinazioni |
| **Confronto visivo metodi** | ❌ Impossibile | ✅ Side-by-side per metodo |
| **Identificazione token specifici** | ❌ Solo conteggi | ✅ Evidenziazione precisa |
| **Comprensione utente** | ❌ Limitata | ✅ Completa |

**Severità**: 🔴 **CRITICO** - Senza queste visualizzazioni, gli utenti non possono vedere **quali** token sono stati identificati

---

## PROBLEMA #2: Manca index.html principale (ALTO)

### 🟡 Problema Identificato

**Gravità**: 🟡 **ALTO**

Esiste un `index.html` solo per `advanced_visualizations/` ma **manca un index.html principale** che permetta di navigare facilmente tra:
- Visualizzazioni comparative (7 esistenti)
- Visualizzazioni token-level (80 mancanti)
- Risultati JSON
- Report

### 📁 Stato Attuale

**File esistente**: `advanced_visualizations/index.html`

Mostra solo:
- Link alle 7 visualizzazioni comparative
- Link ai token_level_multimethod_view (che sono bar chart, NON token evidenziati)

**Mancante**: Index principale a livello di `output_dir/`

### 📝 Struttura Directory Desiderata

```
advanced_methods_comparison/
├── index.html  ← MANCANTE! Index principale
├── complete_comparison_results.json
│
├── advanced_visualizations/  ← Visualizzazioni comparative
│   ├── index.html  ← Esiste
│   ├── methods_comparison_heatmap.html
│   ├── anomaly_counts_comparison.html
│   ├── method_agreement_matrix.html
│   ├── method_performance_radar.html
│   ├── venn_diagram_overlap.html
│   ├── interactive_method_explorer.html
│   └── token_level_multimethod_view_*.html (×10)
│
└── token_visualizations/  ← NUOVO! Visualizzazioni token-level
    ├── index.html  ← NUOVO! Organizza per metodo/esempio
    │
    ├── by_method/  ← Organizzato per metodo
    │   ├── lecprompt/
    │   │   ├── binary_search_buggy.html
    │   │   ├── binary_search_correct.html
    │   │   └── ... (×20 file)
    │   ├── semantic_energy/
    │   │   └── ... (×20 file)
    │   ├── conformal/
    │   │   └── ... (×20 file)
    │   └── attention/
    │       └── ... (×20 file)
    │
    └── by_example/  ← Organizzato per esempio
        ├── binary_search/
        │   ├── lecprompt_buggy.html
        │   ├── lecprompt_correct.html
        │   ├── semantic_energy_buggy.html
        │   ├── semantic_energy_correct.html
        │   ├── conformal_buggy.html
        │   ├── conformal_correct.html
        │   ├── attention_buggy.html
        │   └── attention_correct.html
        └── ... (×10 cartelle)
```

### 🛠️ Index.html Principale Necessario

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Advanced Methods Comparison - Main Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            border: 2px solid #ddd;
            border-radius: 10px;
        }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .card {
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-5px); }
    </style>
</head>
<body>
    <h1>🔬 Advanced Error Detection Methods - Complete Results</h1>

    <div class="section">
        <h2>📊 Comparative Visualizations</h2>
        <p>High-level comparisons across all methods and examples</p>
        <div class="grid">
            <a href="advanced_visualizations/index.html" class="card">
                <h3>🎯 All Comparative Views</h3>
                <p>Interactive dashboards, heatmaps, and performance radars</p>
            </a>
            <a href="advanced_visualizations/interactive_method_explorer.html" class="card">
                <h3>🔍 Quick Explorer</h3>
                <p>Interactive method comparison dashboard</p>
            </a>
        </div>
    </div>

    <div class="section">
        <h2>🎨 Token-Level Visualizations</h2>
        <p>Detailed code highlighting showing exact token anomalies</p>
        <div class="grid">
            <a href="token_visualizations/by_method/index.html" class="card">
                <h3>📋 Browse by Method</h3>
                <p>See all examples for each detection method</p>
            </a>
            <a href="token_visualizations/by_example/index.html" class="card">
                <h3>🎯 Browse by Example</h3>
                <p>Compare all methods on each test example</p>
            </a>
        </div>
    </div>

    <div class="section">
        <h2>📁 Raw Data</h2>
        <div class="grid">
            <a href="complete_comparison_results.json" class="card">
                <h3>💾 Complete Results (JSON)</h3>
                <p>Full dataset with all metrics and analyses</p>
            </a>
        </div>
    </div>
</body>
</html>
```

### 📊 Impatto

| Aspetto | Stato Attuale | Dopo Fix |
|---------|---------------|----------|
| **Navigazione principale** | ❌ Mancante | ✅ Index chiaro |
| **Accesso visualizzazioni** | ⚠️ Disperso | ✅ Centralizzato |
| **Organizzazione** | ⚠️ Confusa | ✅ Strutturata |
| **User experience** | ❌ Difficile | ✅ Intuitiva |

**Severità**: 🟡 **ALTO** - Difficile navigare i risultati senza index principale

---

## PROBLEMA #3: Nessuna integrazione tra visualizer.py e advanced methods (ALTO)

### 🟡 Problema Identificato

**Gravità**: 🟡 **ALTO**

`visualizer.py` ha **già implementato** la logica per visualizzare token evidenziati per le metriche avanzate, MA **non viene mai chiamato** per generare le visualizzazioni per ogni metodo su ogni esempio.

### 📁 Codice Esistente (NON UTILIZZATO)

**File**: `visualizer.py`

**Supporto metodi avanzati** (linee 39-46, 159-200):

```python
class TokenVisualizationMode:
    # ... altri mode

    # Advanced methods from METHODS_OVERVIEW.md
    SEMANTIC_ENERGY = "semantic_energy"  # Method 2
    CONFORMAL_SCORE = "conformal_score"  # Method 3
    ATTENTION_ENTROPY = "attention_entropy"  # Method 4
    ATTENTION_SELF_ATTENTION = "attention_self_attention"
    ATTENTION_VARIANCE = "attention_variance"
    ATTENTION_ANOMALY_SCORE = "attention_anomaly_score"
    SUSPICION_SCORE = "suspicion_score"
```

**Color schemes corretti** (linee 159-200):

```python
self.color_schemes = {
    TokenVisualizationMode.SEMANTIC_ENERGY: {
        'colormap': 'RdYlBu',
        'label': 'Semantic Energy (pre-softmax)',
        'reverse': True,  # ✅ CORRETTO: Alto = incerto = ROSSO
        'description': 'Energy = -logit(token), higher = more uncertain'
    },
    TokenVisualizationMode.CONFORMAL_SCORE: {
        'colormap': 'RdYlBu',
        'label': 'Conformal Prediction Score',
        'reverse': True,  # ✅ CORRETTO: Alto = incerto = ROSSO
        'description': 'Conformal score = 1 - P(token), higher = larger prediction set'
    },
    TokenVisualizationMode.ATTENTION_ENTROPY: {
        'colormap': 'RdYlBu',
        'label': 'Attention Entropy',
        'reverse': True,  # ✅ CORRETTO: Alto = incerto = ROSSO
        'description': 'Entropy of attention distribution, higher = more uncertain'
    },
    TokenVisualizationMode.ATTENTION_SELF_ATTENTION: {
        'colormap': 'RdYlGn',
        'label': 'Self-Attention Weight',
        'reverse': False,  # ✅ CORRETTO: Basso = anomalo = ROSSO
        'description': 'Self-attention weight (a_ii), lower = potentially anomalous'
    },
    # ...
}
```

**Estrazione valori** (linee 311-325):

```python
def create_html_visualization(self, analyses, mode, title):
    # ...

    # Advanced methods from METHODS_OVERVIEW.md
    elif mode == TokenVisualizationMode.SEMANTIC_ENERGY:
        values = [analysis.semantic_energy if analysis.semantic_energy is not None
                 else -analysis.logit for analysis in analyses]
    elif mode == TokenVisualizationMode.CONFORMAL_SCORE:
        values = [analysis.conformal_score if analysis.conformal_score is not None
                 else 1.0 - analysis.probability for analysis in analyses]
    elif mode == TokenVisualizationMode.ATTENTION_ENTROPY:
        values = [analysis.attention_entropy if analysis.attention_entropy is not None
                 else 0.0 for analysis in analyses]
    # ...
```

### ❌ Problema: MAI CHIAMATO

**Nessun codice chiama** `visualizer.create_html_visualization()` con questi mode per gli advanced methods!

**Cosa succede attualmente**:

1. `test_advanced_methods.py` → `AdvancedMethodsComparisonRunner`
2. `AdvancedMethodsComparisonRunner` → `AdvancedMethodsComparator.compare_all_methods()`
3. `compare_all_methods()` → Esegue i 4 metodi, salva **solo conteggi**
4. `AdvancedMethodsVisualizer.create_all_visualizations()` → Crea bar chart/heatmap
5. **FINE** - Nessuna chiamata a `visualizer.create_html_visualization()`

### 🛠️ Integrazione Necessaria

**1. Salvare TokenAnalysis negli advanced methods**

```python
# detectors/advanced_methods.py - SemanticEnergyDetector

def analyze_code(self, code, model, tokenizer, baseline_log_probs):
    """Analizza codice con semantic energy."""

    # Compute energies
    energies = self._compute_energies(...)

    # Detect anomalies
    anomalies, stats = self.detect_anomalies(energies)

    # ✅ NUOVO: Crea TokenAnalysis objects
    from LLM import TokenAnalysis

    token_analyses = []
    for i, (token, energy, is_anom) in enumerate(zip(tokens, energies, anomalies)):
        analysis = TokenAnalysis(
            token=token,
            position=i,
            probability=exp(-energy),  # Converti da energy
            logit=-energy,
            semantic_energy=energy,  # ← METRICA PRINCIPALE
            # ... altre metriche di base
        )
        token_analyses.append(analysis)

    return {
        'method': 'semantic_energy',
        'num_tokens': len(tokens),
        'num_anomalies': sum(anomalies),
        'token_analyses': token_analyses,  # ← NUOVO!
        'code': code,  # ← NUOVO!
        # ...
    }
```

**2. Usare visualizer per generare HTML**

```python
# NUOVO: comparison/token_level_visualizer.py

from visualizer import TokenVisualizer, TokenVisualizationMode

class TokenLevelVisualizer:

    def generate_for_method(self, result, method, code_type, output_dir):
        """Generate token visualization for a method."""

        # Map method to visualization mode
        mode_map = {
            'lecprompt': TokenVisualizationMode.LOGICAL_ERROR_DETECTION,
            'semantic_energy': TokenVisualizationMode.SEMANTIC_ENERGY,
            'conformal': TokenVisualizationMode.CONFORMAL_SCORE,
            'attention': TokenVisualizationMode.ATTENTION_ANOMALY_SCORE
        }

        # Get token analyses
        analyses = result[code_type][method]['token_analyses']
        code = result[code_type][method]['code']

        # Create visualization
        visualizer = TokenVisualizer()
        html = visualizer.create_html_visualization(
            analyses,
            mode=mode_map[method],
            title=f"{result['example_name']} - {method} - {code_type}"
        )

        # Save to file
        filename = f"{result['example_name']}_{method}_{code_type}.html"
        self._save_html(html, os.path.join(output_dir, filename))
```

### 📊 Impatto

| Aspetto | Stato Attuale | Dopo Fix |
|---------|---------------|----------|
| **Codice riutilizzato** | ❌ visualizer.py ignorato | ✅ Utilizzato pienamente |
| **Duplicazione logica** | ⚠️ Rischio alto | ✅ Singola implementazione |
| **Manutenibilità** | ❌ Difficile | ✅ Centralizzata |
| **Consistenza** | ⚠️ Non garantita | ✅ Stessi colori/logica |

**Severità**: 🟡 **ALTO** - Spreco di codice esistente e funzionante

---

## ✅ OK: Logica di evidenziazione token è corretta

### 🟢 Verifica Completata

**Gravità**: 🟢 **NESSUNA** - Tutto corretto

La logica di evidenziazione nel `visualizer.py` è **CORRETTA** per tutte le tecniche.

### ✅ Verifica: Reverse Flag

**File**: `visualizer.py`, linee 60-200

| Metodo | Reverse | Interpretazione | Corretto? |
|--------|---------|-----------------|-----------|
| **Semantic Energy** | `True` | Alto = incerto = **ROSSO** | ✅ Sì |
| **Conformal Score** | `True` | Alto = incerto = **ROSSO** | ✅ Sì |
| **Attention Entropy** | `True` | Alto = incerto = **ROSSO** | ✅ Sì |
| **Attention Self-Attention** | `False` | Basso = anomalo = **ROSSO** | ✅ Sì |
| **Attention Variance** | `True` | Alto = erratico = **ROSSO** | ✅ Sì |
| **Attention Anomaly Score** | `True` | Alto = anomalo = **ROSSO** | ✅ Sì |
| **LecPrompt Error** | `True` | Alto = errore = **ROSSO** | ✅ Sì |

### 📚 Spiegazione Tecnica

**Semantic Energy** (Farquhar et al., NeurIPS 2024):
```
Energy = -logit(token)
Alto energy → Basso logit → Bassa confidence → INCERTO → ROSSO ✅
reverse=True → normalize inverte scala → Funziona correttamente
```

**Conformal Score** (Quach et al., ICLR 2024):
```
Score = 1 - P(token)
Alto score → Bassa probabilità → INCERTO → ROSSO ✅
reverse=True → Funziona correttamente
```

**Attention Entropy** (Ott et al., ICML 2018):
```
Entropia alta → Distribuzione uniforme → Attenzione dispersa → INCERTO → ROSSO ✅
reverse=True → Funziona correttamente
```

**Self-Attention Weight**:
```
Peso basso (token non attende a sé stesso) → ANOMALO → ROSSO ✅
reverse=False → Valori bassi diventano rossi → Funziona correttamente
```

### 🎨 Color Mapping Corretto

**Codice**: `visualizer.py`, linee 204-236

```python
def _normalize_values(self, values: List[float], mode: str) -> List[float]:
    """Normalize values for color mapping."""
    values = np.array(values)

    # Normalize to [0, 1]
    min_val, max_val = np.min(values), np.max(values)
    normalized = (values - min_val) / (max_val - min_val)

    # ✅ Reverse if needed
    if self.color_schemes[mode]['reverse']:
        normalized = 1 - normalized

    return normalized.tolist()
```

**Funzionamento**:

1. Normalizza a [0, 1]
2. Se `reverse=True`, inverte: `1 - normalized`
3. Applica colormap (es. RdYlBu: 0=Blu, 1=Rosso)

**Esempio Semantic Energy**:
```
Valori originali: [5.2, 8.1, 3.4, 9.5, 4.0]  (alto = incerto)
Normalizzati: [0.34, 0.83, 0.00, 1.00, 0.12]
Reversed: [0.66, 0.17, 1.00, 0.00, 0.88]  ← Ora alto originale → basso → BLU
                                             basso originale → alto → ROSSO ✅
```

**ERRATO!** Wait, c'è un problema nella mia analisi. Rileggo...

Ah no, è corretto:
- Valori originali alti (9.5 = molto incerto)
- Dopo normalize: 1.00
- Dopo reverse: 1 - 1.00 = 0.00
- Colormap[0.00] con RdYlBu = **ROSSO** ✅

Perfetto!

### 📊 Verifica Visuale

**Tooltip completi** (linee 366-466):

```python
# Tooltip con TUTTE le metriche
tooltip_parts = [
    f"Token: {html.escape(analysis.token)}",
    f"Position: {analysis.position}",
    "",
    "=== Advanced Detection Methods ===",
    f"Semantic Energy: {analysis.semantic_energy:.4f}",  # ✅
    f"Conformal Score: {analysis.conformal_score:.4f}",  # ✅
    f"Attention Entropy: {analysis.attention_entropy:.4f}",  # ✅
    f"Self-Attention: {analysis.attention_self_attention:.4f}",  # ✅
    f"Attention Variance: {analysis.attention_variance:.4f}",  # ✅
    f"Attention Anomaly Score: {analysis.attention_anomaly_score:.4f}"  # ✅
]
```

**Legenda chiara** (linee 482-528):

```python
def _create_color_legend(self, scheme: Dict, values: List[float]) -> str:
    """Create a color legend for the visualization."""

    legend_html = [
        f"<p><strong>Legend ({scheme['label']}):</strong></p>",
        # ... gradient display
        f"<p><em>{scheme.get('description', '')}</em></p>"  # ✅ Descrizione
    ]
```

**Descrizioni corrette**:
- Semantic Energy: "Energy = -logit(token), **higher = more uncertain**" ✅
- Conformal: "score = 1 - P(token), **higher = larger prediction set**" ✅
- Attention Entropy: "**higher = more uncertain**" ✅
- Self-Attention: "**lower = potentially anomalous**" ✅

### 📊 Conclusione

| Componente | Stato | Note |
|------------|-------|------|
| **Reverse flags** | ✅ Corretti | Tutti configurati bene |
| **Normalize logic** | ✅ Corretta | Inversione funziona |
| **Color mapping** | ✅ Corretto | RdYlBu usato bene |
| **Tooltip descriptions** | ✅ Corrette | Chiare e precise |
| **Legend** | ✅ Corretta | Con descrizioni |

**Severità**: 🟢 **NESSUN PROBLEMA** - Implementazione corretta

---

## 📊 Riepilogo Complessivo

### Problemi per Severità

| Severità | # | Problema | Impatto |
|----------|---|----------|---------|
| 🔴 **CRITICO** | 1 | Manca visualizzazione token-level | Utenti non vedono token specifici |
| 🟡 **ALTO** | 2 | Manca index.html principale | Navigazione difficile |
| 🟡 **ALTO** | 3 | Nessuna integrazione visualizer.py | Codice non utilizzato |
| 🟢 **OK** | - | Logica evidenziazione | Tutto corretto ✅ |

### File da Creare

| File | Scopo | Priorità |
|------|-------|----------|
| `comparison/token_level_visualizer.py` | Generare visualizzazioni token | 🔴 CRITICA |
| `index.html` (root) | Index principale | 🟡 ALTA |
| `token_visualizations/index.html` | Index token-level | 🟡 ALTA |
| `token_visualizations/by_method/index.html` | Browse per metodo | 🟡 ALTA |
| `token_visualizations/by_example/index.html` | Browse per esempio | 🟡 ALTA |

### File da Modificare

| File | Modifiche | Priorità |
|------|-----------|----------|
| `detectors/advanced_methods.py` | Salvare token_analyses | 🔴 CRITICA |
| `comparison/advanced_comparison_runner.py` | Passare analyses a results | 🔴 CRITICA |
| `test_advanced_methods.py` | Chiamare TokenLevelVisualizer | 🔴 CRITICA |

### Stima Implementazione

| Task | Linee Codice | Tempo Stimato |
|------|--------------|---------------|
| TokenLevelVisualizer | ~300 | 3-4 ore |
| Modifiche advanced_methods | ~100 | 1-2 ore |
| Modifiche comparison_runner | ~50 | 1 ora |
| Index HTML files | ~200 | 1-2 ore |
| Testing | - | 2-3 ore |
| **TOTALE** | **~650** | **8-12 ore** |

---

## 💡 Raccomandazioni

### 1. PRIORITÀ MASSIMA: Implementare visualizzazione token-level

Senza questa, il sistema è **incompleto** - gli utenti non possono vedere **quali** token sono stati identificati, rendendo i risultati molto meno utili.

### 2. ALTA PRIORITÀ: Creare index principale

Migliora significativamente la user experience e rende i risultati accessibili.

### 3. ALTA PRIORITÀ: Integrare visualizer.py

Riutilizza codice esistente e garantisce consistenza.

### 4. Testing: Verificare output HTML

Dopo implementazione, verificare:
- ✅ Tutti i 80 file HTML generati
- ✅ Token evidenziati correttamente
- ✅ Colori secondo interpretazione corretta
- ✅ Tooltip funzionanti
- ✅ Legende chiare
- ✅ Navigazione funzionante

---

**Documento creato**: 2025-01-19
**Versione**: 1.0
**Autore**: Analisi automatica codice
