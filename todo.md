# Piano di Miglioramento per il Rilevamento di Bug

Questo documento descrive una serie di miglioramenti strutturati per la codebase, con l'obiettivo di potenziare il calcolo della probabilità che un token o una riga di codice contenga un bug.

---

### Miglioramento 1: Completare l'Implementazione dei Metodi Avanzati (TODO Esistenti)

Questa è la base di partenza per rendere il framework di comparazione completo e robusto.

*   **Stato Attuale:**
    I metodi avanzati (`Semantic Energy`, `Conformal Prediction`, `Attention Anomaly`) sono implementati solo per i modelli Causal LM come `StarCoder2` e `DeepSeek`. I file `codet5_detector.py`, `codebert_error_detector.py` e `logical_error_detector.py` (per Qwen) non hanno ancora queste implementazioni, come indicato nei TODO.

*   **Perché Implementarlo:**
    Per poter effettuare una comparazione completa ("mele con mele") tra le diverse architetture di modelli (Causal LM, Encoder-Decoder, Masked LM). Questo ci permetterà di capire se alcuni metodi di incertezza funzionano meglio su specifiche architetture, un'analisi cruciale per il progetto.

*   **Come Implementarlo (Dettaglio):**
    Dovrai modificare i file dei detector menzionati e implementare i tre metodi (`compute_semantic_energy`, `get_attention_weights`, `compute_conformal_scores`) seguendo la logica specifica dell'architettura, come descritto in `METHODS_OVERVIEW.md`.

    1.  **Per `codet5_detector.py` (Encoder-Decoder):**
        *   **Logits/Energy/Conformal:** L'approccio richiede un forward pass per ogni token. Per il token `i`, l'encoder riceve l'intero codice, mentre il decoder riceve i token da `0` a `i-1`. I logits per il token `i` si estraggono dall'ultima posizione dell'output del decoder.
        *   **Attention:** Usa l'output di `model.encoder(..., output_attentions=True)`. L'auto-attenzione dell'encoder è la più indicata perché ha una visione bidirezionale completa del codice, a differenza di quella del decoder che è causale.

    2.  **Per `codebert_error_detector.py` (Masked LM):**
        *   **Logits/Energy/Conformal:** L'approccio richiede di mascherare un token alla volta. Per ogni token `i`, lo si sostituisce con `[MASK]`, si esegue il modello e si estraggono i logits per la posizione `i` dall'output del modello.
        *   **Attention:** Esegui il modello base (`model.roberta(..., output_attentions=True)`) sull'input non mascherato per estrarre i pattern di attenzione.

    3.  **Per `logical_error_detector.py` (Qwen, Causal LM):**
        *   L'implementazione sarà quasi identica a quella di `starcoder2_detector.py`. Si tratta di un singolo forward pass con `output_attentions=True` per ottenere sia i logits che i pesi di attenzione per tutti i token in una sola volta.

*   **Risultati Attesi:**
    Un framework di benchmark completo `5 modelli x 4 metodi`. Potremo finalmente generare un report che classifica non solo i modelli ma anche i metodi di incertezza in base alla loro efficacia su architetture diverse, ottenendo insight scientifici più profondi.

---

### Miglioramento 2: Calibrazione Completa per la Conformal Prediction

*   **Stato Attuale:**
    Come indicato in `PROJECT_OVERVIEW.md`, l'implementazione attuale è una versione "base" che usa `1 - P(token)` come punteggio di anomalia. Manca una fase di **calibrazione** su un set di dati separato, che è il cuore della Conformal Prediction per fornire garanzie statistiche.

*   **Perché Implementarlo:**
    Per passare da un'euristica a un metodo con **garanzie matematiche reali**. Con la calibrazione, il sistema potrà affermare che "il token corretto si trova nel prediction set con una probabilità del 90%", rendendo la metrica di incertezza molto più affidabile e interpretabile.

*   **Paper di Riferimento:**
    *   Shafer, G., & Vovk, V. (2008). "A Tutorial on Conformal Prediction." *Journal of Machine Learning Research*. (Il paper fondazionale).
    *   Quach, T., et al. (2024). "Conformal Language Modeling." *ICLR 2024*. (Applicazione specifica per LLM).

*   **Come Implementarlo (Dettaglio):**
    1.  **Suddividi il Dataset:** In `test_examples.py`, designa 1-2 esempi come "set di calibrazione" e il resto come "set di test".
    2.  **Modifica `AdvancedMethodsComparisonRunner`:**
        a. Prima di iniziare l'analisi, esegui un ciclo sul set di calibrazione. Per ogni token nel codice **corretto** di questi esempi, calcola il punteggio di non conformità `score = 1 - P(token)`.
        b. Colleziona tutti questi punteggi in una lista.
        c. Calcola il **quantile** `q_alpha` da questa lista. Se `alpha = 0.1` (per una garanzia del 90%), `q_alpha` sarà il 90° percentile dei punteggi di calibrazione.
    3.  **Passa il Quantile:** Passa `q_alpha` come argomento al `ConformalPredictionDetector`.
    4.  **Aggiorna la Logica di Anomalia:** Nel detector, un token non sarà più anomalo se il suo punteggio supera una soglia statistica, ma se `score > q_alpha`. L'incertezza può essere misurata come la dimensione del prediction set `{y | 1 - P(y) <= q_alpha}`.

*   **Risultati Attesi:**
    Metriche di anomalia più stabili e statisticamente valide. Il numero di anomalie non dipenderà più da una soglia arbitraria (basata su media e deviazione standard), ma da un livello di confidenza desiderato (`alpha`), rendendo i risultati più consistenti tra modelli diversi.

---

### Miglioramento 3: Distinguere Incertezza Epistemica e Aleatoria (Monte Carlo Dropout)

*   **Stato Attuale:**
    I metodi attuali misurano un'incertezza "generica". Non distinguono tra:
    *   **Incertezza Aleatoria:** Incertezza intrinseca nei dati (es. un commento ambiguo).
    *   **Incertezza Epistemica:** Incertezza del modello, dovuta a una conoscenza incompleta (quella che vogliamo per trovare i bug).

*   **Perché Implementarlo:**
    I bug sono un segno di **incertezza epistemica**: il modello non sa cosa fare. Separare le due incertezze permette di creare un segnale di "bugginess" molto più pulito, ignorando le parti di codice che sono semplicemente strane o rare ma corrette.

*   **Paper di Riferimento:**
    *   Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML*.

*   **Come Implementarlo (Dettaglio):**
    1.  **Crea un Nuovo Detector:** In `detectors/advanced_methods.py`, crea una nuova classe `MCDropoutDetector`.
    2.  **Attiva Dropout in Inferenza:** All'interno del detector, prima di fare l'analisi, assicurati che i layer di dropout del modello siano attivi usando `model.train()`. (Normalmente sono disattivati con `model.eval()`).
    3.  **Esegui Inferenze Multiple:** Per un dato codice, esegui il forward pass `N` volte (es. N=20). Ogni volta, a causa del dropout, otterrai una stima leggermente diversa delle probabilità dei token.
    4.  **Calcola la Varianza:** Per ogni token, avrai `N` previsioni di probabilità. Calcola la **varianza** di queste `N` previsioni. Questa varianza è una misura diretta dell'incertezza epistemica.
    5.  **Usa la Varianza come Score:** La varianza diventa il nuovo punteggio di anomalia. Un'alta varianza indica che il modello è molto incerto su quel token.

*   **Risultati Attesi:**
    Un rilevamento di bug più preciso. Questo metodo dovrebbe essere meno propenso a segnalare falsi positivi su codice non convenzionale ma corretto e più efficace nell'individuare punti in cui il modello è genuinamente "confuso", che è un forte indicatore di un potenziale errore logico.

---

### Miglioramento 4: Analisi Semantica Contestuale (SimCSE)

*   **Stato Attuale:**
    Il sistema non esegue un'analisi diretta della coerenza semantica tra le righe di codice. L'Attention Anomaly lo fa indirettamente, ma non c'è un metodo che valuti se una riga è "semanticamente fuori posto" rispetto al suo contesto.

*   **Perché Implementarlo:**
    Per catturare una classe di errori che i metodi basati sulla probabilità di un singolo token potrebbero mancare. Un errore logico spesso si manifesta come una riga di codice che, sebbene sintatticamente valida, è semanticamente incoerente con le righe precedenti e successive.

*   **Paper di Riferimento:**
    *   Gao, T., Yao, X., & Chen, D. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." *EMNLP*.

*   **Come Implementarlo (Dettaglio):**
    1.  **Integra un Modello di Embedding:** Aggiungi un modello pre-addestrato per embedding di codice, come una versione di SimCSE fine-tuned su codice o un modello come `CodeBERT` usato solo per produrre embedding.
    2.  **Crea un Nuovo Detector a Livello di Riga:** Questo metodo opererebbe a livello di riga, non di token.
    3.  **Genera Embedding:** Per ogni riga `L_i` nel codice:
        a. Calcola l'embedding `E_i` per la riga `L_i`.
        b. Calcola l'embedding del contesto `E_c`, concatenando le righe circostanti (es. `L_{i-2}` a `L_{i+2}`) e calcolandone l'embedding.
    4.  **Calcola la Similarità:** Calcola la similarità del coseno tra `E_i` e `E_c`.
    5.  **Usa la Dissimilarità come Score:** Il punteggio di anomalia per la riga `L_i` sarà `1 - cosine_similarity(E_i, E_c)`. Un punteggio alto indica che la riga è semanticamente scollegata dal suo contesto.

*   **Risultati Attesi:**
    La capacità di rilevare bug "logici" di alto livello, come un'operazione errata in un algoritmo o una condizione fuori posto, che potrebbero non manifestarsi con una bassa probabilità a livello di singolo token. Questo metodo sarebbe complementare agli altri.

---

### Miglioramento 5: Creare un "Ensemble" Ponderato dei Metodi

*   **Stato Attuale:**
    Il framework esegue i 4 metodi e li confronta, ma non **combina i loro output** per ottenere una previsione finale più accurata. Ogni metodo agisce in modo indipendente.

*   **Perché Implementarlo:**
    Ogni metodo cattura un tipo diverso di segnale di incertezza. Un ensemble (o combinazione) di questi metodi è quasi sempre più robusto e accurato di qualsiasi metodo singolo, in quanto riduce i falsi positivi e sfrutta i punti di forza di ciascuno.

*   **Come Implementarlo (Dettaglio):**
    Modifica `AdvancedMethodsComparisonRunner` per aggiungere una logica di ensemble.

    1.  **Simple Voting (Voto Semplice):**
        *   Dopo aver ottenuto i flag di anomalia (vero/falso) per ogni token da tutti e 4 i metodi, un token viene considerato anomalo solo se almeno `M` metodi (es. `M=3`) sono d'accordo.

    2.  **Weighted Voting (Voto Ponderato):**
        *   Assegna un peso a ciascun metodo in base alla sua performance storica (es. dal report di `test_advanced_methods.py`). Ad esempio: `Semantic Energy: 0.4`, `LecPrompt: 0.25`, `Conformal: 0.2`, `Attention: 0.15`.
        *   Il punteggio di anomalia finale di un token è la somma ponderata dei suoi punteggi di anomalia normalizzati da ciascun metodo.

    3.  **Stacking / Meta-Learner (Avanzato):**
        *   Usa gli output dei 4 metodi (es. i 4 punteggi di anomalia per ogni token) come **feature** per addestrare un semplice modello di machine learning (es. una Regressione Logistica).
        *   Questo "meta-modello" impara a combinare i segnali nel modo ottimale. Il suo output (una probabilità da 0 a 1) diventa il punteggio finale di "bugginess" del token. Questo richiederebbe un set di dati etichettato.

*   **Risultati Attesi:**
    Un **singolo punteggio di "bugginess"** per ogni token, molto più affidabile e con meno rumore rispetto ai punteggi dei singoli metodi. Questo si tradurrebbe in una maggiore precisione (meno falsi positivi) e un migliore recall (meno falsi negativi), rendendo lo strumento molto più utile in pratica.

---

### Miglioramento 6: Rilevamento di Anomalie Semantiche con Embedding Contestuali

*   **Stato Attuale:**
    Il sistema non possiede un metodo che confronti direttamente il significato semantico di una riga di codice con le righe circostanti. Il metodo più simile, Attention Anomaly, analizza i pesi interni del modello piuttosto che il contenuto semantico del testo.

*   **Perché Implementarlo:**
    Questo metodo può catturare errori logici di alto livello che i metodi basati su token singoli potrebbero non vedere. Si concentra sul verificare se una linea di codice ha senso nel flusso generale dell'algoritmo, fornendo un segnale di rilevamento bug diverso e complementare.

*   **Come Implementarlo (Dettaglio):**
    1.  **Crea un Nuovo Detector a Livello di Riga:** Sviluppa una nuova classe, ad esempio `SemanticContextDetector`, che operi su righe di codice invece che su singoli token.
    2.  **Integra un Modello di Embedding:** Utilizza un modello pre-addestrato per generare embedding di codice/testo, come `sentence-transformers/all-MiniLM-L6-v2` o un modello specifico per il codice come `CodeT5`.
    3.  **Processa il Codice:** Per ogni riga `L` nel codice sorgente:
        a. Genera l'embedding per la riga stessa: `E(L)`.
        b. Definisci il contesto `C` come la concatenazione delle 3 righe precedenti e delle 3 righe successive.
        c. Genera l'embedding per il blocco di contesto: `E(C)`.
    4.  **Calcola il Punteggio di Anomalia:** Il punteggio per la riga `L` sarà la distanza coseno tra i due embedding: `1 - cosine_similarity(E(L), E(C))`.
    5.  **Segnala Anomalie:** Le righe con una distanza coseno elevata (ad esempio, sopra una certa soglia percentile o statistica) vengono segnalate come anomale.

*   **Risultati Attesi:**
    Il sistema acquisirà la capacità di identificare una nuova classe di bug logici, in particolare quelli in cui una riga è sintatticamente corretta ma semanticamente fuori posto. Questo migliorerà la robustezza generale del framework di rilevamento bug, aggiungendo una visione semantica a livello macro a quella probabilistica a livello di token già esistente.