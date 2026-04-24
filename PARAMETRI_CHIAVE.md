# 🎛️ Parametri Chiave - Quick Reference

**Guida rapida per modificare hyperparametri e configurazioni**

---

## 📍 Dove Sono i Parametri Principali

### **1. Lunghezza Finestre (Default: 30 secondi)**

**File**: `src/prepare_dataset.py`  
**Linea**: 76

```python
WINDOW_SEC = 30  # ⭐ Modifica qui
```

**Valori suggeriti**: 20, 30, 40, 60  
**Nota**: ⚠️ Dopo la modifica, devi ri-eseguire `prepare_dataset.py`

---

### **2. Finestra Temporale Rolling (Default: 3 → Ottimale: 7)**

**File**: `src/manual_labels/run_analysis_manual.py`  
**Linea**: 50

```python
ROLLING_WINDOW = 7  # ⭐ Modifica qui (3, 5, 7, 9, 11)
```

**Conversione**: 
- 3 → 1.5 minuti
- 5 → 2.5 minuti
- 7 → 3.5 minuti (OTTIMALE)
- 9 → 4.5 minuti
- 11 → 5.5 minuti

---

### **3. Numero Feature Selezionate K (Default: 60 → Ottimale: 70)**

**File**: `src/manual_labels/run_analysis_manual.py`  
**Linea**: 48

```python
K_FEATURES = 70  # ⭐ Modifica qui (40, 50, 60, 70, 80, 100)
```

---

### **4. Soglia Classificazione (Default: 0.30 → Ottimale: 0.25)**

**File**: `src/manual_labels/run_analysis_manual.py`  
**Linea**: 51

```python
CLASSIFICATION_THR = 0.25  # ⭐ Modifica qui (0.20 - 0.50)
```

**Trade-off**:
- ⬇️ Bassa (0.20-0.25): Recall alto (rileva più apnee) → più falsi positivi
- ⬆️ Alta (0.35-0.50): Precision alta (meno falsi allarmi) → più falsi negativi

---

### **5. Random Forest - Numero Alberi**

**File**: `src/run_complete_analysis.py` o `src/manual_labels/run_analysis_manual.py`  
**Linea**: 39 (run_complete_analysis) / vicino a linee 48-51 (run_analysis_manual)

```python
N_ESTIMATORS = 400  # ⭐ Modifica qui (200, 400, 600, 800)
```

**Nota**: Più alberi = migliore stabilità ma più lento

---

### **6. Random Forest - Min Campioni per Foglia**

**File**: `src/run_complete_analysis.py`  
**Linea**: 40

```python
MIN_SAMPLES_LEAF = 3  # ⭐ Modifica qui (1, 3, 5, 10)
```

**Nota**: Valori più alti riducono overfitting

---

### **7. XGBoost - Profondità Alberi**

**File**: `src/run_complete_analysis.py`  
**Linea**: ~310 (dentro la sezione XGBoost)

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,  # ⭐ Modifica qui (4, 6, 8, 10)
    learning_rate=0.1,  # ⭐ Opzionale (0.01, 0.1, 0.3)
    ...
)
```

---

## 🔍 Hyperparameter Tuning Automatico

**File**: `src/manual_labels/hyperparameter_tuning.py`  
**Linee**: ~165-175

```python
# ⭐ Modifica queste griglie per esplorare nuovi valori
K_VALUES = [40, 50, 60, 70, 80]              # Feature selection
WINDOW_SIZES = [3, 5, 7]                     # Finestra temporale
N_ESTIMATORS_VALUES = [200, 400, 600]        # Alberi RF
THRESHOLDS = [0.25, 0.30, 0.35, 0.40]        # Soglia classificazione
XGB_MAX_DEPTHS = [4, 6, 8]                   # Profondità XGB
```

**Esegui**:
```bash
cd src/manual_labels
python hyperparameter_tuning.py
python visualize_tuning_results.py
```

---

## 🧪 Template per Esperimenti Rapidi

### **Esperimento A: Finestre temporali più lunghe**

**File**: `hyperparameter_tuning.py`

```python
WINDOW_SIZES = [7, 9, 11, 13]  # Fino a 6.5 minuti
K_VALUES = [70]                 # Fisso ottimale
N_ESTIMATORS_VALUES = [400]     # Fisso per risparmiare tempo
THRESHOLDS = [0.25]             # Fisso ottimale
```

---

### **Esperimento B: Più feature**

```python
K_VALUES = [70, 80, 90, 100, 120]  # Esplora range alto
WINDOW_SIZES = [7]                 # Fisso ottimale
N_ESTIMATORS_VALUES = [400]        # Fisso
THRESHOLDS = [0.25]                # Fisso
```

---

### **Esperimento C: Soglie ultra-basse (screening massivo)**

```python
THRESHOLDS = [0.15, 0.18, 0.20, 0.22, 0.25]  # Massimizza recall
K_VALUES = [70]                               # Fisso
WINDOW_SIZES = [7]                            # Fisso
N_ESTIMATORS_VALUES = [400]                   # Fisso
```

---

## 📊 Quick Workflow

### **1. Test Singola Configurazione**

```bash
# Modifica parametri in run_analysis_manual.py
cd src/manual_labels
nano run_analysis_manual.py  # Modifica K_FEATURES, ROLLING_WINDOW, etc.
python run_analysis_manual.py
```

---

### **2. Grid Search Esteso**

```bash
# Modifica griglie in hyperparameter_tuning.py
cd src/manual_labels
nano hyperparameter_tuning.py  # Modifica K_VALUES, WINDOW_SIZES, etc.
python hyperparameter_tuning.py
python visualize_tuning_results.py
```

---

### **3. Cambiare Lunghezza Finestre di Base**

```bash
# Modifica WINDOW_SEC in prepare_dataset.py
cd src
nano prepare_dataset.py  # Cambia WINDOW_SEC = 30 → 40
python prepare_dataset.py  # Rigenera dataset
cd manual_labels
python run_analysis_manual.py
```

---

## 🎯 Parametri Ottimali Trovati

| Parametro | Valore Originale | Valore Ottimale | Miglioramento |
|-----------|------------------|-----------------|---------------|
| **K_FEATURES** | 60 | **70** | +2% ROC-AUC |
| **ROLLING_WINDOW** | 3 (1.5 min) | **7 (3.5 min)** | +11% ROC-AUC |
| **CLASSIFICATION_THR** | 0.30 | **0.25** | +15% Recall |
| **N_ESTIMATORS** | 400 | **400** | (già ottimale) |
| **MIN_SAMPLES_LEAF** | 3 | **3** | (già ottimale) |

**Risultato**: ROC-AUC 0.827 → **0.901** (+8.9%)

---

## ⚠️ Avvertenze

1. **WINDOW_SEC**: Cambiare richiede rigenerare tutto il dataset (lungo!)
2. **ROLLING_WINDOW**: Finestre troppo lunghe (>11) potrebbero non catturare eventi brevi
3. **K_FEATURES**: Troppo alto (>100) rischia overfitting
4. **THRESHOLDS**: Troppo basso (<0.15) genera troppi falsi positivi

---

## 📁 File di Configurazione Principali

| Parametro | File Principale | Riga Approssimativa |
|-----------|----------------|---------------------|
| Lunghezza finestre (30s) | `prepare_dataset.py` | 76 |
| Finestra temporale (rolling) | `run_analysis_manual.py` | 50 |
| K feature | `run_analysis_manual.py` | 48 |
| Soglia classificazione | `run_analysis_manual.py` | 51 |
| N alberi RF | `run_analysis_manual.py` | 39 |
| Griglia tuning | `hyperparameter_tuning.py` | 165-175 |

---

**Ultima modifica**: Febbraio 2026  
**Versione**: 1.0
