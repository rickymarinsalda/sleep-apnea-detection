# 🪟 Guida Installazione per Windows

**Come eseguire gli script su Windows - Guida per principianti**

---

## 🎯 Metodo Consigliato: Anaconda (PIÙ SEMPLICE)

### Vantaggi
✅ Interfaccia grafica  
✅ Include già Python e tutte le librerie scientifiche  
✅ Non richiede uso del terminale (opzionale)  
✅ Gestione facilitata degli ambienti virtuali  

---

## 📦 Opzione 1: Anaconda (CONSIGLIATO per principianti)

### Passo 1: Scarica e Installa Anaconda

1. **Vai sul sito ufficiale**: https://www.anaconda.com/download
2. **Scarica la versione per Windows** (circa 600 MB)
3. **Esegui l'installer**:
   - ✅ Accetta tutte le opzioni di default
   - ✅ Seleziona "Add Anaconda to PATH" (se chiesto)
   - ⏱️ L'installazione richiede 5-10 minuti

### Passo 2: Apri Anaconda Navigator

1. Cerca "Anaconda Navigator" nel menu Start
2. Si aprirà un'interfaccia grafica

### Passo 3: Crea un Ambiente per il Progetto

**Opzione A: Via Interfaccia Grafica (più facile)**

1. Clicca su **"Environments"** (barra laterale sinistra)
2. Clicca su **"Create"** (in basso)
3. Nome: `apnea_detection`
4. Seleziona **Python 3.9** o superiore
5. Clicca **"Create"**

**Opzione B: Via Anaconda Prompt (alternativa)**

1. Cerca "Anaconda Prompt" nel menu Start
2. Digita:
```bash
conda create -n apnea_detection python=3.9
conda activate apnea_detection
```

### Passo 4: Installa le Librerie Necessarie

**Opzione A: Via Interfaccia Grafica**

1. Seleziona l'ambiente `apnea_detection` creato
2. Cambia da "Installed" a **"Not Installed"**
3. Cerca e installa (uno alla volta):
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`
   - `xgboost`

**Opzione B: Via Anaconda Prompt (più veloce)**

1. Apri "Anaconda Prompt"
2. Attiva l'ambiente:
```bash
conda activate apnea_detection
```
3. Naviga nella cartella del progetto:
```bash
cd C:\percorso\verso\sleep-apnea-detection
```
4. Installa tutto con un comando:
```bash
pip install -r requirements.txt
```

### Passo 5: Scarica il Codice da GitHub

**Opzione A: Download ZIP (più semplice)**

1. Vai su GitHub: `https://github.com/rickymarinsalda/sleep-apnea-detection`
2. Clicca su **"Code"** → **"Download ZIP"**
3. Estrai lo ZIP in una cartella (es: `C:\Users\TuoNome\Documents\apnea_study`)

**Opzione B: Git Clone (se hai Git installato)**

```bash
cd C:\Users\TuoNome\Documents
git clone https://github.com/rickymarinsalda/sleep-apnea-detection.git
```

### Passo 6: Prepara i Dati

1. Copia i file di dati grezzi nella cartella `data/`:
   - `dataset_apnea_ricky_MAT.csv`
   - `dataset_apnea_ricky_ACC.csv`

### Passo 7: Esegui gli Script

**Metodo 1: Anaconda Prompt (consigliato)**

```bash
# Attiva ambiente
conda activate apnea_detection

# Vai nella cartella del progetto
cd C:\percorso\verso\sleep-apnea-detection\src

# Esegui preparazione dataset
python prepare_dataset.py

# Esegui analisi
python run_complete_analysis.py
```

**Metodo 2: Jupyter Notebook (interfaccia grafica)**

1. Apri Anaconda Navigator
2. Assicurati che l'ambiente `apnea_detection` sia selezionato
3. Lancia **Jupyter Notebook**
4. Naviga alla cartella del progetto
5. Puoi creare un notebook ed eseguire:

```python
# In una cella Jupyter
%run src/prepare_dataset.py
```

**Metodo 3: Spyder IDE (simile a RStudio/MATLAB)**

1. Apri Anaconda Navigator
2. Con ambiente `apnea_detection` attivo, lancia **Spyder**
3. Apri i file `.py` e clicca "Run" (F5)

---

## 💻 Opzione 2: Python Standard + pip (Alternativa più leggera)

### Passo 1: Installa Python

1. Vai su: https://www.python.org/downloads/
2. Scarica **Python 3.9** o superiore per Windows
3. **IMPORTANTE**: Durante installazione, seleziona ✅ **"Add Python to PATH"**

### Passo 2: Apri PowerShell o Command Prompt

1. Premi `Win + R`
2. Digita `powershell` o `cmd`
3. Premi Enter

### Passo 3: Crea Ambiente Virtuale

```bash
# Vai nella cartella del progetto
cd C:\percorso\verso\sleep-apnea-detection

# Crea ambiente virtuale
python -m venv venv

# Attiva ambiente (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# OPPURE, se usi Command Prompt (cmd):
.\venv\Scripts\activate.bat
```

### Passo 4: Installa Dipendenze

```bash
pip install -r requirements.txt
```

### Passo 5: Esegui Script

```bash
cd src
python prepare_dataset.py
python run_complete_analysis.py
```

---

## ☁️ Opzione 3: Google Colab (ZERO installazioni, tutto online)

### Vantaggi
✅ Nessuna installazione necessaria  
✅ GPU gratuita disponibile  
✅ Accessibile da qualsiasi PC/browser  

### Svantaggi
⚠️ Devi caricare i dati ogni volta  
⚠️ Sessione si disconnette dopo inattività  

### Come usare Google Colab

1. Vai su: https://colab.research.google.com/
2. Crea un nuovo notebook
3. Clona il repository:

```python
# Prima cella
!git clone https://github.com/rickymarinsalda/sleep-apnea-detection.git
%cd sleep-apnea-detection
```

4. Carica i dati (bottone Upload sulla sinistra):
   - Carica `dataset_apnea_ricky_MAT.csv` e `dataset_apnea_ricky_ACC.csv` nella cartella `data/`

5. Installa dipendenze:

```python
# Seconda cella
!pip install -r requirements.txt
```

6. Esegui script:

```python
# Terza cella
%cd src
!python prepare_dataset.py
```

```python
# Quarta cella
!python run_complete_analysis.py
```

7. Scarica i risultati:
   - Clicca con tasto destro sui file generati
   - "Download"

---

## 🔧 Risoluzione Problemi Comuni

### Problema: "Python non è riconosciuto come comando"

**Soluzione**:
- Reinstalla Python selezionando ✅ "Add to PATH"
- OPPURE aggiungi manualmente Python al PATH di Windows:
  1. Cerca "Variabili d'ambiente" nel menu Start
  2. Modifica PATH
  3. Aggiungi: `C:\Users\TuoNome\AppData\Local\Programs\Python\Python39`

### Problema: "ModuleNotFoundError: No module named 'pandas'"

**Soluzione**:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Problema: "Permission denied" durante esecuzione script

**Soluzione**:
- Esegui PowerShell/Anaconda Prompt come **Amministratore**
- Click destro → "Esegui come amministratore"

### Problema: Anaconda Prompt non trova i comandi

**Soluzione**:
- Chiudi e riapri Anaconda Prompt
- Verifica che l'ambiente sia attivato: vedrai `(apnea_detection)` prima del prompt

### Problema: Script si blocca o impiega troppo tempo

**Soluzione**:
- Dataset molto grandi richiedono tempo (5-30 minuti)
- Verifica che il PC non sia in modalità risparmio energetico
- Chiudi altri programmi per liberare RAM

---

## 📋 Checklist Veloce per Principianti

### Setup Iniziale (una volta sola)
- [ ] Scarica e installa Anaconda
- [ ] Crea ambiente `apnea_detection`
- [ ] Installa librerie da `requirements.txt`
- [ ] Scarica codice da GitHub
- [ ] Copia file dati in folder `data/`

### Ogni Volta che Vuoi Eseguire gli Script
- [ ] Apri Anaconda Prompt
- [ ] `conda activate apnea_detection`
- [ ] `cd C:\percorso\verso\sleep-apnea-detection`
- [ ] Esegui lo script desiderato: `python src/NOME_SCRIPT.py`

---

## 🎓 Tutorial Video Consigliati (YouTube)

- **"Anaconda Installation on Windows"** - Cerca su YouTube per tutorial visivi
- **"Python Virtual Environments Explained"**
- **"How to use Jupyter Notebook for beginners"**

---

## 🆘 Supporto

Se incontri problemi:

1. **Verifica versioni**:
```bash
python --version     # Deve essere 3.9+
pip --version
conda --version
```

2. **Controlla file requirements.txt**:
```bash
cat requirements.txt  # Anaconda Prompt
type requirements.txt # Command Prompt
```

3. **Test installazione librerie**:
```python
python -c "import pandas; import numpy; import sklearn; print('OK')"
```

Se vedi "OK", le librerie principali sono installate correttamente!

---

## 📊 Performance Attese

| Operazione | Tempo Stimato | RAM Necessaria |
|------------|---------------|----------------|
| Installazione Anaconda | 5-10 min | - |
| Download codice | 1 min | - |
| Installazione libreries | 3-5 min | - |
| `prepare_dataset.py` | 5-15 min | 4-8 GB |
| `run_complete_analysis.py` | 10-30 min | 4-8 GB |
| `hyperparameter_tuning.py` | 1-3 ore | 8-16 GB |

**Requisiti minimi PC**:
- Windows 10/11
- 8 GB RAM (16 GB consigliati per hyperparameter tuning)
- 5 GB spazio disco libero
- Processore dual-core o superiore

---

**Ultima modifica**: Febbraio 2026  
**Versione**: 1.0 - Guida Windows
