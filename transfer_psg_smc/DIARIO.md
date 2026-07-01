# Diario — Transfer learning PSG → materasso (SMC)

Obiettivo del filone: allenare un detector di apnee su un dataset PSG grande
(apnee reali) e trasferirlo ai dati del materasso, passando NON dai segnali
grezzi (modalità fisiche diverse) ma da un **livello di feature fisiologiche
condivise** calcolabili su entrambi i sistemi (sforzo respiratorio, movimento,
cardiaco, posizione). Questo è il "Percorso B".

Decisione sui dati sorgente: **non** usare il dataset della Challenge PhysioNet
2026 finché la competizione è in corso (embargo: uso esclusivo per la challenge,
niente pubblicazione altrove fino a ~CinC 2026; inoltre il task è cognitive
impairment, non apnea). Sorgente target per il transfer: **NSRR (MESA/SHHS)**,
etichettato per apnea e pubblicabile.

---

## 2026-06-28 — Estrazione feature condivise lato materasso

**Cosa ho provato.** Costruito il primo estrattore che riduce i segnali grezzi
SMC (matrice pressione 40 ch @ 8 Hz + 2 ACC @ 80 Hz) al livello fisiologico
condiviso, su finestre di 30 s, per tutti i 23 soggetti.
Script: `extract_shared_features_smc.py` → output `shared_features_smc_30s.csv`.

**Pipeline (per soggetto):**
1. Canali di pressione "attivi" = quelli sotto il corpo (varianza > mediana).
2. Segnale di sforzo respiratorio = media dei canali attivi, filtrato 0.1–0.7 Hz.
3. Segnali torace (ch1–20) e addome (ch21–40) separati → correlazione = proxy
   di movimento paradosso toraco-addominale.
4. ACC: magnitudo → movimento broadband; banda 0.7–2.5 Hz → componente cardiaca/BCG.
5. Finestre 30 s non sovrapposte; label = apnea se ≥50% della finestra è Status==3.
6. Normalizzazione per-soggetto delle feature di ampiezza (suffisso `_rel`), per
   comparabilità cross-recording (stessa cosa che faremo sulla PSG).

**Risultati.** 925 finestre, 45 apnea (4.9%), 23 soggetti.
Potere discriminante univariato (ROC-AUC di ogni feature da sola, apnea vs no):

| feature | AUC | non-apnea | apnea | nota |
|---|---|---|---|---|
| resp_regularity | **0.83** | 0.49 | 0.23 | quota di potenza in banda respiratoria → crolla in apnea |
| updown_corr | 0.71 | 0.23 | 0.53 | torace/addome più correlati in apnea (entrambi piatti: tipo *centrale*) |
| resp_rate | 0.69 | 13.6 | 10.5 | ritmo respiratorio meno definito in apnea |
| resp_amp_std_rel | 0.65 | — | — | variabilità ampiezza |
| acc_move_std_rel | 0.64 | 55 | 379 | burst di movimento (recovery/arousal) a bordo apnea |
| resp_amp_mean | 0.59 | 11.8 | 7.2 | ampiezza respiro che cala |
| hr_est / hr_snr | ~0.50 | — | — | **non discrimina** (vedi limiti) |

Figura: `class_separation.png`.

**Interpretazione.** Il livello di feature condivise separa l'apnea: la più forte
è la **regolarità respiratoria** (AUC 0.83), che è esattamente una feature
calcolabile identica sulle cinghie di sforzo PSG → buon segnale per il transfer.
Le apnee volontarie qui si comportano come **centrali** (sforzo che si ferma,
torace+addome piatti e correlati): il *paradosso* vero (feature `updown_corr`
negativa) ce lo aspettiamo sull'OSA della PSG, non qui.

**Limiti noti (da sistemare).**
- **HR/HRV inutilizzabile** così com'è: stima da ACC dominata da artefatti di
  movimento (AUC ≈ 0.50). Va sostituita con la *pipeline HR validata* (Marinai2025_HR)
  o con BCG dalla matrice. È la prossima priorità: la variazione cardiaca
  (bradi-tachi) è uno dei marker chiave dell'apnea.
- **45 apnee vs 69 del paper**: qui ho usato la colonna `Status` (protocollo),
  non le label manuali; inoltre finestre 30 s non allineate agli eventi perdono
  qualche apnea sotto la soglia 50%. Da riallineare con le label manuali.
- Filtraggio assume campionamento uniforme (gap minori ignorati).

**Prossimi passi.**
1. Integrare HR/HRV dalla pipeline validata.
2. Riallineare le label (manuali, 69 apnee) e valutare soglia overlap.
3. Definire lo schema feature 1:1 da estrarre su MESA/SHHS (canali Chest, ABD, ECG).
4. Baseline: classificatore sulle sole feature condivise (sanity prima del transfer).

---

## 2026-06-28 — v2: migliorie allineate al paper

Dopo aver letto `analysis_manual_labels/prepare_dataset_manual.py` ho corretto due
problemi del primo estrattore e migliorato l'HR.

**Cosa è cambiato.**
1. **Dati**: ora uso i file con etichette MANUALI (`dataset_a_mano/...indiciApneaManuale.csv`),
   non quelli automatici → 69 apnee invece di 45.
2. **Labeling**: adottata la regola del paper — finestra = apnea se majority==3 &
   frac≥0.5, = non-apnea se majority∈{0,1,2} & frac_respiro≥0.5, **altrimenti scartata**
   (movimento/"altro"/miste). Prima marcavo tutto, inquinando la classe non-apnea.
3. **Finestratura**: drop righe tutte-zero + finestre per-indice (240 campioni) come il paper.
4. **HR**: estratto sia dal **BCG della matrice** (banda 0.7–2.5 Hz) sia dagli ACC; per
   ogni finestra scelgo la sorgente con SNR maggiore.

**Risultato chiave**: il dataset ora **coincide col paper — 581 finestre, 69 apnea
(11.9%)**, 341 ambigue scartate. HR preso dalla matrice in 466/581 finestre (più
pulito degli ACC). Separazione univariata migliorata:

| feature | AUC v1 | AUC v2 | dir. apnea |
|---|---|---|---|
| resp_regularity | 0.83 | **0.86** | ↓ (respiro irregolare) |
| resp_amp_std_rel | 0.65 | **0.78** | ↑ (piatto + recovery → alta variabilità) |
| resp_rate | 0.69 | **0.73** | ↓ |
| acc_move_std_rel | 0.64 | **0.70** | ↑ (burst di movimento) |
| updown_corr | 0.71 | 0.63 | ↑ |
| hr_est | ~0.50 | **0.60** | ↑ (lieve tachicardia) |

**Interpretazione.** Con label pulite tutte le feature fisiologiche separano meglio.
L'HR ora è nella direzione giusta (lieve aumento in apnea) ma resta debole: nelle
apnee volontarie di soggetti sani la risposta cardiaca è modesta — diventerà
informativa sull'OSA reale della PSG. Le feature respiratorie restano le portanti.

**Restano aperti.** HRV vera (intervalli battito-battito) non ancora estratta;
HR comunque approssimato (manca pipeline validata). Prossimo: definire lo schema
feature 1:1 su MESA/SHHS e una baseline sulle sole feature condivise.

---

## 2026-06-28 — Baseline + cross-validation robusta

**Cosa ho provato.** (1) Baseline RF sulle SOLE feature condivise (sanity della
nostra idea, prima del transfer). (2) CV robusta su molte partizioni per ottenere
una stima convergente con intervallo di confidenza (consiglio reviewer).
Script: `baseline_cv.py` → `cv_results.csv`, `cv_convergence.png`.
Setup: RF (300 alberi, class_weight balanced_subsample), subject-wise, soglia 0.30.
Confronto feature condivise STATICHE (18) vs + TEMPORALI (74, delta/rolling K=7).

**Risultati.**

| modello | 5-fold ×100 ROC-AUC | 95% CI | PR-AUC | LOSO ROC-AUC | LOSO recall | LOSO bal-acc | F1 |
|---|---|---|---|---|---|---|---|
| shared-static (18) | 0.910 ± 0.005 | [0.899, 0.919] | 0.720 | 0.913 | 0.754 | 0.831 | 0.619 |
| shared-temporal (74) | 0.920 ± 0.004 | [0.912, 0.928] | 0.782 | 0.919 | 0.768 | 0.857 | 0.707 |

Riferimento paper (RF-Temporal, 102 feature): ROC-AUC 0.90 ± 0.03, PR-AUC 0.56.

**Interpretazione (forte).**
- **La nostra idea regge**: con sole **18 feature fisiologiche condivise** si eguaglia
  il modello del paper a 102 feature (ROC-AUC 0.91 vs 0.90), con PR-AUC perfino più
  alta (0.72 vs 0.56). È il feature space ridotto e trasferibile che cercavamo.
- **CV robusta**: su 100 partizioni la media converge (vedi figura) con CI strettissimo
  (±0.005). Il "± 0.03" del paper era la dispersione *tra fold* (eterogeneità soggetti),
  NON l'incertezza della stima: vanno distinti. LOSO (deterministico) dà 0.913,
  conferma indipendente. → il ~0.90 del paper è solido, non frutto di uno split fortunato.
- Le feature temporali aiutano anche qui (+0.01 ROC-AUC, +0.06 PR-AUC), coerente col paper.
- Eterogeneità tra soggetti reale: recall LOSO 0.78 ± 0.27, un soggetto a recall ~0
  (caso difficile da indagare).

**Per il paper.** Stesso harness applicabile a RF-Temporal (102 feature) per riportare:
"5-fold ripetuto ×100 → ROC-AUC convergente con CI; LOSO come conferma esaustiva".
Distinguere sempre dispersione-tra-fold vs incertezza-della-media.

**Prossimo.** Schema feature 1:1 su PSG (MESA/SHHS): Chest+ABD→respiro, ECG→HR/HRV.

---

## 2026-06-28 — HR/HRV battito-battito: cosa si recupera (risultato in parte negativo)

**Cosa ho provato.** Rilevazione dei battiti dal BCG-ACC (80 Hz) per HR e HRV vera,
validata PRIMA su finestre pulite (probe), poi integrata: `hr_hrv()` in
`extract_shared_features_smc.py` (rigetto spike di movimento + pulizia intervalli NN).

**Risultato (onesto).**
- **HR**: plausibile (60–90 bpm) ma affidabile solo nel **7% delle finestre** (conf>0.7);
  nelle altre il movimento rovina la rilevazione dei battiti.
- **HRV**: NON affidabile. Anche dopo pulizia NN, SDNN ~90–110 ms e RMSSD ~130–200 ms
  sono non-fisiologici (atteso RMSSD ~20–50 ms; RMSSD>SDNN = battiti spuri residui).
- **Discriminazione apnea**: il canale cardiaco è a livello del caso —
  hr_est AUC 0.51, hrv_sdnn 0.52, hrv_rmssd 0.51. (La hr_est spettrale precedente
  dava 0.60, ma era probabilmente spuria: catturava armoniche di respiro/movimento,
  non il cuore. Tengo la versione battito-battito perché corretta, anche se nulla qui.)
- Curiosità: `hr_conf` separa lievemente (0.585): in apnea/recovery c'è più movimento
  → meno battiti puliti.

**Interpretazione (importante per la strategia).**
- Sul nostro dataset (apnee VOLONTARIE, soggetti sani) la firma cardiaca dell'apnea è
  debole: niente desaturazione/arousal reali → niente bradi-tachicardia marcata.
  Quindi il canale cardiaco **non aiuta qui**, ed è un risultato atteso, non un bug.
- HR/HRV vanno comunque **tenute nello schema condiviso**: sulla PSG l'HRV si calcola
  in modo banale e affidabile dall'ECG, e sull'OSA reale la variazione ciclica di HR
  (cardiopulmonary coupling) è un marker FORTE. È esattamente uno dei motivi del transfer:
  il "teacher" PSG insegna un canale che il materasso lab non può esprimere bene da solo.
- Sul materasso, le portanti restano respiro + movimento (confermato). Il baseline
  0.91/0.92 non cambia: le feature cardiache, non informative qui, l'RF le ignora.

**Limite/decisione.** HRV affidabile dal materasso richiede la pipeline HR validata
(Marinai2025_HR) o sensori/posizionamento migliori: fuori scope ora. Documentato come
limitazione. Prossimo: schema feature 1:1 su PSG, dove HR/HRV da ECG sono solide.

---

## 2026-06-28 — Lato PSG: schema feature 1:1 (dataset e validazione)

**Dataset scelto.** Per evitare l'embargo della Challenge e il problema "è enorme",
si parte da **UCDDB** (PhysioNet, St. Vincent's/UCD): 25 PSG di pazienti OSA REALI
(AHI 24±20), **aperto** (ODC-BY, niente DUA), **757 MB**. Canali utili: ribcage+abdo
(sforzo, **8 Hz come la matrice**), ECG (128 Hz), Flow+SpO2 (per label), BodyPos.
Eventi apnea/ipopnea annotati in `_respevt.txt`. Per scalare poi a MESA/SHHS: pattern
streaming (1 file → feature → cancella raw), solo i canali utili, sottoinsieme di notti.

**Compute.** Tutto fin qui è CPU (estrazione + RF + CV). La GPU servirà SOLO per il
teacher deep su dataset grande. Per ora si lavora sul PC.

**Cosa ho fatto.** `extract_shared_features_psg.py`: specchio 1:1 di quello materasso,
stesse colonne. Respiro da ribcage+abdo, paradosso = corr(ribcage,abdo), HR/HRV da ECG,
posizione da BodyPos, movimento = proxy (energia alta-freq sforzo). Label = overlap ≥50%
con eventi. Validato su 1 registrazione (ucddb002), poi download delle altre 24.

**Risultato (ucddb002, OSA reale).** 749 finestre, 79 eventi (10.5%), prevalenza simile
al materasso (11.9%). AUC univariata (evento vs no):

| feature | AUC | nota |
|---|---|---|
| resp_amp_std | 0.70 | ampiezza variabile (drop+recovery), come materasso |
| hrv_rmssd | 0.70 | **HRV ora affidabile**: RMSSD ~20-25 ms, SDNN ~42-53 ms (fisiologici!) |
| resp_rate | 0.69 | ↓ nell'evento |
| hr_est | 0.68 | ↓ nell'evento (modulazione autonomica) |
| resp_regularity | 0.67 | **debole qui** (era 0.86 sul materasso!) |

**Interpretazione (cruciale per il transfer).**
- **Parità di feature confermata**: stesso schema calcolabile su entrambi, prevalenze e
  direzioni fisiologiche coerenti (ampiezza↓, HR↓/HRV↑).
- **Il canale cardiaco, morto sul materasso, è ricco qui**: HRV affidabile al 100% e
  discriminante. È esattamente il valore del transfer (il teacher PSG insegna un canale
  che lo studente materasso non sa esprimere da solo).
- **Gap di dominio misurato**: `resp_regularity` forte sul materasso (apnea volontaria =
  cessazione completa) ma debole su UCDDB (ipopnee = riduzione parziale). Questo è IL
  punto da gestire col domain adaptation, ora quantificato.

**Limiti.** Finora 1 soggetto (AUC within-subject). Movimento PSG è un proxy (manca ACC).
Prossimo: estrazione su tutti i 25, confronto distribuzioni materasso↔PSG, primo transfer.

---

## 2026-06-28 — PSG completo (25 soggetti) + primo transfer: il domain gap

**Estrazione completa.** Tutti i 25 pazienti UCDDB: 20.793 finestre, 2.097 eventi
(10.1%), HR affidabile 99%. Direzioni fisiologiche corrette sulla coorte:
resp_amp_mean ↓ (AUC 0.71), hrv_sdnn ↑ (0.66), updown_corr ↓ (paradosso, 0.60).

**Primo esperimento di transfer** (`transfer_experiment.py`, feature core respiro+movimento,
scala allineata; updown_corr esclusa perché di segno opposto tra domini):

| esperimento | ROC-AUC |
|---|---|
| within SMC (CV) | 0.911 |
| within PSG (CV) | 0.798 |
| **transfer PSG → SMC (zero-shot)** | **0.484** (≈ caso) |
| **transfer SMC → PSG (zero-shot)** | **0.699** (parziale) |

**Risultato chiave (onesto).** La parità di feature è condizione *necessaria ma non
sufficiente*: il transfer naïve è **asimmetrico** e in direzione PSG→SMC **fallisce**
(a livello del caso). La figura `domain_comparison.png` spiega perché: forte
**covariate shift**, soprattutto su:
- `resp_regularity`: PSG concentrata ~0.9-1.0 (sonno regolare, ipopnee = riduzioni lievi),
  SMC spalmata 0.1-0.8 (apnee volontarie + protocollo slow/fast). Il PSG **non vede mai**
  il regime a bassa regolarità che caratterizza le apnee del materasso → non sa predirle.
- `resp_rate`: SMC più lenta (protocollo), PSG ritmo di sonno naturale.

**Interpretazione strategica.** Questo NON affossa l'idea: è esattamente il problema che
il paper deve risolvere, ora **quantificato**. Il contributo scientifico è il
**domain adaptation**, non il transfer naïve. Lo shift nasce anche da: (a) etichette
diverse (apnea volontaria vs ipopnea clinica), (b) normalizzazione _rel su baseline
diverse (notte intera vs protocollo strutturato), (c) protocollo SMC con regimi
(slow/fast) assenti nella PSG.

**Prossimo (domain adaptation).** Opzioni in ordine di sforzo:
1. allineamento marginale delle feature (z-score per-dominio / CORAL) prima del transfer;
2. few-shot fine-tuning con pochi soggetti target (scenario clinico realistico);
3. armonizzare le label (apnea-only vs +ipopnea) e la normalizzazione.

---

## 2026-06-28 — Domain adaptation, opzione 1: allineamento feature

`domain_adaptation.py`. Confronto transfer naïve vs allineamento non supervisionato.

| metodo | PSG→SMC (serve) | SMC→PSG |
|---|---|---|
| naïve | 0.484 (caso) | 0.699 |
| z-score per-soggetto | 0.670 | **0.716** |
| **z-score per-dominio** | **0.703** | 0.619 |
| CORAL | 0.609 | 0.669 |
| per-soggetto + CORAL | 0.625 | 0.691 |

**Risultato.** L'allineamento **recupera il transfer dal caso (0.48) a 0.70** nella
direzione utile PSG→SMC. Il semplice z-score (medie/scale) batte CORAL: lo shift
dominante è nelle marginali (es. `resp_regularity`), non nelle covarianze. È la prova
"naïve fallisce → adaptation recupera" — la spina dorsale del paper.
Nota: per-dominio usa statistiche del test (transduttivo); per-soggetto è leakage-free
e più conservativo (0.670) → uso per-soggetto come default negli step successivi.

---

## 2026-06-28 — Domain adaptation, opzione 2: few-shot (risultato NEGATIVO e decisivo)

`fewshot.py`. Allineamento z-score per-soggetto. Per ogni k soggetti SMC etichettati:
A=pooling PSG+SMC, A2=few-shot equo (target ripesato), B=solo SMC.

| k | A: pooling | A2: equo | B: solo SMC |
|---|---|---|---|
| 0 | 0.670 | — | — |
| 1 | 0.688 | 0.699 | **0.822** |
| 3 | 0.710 | 0.744 | **0.884** |
| 8 | 0.764 | 0.807 | **0.894** |

**Risultato (robusto).** Anche col few-shot EQUO (target ripesato per non essere
schiacciato dalle 20k finestre PSG), il transfer resta **sotto** al modello solo-SMC
a ogni k. Conclusione: **quando si hanno etichette materasso, la PSG non aiuta** — il
task del materasso (apnee volontarie) è così omogeneo/facile che 1 solo soggetto dà 0.82.
L'unico regime dove la PSG vince è **k=0** (nessuna etichetta target): 0.67 vs niente.

**Implicazione strategica (importante).**
- Il racconto "pretrain su PSG enorme → migliora il materasso" **non regge su questi dati**.
  → NON inseguire ora il teacher deep su MESA/SHHS+GPU: questi esperimenti predicono che
  non pagherebbe per il dataset lab attuale.
- Il valore del transfer si sblocca solo con (a) **scenario senza etichette** (k=0), o
  (b) un **target più difficile/eterogeneo = pazienti OSA reali sul materasso** (dati da
  raccogliere). Il vincolo vero è il TARGET, come sospettato dall'inizio.
- Restano solidi e pubblicabili: (1) feature space condiviso (18 feat ≈ 102 del paper,
  PR-AUC migliore), (2) CV robusta convergente (0.91, LOSO 0.913) per i reviewer,
  (3) analisi onesta del domain gap PSG↔materasso.

Opzione 3 (armonizzare label apnea-only vs ipopnea) rilevante solo per migliorare il
regime k=0 (la nicchia del transfer), non cambia il verdetto A2<B.

---

## 2026-06-28 — Domain adaptation, opzione 3: armonizzazione label

`option3_labels.py`. Ricalcolo le label PSG usando come positivi **solo le apnee**
(escluse le ipopnee), per avvicinarle alle cessazioni complete del materasso.
PSG positivi: tutti-eventi=2097 → apnea-only=438.

| sorgente PSG (zero-shot PSG→SMC, z-score per-soggetto) | ROC-AUC |
|---|---|
| tutti-eventi (apnea+ipopnea) | 0.670 |
| **apnea-only** | **0.723** |

Conferma: il mismatch di severità diluiva il segnale. Apnea-only + allineamento è la
config migliore per il regime senza etichette.

### Sintesi domain adaptation (tutte e 3 le opzioni)
- **Opz.1 allineamento**: zero-shot PSG→SMC 0.48 → 0.70.
- **Opz.2 few-shot**: il transfer NON batte mai il modello solo-materasso quando ci sono
  etichette (1 soggetto → 0.82). Valore solo a k=0.
- **Opz.3 label apnea-only**: zero-shot k=0 → 0.72 (migliore config leakage-free).

**Conclusione del filone transfer.** Esiste una nicchia reale: **zero-shot senza
etichette materasso ~0.72** (feature condivise + allineamento + apnea-only). Ma con
etichette il modello solo-materasso vince. Il transfer pagherebbe davvero solo con
**pazienti OSA reali sul materasso** (target difficile/eterogeneo) — dati da raccogliere.

**Contributi solidi e pubblicabili ORA (senza inseguire il teacher deep):**
1. feature space fisiologico condiviso (18 feat ≈ 102 del paper, PR-AUC migliore);
2. CV robusta convergente (0.91, CI [0.90,0.92], LOSO 0.913) — risposta ai reviewer;
3. mappatura 1:1 a PSG clinica + analisi onesta del domain gap + zero-shot ~0.72.
