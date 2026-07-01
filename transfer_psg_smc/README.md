# SMC Apnea — Cross-Domain Physiological Feature Study

**An extension of** [`sleep-apnea-detection`](https://github.com/rickymarinsalda/sleep-apnea-detection)
(Smart Mattress Cover, SMC — in-bed sleep apnea detection).

This sub-project investigates a single question:

> *Can large public clinical PSG (polysomnography) apnea datasets be used to improve — or at least inform — apnea detection with the textile Smart Mattress Cover (SMC)?*

The short answer we reached, honestly: **not through naïve transfer, but yes through a shared physiological feature space and, above all, through feature discovery.** The sections below document exactly what was done, what was achieved (with numbers), the honest limitations, and a menu of well-scoped thesis directions at the end.

All experiments here run on **CPU** and read the feature tables produced by the two extractors. No new data collection was involved.

---

## 1. Background & data

| Domain | Dataset | Subjects | Windows (30 s) | Positive class |
|---|---|---|---|---|
| **Target** | SMC (this lab study) | 23 (healthy, *voluntary* apnea) | 581 | 69 apnea (11.9 %) |
| **Source** | [UCDDB](https://physionet.org/content/ucddb/1.0.0/) (St. Vincent's/UCD, PhysioNet, open ODC-BY) | 25 (**real OSA**, AHI 24±20) | 20 793 | 2 097 events (10.1 %) |

**Key sensing fact.** The SMC sees *respiratory effort + body movement + a weak cardiac (BCG) component* — it has **no airflow and no SpO₂**. Clinical apnea is defined by airflow cessation, so this modality has an intrinsic ceiling (~0.80 ROC-AUC on real apnea; see §3).

**The bridge.** Raw signals of the two systems are not comparable (pressure matrix vs. clinical leads). We instead map both to the **same physiological feature layer**, computable on each: respiratory rate/amplitude/regularity, thoraco-abdominal paradox, movement, and HR/HRV. Conveniently, the UCDDB effort belts (`ribcage`, `abdo`) are sampled at **8 Hz — identical to the SMC pressure matrix** — so respiratory features map 1:1 with no resampling.

---

## 2. What was done

Each step is a self-contained script (run with the project `venv`).

| # | Step | Script | Output |
|---|---|---|---|
| 1 | Extract the **shared physiological features** on the SMC (manual labels, 69 apnea) | `extract_shared_features_smc.py` | `shared_features_smc_30s.csv` |
| 2 | Extract the **same features** on UCDDB PSG (real OSA) | `extract_shared_features_psg.py` | `shared_features_psg_30s.csv` |
| 3 | **Robust cross-validation** of the SMC model (repeated 5-fold + LOSO) | `baseline_cv.py` | `cv_results.csv`, `cv_convergence.png` |
| 4 | **Cross-domain transfer** (naïve) + distribution comparison | `transfer_experiment.py` | `domain_comparison.png` |
| 5 | **Domain adaptation** — feature alignment (z-score / CORAL) | `domain_adaptation.py` | — |
| 6 | **Domain adaptation** — few-shot fine-tuning | `fewshot.py` | — |
| 7 | **Domain adaptation** — label harmonization (apnea-only) | `option3_labels.py` | — |
| 8 | **Feature discovery** on PSG (richer effort descriptors) | `exploration/feature_discovery.py` | `exploration/psg_rich_features.csv` |
| 9 | **Port discovered features to the SMC** and re-test | `exploration/port_features_to_mattress.py` | `exploration/smc_rich_features.csv` |
| 10 | Augmentation / pooling trade-off (idea archived as negative) | `exploration/augmentation_morphology.py` | `exploration/tradeoff_augmentation.png` |

Physiological feature layer (computed identically on both domains):
`resp_rate`, `resp_amp_mean`, `resp_amp_std`, `resp_band_power`, `resp_regularity`,
`updown_corr` (thoraco-abdominal paradox), `acc_move_std` (movement),
`hr_est`, `hr_conf`, `hrv_sdnn`, `hrv_rmssd`, plus per-subject-normalized `*_rel` variants.

---

## 3. What was achieved (detailed results)

### 3.1 A compact, interpretable feature space matches the paper model
An 18-feature *shared physiological* set matches the published 102-feature model, with **better PR-AUC**:

| Model | ROC-AUC (5-fold ×100) | 95 % CI | PR-AUC | LOSO ROC-AUC | LOSO recall | F1 |
|---|---|---|---|---|---|---|
| Shared-static (18 feat.) | 0.910 ± 0.005 | [0.899, 0.919] | 0.720 | 0.913 | 0.754 | 0.619 |
| Shared-temporal (74 feat.) | **0.920 ± 0.004** | [0.912, 0.928] | **0.782** | 0.919 | 0.768 | 0.707 |
| *Paper RF-Temporal (102 feat.)* | *0.90 ± 0.03* | — | *0.56* | — | *0.82* | *0.50* |

### 3.2 Robust cross-validation (answer to the reviewers)
Repeating subject-wise 5-fold over **100 random partitions**, the ROC-AUC **converges** to 0.910 with a tight CI ([0.899, 0.919]); Leave-One-Subject-Out gives **0.913** independently. The paper's "± 0.03" was the *between-fold spread* (subject heterogeneity), **not** the uncertainty of the estimate — the two are now clearly separated. See `cv_convergence.png`.

### 3.3 Feature parity with real clinical apnea (UCDDB)
The same features, computed on 25 real-OSA patients, discriminate apnea with physiologically correct directions (amplitude ↓, HRV ↑, paradox ↓). Within-PSG CV ROC-AUC = **0.798** — this is the realistic **effort-only ceiling** for this modality (no airflow/SpO₂). **HR/HRV, unreliable on the SMC, is reliable here** (HRV RMSSD 20–25 ms, physiological), which is precisely the channel a clinical source could contribute.

### 3.4 Transfer: naïve fails, alignment recovers, harmonization helps
Zero-shot **PSG → SMC** (the useful direction):

| Configuration | ROC-AUC |
|---|---|
| Naïve (no alignment) | 0.484 (chance) |
| + feature alignment (z-score per-domain) | 0.703 |
| + label harmonization (apnea-only source) | **0.723** |

Feature alignment lifts the transfer from chance to ~0.70; matching event severity (apnea vs. hypopnea) adds more. Simple marginal alignment beat CORAL, i.e. the dominant shift is in the marginals (e.g. `resp_regularity`), not the covariances.

### 3.5 Few-shot: transfer does **not** beat an SMC-only model when labels exist *(key negative result)*
With even a *fair, target-reweighted* few-shot scheme, PSG pretraining stays below an SMC-only model at every budget *k*:

| k SMC subjects | Transfer (fair few-shot) | SMC-only |
|---|---|---|
| 1 | 0.699 | **0.822** |
| 3 | 0.744 | **0.884** |
| 8 | 0.807 | **0.894** |

The SMC voluntary-apnea task is so homogeneous that a single subject already yields 0.82. External data helps **only in the zero-label regime** (k = 0). This is an honest, decisive finding: it argues *against* investing in a deep PSG "teacher" for the current lab dataset.

### 3.6 Feature discovery is the real lever *(main positive result)*
Ranking richer effort-waveform descriptors on real PSG apnea surfaced features far stronger than those in use, **all portable to the SMC**:

| New feature | AUC on PSG | AUC on SMC |
|---|---|---|
| `effort_kurtosis` (waveform shape / flow-limitation) | 0.86 | **0.90** |
| `breath_amp_cv` (breath-to-breath amplitude variability) | 0.86 | **0.89** |
| `env_min_ratio` (deepest sustained reduction) | 0.80 | 0.75 |

Porting them to the SMC feature set:
- **Lab** ROC-AUC 0.908 → 0.912 (**+0.004**, marginal — the lab metric is saturated & the new features are partly redundant there);
- **Transfer** PSG → SMC 0.670 → 0.724 (**+0.054**, meaningful — the new descriptors are more domain-robust).

**Takeaway:** the useful way to exploit external apnea data is *not* transferring data/models (it does not pay), but **feature discovery** — letting real clinical apnea tell us *which descriptors matter* (waveform shape, breath-to-breath variability) and importing them.

---

## 4. Honest limitations

- **Sensor ceiling.** Effort + movement (no airflow/SpO₂) caps real-apnea detection at ~0.80.
- **Tiny positive count.** 69 apnea windows, **~3 per subject**. Consequence (verified): per-subject recall is quantized/noisy — the "worst subject" is largely a small-sample artifact, **not** a hard case. Per-subject, event-level, and AHI targets are therefore **not measurable on the SMC** with this dataset.
- **Voluntary ≠ clinical apnea.** SMC apneas are voluntary complete cessations (central-like); UCDDB has real obstructive/hypopnea morphology. This is the concrete domain gap.
- **Saturated ROC-AUC.** Lab ROC-AUC ≈ 0.91 is near-ceiling; genuine headroom is in **PR-AUC / precision / event-level / robustness**, not ROC-AUC.

---

## 5. Repository structure & reproduction

```
transfer_psg_smc/
├── extract_shared_features_smc.py     # SMC → shared feature table
├── extract_shared_features_psg.py     # PSG (UCDDB EDF) → shared feature table
├── baseline_cv.py                     # robust CV (repeated 5-fold + LOSO)
├── transfer_experiment.py             # naïve transfer + distribution comparison
├── domain_adaptation.py               # feature alignment (z-score, CORAL)
├── fewshot.py                         # few-shot fine-tuning
├── option3_labels.py                  # label harmonization (apnea-only)
├── exploration/                       # idea experiments (feature discovery, augmentation)
│   ├── feature_discovery.py
│   ├── port_features_to_mattress.py
│   ├── augmentation_morphology.py
│   └── NOTES.md
├── DIARIO.md                          # full chronological lab log (Italian)
└── *.png                              # result figures
```

**Reproduce.** With `numpy/scipy/pandas/scikit-learn/matplotlib/pyedflib`:
1. Produce the SMC features: `python extract_shared_features_smc.py`
2. Download UCDDB (`curl -L -o ucddb.zip https://physionet.org/content/ucddb/get-zip/1.0.0/`, unzip `*.rec`/`*_respevt.txt` into `ucddb/`) and run `python extract_shared_features_psg.py`
3. Run any experiment script above.

> **Data policy.** Raw signals (`ucddb/`) and the derived feature `*.csv` are **git-ignored** (clinical-data governance + pre-publication SMC data), consistent with the base repo. All results are regenerated from the scripts.

---

## 6. Thesis challenges (menu of directions)

Rules of the game to hand to the student:
- The SMC dataset is **fixed** (23 subjects, 69 apnea windows, ~3/subject); **no new subjects**.
- The sensor has **no airflow/SpO₂** → ~0.80 ceiling on real apnea.
- **Do not chase** the (saturated) lab ROC-AUC, nor per-subject recall (noise with ~3 positives).
- **Target metrics with headroom:** PR-AUC, precision at fixed recall, event-level F1, cross-validation robustness.
- Public PSG datasets are used as *reference/validation*, not necessarily transfer.

### A — Improve the front-end signal
1. **Exploit the 4×10 pressure matrix (spatial).** Today it is collapsed to a global/zone signal, discarding spatial structure. Study which cells best capture apnea breathing, spatial patterns, channel selection. *Data: SMC · Metric: PR-AUC/precision · Medium · real headroom.*
2. **Source separation.** The pressure signal mixes respiration + cardiac (BCG) + motion. Separate them (ICA/NMF over the 40 channels, EMD/VMD, adaptive filtering) → cleaner respiration **and** a usable BCG for HR. *SMC · signal quality + PR-AUC · Medium.*

### B — Improve the features
3. **Systematic effort-feature engineering.** Extend the discovered set (breath morphology, inspiration/expiration ratio, entropy, flow-limitation, envelope dynamics). *SMC + PSG for ranking · Low difficulty, guaranteed deliverable.*

### C — Improve the model / temporal structure
4. **Temporal post-processing.** Apnea is temporally contiguous, but windows are classified independently. An HMM / smoothing over per-window probabilities removes isolated false positives. *SMC · event-F1/precision · Easy, near-guaranteed win.*
5. **(optional, GPU) Self-supervised pretraining** of a 1-D respiratory-effort encoder. Future-facing; uncertain gain on the small dataset. *Hard, needs GPU.*

### D — Reformulate the task (to escape the saturated metric)
6. **Event-level detection** (onset/offset) instead of 30-s window classification, with event-based metrics. *PSG-rich; SMC-limited by few events · Medium.*
7. **Recovery-breath / transition detector.** The large post-apnea movement is a robust marker. Build a detector around the apnea→recovery transition. *SMC · robustness · Medium, original.*

### E — Evaluation, robustness, credibility (paper-strengthening)
8. **Cross-dataset benchmarking** of the effort-only ceiling across several open PSG datasets. *Easy.*
9. **Posture / bed / motion robustness.** Quantify degradation by posture; position-invariant features. *SMC · Medium.*
10. **Calibration & operating point.** Well-calibrated probabilities, clinically sensible thresholds, reliable flags. *Easy.*

### F — PSG-side only (portable to SMC later)
11. **Personalization**, developed on PSG (many events/subject) — **not** measurable on the SMC now.
12. **AHI / severity estimation** on PSG (clinical endpoint).

### Explicitly out of scope (to avoid wasted effort)
- New SMC subjects (not currently possible).
- Personalization / event-level / AHI measured **on the SMC** (too few positives).
- Naïve supervised transfer (shown to fail).

### Suggested arc
**Warm-up** (safe, ~3–4 weeks): #4 (temporal post-processing) + #8 (benchmark).
**Core** (real headroom): #1 (spatial) or #2 (source separation), with #3 (features) throughout.
**Original touch:** #7 (recovery-breath).

---

*Full chronological log with every intermediate result is in [`DIARIO.md`](DIARIO.md) and [`exploration/NOTES.md`](exploration/NOTES.md) (Italian).*
