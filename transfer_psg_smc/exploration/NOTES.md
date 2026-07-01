# Esplorazioni (idee 1-2) — separate dallo studio del paper

Esperimenti per capire se i dataset esterni di apnea possono migliorare l'SMC.
Leggono in sola lettura i CSV già prodotti; NON toccano i file del paper né i
deliverable del transfer validato.

## 2026-06-28 — Idea #1: augmentation / pooling (NEGATIVO)

`augmentation_morphology.py` → `tradeoff_augmentation.{csv,png}`.
Mescolo dati materasso + PSG in feature-space con peso α crescente; misuro LAB
(apnea volontaria) e REALE (apnea PSG).

| α | LAB materasso | apnea REALE |
|---|---|---|
| 0 | 0.902 | 0.799 |
| 0.1 | 0.841 | 0.801 |
| 1.0 | 0.821 | 0.801 |

**Esito.** Il pooling in feature-space **peggiora il lab e non muove l'apnea reale**
(già al tetto ~0.80 per effort+movimento). Per ogni target domina il dato in-domain.
Limite: il beneficio vero (robustezza ai *tipi* di apnea sul materasso) NON è
misurabile senza dati OSA reali raccolti col materasso. Col solo pooling: nessun guadagno.

## 2026-06-28 — Idea #2: feature discovery dalla PSG (VINCENTE)

`feature_discovery.py` → `psg_rich_features.csv`. Descrittori ricchi dello sforzo,
classificati per discriminazione dell'apnea reale (tutti portabili sul materasso):

| feature nuova | AUC (PSG) |
|---|---|
| effort_kurtosis | **0.86** |
| breath_amp_cv | **0.86** |
| env_min_ratio | 0.80 |
| effort_entropy | 0.74 |
| longest_flat_s | 0.71 |
| n_breaths | 0.60 |

Migliori core attuali su PSG ~ 0.71. **Kurtosis e breath-amplitude-CV battono tutto.**
Motivo: la media sulla finestra spalma l'evento; *forma dell'onda* e *variabilità
respiro-respiro* lo catturano. → Candidati forti da aggiungere al pipeline materasso.

**Prossimo.** Calcolare queste feature ANCHE sul materasso e ri-testare baseline (lab)
+ transfer. È il lever concreto per migliorare l'SMC e anche il paper (feature migliori,
con credibilità clinica perché scoperte su PSG reale).

## 2026-06-28 — Idea #2bis: feature nuove portate sul MATERASSO

`port_features_to_mattress.py` → `smc_rich_features.csv`. Calcolate sul materasso con
la STESSA finestratura del dataset SMC (merge 581/581, 20793/20793 — allineamento esatto).

**AUC univariata sul materasso (lab):** effort_kurtosis **0.896**, breath_amp_cv **0.890**,
env_min_ratio 0.745, effort_entropy 0.60, longest_flat_s 0.52.
→ kurtosis e breath_amp_cv sono tra le MIGLIORI feature anche sul materasso (≈ resp_regularity 0.86).

**Impatto modelli:**
| | CORE (6) | CORE+NEW | Δ |
|---|---|---|---|
| LAB within-SMC (5-fold ×20) | 0.908 ± 0.005 | 0.912 ± 0.004 | +0.004 |
| TRANSFER zero-shot PSG→SMC | 0.670 | 0.724 | **+0.054** |

**Interpretazione.**
- Le feature nuove sono **individualmente forti** anche sul materasso (≈0.89): idea #2 confermata.
- Sul **lab** il guadagno multivariato è minimo (+0.004): il task è saturo (0.91) e le nuove
  feature sono ridondanti con regularity/amplitude → l'RF già cattura quel segnale.
- Sul **transfer** il guadagno è reale (+0.054): catturano morfologia più robusta tra domini.

**Conclusione esplorazioni.** L'uso utile dei dataset esterni per l'SMC NON è transfer di
dati/modelli (non paga), ma **feature discovery**: scoprire su apnea reale quali descrittori
contano (forma dell'onda, variabilità respiro-respiro) e portarli nel pipeline. Beneficio
misurabile su transfer/robustezza; sul lab saturo poco, ma è un set di feature migliore e
con credibilità clinica → utile anche per il paper.
