#!/usr/bin/env python3
"""
Primo esperimento di TRANSFER sul livello di feature condivise.

1. Confronto distribuzioni materasso (SMC) <-> PSG (UCDDB) sulle feature comparabili.
2. Transfer zero-shot: alleno su un dominio e testo sull'altro (RF), e confronto
   con la performance within-domain. Misura il domain gap.

Feature condivise comparabili (scala allineata: normalizzate _rel + adimensionali).
Cardiaco escluso dal set "core" perche' inaffidabile lato materasso (vedi DIARIO).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
smc = pd.read_csv(f"{ROOT}/shared_features_smc_30s.csv")
psg = pd.read_csv(f"{ROOT}/shared_features_psg_30s.csv")

# feature core: respiro + movimento, scala allineata, direzione coerente tra domini
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]
# updown_corr ESCLUSA dal transfer: direzione opposta tra domini
#   (materasso apnea volontaria -> torace/addome correlati; PSG OSA -> paradosso)


def prep(df):
    d = df.dropna(subset=CORE + ["label_apnea"]).copy()
    return d[CORE].to_numpy(float), d["label_apnea"].to_numpy(int), d["Subject"].to_numpy()


Xs, ys, gs = prep(smc)
Xp, yp, gp = prep(psg)
print(f"SMC: {len(ys)} finestre ({ys.mean()*100:.1f}% evento) | PSG: {len(yp)} finestre ({yp.mean()*100:.1f}% evento)\n")


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


def within(X, y, g):
    """ROC-AUC OOF con GroupKFold (within-domain)."""
    p = np.full(len(y), np.nan)
    k = min(5, len(np.unique(g)))
    for tr, te in GroupKFold(k).split(X, y, g):
        p[te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def cross(Xtr, ytr, Xte, yte):
    """Zero-shot: alleno su un dominio intero, testo sull'altro."""
    return roc_auc_score(yte, rf().fit(Xtr, ytr).predict_proba(Xte)[:, 1])


print("=== Risultati (ROC-AUC) ===")
print(f"within SMC  (CV)        : {within(Xs, ys, gs):.3f}")
print(f"within PSG  (CV)        : {within(Xp, yp, gp):.3f}")
print(f"transfer PSG -> SMC     : {cross(Xp, yp, Xs, ys):.3f}   (zero-shot)")
print(f"transfer SMC -> PSG     : {cross(Xs, ys, Xp, yp):.3f}   (zero-shot)")

# ---- figura distribuzioni ----
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for i, c in enumerate(CORE):
    a = ax[i // 3, i % 3]
    a.hist(smc[c].dropna(), bins=40, density=True, alpha=0.5, label="SMC (materasso)")
    a.hist(psg[c].dropna(), bins=40, density=True, alpha=0.5, label="PSG (UCDDB)")
    a.set_title(c, fontsize=10)
    if i == 0:
        a.legend(fontsize=8)
plt.suptitle("Allineamento del feature space condiviso: materasso vs PSG")
plt.tight_layout()
plt.savefig(f"{ROOT}/domain_comparison.png", dpi=110)
print(f"\nFigura: {ROOT}/domain_comparison.png")
