#!/usr/bin/env python3
"""
IDEA #1 -- Augmentation con morfologia di apnea reale (forma onesta: trade-off).

Il modello materasso conosce solo apnee volontarie (cessazioni complete). La PSG
porta morfologie di apnea reale (ostruttiva, ipopnea). Iniettiamo dati PSG nel
training del materasso con peso alpha crescente e misuriamo DUE metriche:
  - LAB  : detection sul materasso (apnea volontaria) -- NON deve crollare
  - REALE: detection su apnea PSG reale               -- vogliamo che salga
Output = frontiera di trade-off (alpha da 0 = solo materasso, a grande = ~solo PSG).

NB: legge solo i CSV gia' prodotti (read-only). Non tocca lo studio del paper.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
OUT = f"{P}/exploration"
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]
ALPHAS = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def prep(path):
    d = pd.read_csv(path).dropna(subset=CORE + ["label_apnea"])
    return d[CORE].to_numpy(float), d.label_apnea.to_numpy(int), d.Subject.to_numpy()


def zsub(X, g):
    Z = np.zeros_like(X)
    for s in np.unique(g):
        m = g == s; mu, sd = X[m].mean(0), X[m].std(0); sd[sd < 1e-9] = 1
        Z[m] = (X[m] - mu) / sd
    return Z


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


Xs, ys, gs = prep(f"{P}/shared_features_smc_30s.csv")
Xp, yp, gp = prep(f"{P}/shared_features_psg_30s.csv")
Xs, Xp = zsub(Xs, gs), zsub(Xp, gp)


def oof(X_test, y_test, g_test, X_extra, y_extra, alpha):
    """GroupKFold sul dominio-test; aggiunge X_extra (peso alpha) al training."""
    p = np.full(len(y_test), np.nan)
    for tr, te in GroupKFold(5).split(X_test, y_test, g_test):
        if alpha > 0:
            X = np.vstack([X_test[tr], X_extra]); y = np.concatenate([y_test[tr], y_extra])
            w = np.concatenate([np.ones(len(tr)), np.full(len(y_extra), alpha)])
            m = rf().fit(X, y, sample_weight=w)
        else:
            m = rf().fit(X_test[tr], y_test[tr])
        p[te] = m.predict_proba(X_test[te])[:, 1]
    return roc_auc_score(y_test, p)


print(f"{'alpha':>6} | {'LAB (materasso)':>15} | {'REALE (PSG)':>12}")
print("-" * 40)
rows = []
for a in ALPHAS:
    lab = oof(Xs, ys, gs, Xp, yp, a)       # test materasso, extra=PSG
    real = oof(Xp, yp, gp, Xs, ys, a)      # test PSG, extra=materasso
    rows.append((a, lab, real))
    print(f"{a:>6} | {lab:>15.3f} | {real:>12.3f}")

# frontiera di trade-off
fig, ax = plt.subplots(figsize=(7, 5.5))
labv = [r[1] for r in rows]; realv = [r[2] for r in rows]
ax.plot(labv, realv, "-o", color="teal")
for a, l, r in rows:
    ax.annotate(f"α={a}", (l, r), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("AUC sul LAB (apnea volontaria)")
ax.set_ylabel("AUC su apnea REALE (PSG)")
ax.set_title("Trade-off: iniettare morfologia di apnea reale nel modello materasso")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/tradeoff_augmentation.png", dpi=110)
print(f"\nFigura: {OUT}/tradeoff_augmentation.png")
pd.DataFrame(rows, columns=["alpha", "auc_lab", "auc_real"]).to_csv(f"{OUT}/tradeoff_augmentation.csv", index=False)
