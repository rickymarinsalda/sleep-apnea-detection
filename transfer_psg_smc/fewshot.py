#!/usr/bin/env python3
"""
Domain adaptation -- OPZIONE 2: few-shot fine-tuning.

Scenario clinico realistico: ho TANTI dati PSG (sorgente) e POCHI soggetti
materasso etichettati. Domanda: il pretraining su PSG aiuta rispetto ad allenare
solo sui pochi soggetti materasso disponibili?

Per ogni k (n. soggetti materasso usati per adattare):
  A) TRANSFER+few-shot : RF su [PSG] + [k soggetti SMC], test sui restanti SMC
  B) SMC-only          : RF sui soli k soggetti SMC, test sui restanti SMC
Allineamento: z-score per-soggetto (leakage-free). Ripetuto su selezioni casuali.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
smc = pd.read_csv(f"{ROOT}/shared_features_smc_30s.csv")
psg = pd.read_csv(f"{ROOT}/shared_features_psg_30s.csv")
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]
KS = [0, 1, 2, 3, 5, 8]
REPEATS = 15


def prep(df):
    d = df.dropna(subset=CORE + ["label_apnea"]).copy()
    return d[CORE].to_numpy(float), d["label_apnea"].to_numpy(int), d["Subject"].to_numpy()


def zsub(X, g):
    Z = np.zeros_like(X)
    for s in np.unique(g):
        m = g == s
        mu, sd = X[m].mean(0), X[m].std(0)
        sd[sd < 1e-9] = 1.0
        Z[m] = (X[m] - mu) / sd
    return Z


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


Xs, ys, gs = prep(smc)
Xp, yp, gp = prep(psg)
Xs, Xp = zsub(Xs, gs), zsub(Xp, gp)          # allineamento per-soggetto
subs = np.unique(gs)

res = {k: {"A": [], "B": []} for k in KS}
for k in KS:
    for r in range(REPEATS):
        rng = np.random.default_rng(r)
        tr_subj = rng.choice(subs, size=max(k, 1), replace=False) if k > 0 else np.array([])
        te_subj = np.array([s for s in subs if s not in tr_subj])
        te = np.isin(gs, te_subj)
        trm = np.isin(gs, tr_subj)
        # A) transfer + few-shot (pooling naïve)
        if k > 0:
            XA = np.vstack([Xp, Xs[trm]]); yA = np.concatenate([yp, ys[trm]])
        else:
            XA, yA = Xp, yp
        res[k]["A"].append(roc_auc_score(ys[te], rf().fit(XA, yA).predict_proba(Xs[te])[:, 1]))
        # A2) transfer + few-shot EQUO: ripeso i target così pesano quanto la PSG
        if k > 0:
            w = np.concatenate([np.ones(len(Xp)), np.full(trm.sum(), len(Xp) / max(trm.sum(), 1))])
            mA2 = rf().fit(XA, yA, sample_weight=w)
            res[k].setdefault("A2", []).append(roc_auc_score(ys[te], mA2.predict_proba(Xs[te])[:, 1]))
        # B) SMC-only (serve k>=1 e entrambe le classi)
        if k > 0 and ys[trm].sum() > 0 and (ys[trm] == 0).sum() > 0:
            res[k]["B"].append(roc_auc_score(ys[te], rf().fit(Xs[trm], ys[trm]).predict_proba(Xs[te])[:, 1]))

def m(d, k, key):
    return np.mean(d[k][key]) if d[k].get(key) else np.nan

print(f"{'k':>3} | {'A: pooling':>10} | {'A2: equo':>9} | {'B: solo SMC':>11}")
print("-" * 44)
for k in KS:
    a, a2, b = m(res, k, "A"), m(res, k, "A2"), m(res, k, "B")
    f = lambda v: f"{v:.3f}" if v == v else "n/a"
    print(f"{k:>3} | {f(a):>10} | {f(a2):>9} | {f(b):>11}")
