#!/usr/bin/env python3
"""
Domain adaptation -- OPZIONE 1: allineamento delle feature.

Confronta il transfer naïve con tre metodi di allineamento non supervisionato:
  - z-score per-SOGGETTO  (quanto è anomala la finestra rispetto al SUO baseline)
  - z-score per-DOMINIO   (rimuove shift di media/scala tra domini)
  - CORAL                 (allinea anche le covarianze sorgente->target)
Direzioni: PSG -> SMC (quella che ci serve) e SMC -> PSG.
"""
import numpy as np
import pandas as pd
from numpy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
smc = pd.read_csv(f"{ROOT}/shared_features_smc_30s.csv")
psg = pd.read_csv(f"{ROOT}/shared_features_psg_30s.csv")
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]


def prep(df):
    d = df.dropna(subset=CORE + ["label_apnea"]).copy()
    return d[CORE].to_numpy(float), d["label_apnea"].to_numpy(int), d["Subject"].to_numpy()


Xs, ys, gs = prep(smc)
Xp, yp, gp = prep(psg)


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


def zscore_subject(X, g):
    Z = np.zeros_like(X)
    for s in np.unique(g):
        m = g == s
        mu, sd = X[m].mean(0), X[m].std(0)
        sd[sd < 1e-9] = 1.0
        Z[m] = (X[m] - mu) / sd
    return Z


def zscore_global(X, ref=None):
    ref = X if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - mu) / sd


def _sqrtm_psd(C, inv=False):
    w, V = eigh(C)
    w = np.clip(w, 1e-9, None)
    d = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return V @ np.diag(d) @ V.T


def coral(Xsrc, Xtgt):
    """Allinea la covarianza della sorgente a quella del target (dopo z-score globale)."""
    lam = 1e-3
    Cs = np.cov(Xsrc, rowvar=False) + lam * np.eye(Xsrc.shape[1])
    Ct = np.cov(Xtgt, rowvar=False) + lam * np.eye(Xtgt.shape[1])
    return Xsrc @ _sqrtm_psd(Cs, inv=True) @ _sqrtm_psd(Ct)


def transfer(Xsrc, ysrc, Xtgt, ytgt):
    return roc_auc_score(ytgt, rf().fit(Xsrc, ysrc).predict_proba(Xtgt)[:, 1])


def run(direction, Xsrc, ysrc, gsrc, Xtgt, ytgt, gtgt):
    print(f"\n=== {direction} ===")
    # naïve
    print(f"  naïve (nessun allineamento) : {transfer(Xsrc, ysrc, Xtgt, ytgt):.3f}")
    # z-score per-soggetto
    a = transfer(zscore_subject(Xsrc, gsrc), ysrc, zscore_subject(Xtgt, gtgt), ytgt)
    print(f"  z-score per-soggetto        : {a:.3f}")
    # z-score per-dominio
    b = transfer(zscore_global(Xsrc), ysrc, zscore_global(Xtgt), ytgt)
    print(f"  z-score per-dominio         : {b:.3f}")
    # CORAL (su feature z-score per-dominio)
    Zs, Zt = zscore_global(Xsrc), zscore_global(Xtgt)
    c = transfer(coral(Zs, Zt), ysrc, Zt, ytgt)
    print(f"  CORAL                       : {c:.3f}")
    # combo: z-score per-soggetto + CORAL
    Zs2, Zt2 = zscore_subject(Xsrc, gsrc), zscore_subject(Xtgt, gtgt)
    d = transfer(coral(Zs2, Zt2), ysrc, Zt2, ytgt)
    print(f"  per-soggetto + CORAL        : {d:.3f}")


run("PSG -> SMC  (quella che serve)", Xp, yp, gp, Xs, ys, gs)
run("SMC -> PSG", Xs, ys, gs, Xp, yp, gp)
