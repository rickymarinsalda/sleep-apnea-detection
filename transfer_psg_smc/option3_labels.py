#!/usr/bin/env python3
"""
Domain adaptation -- OPZIONE 3: armonizzazione delle label.

Le apnee del materasso sono cessazioni COMPLETE; gli eventi PSG includono molte
ipopnee (riduzioni parziali). Ipotesi: usare come positivi PSG solo le APNEE
(escludendo le ipopnee) avvicina la semantica e migliora il transfer zero-shot.
Ricalcola le label PSG apnea-only (header EDF + eventi) e ri-testa PSG->SMC.
"""
import glob, os, re
import numpy as np
import pandas as pd
import pyedflib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
smc = pd.read_csv(f"{ROOT}/shared_features_smc_30s.csv")
psg = pd.read_csv(f"{ROOT}/shared_features_psg_30s.csv")
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]
WIN = 30.0


def parse_events(path, start_s, apnea_only):
    ev = []
    for line in open(path, errors="ignore"):
        m = re.match(r"\s*(\d{2}):(\d{2}):(\d{2})\s+(\S+)\s+(.*)", line)
        if not m:
            continue
        h, mi, s, typ, rest = m.groups()
        if apnea_only and not re.match(r"APNEA", typ, re.I):
            continue
        if not apnea_only and not re.match(r"(APNEA|HYP)", typ, re.I):
            continue
        t = int(h) * 3600 + int(mi) * 60 + int(s)
        onset = t - start_s + (86400 if t < start_s else 0)
        dur = next((int(x) for x in rest.split() if x.isdigit() and 3 <= int(x) <= 120), 10)
        ev.append((onset, dur))
    return ev


# ricalcola label apnea-only per ogni finestra PSG esistente
start_cache = {}
for rec in glob.glob(f"{ROOT}/ucddb/*.rec"):
    sid = os.path.basename(rec).split(".")[0]
    f = pyedflib.EdfReader(rec)
    sd = f.getStartdatetime(); f._close()
    start_cache[sid] = (sd.hour * 3600 + sd.minute * 60 + sd.second, rec)

lab = []
for sid, sub in psg.groupby("Subject"):
    start_s, rec = start_cache[sid]
    ev = parse_events(rec.replace(".rec", "_respevt.txt"), start_s, apnea_only=True)
    for t0 in sub["t_start"].to_numpy():
        ovl = sum(max(0, min(t0 + WIN, e0 + ed) - max(t0, e0)) for e0, ed in ev)
        lab.append(int(ovl / WIN >= 0.5))
psg = psg.copy(); psg["label_apnea_only"] = lab
print(f"PSG positivi: tutti-eventi={int(psg.label_apnea.sum())}  "
      f"apnea-only={int(psg.label_apnea_only.sum())}")


def zsub(X, g):
    Z = np.zeros_like(X)
    for s in np.unique(g):
        m = g == s; mu, sd = X[m].mean(0), X[m].std(0); sd[sd < 1e-9] = 1
        Z[m] = (X[m] - mu) / sd
    return Z


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


ds = smc.dropna(subset=CORE + ["label_apnea"])
Xs, ys, gs = ds[CORE].to_numpy(float), ds.label_apnea.to_numpy(int), ds.Subject.to_numpy()
Xs = zsub(Xs, gs)


def transfer(label_col):
    d = psg.dropna(subset=CORE + [label_col])
    Xp = zsub(d[CORE].to_numpy(float), d.Subject.to_numpy())
    yp = d[label_col].to_numpy(int)
    if yp.sum() < 10:
        return np.nan
    return roc_auc_score(ys, rf().fit(Xp, yp).predict_proba(Xs)[:, 1])


print("\n=== Transfer zero-shot PSG -> SMC (z-score per-soggetto) ===")
print(f"  PSG tutti-eventi (apnea+ipopnea) : {transfer('label_apnea'):.3f}")
print(f"  PSG apnea-only                   : {transfer('label_apnea_only'):.3f}")
