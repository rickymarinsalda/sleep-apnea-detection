#!/usr/bin/env python3
"""
Porta le feature scoperte (#2) sul MATERASSO e ri-testa lab + transfer.

A) Calcola sul materasso le stesse feature ricche della PSG (effort_kurtosis,
   breath_amp_cv, env_min_ratio, effort_entropy, longest_flat_s), replicando
   ESATTAMENTE la finestratura/etichettatura del dataset SMC esistente.
B) Confronta CORE vs CORE+NEW su:
   - baseline lab (within-SMC, GroupKFold ripetuto)
   - transfer zero-shot PSG -> SMC (z-score per-soggetto)

Isolato in exploration/. Legge i raw manuali e i CSV già prodotti; non tocca il paper.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = "/Users/ricky/Documents/phd/apnea_study"
P = f"{ROOT}/transfer_psg_smc"
MAT_CSV = f"{ROOT}/dataset_a_mano/dataset_apnea_ricky_MAT_filt_indiciApneaManuale.csv"
FS, WIN, WN = 8.0, 30.0, 240
RESP = (0.1, 0.7)
ALL_CH = [f"ch{i}" for i in range(1, 41)]
CORE = ["resp_rate", "resp_regularity", "resp_amp_mean_rel", "resp_amp_std_rel",
        "resp_band_power_rel", "acc_move_std_rel"]
NEW = ["effort_kurtosis", "breath_amp_cv", "env_min_ratio", "effort_entropy", "longest_flat_s"]


def bp(x, band):
    b, a = signal.butter(3, band, btype="band", fs=FS)
    return signal.filtfilt(b, a, x - np.nanmean(x))


def clean(x):
    return pd.Series(x).interpolate(limit_direction="both").fillna(0.0).to_numpy()


def spectral_entropy(x):
    f, pxx = signal.welch(x, fs=FS, nperseg=min(len(x), WN))
    pxx = pxx / (pxx.sum() + 1e-12)
    return -np.sum(pxx * np.log(pxx + 1e-12))


def rich_feats(seg, env, med_env):
    pk, _ = signal.find_peaks(seg, distance=int(1.2 * FS))
    amps = env[pk] if len(pk) else np.array([0.0])
    low = env < 0.30 * med_env
    runs = np.diff(np.where(np.concatenate(([low[0]], np.diff(low.astype(int)) != 0, [True])))[0])
    longest = (runs[::2].max() if low[0] else (runs[1::2].max() if len(runs) > 1 else 0)) / FS
    return dict(
        effort_kurtosis=kurtosis(seg),
        breath_amp_cv=np.std(amps) / (np.mean(amps) + 1e-9),
        env_min_ratio=np.min(env) / (med_env + 1e-9),
        effort_entropy=spectral_entropy(seg),
        longest_flat_s=longest,
    )


# ---------- A) feature ricche sul materasso (stessa finestratura del dataset SMC) ----------
mat = pd.read_csv(MAT_CSV)
mat = mat[~(mat[ALL_CH] == 0).all(axis=1)].copy()
rows = []
for sid, g in mat.groupby("Subject"):
    g = g.sort_values("Time").reset_index(drop=True)
    ch = np.apply_along_axis(clean, 0, g[ALL_CH].to_numpy(float))
    active = np.var(ch, axis=0) > np.percentile(np.var(ch, axis=0), 50)
    glob = ch[:, active].mean(1) if active.any() else ch.mean(1)
    resp = bp(glob, RESP)
    env = np.abs(signal.hilbert(resp))
    med_env = np.median(env)
    status = g["Status"].to_numpy()
    t = g["Time"].to_numpy(float)
    for w in range(len(g) // WN):
        sl = slice(w * WN, (w + 1) * WN)
        st = status[sl]
        maj = np.bincount(st).argmax()
        if maj == 3 and np.mean(st == 3) >= 0.5:
            lab = 1
        elif maj in (0, 1, 2) and np.mean(np.isin(st, [0, 1, 2])) >= 0.5:
            lab = 0
        else:
            continue
        r = rich_feats(resp[sl], env[sl], med_env)
        r.update(Subject=sid, t_start=round(float(t[sl][0]), 2), label_apnea=lab)
        rows.append(r)
smc_rich = pd.DataFrame(rows)
smc_rich.to_csv(f"{P}/exploration/smc_rich_features.csv", index=False)

# ---------- merge core + new ----------
smc_core = pd.read_csv(f"{P}/shared_features_smc_30s.csv")
smc_core["t_start"] = smc_core["t_start"].round(2)
smc = pd.merge(smc_core, smc_rich[["Subject", "t_start"] + NEW], on=["Subject", "t_start"])

psg_core = pd.read_csv(f"{P}/shared_features_psg_30s.csv")
psg_core["t_start"] = psg_core["t_start"].round(2)
psg_rich = pd.read_csv(f"{P}/exploration/psg_rich_features.csv")
psg_rich["t_start"] = psg_rich["t_start"].round(2)
psg = pd.merge(psg_core, psg_rich[["Subject", "t_start"] + NEW], on=["Subject", "t_start"])
print(f"merge: SMC {len(smc)}/{len(smc_core)}  PSG {len(psg)}/{len(psg_core)}")


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=0)


def zsub(X, g):
    Z = np.zeros_like(X)
    for s in np.unique(g):
        m = g == s; mu, sd = X[m].mean(0), X[m].std(0); sd[sd < 1e-9] = 1
        Z[m] = (X[m] - mu) / sd
    return Z


def lab_cv(cols, reps=20):
    d = smc.dropna(subset=cols + ["label_apnea"])
    X, y, gg = d[cols].to_numpy(float), d.label_apnea.to_numpy(int), d.Subject.to_numpy()
    aucs = []
    for r in range(reps):
        subs = np.unique(gg); np.random.default_rng(r).shuffle(subs)
        folds = {s: i % 5 for i, s in enumerate(subs)}
        fold = np.array([folds[s] for s in gg])
        p = np.full(len(y), np.nan)
        for k in range(5):
            te = fold == k
            p[te] = rf().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, p))
    return np.mean(aucs), np.std(aucs)


def transfer(cols):
    ds = smc.dropna(subset=cols + ["label_apnea"]); dp = psg.dropna(subset=cols + ["label_apnea"])
    Xs = zsub(ds[cols].to_numpy(float), ds.Subject.to_numpy()); ys = ds.label_apnea.to_numpy(int)
    Xp = zsub(dp[cols].to_numpy(float), dp.Subject.to_numpy()); yp = dp.label_apnea.to_numpy(int)
    return roc_auc_score(ys, rf().fit(Xp, yp).predict_proba(Xs)[:, 1])


print("\n=== LAB (within-SMC, 5-fold x20) ===")
m0, s0 = lab_cv(CORE); m1, s1 = lab_cv(CORE + NEW)
print(f"  CORE      : {m0:.3f} ± {s0:.3f}")
print(f"  CORE+NEW  : {m1:.3f} ± {s1:.3f}   (Δ {m1-m0:+.3f})")

print("\n=== TRANSFER zero-shot PSG -> SMC ===")
t0 = transfer(CORE); t1 = transfer(CORE + NEW)
print(f"  CORE      : {t0:.3f}")
print(f"  CORE+NEW  : {t1:.3f}   (Δ {t1-t0:+.3f})")

print("\n=== AUC univariata feature NEW sul MATERASSO (lab) ===")
for c in NEW:
    d = smc[["label_apnea", c]].replace([np.inf, -np.inf], np.nan).dropna()
    a = roc_auc_score(d.label_apnea, d[c])
    print(f"  {c:18s} {max(a,1-a):.3f}")
