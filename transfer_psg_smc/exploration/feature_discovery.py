#!/usr/bin/env python3
"""
IDEA #2 -- Feature discovery dalla PSG (apnea reale).

Calcola sullo sforzo respiratorio PSG (ribcage+abdo) descrittori PIU' RICCHI dei
6 core attuali, tutti calcolabili anche dal materasso (sono funzioni di una
waveform di effort), e li classifica per quanto separano l'apnea REALE.
Obiettivo: scoprire feature NUOVE da portare nel pipeline del materasso.

Idea chiave: la media sulla finestra "spalma" l'evento; servono descrittori della
RIDUZIONE SOSTENUTA (durata del tratto piatto, dip minimo, n. respiri...).

NB: legge le PSG (read-only) e le label gia' calcolate. Non tocca lo studio del paper.
"""
import glob, os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis
import pyedflib
from sklearn.metrics import roc_auc_score

P = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
FS = 8.0
WIN = 30.0
WN = int(FS * WIN)
RESP = (0.1, 0.7)

labels = pd.read_csv(f"{P}/shared_features_psg_30s.csv")[["Subject", "t_start", "label_apnea"]]


def bp(x, band):
    b, a = signal.butter(3, band, btype="band", fs=FS)
    return signal.filtfilt(b, a, x - np.nanmean(x))


def spectral_entropy(x):
    f, pxx = signal.welch(x, fs=FS, nperseg=min(len(x), WN))
    pxx = pxx / (pxx.sum() + 1e-12)
    return -np.sum(pxx * np.log(pxx + 1e-12))


def feats(seg, env, med_env):
    """Descrittori ricchi su una finestra di sforzo (seg=bandpassato, env=inviluppo)."""
    # respiri = picchi dell'inviluppo
    pk, _ = signal.find_peaks(seg, distance=int(1.2 * FS))
    amps = env[pk] if len(pk) else np.array([0.0])
    # tratto piatto piu' lungo (riduzione sostenuta sotto 30% del baseline soggetto)
    low = env < 0.30 * med_env
    runs = np.diff(np.where(np.concatenate(([low[0]], np.diff(low.astype(int)) != 0, [True])))[0])
    longest_flat = (runs[::2].max() if low[0] else (runs[1::2].max() if len(runs) > 1 else 0)) / FS
    return dict(
        env_min_ratio=np.min(env) / (med_env + 1e-9),          # dip piu' profondo
        longest_flat_s=longest_flat,                            # durata riduzione sostenuta
        n_breaths=len(pk),                                      # respiri nella finestra
        breath_amp_cv=np.std(amps) / (np.mean(amps) + 1e-9),    # variabilita' ampiezza respiri
        effort_kurtosis=kurtosis(seg),                          # forma (flow limitation)
        effort_entropy=spectral_entropy(seg),                   # irregolarita' spettrale
    )


rows = []
for rec in sorted(glob.glob(f"{P}/ucddb/*.rec")):
    sid = os.path.basename(rec).split(".")[0]
    f = pyedflib.EdfReader(rec)
    idx = {f.getLabel(i).lower(): i for i in range(f.signals_in_file)}
    glob_sig = f.readSignal(idx["ribcage"]) + f.readSignal(idx["abdo"])
    f._close()
    resp = bp(glob_sig, RESP)
    env = np.abs(signal.hilbert(resp))
    med_env = np.median(env)
    for w in range(len(resp) // WN):
        sl = slice(w * WN, (w + 1) * WN)
        r = feats(resp[sl], env[sl], med_env)
        r.update(Subject=sid, t_start=w * WIN)
        rows.append(r)

df = pd.merge(pd.DataFrame(rows), labels, on=["Subject", "t_start"], how="inner")
NEW = ["env_min_ratio", "longest_flat_s", "n_breaths", "breath_amp_cv",
       "effort_kurtosis", "effort_entropy"]
print(f"Finestre: {len(df)} | eventi: {int(df.label_apnea.sum())}\n")
print(f"{'feature NUOVA':18s} {'AUC':>6s}  {'non-evento':>11s}  {'evento':>10s}")
print("-" * 50)
rank = []
for c in NEW:
    d = df[["label_apnea", c]].replace([np.inf, -np.inf], np.nan).dropna()
    auc = roc_auc_score(d.label_apnea, d[c])
    rank.append((c, max(auc, 1 - auc),
                 df.loc[df.label_apnea == 0, c].median(), df.loc[df.label_apnea == 1, c].median()))
for c, auc, m0, m1 in sorted(rank, key=lambda x: -x[1]):
    print(f"{c:18s} {auc:6.3f}  {m0:11.3f}  {m1:10.3f}")
print("\n(confronto: migliori core attuali su PSG ~ resp_amp 0.71, resp_regularity 0.67)")
df.to_csv(f"{P}/exploration/psg_rich_features.csv", index=False)
