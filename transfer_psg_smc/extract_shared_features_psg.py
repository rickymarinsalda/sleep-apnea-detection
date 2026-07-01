#!/usr/bin/env python3
"""
Estrattore feature fisiologiche CONDIVISE lato PSG (UCDDB).

Specchio 1:1 di extract_shared_features_smc.py: calcola le STESSE colonne, ma dai
canali PSG. Mappatura del livello condiviso:
  - sforzo respiratorio  <- ribcage + abdo (qui: Sum / ribcage+abdo)  [8 Hz, come matrice]
  - paradosso toraco-addom. <- corr(ribcage, abdo)   [paradosso VERO dell'OSA]
  - HR / HRV             <- ECG  [128 Hz, battiti affidabili]
  - posizione            <- BodyPos
  - movimento (proxy)    <- energia alta-frequenza dello sforzo  [NB: proxy, vedi DIARIO]
Label: finestra = evento se overlap >=50% con apnea/ipopnea annotata (_respevt.txt).

Uso: processa tutti i .rec presenti in ucddb/ -> shared_features_psg_30s.csv
"""
import glob, os, re, datetime
import numpy as np
import pandas as pd
from scipy import signal
import pyedflib
from sklearn.metrics import roc_auc_score

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
UCDDB = f"{ROOT}/ucddb"
FS_R = 8.0            # sforzo respiratorio (come matrice)
WIN = 30.0
WIN_R = int(FS_R * WIN)
RESP_BAND = (0.1, 0.7)
MIN_OVL = 0.5


# ----- funzioni feature (identiche al lato materasso) -----
def bandpass(x, band, fs, order=3):
    b, a = signal.butter(order, band, btype="band", fs=fs)
    return signal.filtfilt(b, a, x - np.nanmean(x))


def peak_freq(x, fs, band):
    if len(x) < fs * 4:
        return np.nan
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * WIN)))
    m = (f >= band[0]) & (f <= band[1])
    return f[m][np.argmax(p[m])] if m.any() and np.any(p[m]) else np.nan


def band_power_ratio(x, fs, band):
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * WIN)))
    m = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(p[m], f[m]) / (np.trapezoid(p, f) + 1e-12)


def ecg_hr_hrv(ecg, fs):
    """HR/HRV da ECG: rilevazione R-peak (qui affidabile) + pulizia NN."""
    x = bandpass(ecg, (5.0, 18.0), fs)
    x = x ** 2
    pk, _ = signal.find_peaks(x, distance=int(0.33 * fs), height=np.median(x) + 2 * np.std(x))
    if len(pk) < 6:
        return np.nan, 0.0, np.nan, np.nan
    ibi = np.diff(pk) / fs
    plaus = ibi[(ibi > 0.4) & (ibi < 1.5)]
    if len(plaus) < 5:
        return np.nan, 0.0, np.nan, np.nan
    med = np.median(plaus)
    nn = ibi[(ibi > 0.4) & (ibi < 1.5) & (np.abs(ibi - med) < 0.30 * med)]
    if len(nn) < 5:
        return np.nan, 0.0, np.nan, np.nan
    return (60.0 / np.median(nn), len(nn) / (len(x) / fs / np.median(nn)),
            np.std(nn) * 1000, np.sqrt(np.mean(np.diff(nn) ** 2)) * 1000)


# ----- parsing eventi respiratori -----
def parse_events(path, start_dt):
    """Ritorna lista (onset_s, dur_s) degli eventi respiratori dal _respevt.txt."""
    start_s = start_dt.hour * 3600 + start_dt.minute * 60 + start_dt.second
    ev = []
    for line in open(path, errors="ignore"):
        m = re.match(r"\s*(\d{2}):(\d{2}):(\d{2})\s+(\S+)\s+(.*)", line)
        if not m:
            continue
        h, mi, s, typ, rest = m.groups()
        if not re.match(r"(APNEA|HYP)", typ, re.I):
            continue
        t = int(h) * 3600 + int(mi) * 60 + int(s)
        onset = t - start_s + (86400 if t < start_s else 0)
        # durata = primo intero plausibile (5-120 s) nei campi seguenti
        dur = next((int(tok) for tok in rest.split() if tok.isdigit() and 3 <= int(tok) <= 120), 10)
        ev.append((onset, dur))
    return ev


def process_rec(rec):
    f = pyedflib.EdfReader(rec)
    labels = [f.getLabel(i) for i in range(f.signals_in_file)]
    idx = {l.lower(): i for i, l in enumerate(labels)}

    def sig(name):
        return f.readSignal(idx[name]), f.getSampleFrequency(idx[name])

    rib, _ = sig("ribcage")
    abd, _ = sig("abdo")
    ecg, fs_e = sig("ecg")
    pos = f.readSignal(idx["bodypos"]) if "bodypos" in idx else np.zeros_like(rib)
    start_dt = f.getStartdatetime()
    dur_total = f.getFileDuration()
    f._close()

    sid = os.path.basename(rec).split(".")[0]
    events = parse_events(rec.replace(".rec", "_respevt.txt"), start_dt)

    # segnali respiratori (come matrice: glob=somma, up=ribcage, lo=abdo)
    glob = rib + abd
    resp = bandpass(glob, RESP_BAND, FS_R)
    resp_up = bandpass(rib, RESP_BAND, FS_R)
    resp_lo = bandpass(abd, RESP_BAND, FS_R)
    env = np.abs(signal.hilbert(resp))
    move_proxy = bandpass(glob, (1.0, 3.5), FS_R)        # proxy movimento (alta freq sforzo)

    rows = []
    n_win = int(dur_total // WIN)
    for w in range(n_win):
        t0, t1 = w * WIN, (w + 1) * WIN
        sl = slice(int(t0 * FS_R), int(t1 * FS_R))
        if sl.stop > len(resp):
            break
        # label: overlap con eventi
        ovl = sum(max(0, min(t1, e0 + ed) - max(t0, e0)) for e0, ed in events)
        label = int(ovl / WIN >= MIN_OVL)

        es = slice(int(t0 * fs_e), int(t1 * fs_e))
        hr, hrc, sdnn, rmssd = ecg_hr_hrv(ecg[es], fs_e)
        pv = pos[sl]
        rows.append(dict(
            Subject=sid, t_start=t0, label_apnea=label,
            position=int(np.median(pv)) if len(pv) else -1,
            resp_rate=peak_freq(resp[sl], FS_R, RESP_BAND) * 60,
            resp_amp_mean=np.mean(env[sl]),
            resp_amp_std=np.std(env[sl]),
            resp_band_power=np.var(resp[sl]),
            resp_regularity=band_power_ratio(glob[sl], FS_R, RESP_BAND),
            updown_corr=np.corrcoef(resp_up[sl], resp_lo[sl])[0, 1],
            acc_move_std=np.std(move_proxy[sl]),
            hr_est=hr, hr_conf=hrc, hrv_sdnn=sdnn, hrv_rmssd=rmssd,
        ))
    df = pd.DataFrame(rows)
    for c in ["resp_amp_mean", "resp_amp_std", "resp_band_power", "acc_move_std"]:
        m = df[c].median()
        df[c + "_rel"] = df[c] / m if m else np.nan
    return df


def main():
    recs = sorted(glob.glob(f"{UCDDB}/*.rec"))
    print(f"Registrazioni trovate: {len(recs)}")
    out = [process_rec(r) for r in recs]
    feats = pd.concat(out, ignore_index=True)
    path = f"{ROOT}/shared_features_psg_30s.csv"
    feats.to_csv(path, index=False)
    n, na = len(feats), int(feats.label_apnea.sum())
    print(f"Finestre: {n} | evento: {na} ({100*na/n:.1f}%) | soggetti: {feats.Subject.nunique()}")
    print(f"HR valido (conf>0.7): {(feats.hr_conf>0.7).mean()*100:.0f}%  | salvato: {path}\n")

    skip = ("Subject", "t_start", "label_apnea", "position")
    print(f"{'feature':22s} {'AUC':>6s}  {'non-event':>12s}  {'event':>12s}")
    print("-" * 58)
    rank = []
    for c in [c for c in feats.columns if c not in skip]:
        d = feats[["label_apnea", c]].dropna()
        if d[c].nunique() < 3 or d.label_apnea.nunique() < 2:
            continue
        auc = roc_auc_score(d.label_apnea, d[c])
        rank.append((c, max(auc, 1 - auc),
                     feats.loc[feats.label_apnea == 0, c].mean(),
                     feats.loc[feats.label_apnea == 1, c].mean()))
    for c, auc, m0, m1 in sorted(rank, key=lambda x: -x[1]):
        print(f"{c:22s} {auc:6.3f}  {m0:12.3f}  {m1:12.3f}")


if __name__ == "__main__":
    main()
