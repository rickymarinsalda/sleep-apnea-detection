#!/usr/bin/env python3
"""
Estrattore feature fisiologiche CONDIVISE lato materasso (SMC).  -- v2 --

Obiettivo: ridurre i segnali grezzi del materasso (matrice di pressione 8 Hz +
2 accelerometri 80 Hz) allo stesso "livello fisiologico" calcolabile anche dalla
PSG (cinghie di sforzo toraco-addominale + ECG). Su questo livello condiviso si
baserà il transfer learning PSG -> materasso.

MIGLIORIE v2 (allineate alla metodologia del paper, prepare_dataset_manual.py):
  1. usa i file con ETICHETTE MANUALI (69 apnee) in ../dataset_a_mano
  2. labeling del paper: finestre ambigue (majority "altro"/miste) SCARTATE
  3. drop righe tutte-zero + finestratura per-indice (finestre = quelle del paper)
  4. HR da BCG-matrice E da ACC, sceglie per finestra la sorgente a SNR maggiore

Output: shared_features_smc_30s.csv
"""
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ricky/Documents/phd/apnea_study"
OUTDIR = f"{ROOT}/transfer_psg_smc"
MAT_CSV = f"{ROOT}/dataset_a_mano/dataset_apnea_ricky_MAT_filt_indiciApneaManuale.csv"
ACC_CSV = f"{ROOT}/dataset_a_mano/dataset_apnea_ricky_ACC_indiciApneaManuale.csv"

FS_MAT, FS_ACC = 8.0, 80.0
WIN = 30.0
WIN_MAT = int(FS_MAT * WIN)            # 240 campioni/finestra
RESP_BAND = (0.1, 0.7)                 # 6-42 respiri/min
CARD_BAND = (0.7, 2.5)                 # 42-150 battiti/min
MIN_FRAC = 0.5                         # soglia overlap (come il paper)

ALL_CH = [f"ch{i}" for i in range(1, 41)]
UPPER_CH = [f"ch{i}" for i in range(1, 21)]    # torace (UL+UR)
LOWER_CH = [f"ch{i}" for i in range(21, 41)]   # addome/bacino (LL+LR)


def bandpass(x, band, fs, order=3):
    b, a = signal.butter(order, band, btype="band", fs=fs)
    return signal.filtfilt(b, a, x - np.nanmean(x))


def clean(x):
    return pd.Series(x).interpolate(limit_direction="both").fillna(0.0).to_numpy()


def peak_freq(x, fs, band):
    """Frequenza dominante (Hz) e SNR-in-banda (picco/mediana) via Welch."""
    if len(x) < fs * 4:
        return np.nan, np.nan
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * WIN)))
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or np.all(p[m] == 0):
        return np.nan, np.nan
    return f[m][np.argmax(p[m])], p[m].max() / (np.median(p[m]) + 1e-12)


def band_power_ratio(x, fs, band):
    f, p = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * WIN)))
    m = (f >= band[0]) & (f <= band[1])
    return np.trapezoid(p[m], f[m]) / (np.trapezoid(p, f) + 1e-12)


def bcg_envelope(mag, fs):
    """Inviluppo di energia battito (BCG) dalla magnitudo ACC."""
    b, a = signal.butter(3, [2.0, 12.0], btype="band", fs=fs)
    return np.abs(signal.hilbert(signal.filtfilt(b, a, mag - mag.mean())))


def hr_hrv(env, fs):
    """HR robusto (mediana IBI) + HRV con rigetto artefatti e pulizia NN.
    NB: HRV (sdnn/rmssd) resta poco affidabile sul materasso -> usare hr_conf."""
    thr = np.median(env) + 0.5 * (np.percentile(env, 75) - np.percentile(env, 25))
    pk, _ = signal.find_peaks(env, distance=int(0.33 * fs), height=thr)
    if len(pk) < 6:
        return np.nan, 0.0, np.nan, np.nan
    pk = pk[env[pk] < 3.0 * np.median(env[pk])]          # rigetto spike di movimento
    ibi = np.diff(pk) / fs
    plaus = ibi[(ibi > 0.4) & (ibi < 1.5)]
    if len(plaus) < 5:
        return np.nan, 0.0, np.nan, np.nan
    med = np.median(plaus)
    nn = ibi[(ibi > 0.4) & (ibi < 1.5) & (np.abs(ibi - med) < 0.30 * med)]
    if len(nn) < 5:
        return np.nan, 0.0, np.nan, np.nan
    hr = 60.0 / np.median(nn)
    conf = len(nn) / (len(env) / fs / np.median(nn))     # NN puliti / attesi
    return hr, conf, np.std(nn) * 1000, np.sqrt(np.mean(np.diff(nn) ** 2)) * 1000


def process_subject(sid, mat, acc):
    mat = mat.sort_values("Time").reset_index(drop=True)
    ch = np.apply_along_axis(clean, 0, mat[ALL_CH].to_numpy(float))

    # canali attivi (sotto il corpo) + segnali respiro: filtro sul continuo, poi taglio
    active = np.var(ch, axis=0) > np.percentile(np.var(ch, axis=0), 50)
    glob = ch[:, active].mean(1) if active.any() else ch.mean(1)
    up = ch[:, [i for i, c in enumerate(ALL_CH) if c in UPPER_CH]].mean(1)
    lo = ch[:, [i for i, c in enumerate(ALL_CH) if c in LOWER_CH]].mean(1)
    resp = bandpass(glob, RESP_BAND, FS_MAT)
    resp_up = bandpass(up, RESP_BAND, FS_MAT)
    resp_lo = bandpass(lo, RESP_BAND, FS_MAT)
    env = np.abs(signal.hilbert(resp))
    card_mat = bandpass(glob, CARD_BAND, FS_MAT)          # BCG dalla matrice

    status = mat["Status"].to_numpy()
    pos = mat["Position"].astype(str).to_numpy()
    t_mat = mat["Time"].to_numpy(float)

    acc = acc.sort_values("Time_acc1").reset_index(drop=True)
    a = np.apply_along_axis(clean, 0, acc[["X1", "Y1", "Z1", "X2", "Y2", "Z2"]].to_numpy(float))
    mag = np.sqrt((a[:, :3] ** 2).sum(1)) + np.sqrt((a[:, 3:] ** 2).sum(1))
    move = mag - np.mean(mag)
    env_bcg = bcg_envelope(mag, FS_ACC)               # per HR/HRV battito-battito
    t_acc = acc["Time_acc1"].to_numpy(float)

    rows, skipped = [], 0
    for w in range(len(mat) // WIN_MAT):
        sl = slice(w * WIN_MAT, (w + 1) * WIN_MAT)
        st = status[sl]
        frac_ap = np.mean(st == 3)
        frac_resp = np.mean(np.isin(st, [0, 1, 2]))
        maj = np.bincount(st).argmax()
        # --- labeling del paper: scarta ambigue ---
        if maj == 3 and frac_ap >= MIN_FRAC:
            label = 1
        elif maj in (0, 1, 2) and frac_resp >= MIN_FRAC:
            label = 0
        else:
            skipped += 1
            continue

        t0, t1 = t_mat[sl][0], t_mat[sl][-1]
        am = (t_acc >= t0) & (t_acc <= t1)

        rr, _ = peak_freq(resp[sl], FS_MAT, RESP_BAND)
        ud = np.corrcoef(resp_up[sl], resp_lo[sl])[0, 1]
        # HR/HRV battito-battito dal BCG-ACC
        if am.sum() > FS_ACC * 8:
            hr, hr_conf, sdnn, rmssd = hr_hrv(env_bcg[am], FS_ACC)
        else:
            hr, hr_conf, sdnn, rmssd = np.nan, 0.0, np.nan, np.nan

        pv, pc = np.unique(pos[sl][pos[sl] != "nan"], return_counts=True)
        rows.append(dict(
            Subject=sid, t_start=t0, t_end=t1, label_apnea=label,
            frac_apnea=frac_ap, majority_status=int(maj),
            position=pv[np.argmax(pc)] if len(pv) else "nan",
            resp_rate=rr * 60 if rr == rr else np.nan,
            resp_amp_mean=np.mean(env[sl]),
            resp_amp_std=np.std(env[sl]),
            resp_band_power=np.var(resp[sl]),
            resp_regularity=band_power_ratio(glob[sl], FS_MAT, RESP_BAND),
            updown_corr=ud,
            acc_move_std=np.std(move[am]) if am.sum() else np.nan,
            acc_move_max=np.max(np.abs(np.diff(move[am]))) if am.sum() > 1 else np.nan,
            hr_est=hr, hr_conf=hr_conf, hrv_sdnn=sdnn, hrv_rmssd=rmssd,
        ))
    df = pd.DataFrame(rows)
    for c in ["resp_amp_mean", "resp_amp_std", "resp_band_power", "acc_move_std"]:
        med = df[c].median()
        df[c + "_rel"] = df[c] / med if med and med == med else np.nan
    return df, skipped


def main():
    mat_all = pd.read_csv(MAT_CSV)
    acc_all = pd.read_csv(ACC_CSV)
    # drop righe tutte-zero (come il paper)
    z = (mat_all[ALL_CH] == 0).all(axis=1)
    mat_all = mat_all[~z].copy()

    out, tot_skip = [], 0
    for sid in sorted(mat_all.Subject.unique()):
        df, sk = process_subject(sid,
                                 mat_all[mat_all.Subject == sid],
                                 acc_all[acc_all.Subject == sid])
        out.append(df)
        tot_skip += sk
    feats = pd.concat(out, ignore_index=True)
    path = f"{OUTDIR}/shared_features_smc_30s.csv"
    feats.to_csv(path, index=False)

    n, na = len(feats), int(feats.label_apnea.sum())
    print(f"Finestre: {n} | apnea: {na} ({100*na/n:.1f}%) | scartate(ambigue): {tot_skip} | soggetti: {feats.Subject.nunique()}")
    print(f"HR valido (conf>0.7): {(feats.hr_conf > 0.7).mean()*100:.0f}% finestre\n")
    print(f"Salvato: {path}\n")

    skip = ("Subject", "t_start", "t_end", "label_apnea", "frac_apnea",
            "majority_status", "position")
    rank = []
    for c in [c for c in feats.columns if c not in skip]:
        d = feats[["label_apnea", c]].dropna()
        if d[c].nunique() < 3 or d.label_apnea.nunique() < 2:
            continue
        auc = roc_auc_score(d.label_apnea, d[c])
        rank.append((c, max(auc, 1 - auc),
                     feats.loc[feats.label_apnea == 0, c].mean(),
                     feats.loc[feats.label_apnea == 1, c].mean()))
    print(f"{'feature':22s} {'AUC':>6s}  {'non-apnea':>12s}  {'apnea':>12s}")
    print("-" * 58)
    for c, auc, m0, m1 in sorted(rank, key=lambda x: -x[1]):
        print(f"{c:22s} {auc:6.3f}  {m0:12.3f}  {m1:12.3f}")

    top = [c for c, *_ in sorted(rank, key=lambda x: -x[1])[:4]]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for i, c in enumerate(top):
        ax[i].boxplot([feats.loc[feats.label_apnea == 0, c].dropna(),
                       feats.loc[feats.label_apnea == 1, c].dropna()],
                      tick_labels=["non-apnea", "apnea"], showfliers=False)
        ax[i].set_title(c, fontsize=10)
    plt.suptitle("Separazione apnea vs non-apnea — feature fisiologiche condivise (SMC, label manuali)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/class_separation.png", dpi=110)
    print(f"\nFigura: {OUTDIR}/class_separation.png")


if __name__ == "__main__":
    main()
