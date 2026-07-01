#!/usr/bin/env python3
"""
Baseline sulle feature fisiologiche CONDIVISE + cross-validation robusta.

Risponde a due obiettivi:
 1. SANITY della "nostra idea": un classificatore leggero (RF) sulle sole feature
    condivise separa l'apnea? (prerequisito al transfer da PSG)
 2. CV ROBUSTA (consiglio reviewer): invece di un singolo 5-fold, valutiamo su
    MOLTE partizioni per ottenere una stima CONVERGENTE e un intervallo di
    confidenza -> risultati solidi, non frutto di uno split fortunato.

Schemi di CV (tutti subject-wise, nessun leakage tra soggetti):
 - 5-fold ripetuto 100 volte con partizioni di soggetti diverse (-> convergenza, CI)
 - Leave-One-Subject-Out (23 fold): la valutazione esaustiva sul singolo soggetto

Confronto: feature condivise STATICHE vs + TEMPORALI (delta/rolling, come il paper).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, balanced_accuracy_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/ricky/Documents/phd/apnea_study/transfer_psg_smc"
CSV = f"{ROOT}/shared_features_smc_30s.csv"
K_ROLL = 7          # finestra rolling temporale (come il paper)
N_REPEATS = 100     # ripetizioni del 5-fold per la convergenza
N_FOLDS = 5
THR = 0.30          # soglia per metriche orientate a screening
RNG = np.random.default_rng(0)

META = ["Subject", "t_start", "t_end", "label_apnea", "frac_apnea",
        "majority_status", "hr_source"]


def build_features(df):
    df = df.sort_values(["Subject", "t_start"]).reset_index(drop=True)
    num = [c for c in df.columns if c not in META + ["position"]]
    # one-hot posizione
    pos = pd.get_dummies(df["position"], prefix="pos")
    X_static = pd.concat([df[num].fillna(df[num].median()), pos], axis=1)

    # --- augmentation temporale (per soggetto, ordinata nel tempo) ---
    tmp = {}
    g = df.groupby("Subject")
    for f in num:
        tmp[f"delta_{f}"] = g[f].diff().fillna(0)
        rm = g[f].transform(lambda x: x.rolling(K_ROLL, min_periods=1).mean())
        rs = g[f].transform(lambda x: x.rolling(K_ROLL, min_periods=1).std()).fillna(0)
        tmp[f"rollmean_{f}"] = rm
        tmp[f"rollstd_{f}"] = rs
        tmp[f"trend_{f}"] = df[f].fillna(df[f].median()) - rm
    X_temporal = pd.concat([X_static, pd.DataFrame(tmp, index=df.index)], axis=1)
    return X_static.to_numpy(float), X_temporal.fillna(0).to_numpy(float), \
        df["label_apnea"].to_numpy(int), df["Subject"].to_numpy()


def rf():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=0)


def metrics(y, p):
    yhat = (p >= THR).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
        recall=recall_score(y, yhat, zero_division=0),
        precision=precision_score(y, yhat, zero_division=0),
        bal_acc=balanced_accuracy_score(y, yhat),
        f1=f1_score(y, yhat, zero_division=0))


def oof_predict(X, y, groups, folds):
    """folds: lista di array di soggetti-test. Ritorna proba OOF su tutti i campioni."""
    p = np.full(len(y), np.nan)
    for test_subj in folds:
        te = np.isin(groups, test_subj)
        tr = ~te
        m = rf().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def repeated_kfold(X, y, groups, n_repeats=N_REPEATS, k=N_FOLDS):
    subs = np.unique(groups)
    aucs, prs = [], []
    for r in range(n_repeats):
        s = subs.copy(); np.random.default_rng(r).shuffle(s)
        folds = np.array_split(s, k)
        p = oof_predict(X, y, groups, folds)
        aucs.append(roc_auc_score(y, p)); prs.append(average_precision_score(y, p))
    return np.array(aucs), np.array(prs)


def loso(X, y, groups):
    logo = LeaveOneGroupOut()
    folds = [groups[te][:1] for _, te in logo.split(X, y, groups)]
    p = oof_predict(X, y, groups, folds)
    # recall per soggetto (solo chi ha apnee)
    per = []
    for s in np.unique(groups):
        m = groups == s
        if y[m].sum() > 0:
            per.append(recall_score(y[m], (p[m] >= THR).astype(int), zero_division=0))
    return metrics(y, p), np.array(per)


def main():
    df = pd.read_csv(CSV)
    Xs, Xt, y, g = build_features(df)
    print(f"Campioni: {len(y)} | apnea: {y.sum()} | soggetti: {len(np.unique(g))}")
    print(f"Feature: statiche={Xs.shape[1]}  temporali={Xt.shape[1]}\n")

    rows = []
    for name, X in [("shared-static", Xs), ("shared-temporal", Xt)]:
        a, pr = repeated_kfold(X, y, g)
        m_loso, per = loso(X, y, g)
        rows.append((name, a, pr, m_loso, per))
        ci = np.percentile(a, [2.5, 97.5])
        print(f"== {name} ==")
        print(f"  5-fold x{N_REPEATS}: ROC-AUC {a.mean():.3f} ± {a.std():.3f} "
              f"(95% CI [{ci[0]:.3f}, {ci[1]:.3f}])  PR-AUC {pr.mean():.3f} ± {pr.std():.3f}")
        print(f"  LOSO       : ROC-AUC {m_loso['roc_auc']:.3f}  PR-AUC {m_loso['pr_auc']:.3f}  "
              f"recall {m_loso['recall']:.3f}  prec {m_loso['precision']:.3f}  "
              f"bal_acc {m_loso['bal_acc']:.3f}  F1 {m_loso['f1']:.3f}")
        print(f"  LOSO recall/soggetto: {per.mean():.2f} ± {per.std():.2f} (min {per.min():.2f})\n")

    # ---- figura convergenza (cumulative mean del 5-fold ripetuto) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, a, pr, *_ in rows:
        cummean = np.cumsum(a) / np.arange(1, len(a) + 1)
        ax[0].plot(range(1, len(a) + 1), cummean, label=name)
    ax[0].set_xlabel("n. ripetizioni 5-fold"); ax[0].set_ylabel("ROC-AUC (media cumulata)")
    ax[0].set_title("Convergenza della stima"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].boxplot([r[1] for r in rows], tick_labels=[r[0] for r in rows], showfliers=False)
    ax[1].set_ylabel("ROC-AUC per ripetizione"); ax[1].set_title(f"Distribuzione su {N_REPEATS} partizioni")
    ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{ROOT}/cv_convergence.png", dpi=110)
    print(f"Figura: {ROOT}/cv_convergence.png")

    pd.DataFrame([{
        "model": n, "kfold_roc_mean": a.mean(), "kfold_roc_std": a.std(),
        "kfold_ci_lo": np.percentile(a, 2.5), "kfold_ci_hi": np.percentile(a, 97.5),
        "kfold_pr_mean": pr.mean(), "loso_roc": m["roc_auc"], "loso_pr": m["pr_auc"],
        "loso_recall": m["recall"], "loso_bal_acc": m["bal_acc"], "loso_f1": m["f1"],
    } for n, a, pr, m, _ in rows]).to_csv(f"{ROOT}/cv_results.csv", index=False)
    print(f"Risultati: {ROOT}/cv_results.csv")


if __name__ == "__main__":
    main()
