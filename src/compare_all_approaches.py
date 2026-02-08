#!/usr/bin/env python3
"""
Confronto tra tutti gli approcci di validazione:

1. STANDARD (originale): K scelto su tutto il dataset (data leakage)
2. NESTED CV (F1): Tuning onesto ma conservativo
3. NESTED CV (Recall-opt): Tuning onesto ottimizzato per recall

Questo script esegue tutti e tre e produce un report comparativo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
    fbeta_score
)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
RESULTS_DIR = BASE_DIR / "results_comparison"
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_NAME = "dataset_apnea_windows_30s_features_zones_mat_acc_CORRECTED.csv"
SEARCH_PATHS = [
    BASE_DIR / "preprocessing_output" / DATASET_NAME,
    BASE_DIR / ".." / "data" / DATASET_NAME,
    Path("/Users/ricky/Documents/phd/apnea_study") / DATASET_NAME,
]

def find_dataset():
    for path in SEARCH_PATHS:
        if path.exists():
            return str(path)
    return None

N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3
WINDOW_SIZE = 3

def add_temporal_features(df, window_size=3):
    base_features = [
        'global_mean', 'global_std', 'global_max',
        'zone_UL_mean_mean', 'zone_UR_mean_mean',
        'zone_LL_mean_mean', 'zone_LR_mean_mean',
        'zone_UL_diff_std_mean', 'zone_UR_diff_std_mean',
        'zone_LL_diff_std_mean', 'zone_LR_diff_std_mean',
        'acc_global_mean', 'acc_global_std',
    ]
    base_features = [f for f in base_features if f in df.columns]
    temporal_features = pd.DataFrame(index=df.index)

    for feat in base_features:
        temporal_features[f'delta_{feat}'] = df.groupby('Subject')[feat].diff().fillna(0)
        temporal_features[f'roll{window_size}_mean_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )
        temporal_features[f'roll{window_size}_std_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        ).fillna(0)
        temporal_features[f'trend_{feat}'] = (
            df[feat] - temporal_features[f'roll{window_size}_mean_{feat}']
        )

    return temporal_features


def evaluate_fold(X_train, y_train, X_test, y_test, k, threshold):
    """Train and evaluate a single fold with given hyperparameters."""
    # Feature selection
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Oversample
    pos_idx = y_train == 1
    neg_idx = y_train == 0
    X_pos = X_train_sel[pos_idx]
    X_neg = X_train_sel[neg_idx]
    y_pos = y_train[pos_idx]
    y_neg = y_train[neg_idx]

    X_pos_res, y_pos_res = resample(X_pos, y_pos, n_samples=len(X_neg),
                                     random_state=RANDOM_STATE, replace=True)
    X_train_bal = np.vstack([X_neg, X_pos_res])
    y_train_bal = np.hstack([y_neg, y_pos_res])

    # Train
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
    }

    if len(np.unique(y_test)) > 1:
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['tp'] = tp
            metrics['fp'] = fp
            metrics['fn'] = fn
            metrics['tn'] = tn

    return metrics


def inner_cv_tune(X_dev, y_dev, subjects_dev, k_values, threshold_values, optimize_for='f1'):
    """Inner CV for hyperparameter tuning."""
    best_score = -np.inf
    best_k = 60
    best_threshold = 0.30

    inner_cv = GroupKFold(n_splits=4)

    for k in k_values:
        for threshold in threshold_values:
            scores = []

            for train_idx, val_idx in inner_cv.split(X_dev, y_dev, groups=subjects_dev):
                X_tr = X_dev[train_idx]
                X_val = X_dev[val_idx]
                y_tr = y_dev[train_idx]
                y_val = y_dev[val_idx]

                if y_tr.sum() == 0 or y_val.sum() == 0:
                    continue

                # Feature selection
                selector = SelectKBest(f_classif, k=min(k, X_tr.shape[1]))
                X_tr_sel = selector.fit_transform(X_tr, y_tr)
                X_val_sel = selector.transform(X_val)

                # Oversample
                pos_idx = y_tr == 1
                neg_idx = y_tr == 0
                X_pos = X_tr_sel[pos_idx]
                X_neg = X_tr_sel[neg_idx]
                y_pos = y_tr[pos_idx]
                y_neg = y_tr[neg_idx]

                X_pos_res, y_pos_res = resample(X_pos, y_pos, n_samples=len(X_neg),
                                                 random_state=RANDOM_STATE, replace=True)
                X_tr_bal = np.vstack([X_neg, X_pos_res])
                y_tr_bal = np.hstack([y_neg, y_pos_res])

                # Train
                model = RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
                model.fit(X_tr_bal, y_tr_bal)

                # Evaluate
                y_pred_proba = model.predict_proba(X_val_sel)[:, 1]
                y_pred = (y_pred_proba >= threshold).astype(int)

                if optimize_for == 'f1':
                    score = f1_score(y_val, y_pred, zero_division=0)
                elif optimize_for == 'fbeta':
                    score = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
                elif optimize_for == 'recall':
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    if prec >= 0.15:  # Min precision constraint
                        score = recall_score(y_val, y_pred, zero_division=0)
                    else:
                        score = -1  # Penalize
                else:
                    score = f1_score(y_val, y_pred, zero_division=0)

                scores.append(score)

            if len(scores) > 0:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_k = k
                    best_threshold = threshold

    return best_k, best_threshold


def run_approach_1_standard(X_full, y_full, subjects_full, k=60, threshold=0.30):
    """Approach 1: Fixed K and threshold (simulates data leakage)."""
    print("\n" + "="*70)
    print("APPROACH 1: STANDARD (K=60, threshold=0.30 fixed)")
    print("="*70)

    outer_cv = GroupKFold(n_splits=5)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups=subjects_full), 1):
        X_train = X_full[train_idx]
        X_test = X_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        metrics = evaluate_fold(X_train, y_train, X_test, y_test, k, threshold)
        metrics['fold'] = fold_idx
        results.append(metrics)
        print(f"  Fold {fold_idx}: ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}")

    return pd.DataFrame(results)


def run_approach_2_nested_f1(X_full, y_full, subjects_full):
    """Approach 2: Nested CV optimizing for F1."""
    print("\n" + "="*70)
    print("APPROACH 2: NESTED CV (optimize F1)")
    print("="*70)

    k_values = [40, 50, 60, 70, 80]
    threshold_values = [0.25, 0.30, 0.35, 0.40]

    outer_cv = GroupKFold(n_splits=5)
    results = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups=subjects_full), 1):
        X_dev = X_full[dev_idx]
        X_test = X_full[test_idx]
        y_dev = y_full[dev_idx]
        y_test = y_full[test_idx]
        subjects_dev = subjects_full[dev_idx]

        if y_dev.sum() == 0 or y_test.sum() == 0:
            continue

        # Inner CV tuning
        best_k, best_threshold = inner_cv_tune(X_dev, y_dev, subjects_dev,
                                                k_values, threshold_values, optimize_for='f1')

        # Evaluate on test
        metrics = evaluate_fold(X_dev, y_dev, X_test, y_test, best_k, best_threshold)
        metrics['fold'] = fold_idx
        metrics['best_k'] = best_k
        metrics['best_threshold'] = best_threshold
        results.append(metrics)
        print(f"  Fold {fold_idx}: K={best_k}, thr={best_threshold}, ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}")

    return pd.DataFrame(results)


def run_approach_3_nested_recall(X_full, y_full, subjects_full):
    """Approach 3: Nested CV optimizing for Recall (with F-beta)."""
    print("\n" + "="*70)
    print("APPROACH 3: NESTED CV (optimize F-beta, favor recall)")
    print("="*70)

    k_values = [30, 40, 50, 60, 70, 80, 90]
    threshold_values = [0.15, 0.20, 0.25, 0.30, 0.35]

    outer_cv = GroupKFold(n_splits=5)
    results = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups=subjects_full), 1):
        X_dev = X_full[dev_idx]
        X_test = X_full[test_idx]
        y_dev = y_full[dev_idx]
        y_test = y_full[test_idx]
        subjects_dev = subjects_full[dev_idx]

        if y_dev.sum() == 0 or y_test.sum() == 0:
            continue

        # Inner CV tuning
        best_k, best_threshold = inner_cv_tune(X_dev, y_dev, subjects_dev,
                                                k_values, threshold_values, optimize_for='fbeta')

        # Evaluate on test
        metrics = evaluate_fold(X_dev, y_dev, X_test, y_test, best_k, best_threshold)
        metrics['fold'] = fold_idx
        metrics['best_k'] = best_k
        metrics['best_threshold'] = best_threshold
        results.append(metrics)
        print(f"  Fold {fold_idx}: K={best_k}, thr={best_threshold}, ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}")

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print(" " * 20 + "CONFRONTO TRA APPROCCI DI VALIDAZIONE")
    print("=" * 80)

    # Load dataset
    dataset_path = find_dataset()
    if dataset_path is None:
        print("ERROR: Dataset not found!")
        return

    print(f"\nLoading: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Samples: {len(df)}, Subjects: {df['Subject'].nunique()}")
    print(f"Apnea: {df['label_apnea'].sum()} ({100*df['label_apnea'].sum()/len(df):.1f}%)")

    # Prepare features
    meta_cols = ['Subject', 'start_time', 'end_time', 'Position_mode',
                 'majority_status', 'label_apnea', 'frac_apnea',
                 'frac_altro', 'frac_respiro']

    baseline_features = [c for c in df.columns if c not in meta_cols]
    temporal_feats = add_temporal_features(df, window_size=WINDOW_SIZE)
    df_full = pd.concat([df[baseline_features], temporal_feats], axis=1)

    X_full = df_full.values
    y_full = df['label_apnea'].values
    subjects_full = df['Subject'].values

    # Run all approaches
    df_standard = run_approach_1_standard(X_full, y_full, subjects_full)
    df_nested_f1 = run_approach_2_nested_f1(X_full, y_full, subjects_full)
    df_nested_recall = run_approach_3_nested_recall(X_full, y_full, subjects_full)

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 80)
    print(" " * 25 + "RISULTATI COMPARATIVI")
    print("=" * 80)

    comparison = []

    for name, df_res in [
        ("1. Standard (K=60, thr=0.30)", df_standard),
        ("2. Nested CV (F1)", df_nested_f1),
        ("3. Nested CV (Recall-opt)", df_nested_recall)
    ]:
        row = {
            'Approach': name,
            'ROC-AUC': f"{df_res['roc_auc'].mean():.4f} ± {df_res['roc_auc'].std():.4f}",
            'F1': f"{df_res['f1'].mean():.4f} ± {df_res['f1'].std():.4f}",
            'Precision': f"{df_res['precision'].mean():.4f} ± {df_res['precision'].std():.4f}",
            'Recall': f"{df_res['recall'].mean():.4f} ± {df_res['recall'].std():.4f}",
        }
        comparison.append(row)

    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))

    # ========================================================================
    # DETAILED ANALYSIS
    # ========================================================================
    print("\n\n" + "=" * 80)
    print(" " * 28 + "ANALISI DETTAGLIATA")
    print("=" * 80)

    recall_standard = df_standard['recall'].mean()
    recall_nested_f1 = df_nested_f1['recall'].mean()
    recall_nested_recall = df_nested_recall['recall'].mean()

    print(f"""
RECALL COMPARISON:
  Standard (biased):        {recall_standard:.4f}
  Nested CV (F1):           {recall_nested_f1:.4f}  (diff: {recall_nested_f1 - recall_standard:+.4f})
  Nested CV (Recall-opt):   {recall_nested_recall:.4f}  (diff: {recall_nested_recall - recall_standard:+.4f})

INTERPRETAZIONE:
""")

    if recall_standard - recall_nested_f1 > 0.1:
        print("  - La differenza Standard vs Nested F1 conferma DATA LEAKAGE")
        print(f"    (bias stimato: ~{100*(recall_standard - recall_nested_f1)/recall_standard:.1f}%)")
    else:
        print("  - La differenza è contenuta, il data leakage era limitato")

    improvement = recall_nested_recall - recall_nested_f1
    if improvement > 0:
        print(f"\n  - L'ottimizzazione per recall ha recuperato {improvement:.4f} punti")
        print(f"    (+{100*improvement/recall_nested_f1:.1f}% rispetto a Nested F1)")

    # Save results
    df_standard.to_csv(RESULTS_DIR / 'results_standard.csv', index=False)
    df_nested_f1.to_csv(RESULTS_DIR / 'results_nested_f1.csv', index=False)
    df_nested_recall.to_csv(RESULTS_DIR / 'results_nested_recall.csv', index=False)
    df_comparison.to_csv(RESULTS_DIR / 'comparison_summary.csv', index=False)

    print(f"\n\n{'='*80}")
    print("RACCOMANDAZIONI PER IL PAPER:")
    print("="*80)
    print("""
1. OPZIONE CONSERVATIVA (più onesta scientificamente):
   - Riporta i risultati della Nested CV (F1 o Recall-opt)
   - Spiega che i risultati originali avevano un bias da hyperparameter tuning

2. OPZIONE PRAGMATICA:
   - Riporta entrambi i risultati (originale e nested CV)
   - Spiega la differenza come "optimism due to hyperparameter selection"
   - Usa la nested CV come "lower bound" delle performance attese

3. SE DEVI SCEGLIERE UN SOLO RISULTATO:
   - Usa "Nested CV (Recall-opt)" se la recall è prioritaria
   - Usa "Nested CV (F1)" se vuoi bilanciare precision e recall
""")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
