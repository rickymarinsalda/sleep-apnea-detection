#!/usr/bin/env python3
"""
NESTED CROSS-VALIDATION per Manual Labels Dataset

Confronta:
1. STANDARD CV (come hyperparameter_tuning.py): K scelto su tutto il dataset
2. NESTED CV: K scelto nell'inner loop per ogni fold esterno

Questo script mostra la differenza reale causata dal data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, fbeta_score
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
RESULTS_DIR = BASE_DIR / "results_nested_cv"
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset
DATASET_NAME = "dataset_windows_30s_features_zones_MANUAL.csv"
SEARCH_PATHS = [
    BASE_DIR / "preprocessing_output" / DATASET_NAME,
    Path("/Users/ricky/Documents/phd/apnea_study/analysis_manual_labels/preprocessing_output") / DATASET_NAME,
    BASE_DIR / DATASET_NAME,
]

DATASET_PATH = None
for p in SEARCH_PATHS:
    if p.exists():
        DATASET_PATH = p
        break

# Model params
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3

# Grid per tuning
K_VALUES = [40, 50, 60, 70, 80]
WINDOW_SIZES = [3, 5, 7]
THRESHOLDS = [0.20, 0.25, 0.30, 0.35]

print("=" * 80)
print(" " * 15 + "NESTED CV vs STANDARD CV COMPARISON")
print(" " * 15 + "(Manual Labels Dataset)")
print("=" * 80)


def add_temporal_features(df, window_size=3):
    """Generate temporal features."""
    base_feats = [
        'global_mean', 'global_std', 'global_max',
        'zone_UL_mean_mean', 'zone_UR_mean_mean',
        'zone_LL_mean_mean', 'zone_LR_mean_mean',
        'zone_UL_diff_std_mean', 'zone_UR_diff_std_mean',
        'zone_LL_diff_std_mean', 'zone_LR_diff_std_mean',
        'acc_global_mean', 'acc_global_std',
    ]
    base_feats = [f for f in base_feats if f in df.columns]
    temporal_features = pd.DataFrame(index=df.index)

    for feat in base_feats:
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


def evaluate_config(X_train, y_train, X_test, y_test, k, threshold):
    """Train and evaluate with given K and threshold."""
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

    if len(y_pos) == 0:
        return None

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

    # Predict
    y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def inner_cv_tune(X_dev, y_dev, subjects_dev, temporal_cache, baseline_features,
                  optimize_for='f1'):
    """
    Inner CV per trovare i migliori iperparametri.

    Args:
        optimize_for: 'f1', 'fbeta' (beta=2), o 'recall'
    """
    best_score = -np.inf
    best_config = {'k': 60, 'window': 3, 'threshold': 0.30}

    inner_cv = GroupKFold(n_splits=4)

    for k, ws, thr in product(K_VALUES, WINDOW_SIZES, THRESHOLDS):
        scores = []

        # Get temporal features for this window size
        temporal_df = temporal_cache[ws]

        for train_idx, val_idx in inner_cv.split(X_dev, y_dev, groups=subjects_dev):
            # Prepare data
            X_train_base = X_dev[baseline_features].iloc[train_idx]
            X_val_base = X_dev[baseline_features].iloc[val_idx]
            temp_train = temporal_df.iloc[train_idx]
            temp_val = temporal_df.iloc[val_idx]

            X_train = pd.concat([X_train_base.reset_index(drop=True),
                                  temp_train.reset_index(drop=True)], axis=1).values
            X_val = pd.concat([X_val_base.reset_index(drop=True),
                                temp_val.reset_index(drop=True)], axis=1).values

            y_train = y_dev.iloc[train_idx].values
            y_val = y_dev.iloc[val_idx].values

            if y_train.sum() == 0 or y_val.sum() == 0:
                continue

            result = evaluate_config(X_train, y_train, X_val, y_val, k, thr)
            if result is None:
                continue

            if optimize_for == 'f1':
                score = result['f1']
            elif optimize_for == 'fbeta':
                score = fbeta_score(y_val, result['y_pred'], beta=2, zero_division=0)
            elif optimize_for == 'recall':
                # Only accept if precision >= 0.15
                if result['precision'] >= 0.15:
                    score = result['recall']
                else:
                    score = -1
            else:
                score = result['f1']

            scores.append(score)

        if len(scores) > 0:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_config = {'k': k, 'window': ws, 'threshold': thr}

    return best_config


def run_standard_cv(df, baseline_features, temporal_cache):
    """
    STANDARD CV: Come hyperparameter_tuning.py
    Trova i migliori iperparametri su tutti i fold, poi riusa quelli.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: STANDARD CV (tuning su tutto il dataset)")
    print("=" * 70)

    X_full = df[baseline_features]
    y_full = df['label_apnea']
    subjects_full = df['Subject']

    # Step 1: Find best config across ALL data (this is the data leakage!)
    print("\n[1] Finding best hyperparameters across ALL folds...")

    best_overall = {'score': -np.inf, 'config': None}

    for k, ws, thr in product(K_VALUES, WINDOW_SIZES, THRESHOLDS):
        temporal_df = temporal_cache[ws]
        X_full_temp = pd.concat([X_full.reset_index(drop=True),
                                  temporal_df.reset_index(drop=True)], axis=1)

        gkf = GroupKFold(n_splits=5)
        fold_scores = []

        for train_idx, test_idx in gkf.split(X_full, y_full, groups=subjects_full):
            X_train = X_full_temp.iloc[train_idx].values
            X_test = X_full_temp.iloc[test_idx].values
            y_train = y_full.iloc[train_idx].values
            y_test = y_full.iloc[test_idx].values

            if y_train.sum() == 0 or y_test.sum() == 0:
                continue

            result = evaluate_config(X_train, y_train, X_test, y_test, k, thr)
            if result:
                fold_scores.append(result['f1'])

        if len(fold_scores) > 0:
            mean_score = np.mean(fold_scores)
            if mean_score > best_overall['score']:
                best_overall['score'] = mean_score
                best_overall['config'] = {'k': k, 'window': ws, 'threshold': thr}

    best_k = best_overall['config']['k']
    best_ws = best_overall['config']['window']
    best_thr = best_overall['config']['threshold']

    print(f"    Best config: K={best_k}, Window={best_ws}, Threshold={best_thr}")
    print(f"    Best CV F1: {best_overall['score']:.4f}")

    # Step 2: Evaluate with these fixed params
    print("\n[2] Evaluating with fixed best params...")

    temporal_df = temporal_cache[best_ws]
    X_full_temp = pd.concat([X_full.reset_index(drop=True),
                              temporal_df.reset_index(drop=True)], axis=1)

    gkf = GroupKFold(n_splits=5)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_full, groups=subjects_full), 1):
        X_train = X_full_temp.iloc[train_idx].values
        X_test = X_full_temp.iloc[test_idx].values
        y_train = y_full.iloc[train_idx].values
        y_test = y_full.iloc[test_idx].values

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        result = evaluate_config(X_train, y_train, X_test, y_test, best_k, best_thr)
        if result:
            result['fold'] = fold_idx
            result['best_k'] = best_k
            result['best_window'] = best_ws
            result['best_threshold'] = best_thr
            results.append(result)
            print(f"    Fold {fold_idx}: ROC-AUC={result['roc_auc']:.4f}, "
                  f"F1={result['f1']:.4f}, Recall={result['recall']:.4f}")

    return pd.DataFrame(results), best_overall['config']


def run_nested_cv(df, baseline_features, temporal_cache, optimize_for='f1'):
    """
    NESTED CV: Tuning nell'inner loop per ogni fold esterno.
    Questo è l'approccio CORRETTO senza data leakage.
    """
    opt_name = {'f1': 'F1', 'fbeta': 'F-beta (recall)', 'recall': 'Recall'}[optimize_for]

    print("\n" + "=" * 70)
    print(f"APPROACH 2: NESTED CV (optimize {opt_name})")
    print("=" * 70)

    X_full = df[baseline_features]
    y_full = df['label_apnea']
    subjects_full = df['Subject']

    outer_cv = GroupKFold(n_splits=5)
    results = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups=subjects_full), 1):
        print(f"\n  Outer Fold {fold_idx}/5:")

        # Development and test sets
        X_dev = df.iloc[dev_idx]
        X_test_base = df[baseline_features].iloc[test_idx]
        y_dev = df['label_apnea'].iloc[dev_idx]
        y_test = df['label_apnea'].iloc[test_idx].values
        subjects_dev = df['Subject'].iloc[dev_idx]

        if y_dev.sum() == 0 or y_test.sum() == 0:
            print("    Skipping: no apnea samples")
            continue

        # Inner CV: find best config for THIS fold
        print("    [Inner CV] Finding best hyperparameters...")
        best_config = inner_cv_tune(X_dev, y_dev, subjects_dev, temporal_cache,
                                     baseline_features, optimize_for=optimize_for)

        best_k = best_config['k']
        best_ws = best_config['window']
        best_thr = best_config['threshold']
        print(f"    Best: K={best_k}, Window={best_ws}, Threshold={best_thr}")

        # Evaluate on outer test set
        temporal_dev = temporal_cache[best_ws].iloc[dev_idx]
        temporal_test = temporal_cache[best_ws].iloc[test_idx]

        X_dev_full = pd.concat([X_dev[baseline_features].reset_index(drop=True),
                                 temporal_dev.reset_index(drop=True)], axis=1).values
        X_test_full = pd.concat([X_test_base.reset_index(drop=True),
                                  temporal_test.reset_index(drop=True)], axis=1).values
        y_dev_arr = y_dev.values

        result = evaluate_config(X_dev_full, y_dev_arr, X_test_full, y_test, best_k, best_thr)
        if result:
            result['fold'] = fold_idx
            result['best_k'] = best_k
            result['best_window'] = best_ws
            result['best_threshold'] = best_thr
            results.append(result)
            print(f"    Result: ROC-AUC={result['roc_auc']:.4f}, "
                  f"F1={result['f1']:.4f}, Recall={result['recall']:.4f}")

    return pd.DataFrame(results)


def main():
    # Load dataset
    if DATASET_PATH is None or not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found!")
        print("Searched in:")
        for p in SEARCH_PATHS:
            print(f"  - {p}")
        print("Run prepare_dataset_manual.py first")
        return

    print(f"\n[DATASET] Loading: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"          Samples: {len(df)}, Subjects: {df['Subject'].nunique()}")
    print(f"          Apnea: {df['label_apnea'].sum()} ({100*df['label_apnea'].sum()/len(df):.1f}%)")

    # Prepare features
    exclude_cols = ['Subject', 'label_apnea', 'start_time', 'end_time',
                    'Position_mode', 'majority_status', 'frac_apnea',
                    'frac_altro', 'frac_respiro']
    baseline_features = [c for c in df.columns if c not in exclude_cols]

    # Pre-generate temporal features
    print("\n[FEATURES] Generating temporal features...")
    temporal_cache = {}
    for ws in WINDOW_SIZES:
        temporal_cache[ws] = add_temporal_features(df, window_size=ws)
        print(f"           Window={ws}: {len(temporal_cache[ws].columns)} features")

    # Run both approaches
    df_standard, best_std_config = run_standard_cv(df, baseline_features, temporal_cache)
    df_nested_f1 = run_nested_cv(df, baseline_features, temporal_cache, optimize_for='f1')
    df_nested_recall = run_nested_cv(df, baseline_features, temporal_cache, optimize_for='fbeta')

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n\n" + "=" * 80)
    print(" " * 25 + "RISULTATI COMPARATIVI")
    print("=" * 80)

    def summarize(df_res, name):
        if len(df_res) == 0:
            return None
        return {
            'Approach': name,
            'ROC-AUC': f"{df_res['roc_auc'].mean():.4f} +/- {df_res['roc_auc'].std():.4f}",
            'F1': f"{df_res['f1'].mean():.4f} +/- {df_res['f1'].std():.4f}",
            'Precision': f"{df_res['precision'].mean():.4f} +/- {df_res['precision'].std():.4f}",
            'Recall': f"{df_res['recall'].mean():.4f} +/- {df_res['recall'].std():.4f}",
            'recall_mean': df_res['recall'].mean(),
            'f1_mean': df_res['f1'].mean(),
        }

    summary_std = summarize(df_standard, "1. Standard CV (biased)")
    summary_nested_f1 = summarize(df_nested_f1, "2. Nested CV (F1)")
    summary_nested_recall = summarize(df_nested_recall, "3. Nested CV (Recall)")

    print("\n" + "-" * 80)
    print(f"{'Approach':<30} {'ROC-AUC':>20} {'F1':>20} {'Recall':>20}")
    print("-" * 80)

    for s in [summary_std, summary_nested_f1, summary_nested_recall]:
        if s:
            print(f"{s['Approach']:<30} {s['ROC-AUC']:>20} {s['F1']:>20} {s['Recall']:>20}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALISI")
    print("=" * 80)

    if summary_std and summary_nested_f1:
        recall_diff = summary_std['recall_mean'] - summary_nested_f1['recall_mean']
        f1_diff = summary_std['f1_mean'] - summary_nested_f1['f1_mean']

        print(f"""
DIFFERENZA STANDARD vs NESTED (F1):
  Recall: {summary_std['recall_mean']:.4f} -> {summary_nested_f1['recall_mean']:.4f} (diff: {recall_diff:+.4f})
  F1:     {summary_std['f1_mean']:.4f} -> {summary_nested_f1['f1_mean']:.4f} (diff: {f1_diff:+.4f})
""")

        if recall_diff > 0.05:
            print(f"  -> La recall e' calata del {100*recall_diff/summary_std['recall_mean']:.1f}%")
            print(f"     Questo conferma DATA LEAKAGE nel tuning originale!")
        else:
            print(f"  -> La differenza e' contenuta ({100*recall_diff/summary_std['recall_mean']:.1f}%)")
            print(f"     Il data leakage era limitato.")

    if summary_nested_recall:
        print(f"""
NESTED CV (Recall-optimized):
  Recall: {summary_nested_recall['recall_mean']:.4f}
  F1:     {summary_nested_recall['f1_mean']:.4f}
""")
        if summary_nested_f1:
            improvement = summary_nested_recall['recall_mean'] - summary_nested_f1['recall_mean']
            if improvement > 0:
                print(f"  -> Ottimizzando per recall, recuperiamo {improvement:.4f} punti di recall")

    # Hyperparameters per fold
    print("\n" + "-" * 80)
    print("IPERPARAMETRI SCELTI PER FOLD:")
    print("-" * 80)

    print(f"\nStandard CV: K={best_std_config['k']}, Window={best_std_config['window']}, "
          f"Threshold={best_std_config['threshold']} (fisso per tutti i fold)")

    print("\nNested CV (F1):")
    for _, row in df_nested_f1.iterrows():
        print(f"  Fold {int(row['fold'])}: K={int(row['best_k'])}, "
              f"Window={int(row['best_window'])}, Threshold={row['best_threshold']}")

    print("\nNested CV (Recall):")
    for _, row in df_nested_recall.iterrows():
        print(f"  Fold {int(row['fold'])}: K={int(row['best_k'])}, "
              f"Window={int(row['best_window'])}, Threshold={row['best_threshold']}")

    # Save results
    df_standard.to_csv(RESULTS_DIR / 'standard_cv_results.csv', index=False)
    df_nested_f1.to_csv(RESULTS_DIR / 'nested_cv_f1_results.csv', index=False)
    df_nested_recall.to_csv(RESULTS_DIR / 'nested_cv_recall_results.csv', index=False)

    print(f"\n\n{'='*80}")
    print("RACCOMANDAZIONI:")
    print("="*80)
    print("""
1. I risultati NESTED CV sono quelli REALI da aspettarsi su nuovi pazienti

2. La differenza tra Standard e Nested mostra il BIAS del tuning globale

3. Per il paper:
   - Riporta i risultati Nested CV come stima realistica
   - Puoi menzionare i risultati Standard come "upper bound ottimistico"

4. Se la recall e' prioritaria, usa i risultati "Nested CV (Recall)"
""")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
