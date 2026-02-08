#!/usr/bin/env python3
"""
Nested Cross-Validation OTTIMIZZATA PER RECALL

Questo script è una variante della nested CV che ottimizza esplicitamente
per la RECALL invece che per F1, con vincoli minimi su precision.

MOTIVAZIONE:
- La nested CV standard con F1 tende a essere conservativa
- Per screening medico, la recall (sensibilità) è più importante
- Usiamo F-beta con beta=2 (recall vale 2x più di precision)

STRATEGIA:
1. Ottimizzazione per F-beta (beta=2) invece di F1
2. Range più ampio di threshold (più bassi per favorire recall)
3. Vincolo minimo su precision per evitare troppi falsi positivi
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

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
RESULTS_DIR = BASE_DIR / "results_nested_cv_recall_optimized"
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset
DATASET_NAME = "dataset_apnea_windows_30s_features_zones_mat_acc_CORRECTED.csv"
SEARCH_PATHS = [
    BASE_DIR / "preprocessing_output" / DATASET_NAME,
    BASE_DIR / ".." / "data" / DATASET_NAME,
    BASE_DIR / ".." / ".." / DATASET_NAME,
    Path("/Users/ricky/Documents/phd/apnea_study") / DATASET_NAME,
]

def find_dataset():
    for path in SEARCH_PATHS:
        if path.exists():
            return str(path)
    return None

# ============================================================================
# HYPERPARAMETER GRID - OTTIMIZZATO PER RECALL
# ============================================================================
# Range più ampio di K
K_VALUES = [30, 40, 50, 60, 70, 80, 90]

# Threshold più bassi per favorire recall (più predizioni positive)
THRESHOLD_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Vincolo minimo su precision (evita troppi falsi positivi)
MIN_PRECISION = 0.15

# Beta per F-beta score (beta=2 significa recall conta 2x più di precision)
BETA = 2.0

# Fixed hyperparameters
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3
WINDOW_SIZE = 3

print("=" * 80)
print(" " * 15 + "NESTED CROSS-VALIDATION (RECALL OPTIMIZED)")
print(" " * 20 + "Sleep Apnea Detection Analysis")
print("=" * 80)
print(f"\nOBIETTIVO: Massimizzare RECALL mantenendo precision >= {MIN_PRECISION}")
print(f"METRICA DI TUNING: F-beta score (beta={BETA})")
print(f"THRESHOLD RANGE: {THRESHOLD_VALUES} (più bassi = più recall)")
print("=" * 80)

# ============================================================================
# TEMPORAL FEATURES
# ============================================================================
def add_temporal_features(df, window_size=3):
    """Generate temporal features with rolling windows."""
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

# ============================================================================
# INNER CV: HYPERPARAMETER TUNING (OTTIMIZZATO PER RECALL)
# ============================================================================
def inner_cv_tuning(X_dev, y_dev, subjects_dev, model_type='rf'):
    """
    Inner cross-validation ottimizzata per RECALL.

    Usa F-beta score (beta=2) invece di F1 per dare più peso alla recall.
    Applica anche un vincolo minimo su precision.
    """
    best_score = -np.inf
    best_k = 60
    best_threshold = 0.25  # Default più basso per favorire recall

    inner_cv = GroupKFold(n_splits=4)

    # Track all results for analysis
    all_results = []

    for k in K_VALUES:
        for threshold in THRESHOLD_VALUES:
            fold_scores = []
            fold_recalls = []
            fold_precisions = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_dev, y_dev, groups=subjects_dev):
                X_inner_train = X_dev[inner_train_idx]
                X_inner_val = X_dev[inner_val_idx]
                y_inner_train = y_dev[inner_train_idx]
                y_inner_val = y_dev[inner_val_idx]

                if y_inner_train.sum() == 0 or y_inner_val.sum() == 0:
                    continue

                # Feature selection
                selector = SelectKBest(f_classif, k=min(k, X_inner_train.shape[1]))
                X_inner_train_sel = selector.fit_transform(X_inner_train, y_inner_train)
                X_inner_val_sel = selector.transform(X_inner_val)

                # Oversample
                pos_idx = y_inner_train == 1
                neg_idx = y_inner_train == 0

                X_pos = X_inner_train_sel[pos_idx]
                X_neg = X_inner_train_sel[neg_idx]
                y_pos = y_inner_train[pos_idx]
                y_neg = y_inner_train[neg_idx]

                X_pos_res, y_pos_res = resample(
                    X_pos, y_pos,
                    n_samples=len(X_neg),
                    random_state=RANDOM_STATE,
                    replace=True
                )

                X_inner_train_bal = np.vstack([X_neg, X_pos_res])
                y_inner_train_bal = np.hstack([y_neg, y_pos_res])

                # Train model
                if model_type == 'rf':
                    model = RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=MIN_SAMPLES_LEAF,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=N_ESTIMATORS,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric='logloss'
                    )

                model.fit(X_inner_train_bal, y_inner_train_bal)

                # Evaluate
                y_pred_proba = model.predict_proba(X_inner_val_sel)[:, 1]
                y_pred = (y_pred_proba >= threshold).astype(int)

                # Compute metrics
                recall = recall_score(y_inner_val, y_pred, zero_division=0)
                precision = precision_score(y_inner_val, y_pred, zero_division=0)

                # F-beta score (beta=2 favors recall)
                fbeta = fbeta_score(y_inner_val, y_pred, beta=BETA, zero_division=0)

                fold_scores.append(fbeta)
                fold_recalls.append(recall)
                fold_precisions.append(precision)

            if len(fold_scores) > 0:
                mean_score = np.mean(fold_scores)
                mean_recall = np.mean(fold_recalls)
                mean_precision = np.mean(fold_precisions)

                all_results.append({
                    'k': k,
                    'threshold': threshold,
                    'fbeta': mean_score,
                    'recall': mean_recall,
                    'precision': mean_precision
                })

                # Accept only if precision >= minimum constraint
                if mean_precision >= MIN_PRECISION and mean_score > best_score:
                    best_score = mean_score
                    best_k = k
                    best_threshold = threshold

    # If no configuration met the precision constraint, take the one with highest recall
    # that has at least SOME precision
    if best_score == -np.inf and len(all_results) > 0:
        valid_results = [r for r in all_results if r['precision'] > 0.05]
        if valid_results:
            best_config = max(valid_results, key=lambda x: x['recall'])
            best_k = best_config['k']
            best_threshold = best_config['threshold']

    return best_k, best_threshold, all_results

# ============================================================================
# OUTER CV: PERFORMANCE EVALUATION
# ============================================================================
def outer_cv_evaluation(X_full, y_full, subjects_full, model_type='rf'):
    """
    Outer cross-validation per stimare le performance finali.
    """
    outer_cv = GroupKFold(n_splits=5)
    results = []
    tuning_info = []

    for fold_idx, (dev_idx, test_idx) in enumerate(outer_cv.split(X_full, y_full, groups=subjects_full), 1):
        print(f"\n{'='*70}")
        print(f"OUTER FOLD {fold_idx}/5")
        print(f"{'='*70}")

        # Split data
        X_dev = X_full[dev_idx]
        X_test = X_full[test_idx]
        y_dev = y_full[dev_idx]
        y_test = y_full[test_idx]
        subjects_dev = subjects_full[dev_idx]
        subjects_test = subjects_full[test_idx]

        test_subjects_list = np.unique(subjects_test)

        print(f"Development set: {len(y_dev)} samples, {len(np.unique(subjects_dev))} subjects")
        print(f"Test set: {len(y_test)} samples, {len(np.unique(subjects_test))} subjects")
        print(f"Test subjects: {list(test_subjects_list)}")
        print(f"Apnea in dev: {y_dev.sum()} ({100*y_dev.sum()/len(y_dev):.1f}%)")
        print(f"Apnea in test: {y_test.sum()} ({100*y_test.sum()/len(y_test):.1f}%)")

        if y_dev.sum() == 0 or y_test.sum() == 0:
            print("WARNING: No apnea in dev or test, skipping fold")
            continue

        # INNER CV: Hyperparameter tuning
        print(f"\n[1] Inner CV tuning (optimizing F-beta with beta={BETA})...")
        best_k, best_threshold, tuning_results = inner_cv_tuning(X_dev, y_dev, subjects_dev, model_type)
        print(f"    Best K: {best_k}")
        print(f"    Best threshold: {best_threshold}")

        tuning_info.append({
            'fold': fold_idx,
            'best_k': best_k,
            'best_threshold': best_threshold,
            'tuning_results': tuning_results
        })

        # OUTER TRAINING
        print("\n[2] Training on full development set...")

        # Feature selection
        selector = SelectKBest(f_classif, k=min(best_k, X_dev.shape[1]))
        X_dev_sel = selector.fit_transform(X_dev, y_dev)
        X_test_sel = selector.transform(X_test)

        # Oversample
        pos_idx = y_dev == 1
        neg_idx = y_dev == 0

        X_pos = X_dev_sel[pos_idx]
        X_neg = X_dev_sel[neg_idx]
        y_pos = y_dev[pos_idx]
        y_neg = y_dev[neg_idx]

        X_pos_res, y_pos_res = resample(
            X_pos, y_pos,
            n_samples=len(X_neg),
            random_state=RANDOM_STATE,
            replace=True
        )

        X_dev_bal = np.vstack([X_neg, X_pos_res])
        y_dev_bal = np.hstack([y_neg, y_pos_res])

        # Train final model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )

        model.fit(X_dev_bal, y_dev_bal)

        # OUTER EVALUATION
        print("\n[3] Evaluating on held-out test set...")
        y_pred_proba = model.predict_proba(X_test_sel)[:, 1]
        y_pred = (y_pred_proba >= best_threshold).astype(int)

        # Compute metrics
        metrics = {
            'fold': fold_idx,
            'test_subjects': list(test_subjects_list),
            'best_k': best_k,
            'best_threshold': best_threshold,
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'fbeta': fbeta_score(y_test, y_pred, beta=BETA, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
        }

        # Confusion matrix
        if len(np.unique(y_test)) > 1:
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['tp'] = tp
                metrics['fp'] = fp
                metrics['fn'] = fn
                metrics['tn'] = tn

        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    F-beta (β={BETA}): {metrics['fbeta']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    RECALL: {metrics['recall']:.4f}  <-- TARGET METRIC")

        results.append(metrics)

    return results, tuning_info

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Load dataset
    dataset_path = find_dataset()
    if dataset_path is None:
        print("ERROR: Dataset not found!")
        return

    print(f"\n[DATASET] Loading: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"          Samples: {len(df)}")
    print(f"          Subjects: {df['Subject'].nunique()}")
    print(f"          Apnea: {df['label_apnea'].sum()} / {len(df)} ({100*df['label_apnea'].sum()/len(df):.1f}%)")

    # Prepare features
    meta_cols = ['Subject', 'start_time', 'end_time', 'Position_mode',
                 'majority_status', 'label_apnea', 'frac_apnea',
                 'frac_altro', 'frac_respiro']

    baseline_features = [c for c in df.columns if c not in meta_cols]

    print(f"\n[FEATURES] Generating temporal features (window={WINDOW_SIZE})...")
    temporal_feats = add_temporal_features(df, window_size=WINDOW_SIZE)

    df_full = pd.concat([df[baseline_features], temporal_feats], axis=1)

    print(f"           Baseline features: {len(baseline_features)}")
    print(f"           Temporal features: {len(temporal_feats.columns)}")
    print(f"           Total features: {len(df_full.columns)}")

    X_full = df_full.values
    y_full = df['label_apnea'].values
    subjects_full = df['Subject'].values

    # Run nested CV for Random Forest
    print(f"\n{'='*80}")
    print(" " * 15 + "RANDOM FOREST - NESTED CV (RECALL OPTIMIZED)")
    print(f"{'='*80}")

    results_rf, tuning_rf = outer_cv_evaluation(X_full, y_full, subjects_full, model_type='rf')

    # Run nested CV for XGBoost (if available)
    results_xgb, tuning_xgb = None, None
    if HAS_XGB:
        print(f"\n{'='*80}")
        print(" " * 15 + "XGBOOST - NESTED CV (RECALL OPTIMIZED)")
        print(f"{'='*80}")

        results_xgb, tuning_xgb = outer_cv_evaluation(X_full, y_full, subjects_full, model_type='xgb')

    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    print(f"\n\n{'='*80}")
    print(" " * 20 + "NESTED CV RESULTS SUMMARY")
    print(" " * 20 + "(RECALL OPTIMIZED)")
    print(f"{'='*80}")

    def print_summary(results, tuning_info, model_name):
        if not results:
            print(f"\n{model_name}: No valid folds")
            return

        print(f"\n{model_name}")
        print("-" * 80)

        # Print tuning info
        print("\nHyperparameters selected per fold (via inner CV):")
        for info in tuning_info:
            print(f"  Fold {info['fold']}: K={info['best_k']}, Threshold={info['best_threshold']}")

        # Aggregate metrics
        df_results = pd.DataFrame(results)

        print(f"\nPerformance metrics (mean ± std across {len(results)} folds):")
        print("-" * 80)

        for metric in ['roc_auc', 'fbeta', 'f1', 'precision', 'recall', 'specificity', 'accuracy']:
            if metric in df_results.columns:
                values = df_results[metric].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    std = values.std()
                    marker = "  <-- TARGET" if metric == 'recall' else ""
                    print(f"  {metric.upper():15s}: {mean:.4f} ± {std:.4f}{marker}")

        # Confusion matrix aggregated
        if 'tp' in df_results.columns:
            tp_total = df_results['tp'].sum()
            fp_total = df_results['fp'].sum()
            fn_total = df_results['fn'].sum()
            tn_total = df_results['tn'].sum()
            print(f"\n  Aggregated Confusion Matrix:")
            print(f"    TP={tp_total}, FP={fp_total}, FN={fn_total}, TN={tn_total}")

        # Save results
        results_file = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_nested_cv_recall_opt.csv"
        df_results.to_csv(results_file, index=False)
        print(f"\n  Results saved to: {results_file}")

        return df_results

    df_rf = print_summary(results_rf, tuning_rf, "Random Forest")

    df_xgb = None
    if results_xgb:
        df_xgb = print_summary(results_xgb, tuning_xgb, "XGBoost")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print(f"\n\n{'='*80}")
    print(" " * 25 + "ANALISI COMPARATIVA")
    print(f"{'='*80}")

    print("""
CONFRONTO TRA APPROCCI:

1. APPROCCIO ORIGINALE (K scelto su tutto il dataset):
   - K=60, threshold=0.30 scelti globalmente
   - Performance ottimistiche (data leakage)
   - Recall alta ma non realistica

2. NESTED CV STANDARD (F1):
   - Ottimizza per F1 (bilancia precision/recall)
   - Tende a essere conservativo
   - Recall spesso più bassa

3. NESTED CV RECALL-OPTIMIZED (questo script):
   - Ottimizza per F-beta (beta=2)
   - Threshold più bassi (0.15-0.40)
   - Vincolo minimo su precision
   - Recall migliore mantenendo validità statistica

PERCHÉ LA NESTED CV DÀ RISULTATI PEGGIORI:
- I risultati originali erano ottimistici per data leakage
- La differenza mostra quanto il tuning globale "barasse"
- I risultati nested CV sono quelli REALI da aspettarsi

RACCOMANDAZIONI:
- Per il paper: riporta i risultati nested CV (sono onesti)
- Per l'applicazione clinica: usa i threshold ottimizzati per recall
- La differenza tra originale e nested CV è il "bias da ottimizzazione"
    """)

    # Save comparison
    if df_rf is not None:
        comparison = {
            'approach': ['Nested CV (Recall Optimized)'],
            'recall_mean': [df_rf['recall'].mean()],
            'recall_std': [df_rf['recall'].std()],
            'precision_mean': [df_rf['precision'].mean()],
            'precision_std': [df_rf['precision'].std()],
            'f1_mean': [df_rf['f1'].mean()],
            'f1_std': [df_rf['f1'].std()],
            'roc_auc_mean': [df_rf['roc_auc'].mean()],
            'roc_auc_std': [df_rf['roc_auc'].std()],
        }
        pd.DataFrame(comparison).to_csv(RESULTS_DIR / 'comparison_summary.csv', index=False)

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
