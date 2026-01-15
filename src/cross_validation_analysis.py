#!/usr/bin/env python3
"""
Cross-validation analysis con GroupKFold per subject.
Verifica la robustezza dei risultati senza distruggere i dati attuali.

Esegue 5-fold cross-validation con:
- RF + Temporal features (K=60)
- XGBoost + Temporal features (K=60)
- Baseline (50 features)

Output: Mean ± Std per tutte le metriche.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available, skipping XGB CV")

# Config
from pathlib import Path

DATASET_NAME = "dataset_apnea_windows_30s_features_zones_mat_acc_CORRECTED.csv"
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

# Search paths for dataset
SEARCH_PATHS = [
    BASE_DIR / ".." / "data" / DATASET_NAME,
    BASE_DIR / ".." / DATASET_NAME,
    BASE_DIR / "preprocessing_output" / "dataset_windows_30s_features_zones_CORRECTED.csv",
    Path(".") / DATASET_NAME,
]

def find_dataset():
    for path in SEARCH_PATHS:
        if path.exists():
            return str(path)
    return None

INPUT_CSV = find_dataset()
if INPUT_CSV is None:
    print("ERROR: Dataset not found! Run prepare_dataset.py first.")
    exit(1)

K_FEATURES = 60
N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.30

def add_temporal_features(df):
    """Generate temporal features."""
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

        roll_mean = df.groupby('Subject')[feat].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        temporal_features[f'roll3_mean_{feat}'] = roll_mean

        roll_std = df.groupby('Subject')[feat].rolling(3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
        temporal_features[f'roll3_std_{feat}'] = roll_std

        temporal_features[f'trend_{feat}'] = df[feat] - roll_mean

    return temporal_features

def add_interaction_features(df):
    """Generate interaction features."""
    interaction_features = pd.DataFrame(index=df.index)

    if 'zone_UL_mean_mean' in df.columns:
        left = df['zone_UL_mean_mean'] + df['zone_LL_mean_mean']
        right = df['zone_UR_mean_mean'] + df['zone_LR_mean_mean']
        interaction_features['contrast_left_right'] = left - right
        interaction_features['ratio_left_right'] = left / (right + 1e-6)

        upper = df['zone_UL_mean_mean'] + df['zone_UR_mean_mean']
        lower = df['zone_LL_mean_mean'] + df['zone_LR_mean_mean']
        interaction_features['contrast_upper_lower'] = upper - lower
        interaction_features['ratio_upper_lower'] = upper / (lower + 1e-6)

        interaction_features['zone_variability_global'] = (
            df['zone_UL_diff_std_mean'] + df['zone_UR_diff_std_mean'] +
            df['zone_LL_diff_std_mean'] + df['zone_LR_diff_std_mean']
        ) / 4

        diag1 = df['zone_UL_mean_mean'] + df['zone_LR_mean_mean']
        diag2 = df['zone_UR_mean_mean'] + df['zone_LL_mean_mean']
        interaction_features['contrast_diag1_diag2'] = diag1 - diag2

    return interaction_features

def train_and_evaluate_baseline(X_train, y_train, X_test, y_test):
    """Train baseline RF without temporal features."""
    # Oversample
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    y_pos = y_train[y_train == 1]
    y_neg = y_train[y_train == 0]

    if len(y_pos) == 0:
        return None  # No positive samples in this fold

    X_pos_res, y_pos_res = resample(X_pos, y_pos, n_samples=len(X_neg), random_state=RANDOM_STATE, replace=True)
    X_train_bal = np.vstack([X_neg, X_pos_res])
    y_train_bal = np.hstack([y_neg, y_pos_res])

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_pred_proba)

def train_and_evaluate_rf_temporal(X_train, y_train, X_test, y_test, feature_names):
    """Train RF with temporal features and feature selection."""
    # Feature selection
    selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    # Oversample
    X_pos = X_train_sel[y_train == 1]
    X_neg = X_train_sel[y_train == 0]
    y_pos = y_train[y_train == 1]
    y_neg = y_train[y_train == 0]

    if len(y_pos) == 0:
        return None

    X_pos_res, y_pos_res = resample(X_pos, y_pos, n_samples=len(X_neg), random_state=RANDOM_STATE, replace=True)
    X_train_bal = np.vstack([X_neg, X_pos_res])
    y_train_bal = np.hstack([y_neg, y_pos_res])

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)

    y_pred_proba = rf.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_pred_proba)

def train_and_evaluate_xgb_temporal(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost with temporal features and feature selection."""
    if not HAS_XGB:
        return None

    # Feature selection
    selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    if len(y_train[y_train == 1]) == 0:
        return None

    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_sel, y_train)

    y_pred_proba = xgb_model.predict_proba(X_test_sel)[:, 1]
    y_pred = (y_pred_proba >= THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_pred_proba)

def compute_metrics(y_test, y_pred, y_pred_proba):
    """Compute all metrics."""
    metrics = {}

    # Basic metrics
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)

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
        else:
            metrics['specificity'] = np.nan
            metrics['tp'] = metrics['fp'] = metrics['fn'] = metrics['tn'] = 0
    else:
        metrics['specificity'] = np.nan
        metrics['tp'] = metrics['fp'] = metrics['fn'] = metrics['tn'] = 0

    return metrics

def main():
    print("="*70)
    print("CROSS-VALIDATION ANALYSIS (5-Fold GroupKFold by Subject)")
    print("="*70)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"\nDataset: {df.shape}")
    print(f"Subjects: {df['Subject'].nunique()}")
    print(f"Apnea windows: {df['label_apnea'].sum()} / {len(df)}")

    # Prepare features
    meta_cols = ['Subject', 'start_time', 'end_time', 'Position_mode',
                 'majority_status', 'label_apnea', 'frac_apnea', 'frac_altro', 'frac_respiro']

    # Baseline features
    baseline_features = [c for c in df.columns if c not in meta_cols]
    X_baseline = df[baseline_features].values

    # Generate temporal features
    print("\nGenerating temporal features...")
    temporal_feats = add_temporal_features(df)
    interaction_feats = add_interaction_features(df)
    df_full = pd.concat([df, temporal_feats, interaction_feats], axis=1)

    full_features = [c for c in df_full.columns if c not in meta_cols]
    X_full = df_full[full_features].values

    y = df['label_apnea'].values
    subjects = df['Subject'].values

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Full features (with temporal): {len(full_features)}")

    # Cross-validation
    print(f"\nRunning {N_SPLITS}-fold cross-validation...")
    gkf = GroupKFold(n_splits=N_SPLITS)

    results_baseline = []
    results_rf_temporal = []
    results_xgb_temporal = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_baseline, y, groups=subjects), 1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")

        # Get train/test for this fold
        X_train_base, X_test_base = X_baseline[train_idx], X_baseline[test_idx]
        X_train_full, X_test_full = X_full[train_idx], X_full[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_subjects = np.unique(subjects[test_idx])
        n_apnea_train = y_train.sum()
        n_apnea_test = y_test.sum()

        print(f"  Test subjects: {list(test_subjects)}")
        print(f"  Train: {len(y_train)} samples ({n_apnea_train} apneas)")
        print(f"  Test: {len(y_test)} samples ({n_apnea_test} apneas)")

        if n_apnea_train == 0 or n_apnea_test == 0:
            print("  WARNING: No apnea samples in train or test, skipping fold")
            continue

        # Train and evaluate all models
        print("  Training baseline RF...")
        metrics_base = train_and_evaluate_baseline(X_train_base, y_train, X_test_base, y_test)
        if metrics_base:
            results_baseline.append(metrics_base)
            print(f"    ROC-AUC: {metrics_base['roc_auc']:.4f}, F1: {metrics_base['f1']:.4f}")

        print("  Training RF + Temporal...")
        metrics_rf = train_and_evaluate_rf_temporal(X_train_full, y_train, X_test_full, y_test, full_features)
        if metrics_rf:
            results_rf_temporal.append(metrics_rf)
            print(f"    ROC-AUC: {metrics_rf['roc_auc']:.4f}, F1: {metrics_rf['f1']:.4f}")

        if HAS_XGB:
            print("  Training XGBoost + Temporal...")
            metrics_xgb = train_and_evaluate_xgb_temporal(X_train_full, y_train, X_test_full, y_test, full_features)
            if metrics_xgb:
                results_xgb_temporal.append(metrics_xgb)
                print(f"    ROC-AUC: {metrics_xgb['roc_auc']:.4f}, F1: {metrics_xgb['f1']:.4f}")

    # Aggregate results
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)

    def print_results(results, model_name):
        if not results:
            print(f"\n{model_name}: No valid folds")
            return

        print(f"\n{model_name} ({len(results)} folds)")
        print("-" * 70)

        metrics_df = pd.DataFrame(results)

        for metric in ['roc_auc', 'f1', 'precision', 'recall', 'specificity', 'accuracy']:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                print(f"  {metric.upper():15s}: {mean:.4f} ± {std:.4f}")

        # Confusion matrix aggregated
        tp_total = metrics_df['tp'].sum()
        fp_total = metrics_df['fp'].sum()
        fn_total = metrics_df['fn'].sum()
        tn_total = metrics_df['tn'].sum()

        print(f"\n  Aggregated Confusion Matrix:")
        print(f"    TP={tp_total}, FP={fp_total}, FN={fn_total}, TN={tn_total}")

        return metrics_df

    df_baseline = print_results(results_baseline, "BASELINE (50 features)")
    df_rf = print_results(results_rf_temporal, "RF + TEMPORAL (K=60)")
    if HAS_XGB:
        df_xgb = print_results(results_xgb_temporal, "XGBOOST + TEMPORAL (K=60)")

    # Save results to CSV
    print("\n" + "="*70)
    print("Saving results...")

    if results_baseline:
        pd.DataFrame(results_baseline).to_csv('results_analysis/cv_baseline.csv', index=False)
        print("✓ Saved: results_analysis/cv_baseline.csv")

    if results_rf_temporal:
        pd.DataFrame(results_rf_temporal).to_csv('results_analysis/cv_rf_temporal.csv', index=False)
        print("✓ Saved: results_analysis/cv_rf_temporal.csv")

    if HAS_XGB and results_xgb_temporal:
        pd.DataFrame(results_xgb_temporal).to_csv('results_analysis/cv_xgb_temporal.csv', index=False)
        print("✓ Saved: results_analysis/cv_xgb_temporal.csv")

    # Summary table
    print("\n" + "="*70)
    print("COMPARISON WITH SINGLE SPLIT")
    print("="*70)

    print("\nSingle split (original):")
    print("  Baseline:      ROC-AUC 0.6929, F1 0.242")
    print("  RF+Temporal:   ROC-AUC 0.9470, F1 0.556")
    print("  XGB+Temporal:  ROC-AUC 0.8981, F1 0.500")

    if results_rf_temporal:
        rf_auc_mean = df_rf['roc_auc'].mean()
        rf_f1_mean = df_rf['f1'].mean()
        print(f"\nCross-validation (mean):")
        print(f"  RF+Temporal:   ROC-AUC {rf_auc_mean:.4f}, F1 {rf_f1_mean:.4f}")

        if rf_auc_mean > 0.85:
            print("\n✓ Results are ROBUST: CV performance remains high")
        elif rf_auc_mean > 0.75:
            print("\n⚠ Results show MODERATE robustness: some variation across folds")
        else:
            print("\n❌ WARNING: Results may be OVERFITTED to original split")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
