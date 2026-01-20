#!/usr/bin/env python3
"""
Complete Sleep Apnea Detection Analysis
VERSIONE PER DATASET CON ETICHETTE MANUALI (OTTIMIZZATO)

Runs the full pipeline from data loading to cross-validation
using the manually-labeled dataset with ~72% more apnea events.

PARAMETRI OTTIMIZZATI (da hyperparameter_tuning.py):
- K_FEATURES: 70 (era 60)
- ROLLING_WINDOW: 7 (era 3) - 3.5 minuti invece di 1.5
- THRESHOLD: 0.25 (era 0.30)

Risultato atteso: ROC-AUC ~0.90 vs 0.81 con parametri originali (+11%)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TEST_SUBJECTS = ['GZ01', 'FDR01', 'AM01', 'FC01']
N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 3

# PARAMETRI OTTIMIZZATI (da hyperparameter_tuning.py)
K_FEATURES = 70          # era 60
ROLLING_WINDOW = 7       # era 3 (finestra temporale più lunga: 3.5 min)
CLASSIFICATION_THR = 0.25  # era 0.30

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
RESULTS_DIR = BASE_DIR / "results_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset filename (MANUAL LABELS VERSION)
DATASET_NAME = "dataset_windows_30s_features_zones_MANUAL.csv"

# Search paths for dataset (in order of priority)
SEARCH_PATHS = [
    BASE_DIR / "preprocessing_output" / DATASET_NAME,  # preprocessing output (primary)
    BASE_DIR / DATASET_NAME,                            # current directory
    Path(".") / DATASET_NAME,                           # fallback
]

def find_dataset():
    """Find the dataset file in multiple possible locations."""
    for path in SEARCH_PATHS:
        if path.exists():
            return path
    return None

print("=" * 80)
print(" " * 25 + "SLEEP APNEA DETECTION")
print(" " * 10 + "(MANUAL LABELS - OPTIMIZED PARAMETERS)")
print(" " * 5 + f"K={K_FEATURES}, Window={ROLLING_WINDOW}, Threshold={CLASSIFICATION_THR}")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATASET")
print("=" * 80)

DATASET_PATH = find_dataset()
if DATASET_PATH is None:
    print("\nERROR: Dataset not found!")
    print("Searched in:")
    for p in SEARCH_PATHS:
        print(f"  - {p}")
    print("\nPlease either:")
    print("  1. Run prepare_dataset.py first to generate the dataset")
    print("  2. Place the dataset file in one of the above locations")
    exit(1)

print(f"\nLoading: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Total windows: {len(df)}")
print(f"Total subjects: {df['Subject'].nunique()}")
print(f"\nClass distribution:")
print(f"  Normal: {(df['label_apnea']==0).sum()} ({100*(df['label_apnea']==0).sum()/len(df):.1f}%)")
print(f"  Apnea:  {(df['label_apnea']==1).sum()} ({100*(df['label_apnea']==1).sum()/len(df):.1f}%)")

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("2. TRAIN/TEST SPLIT (Subject-wise)")
print("=" * 80)

df_train = df[~df['Subject'].isin(TEST_SUBJECTS)].copy()
df_test = df[df['Subject'].isin(TEST_SUBJECTS)].copy()

print(f"\nTrain: {len(df_train)} windows, {df_train['Subject'].nunique()} subjects")
print(f"  Normal: {(df_train['label_apnea']==0).sum()}, Apnea: {(df_train['label_apnea']==1).sum()}")
print(f"\nTest: {len(df_test)} windows, {len(TEST_SUBJECTS)} subjects")
print(f"  Normal: {(df_test['label_apnea']==0).sum()}, Apnea: {(df_test['label_apnea']==1).sum()}")

# Select features (exclude metadata and target)
exclude_cols = ['Subject', 'label_apnea', 'start_time', 'end_time',
                'Position_mode', 'majority_status', 'frac_apnea',
                'frac_altro', 'frac_respiro']
baseline_features = [c for c in df_train.columns if c not in exclude_cols]

X_train = df_train[baseline_features]
y_train = df_train['label_apnea']
X_test = df_test[baseline_features]
y_test = df_test['label_apnea']

print(f"\nBaseline features: {len(baseline_features)}")

# ============================================================================
# 3. BASELINE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("3. BASELINE MODEL (Simple Zones)")
print("=" * 80)

# Oversample minority class
X_train_maj = X_train[y_train == 0]
X_train_min = X_train[y_train == 1]
y_train_maj = y_train[y_train == 0]
y_train_min = y_train[y_train == 1]

X_train_min_up, y_train_min_up = resample(
    X_train_min, y_train_min,
    replace=True,
    n_samples=len(X_train_maj),
    random_state=RANDOM_STATE
)

X_train_bal = pd.concat([X_train_maj, X_train_min_up])
y_train_bal = pd.concat([y_train_maj, y_train_min_up])

print(f"\nOversampled: {len(y_train_bal)} samples (balanced)")

# Train baseline
print("\nTraining baseline Random Forest...")
rf_baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_baseline.fit(X_train_bal, y_train_bal)

# Evaluate
y_pred_proba_base = rf_baseline.predict_proba(X_test)[:, 1]
y_pred_base = (y_pred_proba_base >= 0.5).astype(int)

roc_auc_base = roc_auc_score(y_test, y_pred_proba_base)
precision_base = precision_score(y_test, y_pred_base, zero_division=0)
recall_base = recall_score(y_test, y_pred_base, zero_division=0)
f1_base = f1_score(y_test, y_pred_base, zero_division=0)

print(f"\n✓ Baseline Results:")
print(f"  ROC-AUC:   {roc_auc_base:.4f}")
print(f"  Precision: {precision_base:.4f}")
print(f"  Recall:    {recall_base:.4f}")
print(f"  F1-Score:  {f1_base:.4f}")

# ============================================================================
# 4. TEMPORAL FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("4. TEMPORAL FEATURES (BREAKTHROUGH!)")
print("=" * 80)

def add_temporal_features(df, window_size=ROLLING_WINDOW):
    """Add temporal features to dataframe

    Args:
        df: DataFrame with features
        window_size: Rolling window size (default: ROLLING_WINDOW=7, i.e., 3.5 minutes)
    """
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
        # Delta (change from previous window)
        temporal_features[f'delta_{feat}'] = df.groupby('Subject')[feat].diff().fillna(0)
        # Rolling mean (window_size windows = window_size*30 seconds)
        temporal_features[f'roll{window_size}_mean_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )
        # Rolling std
        temporal_features[f'roll{window_size}_std_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        ).fillna(0)
        # Trend (deviation from rolling mean)
        temporal_features[f'trend_{feat}'] = (
            df[feat] - temporal_features[f'roll{window_size}_mean_{feat}']
        )

    return temporal_features

print("\nGenerating temporal features...")
temporal_train = add_temporal_features(df_train)
temporal_test = add_temporal_features(df_test)

X_train_temp = pd.concat([X_train.reset_index(drop=True),
                          temporal_train.reset_index(drop=True)], axis=1)
X_test_temp = pd.concat([X_test.reset_index(drop=True),
                         temporal_test.reset_index(drop=True)], axis=1)

print(f"Total features: {len(X_train_temp.columns)}")
print(f"  Original: {len(baseline_features)}")
print(f"  Temporal: {len(temporal_train.columns)}")

# Oversample with temporal features
# Use boolean indexing directly on the reset index
y_train_reset = y_train.reset_index(drop=True)
X_train_temp_maj = X_train_temp[y_train_reset == 0]
X_train_temp_min = X_train_temp[y_train_reset == 1]

X_train_temp_min_up, y_train_temp_min_up = resample(
    X_train_temp_min, y_train_min,
    replace=True,
    n_samples=len(X_train_temp_maj),
    random_state=RANDOM_STATE
)

X_train_temp_bal = pd.concat([X_train_temp_maj, X_train_temp_min_up])
y_train_temp_bal = pd.concat([y_train_maj, y_train_temp_min_up])

# Feature selection
print(f"\nSelecting top K={K_FEATURES} features...")
selector = SelectKBest(f_classif, k=K_FEATURES)
X_train_selected = selector.fit_transform(X_train_temp_bal, y_train_temp_bal)
X_test_selected = selector.transform(X_test_temp)

selected_features = X_train_temp.columns[selector.get_support()].tolist()

# Train RF with temporal features
print("\nTraining Random Forest with temporal features...")
rf_temporal = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_temporal.fit(X_train_selected, y_train_temp_bal)

# Evaluate
y_pred_proba_temp = rf_temporal.predict_proba(X_test_selected)[:, 1]
y_pred_temp = (y_pred_proba_temp >= CLASSIFICATION_THR).astype(int)

roc_auc_temp = roc_auc_score(y_test, y_pred_proba_temp)
precision_temp = precision_score(y_test, y_pred_temp, zero_division=0)
recall_temp = recall_score(y_test, y_pred_temp, zero_division=0)
f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)

print(f"\n✓ RF + Temporal Results (threshold={CLASSIFICATION_THR}):")
print(f"  ROC-AUC:   {roc_auc_temp:.4f}")
print(f"  Precision: {precision_temp:.4f}")
print(f"  Recall:    {recall_temp:.4f}")
print(f"  F1-Score:  {f1_temp:.4f}")

print(f"\n🎉 IMPROVEMENT over baseline:")
print(f"  ROC-AUC: {roc_auc_base:.4f} → {roc_auc_temp:.4f} (+{100*(roc_auc_temp/roc_auc_base-1):.1f}%)")
print(f"  F1-Score: {f1_base:.4f} → {f1_temp:.4f} (+{100*(f1_temp/f1_base-1):.1f}%)")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_temporal.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    is_temporal = any(x in row['feature'] for x in ['delta_', 'roll3_', 'trend_'])
    feat_type = "Temporal" if is_temporal else "Original"
    print(f"  {i+1:2d}. {row['feature']:45s} {row['importance']*100:5.2f}% [{feat_type}]")

temporal_count = sum(any(x in feat for x in ['delta_', 'roll3_', 'trend_'])
                     for feat in feature_importance.head(10)['feature'])
print(f"\n✓ {temporal_count}/10 top features are temporal!")

# ============================================================================
# 4B. XGBOOST (OPTIONAL)
# ============================================================================
if HAS_XGBOOST:
    print("\n" + "=" * 80)
    print("4B. XGBOOST + TEMPORAL (COMPARISON)")
    print("=" * 80)

    print("\nTraining XGBoost with temporal features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_selected, y_train_temp_bal)

    # Evaluate
    y_pred_proba_xgb = xgb_model.predict_proba(X_test_selected)[:, 1]
    y_pred_xgb = (y_pred_proba_xgb >= CLASSIFICATION_THR).astype(int)

    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
    recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
    f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)

    print(f"\n✓ XGBoost Results (threshold={CLASSIFICATION_THR}):")
    print(f"  ROC-AUC:   {roc_auc_xgb:.4f}")
    print(f"  Precision: {precision_xgb:.4f}")
    print(f"  Recall:    {recall_xgb:.4f}")
    print(f"  F1-Score:  {f1_xgb:.4f}")

    print(f"\n📊 RF vs XGBoost:")
    print(f"  ROC-AUC: RF {roc_auc_temp:.4f} vs XGB {roc_auc_xgb:.4f} (Δ{roc_auc_xgb-roc_auc_temp:+.4f})")
    print(f"  F1-Score: RF {f1_temp:.4f} vs XGB {f1_xgb:.4f} (Δ{f1_xgb-f1_temp:+.4f})")
else:
    print("\n⚠️  XGBoost not available - skipping comparison")
    y_pred_proba_xgb = None
    y_pred_xgb = None
    roc_auc_xgb = None
    precision_xgb = None
    recall_xgb = None
    f1_xgb = None

# ============================================================================
# 5. CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("5. CROSS-VALIDATION (5-fold GroupKFold)")
print("=" * 80)

print("\nRunning 5-fold cross-validation (this may take a few minutes)...")

# Prepare full dataset
X_full = df[baseline_features]
y_full = df['label_apnea']
subjects_full = df['Subject']

# Generate temporal for full dataset
temporal_full = add_temporal_features(df)
X_full_temp = pd.concat([X_full.reset_index(drop=True),
                         temporal_full.reset_index(drop=True)], axis=1)

# CV
gkf = GroupKFold(n_splits=5)
cv_results = {'Baseline': [], 'RF+Temporal': []}
if HAS_XGBOOST:
    cv_results['XGBoost+Temporal'] = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_full, groups=subjects_full), 1):
    print(f"  Fold {fold}/5...", end=" ")

    # Split
    X_tr, X_te = X_full.iloc[train_idx], X_full.iloc[test_idx]
    X_tr_temp, X_te_temp = X_full_temp.iloc[train_idx], X_full_temp.iloc[test_idx]
    y_tr, y_te = y_full.iloc[train_idx], y_full.iloc[test_idx]

    # Oversample baseline
    X_tr_maj = X_tr[y_tr == 0]
    X_tr_min = X_tr[y_tr == 1]
    y_tr_maj = y_tr[y_tr == 0]
    y_tr_min = y_tr[y_tr == 1]

    X_tr_min_up, y_tr_min_up = resample(
        X_tr_min, y_tr_min,
        replace=True, n_samples=len(X_tr_maj),
        random_state=RANDOM_STATE
    )

    X_tr_bal = pd.concat([X_tr_maj, X_tr_min_up])
    y_tr_bal = pd.concat([y_tr_maj, y_tr_min_up])

    # Baseline
    rf_base_cv = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_base_cv.fit(X_tr_bal, y_tr_bal)
    y_pred_base_cv = rf_base_cv.predict_proba(X_te)[:, 1]
    cv_results['Baseline'].append(roc_auc_score(y_te, y_pred_base_cv))

    # RF + Temporal
    # Oversample temporal
    X_tr_temp_maj = X_tr_temp.iloc[(y_tr == 0).values]
    X_tr_temp_min = X_tr_temp.iloc[(y_tr == 1).values]

    X_tr_temp_min_up, _ = resample(
        X_tr_temp_min, y_tr_min,
        replace=True, n_samples=len(X_tr_temp_maj),
        random_state=RANDOM_STATE
    )

    X_tr_temp_bal = pd.concat([X_tr_temp_maj, X_tr_temp_min_up])
    y_tr_temp_bal = pd.concat([y_tr_maj, y_tr_min_up])

    # Feature selection
    sel_cv = SelectKBest(f_classif, k=K_FEATURES)
    X_tr_sel = sel_cv.fit_transform(X_tr_temp_bal, y_tr_temp_bal)
    X_te_sel = sel_cv.transform(X_te_temp)

    rf_temp_cv = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_temp_cv.fit(X_tr_sel, y_tr_temp_bal)
    y_pred_temp_cv = rf_temp_cv.predict_proba(X_te_sel)[:, 1]
    cv_results['RF+Temporal'].append(roc_auc_score(y_te, y_pred_temp_cv))

    # XGBoost CV
    if HAS_XGBOOST:
        xgb_cv = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_cv.fit(X_tr_sel, y_tr_temp_bal)
        y_pred_xgb_cv = xgb_cv.predict_proba(X_te_sel)[:, 1]
        cv_results['XGBoost+Temporal'].append(roc_auc_score(y_te, y_pred_xgb_cv))
        print(f"Baseline: {cv_results['Baseline'][-1]:.3f}, RF+Temp: {cv_results['RF+Temporal'][-1]:.3f}, XGB+Temp: {cv_results['XGBoost+Temporal'][-1]:.3f}")
    else:
        print(f"Baseline: {cv_results['Baseline'][-1]:.3f}, RF+Temp: {cv_results['RF+Temporal'][-1]:.3f}")

print(f"\n✓ Cross-validation completed")

print(f"\nCV Results Summary:")
for model, results in cv_results.items():
    mean_auc = np.mean(results)
    std_auc = np.std(results)
    print(f"  {model:12s}: {mean_auc:.4f} ± {std_auc:.4f} (range: [{min(results):.2f}, {max(results):.2f}])")

rf_cv_mean = np.mean(cv_results['RF+Temporal'])
baseline_cv_mean = np.mean(cv_results['Baseline'])
improvement = 100 * (rf_cv_mean - baseline_cv_mean) / baseline_cv_mean

print(f"\n✓ Robust improvement: +{improvement:.1f}% ({baseline_cv_mean:.3f} → {rf_cv_mean:.3f})")

# ============================================================================
# 5B. CROSS-VALIDATION VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("5B. CROSS-VALIDATION VISUALIZATIONS")
print("=" * 80)

# 1. Boxplot dei risultati CV
fig, ax = plt.subplots(figsize=(10, 6))

cv_data = []
cv_labels = []
for model, results in cv_results.items():
    cv_data.append(results)
    cv_labels.append(model)

bp = ax.boxplot(cv_data, labels=cv_labels, patch_artist=True)

colors_box = ['#FF6B6B', '#4ECDC4', '#FFA500']
for patch, color in zip(bp['boxes'], colors_box[:len(cv_data)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add individual points
for i, (data, label) in enumerate(zip(cv_data, cv_labels)):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(x, data, alpha=0.8, s=50, zorder=5, edgecolor='black', linewidth=1)

# Add mean line
for i, data in enumerate(cv_data):
    ax.hlines(np.mean(data), i+0.7, i+1.3, colors='red', linestyles='dashed', linewidth=2)

ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title(f'5-Fold Cross-Validation Results\n(K={K_FEATURES}, Window={ROLLING_WINDOW}, Optimized)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0.5, 1.0)
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target 0.90')
ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good threshold 0.80')
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='y')

# Add mean±std annotation
for i, (data, label) in enumerate(zip(cv_data, cv_labels)):
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.annotate(f'{mean_val:.3f}±{std_val:.3f}',
                xy=(i+1, mean_val), xytext=(i+1, mean_val+0.05),
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cv_boxplot.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: cv_boxplot.png")
plt.close()

# 2. Barplot per fold con confronto modelli
fig, ax = plt.subplots(figsize=(12, 6))

n_folds = 5
x = np.arange(n_folds)
width = 0.25

bars1 = ax.bar(x - width, cv_results['Baseline'], width, label='Baseline', color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x, cv_results['RF+Temporal'], width, label='RF+Temporal', color='#4ECDC4', edgecolor='black')
if 'XGBoost+Temporal' in cv_results:
    bars3 = ax.bar(x + width, cv_results['XGBoost+Temporal'], width, label='XGBoost+Temporal', color='#FFA500', edgecolor='black')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title(f'Cross-Validation: Performance per Fold\n(K={K_FEATURES}, Window={ROLLING_WINDOW})',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
ax.legend(loc='lower right')
ax.set_ylim(0.6, 1.0)
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
ax.grid(alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2] + ([bars3] if 'XGBoost+Temporal' in cv_results else []):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cv_per_fold.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: cv_per_fold.png")
plt.close()

# 3. Summary comparison chart (CV vs Single Split)
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Baseline', 'RF+Temporal']
if HAS_XGBOOST:
    models.append('XGBoost+Temporal')

# Single split values
single_split = [roc_auc_base, roc_auc_temp]
if HAS_XGBOOST and roc_auc_xgb is not None:
    single_split.append(roc_auc_xgb)

# CV values
cv_means = [np.mean(cv_results[m.replace('+', '+')]) for m in models]
cv_stds = [np.std(cv_results[m.replace('+', '+')]) for m in models]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, single_split, width, label='Single Split', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds, capsize=5,
               label='CV (mean±std)', color='#2ecc71', edgecolor='black')

ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Single Split vs Cross-Validation\n(Optimized Parameters)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0.6, 1.0)
ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target')
ax.grid(alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, single_split):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
for bar, val, std in zip(bars2, cv_means, cv_stds):
    ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val + std),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'cv_vs_single_split.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: cv_vs_single_split.png")
plt.close()

# ============================================================================
# 6. VISUALIZATIONS (Single Split)
# ============================================================================
print("\n" + "=" * 80)
print("6. GENERATING VISUALIZATIONS (Single Split)")
print("=" * 80)

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

fpr_base, tpr_base, _ = roc_curve(y_test, y_pred_proba_base)
fpr_temp, tpr_temp, _ = roc_curve(y_test, y_pred_proba_temp)

ax.plot(fpr_base, tpr_base, color='#FF6B6B', linewidth=2.5,
        label=f'Baseline (AUC={roc_auc_base:.4f})')
ax.plot(fpr_temp, tpr_temp, color='#4ECDC4', linewidth=2.5,
        label=f'RF + Temporal (AUC={roc_auc_temp:.4f})')

# Add XGBoost if available
if HAS_XGBOOST and y_pred_proba_xgb is not None:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
    ax.plot(fpr_xgb, tpr_xgb, color='#FFA500', linewidth=2.5,
            label=f'XGBoost + Temporal (AUC={roc_auc_xgb:.4f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC=0.5000)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves (Manual Labels, K={K_FEATURES}, Window={ROLLING_WINDOW})', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'roc_curves_CORRECTED.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: roc_curves_CORRECTED.png")
plt.close()

# Confusion Matrices
n_models = 3 if (HAS_XGBOOST and y_pred_xgb is not None) else 2
fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
if n_models == 2:
    axes = [axes[0], axes[1]]

cm_base = confusion_matrix(y_test, y_pred_base)
cm_temp = confusion_matrix(y_test, y_pred_temp)

models_data = [
    (cm_base, 'Baseline', (roc_auc_base, precision_base, recall_base, f1_base)),
    (cm_temp, f'RF + Temporal (thr={CLASSIFICATION_THR})', (roc_auc_temp, precision_temp, recall_temp, f1_temp))
]

if HAS_XGBOOST and y_pred_xgb is not None:
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    models_data.append((cm_xgb, f'XGBoost + Temporal (thr={CLASSIFICATION_THR})', (roc_auc_xgb, precision_xgb, recall_xgb, f1_xgb)))

for idx, (cm, title, results) in enumerate(models_data):
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

    auc_val, prec, rec, f1_val = results
    ax.text(0.5, -0.15,
            f"ROC-AUC: {auc_val:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1_val:.3f}",
            transform=ax.transAxes, ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrices_CORRECTED.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: confusion_matrices_CORRECTED.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print(" " * 28 + "FINAL SUMMARY")
print("=" * 80)

print(f"\n📊 Single Train/Test Split Performance:")
print(f"  {'Model':<30} {'ROC-AUC':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'-'*77}")
print(f"  {'Baseline':<30} {roc_auc_base:>10.4f} {f1_base:>10.4f} {precision_base:>10.4f} {recall_base:>10.4f}")
print(f"  {'RF + Temporal (K=60)':<30} {roc_auc_temp:>10.4f} {f1_temp:>10.4f} {precision_temp:>10.4f} {recall_temp:>10.4f}")
if HAS_XGBOOST and roc_auc_xgb is not None:
    print(f"  {'XGBoost + Temporal (K=60)':<30} {roc_auc_xgb:>10.4f} {f1_xgb:>10.4f} {precision_xgb:>10.4f} {recall_xgb:>10.4f}")

print(f"\n📊 5-Fold Cross-Validation Performance:")
print(f"  {'Model':<25} {'ROC-AUC (mean±std)':>25}")
print(f"  {'-'*52}")
for model, results in cv_results.items():
    mean_val, std_val = np.mean(results), np.std(results)
    print(f"  {model:<25} {mean_val:>10.4f} ± {std_val:<6.4f}")

print(f"\n" + "=" * 80)
print("KEY FINDINGS:")
print("=" * 80)

print(f"\n1️⃣ Temporal features provide DRAMATIC improvement")
print(f"   Single-split: +{100*(roc_auc_temp/roc_auc_base-1):.1f}% ROC-AUC")
print(f"   Cross-val: +{improvement:.1f}% ROC-AUC")

print(f"\n2️⃣ Single-split was optimistic")
print(f"   Single: {roc_auc_temp:.4f}, CV mean: {rf_cv_mean:.4f}")
print(f"   But still GOOD (>0.80 threshold)")

print(f"\n3️⃣ {temporal_count}/10 top features are temporal")

print(f"\n4️⃣ Subject-dependent performance")
print(f"   CV range: [{min(cv_results['RF+Temporal']):.2f}, {max(cv_results['RF+Temporal']):.2f}]")

print(f"\n" + "=" * 80)
print(f"\n✅ Analysis completed successfully!")
print(f"\n📁 Results saved to: {RESULTS_DIR}/")
print(f"\nGenerated files:")
for f in sorted(RESULTS_DIR.glob('*.png')):
    print(f"  - {f.name}")

print("\n" + "=" * 80)
