#!/usr/bin/env python3
"""
Hyperparameter Tuning for Sleep Apnea Detection
VERSIONE PER DATASET CON ETICHETTE MANUALI

Esplora diverse configurazioni di:
- K (numero di feature selezionate): 40, 50, 60, 70, 80
- Rolling window size: 3, 5, 7
- N_estimators: 200, 400, 600
- XGBoost max_depth: 4, 6, 8
- Threshold di classificazione: 0.25, 0.30, 0.35, 0.40
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
RESULTS_DIR = BASE_DIR / "tuning_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset
DATASET_NAME = "dataset_windows_30s_features_zones_MANUAL.csv"
DATASET_PATH = BASE_DIR / "preprocessing_output" / DATASET_NAME

print("=" * 80)
print(" " * 20 + "HYPERPARAMETER TUNING")
print(" " * 15 + "(Manual Labels Dataset)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"    Dataset: {len(df)} windows, {df['Subject'].nunique()} subjects")
print(f"    Apnea: {(df['label_apnea']==1).sum()} ({100*(df['label_apnea']==1).sum()/len(df):.1f}%)")

# Select features
exclude_cols = ['Subject', 'label_apnea', 'start_time', 'end_time',
                'Position_mode', 'majority_status', 'frac_apnea',
                'frac_altro', 'frac_respiro']
baseline_features = [c for c in df.columns if c not in exclude_cols]

X_full = df[baseline_features]
y_full = df['label_apnea']
subjects_full = df['Subject']

print(f"    Baseline features: {len(baseline_features)}")

# ============================================================================
# 2. TEMPORAL FEATURE GENERATION FUNCTION
# ============================================================================
def add_temporal_features(df, window_size=3):
    """Add temporal features with configurable window size"""
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
        # Delta (always window=1)
        temporal_features[f'delta_{feat}'] = df.groupby('Subject')[feat].diff().fillna(0)
        # Rolling mean
        temporal_features[f'roll{window_size}_mean_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )
        # Rolling std
        temporal_features[f'roll{window_size}_std_{feat}'] = (
            df.groupby('Subject')[feat]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).std())
        ).fillna(0)
        # Trend
        temporal_features[f'trend_{feat}'] = (
            df[feat] - temporal_features[f'roll{window_size}_mean_{feat}']
        )

    return temporal_features

# ============================================================================
# 3. CROSS-VALIDATION FUNCTION
# ============================================================================
def run_cv(X_full, y_full, subjects_full, temporal_full,
           k_features, n_estimators, model_type='rf',
           xgb_max_depth=6, threshold=0.30, n_splits=5):
    """Run cross-validation with given parameters"""

    X_full_temp = pd.concat([X_full.reset_index(drop=True),
                             temporal_full.reset_index(drop=True)], axis=1)

    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for train_idx, test_idx in gkf.split(X_full, y_full, groups=subjects_full):
        # Split
        X_tr = X_full.iloc[train_idx]
        X_tr_temp = X_full_temp.iloc[train_idx]
        X_te_temp = X_full_temp.iloc[test_idx]
        y_tr = y_full.iloc[train_idx]
        y_te = y_full.iloc[test_idx]

        # Oversample
        X_tr_temp_maj = X_tr_temp.iloc[(y_tr == 0).values]
        X_tr_temp_min = X_tr_temp.iloc[(y_tr == 1).values]
        y_tr_maj = y_tr[y_tr == 0]
        y_tr_min = y_tr[y_tr == 1]

        X_tr_temp_min_up, y_tr_min_up = resample(
            X_tr_temp_min, y_tr_min,
            replace=True, n_samples=len(X_tr_temp_maj),
            random_state=RANDOM_STATE
        )

        X_tr_temp_bal = pd.concat([X_tr_temp_maj, X_tr_temp_min_up])
        y_tr_temp_bal = pd.concat([y_tr_maj, y_tr_min_up])

        # Feature selection
        k = min(k_features, X_tr_temp_bal.shape[1])
        selector = SelectKBest(f_classif, k=k)
        X_tr_sel = selector.fit_transform(X_tr_temp_bal, y_tr_temp_bal)
        X_te_sel = selector.transform(X_te_temp)

        # Train model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        else:  # xgboost
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            )

        model.fit(X_tr_sel, y_tr_temp_bal)

        # Evaluate
        y_pred_proba = model.predict_proba(X_te_sel)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        roc_auc = roc_auc_score(y_te, y_pred_proba)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        precision = precision_score(y_te, y_pred, zero_division=0)
        recall = recall_score(y_te, y_pred, zero_division=0)

        results.append({
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

    return {
        'roc_auc_mean': np.mean([r['roc_auc'] for r in results]),
        'roc_auc_std': np.std([r['roc_auc'] for r in results]),
        'f1_mean': np.mean([r['f1'] for r in results]),
        'f1_std': np.std([r['f1'] for r in results]),
        'precision_mean': np.mean([r['precision'] for r in results]),
        'recall_mean': np.mean([r['recall'] for r in results]),
    }

# ============================================================================
# 4. GRID SEARCH
# ============================================================================
print("\n[2] Starting hyperparameter search...")

# Parameters to explore
K_VALUES = [40, 50, 60, 70, 80]
WINDOW_SIZES = [3, 5, 7]
N_ESTIMATORS_VALUES = [200, 400]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40]
XGB_MAX_DEPTHS = [4, 6, 8]

# Store all results
all_results = []

# Pre-generate temporal features for each window size
print("\n    Generating temporal features for different window sizes...")
temporal_cache = {}
for ws in WINDOW_SIZES:
    temporal_cache[ws] = add_temporal_features(df, window_size=ws)
    print(f"      Window size {ws}: {len(temporal_cache[ws].columns)} temporal features")

# Total combinations
total_rf = len(K_VALUES) * len(WINDOW_SIZES) * len(N_ESTIMATORS_VALUES) * len(THRESHOLDS)
total_xgb = len(K_VALUES) * len(WINDOW_SIZES) * len(N_ESTIMATORS_VALUES) * len(THRESHOLDS) * len(XGB_MAX_DEPTHS) if HAS_XGBOOST else 0
print(f"\n    Total RF configurations: {total_rf}")
print(f"    Total XGBoost configurations: {total_xgb}")
print(f"    Total: {total_rf + total_xgb}")

# Random Forest Grid Search
print("\n" + "=" * 80)
print("RANDOM FOREST TUNING")
print("=" * 80)

count = 0
for k, ws, n_est, thr in product(K_VALUES, WINDOW_SIZES, N_ESTIMATORS_VALUES, THRESHOLDS):
    count += 1
    print(f"\r    [{count}/{total_rf}] K={k}, window={ws}, n_est={n_est}, thr={thr}...", end="", flush=True)

    result = run_cv(
        X_full, y_full, subjects_full, temporal_cache[ws],
        k_features=k, n_estimators=n_est, model_type='rf',
        threshold=thr
    )

    all_results.append({
        'model': 'RF',
        'k_features': k,
        'window_size': ws,
        'n_estimators': n_est,
        'threshold': thr,
        'xgb_max_depth': None,
        **result
    })

print(f"\n    ✓ RF tuning completed")

# XGBoost Grid Search
if HAS_XGBOOST:
    print("\n" + "=" * 80)
    print("XGBOOST TUNING")
    print("=" * 80)

    count = 0
    for k, ws, n_est, thr, depth in product(K_VALUES, WINDOW_SIZES, N_ESTIMATORS_VALUES, THRESHOLDS, XGB_MAX_DEPTHS):
        count += 1
        print(f"\r    [{count}/{total_xgb}] K={k}, window={ws}, n_est={n_est}, thr={thr}, depth={depth}...", end="", flush=True)

        result = run_cv(
            X_full, y_full, subjects_full, temporal_cache[ws],
            k_features=k, n_estimators=n_est, model_type='xgb',
            xgb_max_depth=depth, threshold=thr
        )

        all_results.append({
            'model': 'XGBoost',
            'k_features': k,
            'window_size': ws,
            'n_estimators': n_est,
            'threshold': thr,
            'xgb_max_depth': depth,
            **result
        })

    print(f"\n    ✓ XGBoost tuning completed")

# ============================================================================
# 5. ANALYZE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / 'tuning_results_all.csv', index=False)
print(f"\n✓ Saved all results to tuning_results_all.csv ({len(results_df)} configurations)")

# Best RF configuration
rf_results = results_df[results_df['model'] == 'RF']
best_rf_idx = rf_results['roc_auc_mean'].idxmax()
best_rf = rf_results.loc[best_rf_idx]

print(f"\n" + "-" * 60)
print("BEST RANDOM FOREST CONFIGURATION:")
print("-" * 60)
print(f"  K features:    {int(best_rf['k_features'])}")
print(f"  Window size:   {int(best_rf['window_size'])}")
print(f"  N estimators:  {int(best_rf['n_estimators'])}")
print(f"  Threshold:     {best_rf['threshold']}")
print(f"\n  ROC-AUC:       {best_rf['roc_auc_mean']:.4f} ± {best_rf['roc_auc_std']:.4f}")
print(f"  F1-Score:      {best_rf['f1_mean']:.4f}")
print(f"  Precision:     {best_rf['precision_mean']:.4f}")
print(f"  Recall:        {best_rf['recall_mean']:.4f}")

# Compare with default (K=60, window=3, n_est=400, thr=0.30)
default_rf = rf_results[
    (rf_results['k_features'] == 60) &
    (rf_results['window_size'] == 3) &
    (rf_results['n_estimators'] == 400) &
    (rf_results['threshold'] == 0.30)
]
if len(default_rf) > 0:
    default_rf = default_rf.iloc[0]
    improvement = (best_rf['roc_auc_mean'] - default_rf['roc_auc_mean']) / default_rf['roc_auc_mean'] * 100
    print(f"\n  vs Default (K=60, w=3, n=400, t=0.30):")
    print(f"     Default ROC-AUC: {default_rf['roc_auc_mean']:.4f}")
    print(f"     Improvement: {improvement:+.2f}%")

# Best XGBoost configuration
if HAS_XGBOOST:
    xgb_results = results_df[results_df['model'] == 'XGBoost']
    best_xgb_idx = xgb_results['roc_auc_mean'].idxmax()
    best_xgb = xgb_results.loc[best_xgb_idx]

    print(f"\n" + "-" * 60)
    print("BEST XGBOOST CONFIGURATION:")
    print("-" * 60)
    print(f"  K features:    {int(best_xgb['k_features'])}")
    print(f"  Window size:   {int(best_xgb['window_size'])}")
    print(f"  N estimators:  {int(best_xgb['n_estimators'])}")
    print(f"  Max depth:     {int(best_xgb['xgb_max_depth'])}")
    print(f"  Threshold:     {best_xgb['threshold']}")
    print(f"\n  ROC-AUC:       {best_xgb['roc_auc_mean']:.4f} ± {best_xgb['roc_auc_std']:.4f}")
    print(f"  F1-Score:      {best_xgb['f1_mean']:.4f}")
    print(f"  Precision:     {best_xgb['precision_mean']:.4f}")
    print(f"  Recall:        {best_xgb['recall_mean']:.4f}")

# ============================================================================
# 6. TOP 10 CONFIGURATIONS
# ============================================================================
print(f"\n" + "-" * 60)
print("TOP 10 CONFIGURATIONS (by ROC-AUC):")
print("-" * 60)

top10 = results_df.nlargest(10, 'roc_auc_mean')
print(f"\n{'#':>2} {'Model':>8} {'K':>3} {'Win':>4} {'N_est':>6} {'Thr':>5} {'Depth':>5} {'ROC-AUC':>12} {'F1':>8}")
print("-" * 70)
for i, (_, row) in enumerate(top10.iterrows(), 1):
    depth_str = f"{int(row['xgb_max_depth'])}" if pd.notna(row['xgb_max_depth']) else "-"
    print(f"{i:>2} {row['model']:>8} {int(row['k_features']):>3} {int(row['window_size']):>4} "
          f"{int(row['n_estimators']):>6} {row['threshold']:>5.2f} {depth_str:>5} "
          f"{row['roc_auc_mean']:>6.4f}±{row['roc_auc_std']:.3f} {row['f1_mean']:>8.4f}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# 1. Heatmap: K vs Window Size (RF only, aggregated over other params)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF Heatmap
rf_pivot = rf_results.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().unstack()
sns.heatmap(rf_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[0])
axes[0].set_title('RF: Best ROC-AUC by K and Window Size', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Window Size')
axes[0].set_ylabel('K Features')

# XGBoost Heatmap (if available)
if HAS_XGBOOST:
    xgb_pivot = xgb_results.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().unstack()
    sns.heatmap(xgb_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('XGBoost: Best ROC-AUC by K and Window Size', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Window Size')
    axes[1].set_ylabel('K Features')
else:
    axes[1].text(0.5, 0.5, 'XGBoost not available', ha='center', va='center')
    axes[1].set_title('XGBoost: Not Available')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'heatmap_k_vs_window.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: heatmap_k_vs_window.png")
plt.close()

# 2. Line plot: ROC-AUC vs K for different window sizes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ws in WINDOW_SIZES:
    rf_ws = rf_results[rf_results['window_size'] == ws].groupby('k_features')['roc_auc_mean'].max()
    axes[0].plot(rf_ws.index, rf_ws.values, marker='o', label=f'Window={ws}')

axes[0].set_xlabel('K Features')
axes[0].set_ylabel('ROC-AUC (CV mean)')
axes[0].set_title('RF: ROC-AUC vs K Features', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

if HAS_XGBOOST:
    for ws in WINDOW_SIZES:
        xgb_ws = xgb_results[xgb_results['window_size'] == ws].groupby('k_features')['roc_auc_mean'].max()
        axes[1].plot(xgb_ws.index, xgb_ws.values, marker='s', label=f'Window={ws}')

    axes[1].set_xlabel('K Features')
    axes[1].set_ylabel('ROC-AUC (CV mean)')
    axes[1].set_title('XGBoost: ROC-AUC vs K Features', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lineplot_k_features.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: lineplot_k_features.png")
plt.close()

# 3. Threshold analysis
fig, ax = plt.subplots(figsize=(10, 6))

for thr in THRESHOLDS:
    rf_thr = rf_results[rf_results['threshold'] == thr]
    best_per_thr = rf_thr.groupby('threshold').agg({
        'roc_auc_mean': 'max',
        'f1_mean': 'max',
        'precision_mean': 'max',
        'recall_mean': 'max'
    })

rf_by_thr = rf_results.groupby('threshold').agg({
    'roc_auc_mean': 'max',
    'f1_mean': 'max',
    'precision_mean': 'max',
    'recall_mean': 'max'
}).reset_index()

x = np.arange(len(THRESHOLDS))
width = 0.2

ax.bar(x - 1.5*width, rf_by_thr['roc_auc_mean'], width, label='ROC-AUC', color='#4ECDC4')
ax.bar(x - 0.5*width, rf_by_thr['f1_mean'], width, label='F1', color='#FF6B6B')
ax.bar(x + 0.5*width, rf_by_thr['precision_mean'], width, label='Precision', color='#45B7D1')
ax.bar(x + 1.5*width, rf_by_thr['recall_mean'], width, label='Recall', color='#96CEB4')

ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('RF: Metrics by Classification Threshold', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(THRESHOLDS)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: threshold_analysis.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print(" " * 25 + "FINAL SUMMARY")
print("=" * 80)

print(f"\n📊 Configurations tested: {len(results_df)}")
print(f"   - Random Forest: {len(rf_results)}")
if HAS_XGBOOST:
    print(f"   - XGBoost: {len(xgb_results)}")

print(f"\n🏆 BEST OVERALL:")
best_overall = results_df.loc[results_df['roc_auc_mean'].idxmax()]
print(f"   Model: {best_overall['model']}")
print(f"   K={int(best_overall['k_features'])}, Window={int(best_overall['window_size'])}, "
      f"N_est={int(best_overall['n_estimators'])}, Threshold={best_overall['threshold']}")
if pd.notna(best_overall['xgb_max_depth']):
    print(f"   Max depth={int(best_overall['xgb_max_depth'])}")
print(f"\n   ROC-AUC: {best_overall['roc_auc_mean']:.4f} ± {best_overall['roc_auc_std']:.4f}")
print(f"   F1-Score: {best_overall['f1_mean']:.4f}")

# Compare with original default results
print(f"\n📈 COMPARISON WITH ORIGINAL SETTINGS:")
print(f"   Original (K=60, w=3): ROC-AUC = 0.808")
print(f"   Best found:           ROC-AUC = {best_overall['roc_auc_mean']:.4f}")
improvement_pct = (best_overall['roc_auc_mean'] - 0.808) / 0.808 * 100
print(f"   Improvement: {improvement_pct:+.2f}%")

print(f"\n📁 Results saved to: {RESULTS_DIR}/")
print(f"\nGenerated files:")
for f in sorted(RESULTS_DIR.glob('*')):
    print(f"  - {f.name}")

print("\n" + "=" * 80)
print("✅ Hyperparameter tuning completed!")
print("=" * 80)
