#!/usr/bin/env python3
"""
Visualizzazione dei risultati dell'Hyperparameter Tuning
per Sleep Apnea Detection (Dataset con etichette manuali)

Genera grafici che mostrano il processo di ottimizzazione
e come si è arrivati ai parametri ottimali:
- K_FEATURES: 70
- ROLLING_WINDOW: 7
- THRESHOLD: 0.25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
TUNING_DIR = BASE_DIR / "tuning_results"
FIGURES_DIR = BASE_DIR / "tuning_figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
print("Loading tuning results...")
df = pd.read_csv(TUNING_DIR / "tuning_results_all.csv")
print(f"Loaded {len(df)} configurations")

# Separate RF and XGBoost
rf_df = df[df['model'] == 'RF'].copy()
xgb_df = df[df['model'] == 'XGBoost'].copy()

# Best configuration
best_idx = df['roc_auc_mean'].idxmax()
best = df.loc[best_idx]
print(f"\nBest config: {best['model']}, K={int(best['k_features'])}, "
      f"Window={int(best['window_size'])}, Threshold={best['threshold']}")
print(f"ROC-AUC: {best['roc_auc_mean']:.4f}")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'RF': '#2ecc71', 'XGBoost': '#e74c3c'}

# ============================================================================
# FIGURA 1: Effetto della Window Size (il fattore più importante)
# ============================================================================
print("\n[1/6] Generating: Window Size Effect...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF - Window size effect
rf_by_window = rf_df.groupby('window_size')['roc_auc_mean'].agg(['mean', 'std', 'max']).reset_index()

ax = axes[0]
bars = ax.bar(rf_by_window['window_size'].astype(str), rf_by_window['max'],
              color=['#bdc3c7', '#95a5a6', '#2ecc71'], edgecolor='black', linewidth=1.5)
ax.errorbar(rf_by_window['window_size'].astype(str), rf_by_window['mean'],
            yerr=rf_by_window['std'], fmt='o', color='black', capsize=5, label='Mean ± Std')

# Annotate best
ax.annotate('BEST\n+11%', xy=(2, rf_by_window['max'].iloc[2]),
            xytext=(2, rf_by_window['max'].iloc[2] + 0.02),
            ha='center', fontsize=11, fontweight='bold', color='#27ae60')

ax.set_xlabel('Rolling Window Size (×30 sec)', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Random Forest: Effetto della Window Size', fontsize=14, fontweight='bold')
ax.set_ylim(0.75, 0.95)
ax.legend()

# Add time labels
for i, ws in enumerate([3, 5, 7]):
    ax.text(i, 0.76, f'{ws*30}s\n({ws*0.5:.1f}min)', ha='center', fontsize=9, color='gray')

# XGBoost - Window size effect
xgb_by_window = xgb_df.groupby('window_size')['roc_auc_mean'].agg(['mean', 'std', 'max']).reset_index()

ax = axes[1]
bars = ax.bar(xgb_by_window['window_size'].astype(str), xgb_by_window['max'],
              color=['#f5b7b1', '#f1948a', '#e74c3c'], edgecolor='black', linewidth=1.5)
ax.errorbar(xgb_by_window['window_size'].astype(str), xgb_by_window['mean'],
            yerr=xgb_by_window['std'], fmt='o', color='black', capsize=5, label='Mean ± Std')

ax.set_xlabel('Rolling Window Size (×30 sec)', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('XGBoost: Effetto della Window Size', fontsize=14, fontweight='bold')
ax.set_ylim(0.75, 0.95)
ax.legend()

for i, ws in enumerate([3, 5, 7]):
    ax.text(i, 0.76, f'{ws*30}s\n({ws*0.5:.1f}min)', ha='center', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '1_window_size_effect.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 1_window_size_effect.png")
plt.close()

# ============================================================================
# FIGURA 2: Effetto di K (numero di feature selezionate)
# ============================================================================
print("[2/6] Generating: K Features Effect...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF - K effect (for window=7 only, the best window)
rf_w7 = rf_df[rf_df['window_size'] == 7]
rf_by_k = rf_w7.groupby('k_features')['roc_auc_mean'].agg(['mean', 'std', 'max']).reset_index()

ax = axes[0]
ax.plot(rf_by_k['k_features'], rf_by_k['max'], 'o-', color='#2ecc71',
        linewidth=2.5, markersize=10, label='Best ROC-AUC')
ax.fill_between(rf_by_k['k_features'],
                rf_by_k['mean'] - rf_by_k['std'],
                rf_by_k['mean'] + rf_by_k['std'],
                alpha=0.3, color='#2ecc71', label='Mean ± Std')

# Mark optimum
best_k_idx = rf_by_k['max'].idxmax()
best_k = rf_by_k.loc[best_k_idx]
ax.axvline(x=best_k['k_features'], color='red', linestyle='--', alpha=0.7)
ax.annotate(f"Optimal K={int(best_k['k_features'])}\nROC-AUC={best_k['max']:.3f}",
            xy=(best_k['k_features'], best_k['max']),
            xytext=(best_k['k_features']+5, best_k['max']-0.01),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('Number of Selected Features (K)', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('RF (Window=7): Effetto del Numero di Feature', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xticks([40, 50, 60, 70, 80])
ax.set_ylim(0.86, 0.92)

# XGBoost - K effect (for window=7)
xgb_w7 = xgb_df[xgb_df['window_size'] == 7]
xgb_by_k = xgb_w7.groupby('k_features')['roc_auc_mean'].agg(['mean', 'std', 'max']).reset_index()

ax = axes[1]
ax.plot(xgb_by_k['k_features'], xgb_by_k['max'], 's-', color='#e74c3c',
        linewidth=2.5, markersize=10, label='Best ROC-AUC')
ax.fill_between(xgb_by_k['k_features'],
                xgb_by_k['mean'] - xgb_by_k['std'],
                xgb_by_k['mean'] + xgb_by_k['std'],
                alpha=0.3, color='#e74c3c', label='Mean ± Std')

ax.set_xlabel('Number of Selected Features (K)', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('XGBoost (Window=7): Effetto del Numero di Feature', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xticks([40, 50, 60, 70, 80])
ax.set_ylim(0.83, 0.89)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '2_k_features_effect.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 2_k_features_effect.png")
plt.close()

# ============================================================================
# FIGURA 3: Heatmap K vs Window Size
# ============================================================================
print("[3/6] Generating: Heatmap K vs Window...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RF Heatmap
rf_pivot = rf_df.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().unstack()
ax = axes[0]
sns.heatmap(rf_pivot, annot=True, fmt='.3f', cmap='Greens', ax=ax,
            cbar_kws={'label': 'ROC-AUC'}, linewidths=0.5)
ax.set_title('Random Forest: ROC-AUC per K e Window Size', fontsize=14, fontweight='bold')
ax.set_xlabel('Window Size', fontsize=12)
ax.set_ylabel('K Features', fontsize=12)

# Highlight best cell
best_rf = rf_df.loc[rf_df['roc_auc_mean'].idxmax()]
best_k_pos = list(rf_pivot.index).index(best_rf['k_features'])
best_w_pos = list(rf_pivot.columns).index(best_rf['window_size'])
ax.add_patch(plt.Rectangle((best_w_pos, best_k_pos), 1, 1, fill=False,
                            edgecolor='red', linewidth=3))

# XGBoost Heatmap
xgb_pivot = xgb_df.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().unstack()
ax = axes[1]
sns.heatmap(xgb_pivot, annot=True, fmt='.3f', cmap='Reds', ax=ax,
            cbar_kws={'label': 'ROC-AUC'}, linewidths=0.5)
ax.set_title('XGBoost: ROC-AUC per K e Window Size', fontsize=14, fontweight='bold')
ax.set_xlabel('Window Size', fontsize=12)
ax.set_ylabel('K Features', fontsize=12)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '3_heatmap_k_window.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 3_heatmap_k_window.png")
plt.close()

# ============================================================================
# FIGURA 4: Threshold Analysis (Precision-Recall Trade-off)
# ============================================================================
print("[4/6] Generating: Threshold Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# For best config (K=70, W=7)
rf_best_config = rf_df[(rf_df['k_features'] == 70) & (rf_df['window_size'] == 7)]

ax = axes[0]
thresholds = rf_best_config['threshold'].values
precision = rf_best_config['precision_mean'].values
recall = rf_best_config['recall_mean'].values
f1 = rf_best_config['f1_mean'].values

ax.plot(thresholds, precision, 'o-', color='#3498db', linewidth=2, markersize=8, label='Precision')
ax.plot(thresholds, recall, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Recall')
ax.plot(thresholds, f1, '^-', color='#2ecc71', linewidth=2, markersize=8, label='F1-Score')

# Mark optimal threshold
ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.7)
ax.annotate('Optimal\nThreshold=0.25', xy=(0.25, 0.85), xytext=(0.30, 0.85),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('Classification Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('RF (K=70, W=7): Precision-Recall Trade-off', fontsize=14, fontweight='bold')
ax.legend()
ax.set_xlim(0.2, 0.45)
ax.set_ylim(0.3, 0.9)

# Bar chart comparison
ax = axes[1]
metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1']
thr_025 = rf_best_config[rf_best_config['threshold'] == 0.25].iloc[0]
thr_030 = rf_best_config[rf_best_config['threshold'] == 0.30].iloc[0]
thr_035 = rf_best_config[rf_best_config['threshold'] == 0.35].iloc[0]

x = np.arange(len(metrics))
width = 0.25

values_025 = [thr_025['roc_auc_mean'], thr_025['precision_mean'],
              thr_025['recall_mean'], thr_025['f1_mean']]
values_030 = [thr_030['roc_auc_mean'], thr_030['precision_mean'],
              thr_030['recall_mean'], thr_030['f1_mean']]
values_035 = [thr_035['roc_auc_mean'], thr_035['precision_mean'],
              thr_035['recall_mean'], thr_035['f1_mean']]

bars1 = ax.bar(x - width, values_025, width, label='Threshold=0.25', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x, values_030, width, label='Threshold=0.30', color='#3498db', edgecolor='black')
bars3 = ax.bar(x + width, values_035, width, label='Threshold=0.35', color='#9b59b6', edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Confronto Metriche per Threshold', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '4_threshold_analysis.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 4_threshold_analysis.png")
plt.close()

# ============================================================================
# FIGURA 5: RF vs XGBoost Comparison
# ============================================================================
print("[5/6] Generating: RF vs XGBoost Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: RF vs XGBoost for same configurations
# Match RF and XGBoost results for same (k, window, threshold)
rf_for_compare = rf_df.copy()
rf_for_compare['config'] = rf_for_compare.apply(
    lambda r: f"{int(r['k_features'])}_{int(r['window_size'])}_{r['threshold']}", axis=1)

# XGBoost: get best per config (best max_depth)
xgb_best = xgb_df.loc[xgb_df.groupby(['k_features', 'window_size', 'threshold'])['roc_auc_mean'].idxmax()]
xgb_best['config'] = xgb_best.apply(
    lambda r: f"{int(r['k_features'])}_{int(r['window_size'])}_{r['threshold']}", axis=1)

merged = pd.merge(rf_for_compare[['config', 'roc_auc_mean']],
                  xgb_best[['config', 'roc_auc_mean']],
                  on='config', suffixes=('_rf', '_xgb'))

ax = axes[0]
ax.scatter(merged['roc_auc_mean_rf'], merged['roc_auc_mean_xgb'],
           alpha=0.6, s=60, c=merged['roc_auc_mean_rf'], cmap='viridis')
ax.plot([0.75, 0.95], [0.75, 0.95], 'k--', alpha=0.5, label='RF = XGBoost')
ax.fill_between([0.75, 0.95], [0.75, 0.95], [0.95, 0.95], alpha=0.1, color='green', label='RF better')
ax.fill_between([0.75, 0.95], [0.75, 0.75], [0.75, 0.95], alpha=0.1, color='red', label='XGBoost better')

ax.set_xlabel('Random Forest ROC-AUC', fontsize=12)
ax.set_ylabel('XGBoost ROC-AUC', fontsize=12)
ax.set_title('RF vs XGBoost: Confronto Diretto', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0.75, 0.92)
ax.set_ylim(0.75, 0.92)

# Bar chart: Best of each
ax = axes[1]
best_rf = rf_df.loc[rf_df['roc_auc_mean'].idxmax()]
best_xgb = xgb_df.loc[xgb_df['roc_auc_mean'].idxmax()]

models = ['Random Forest', 'XGBoost']
roc_aucs = [best_rf['roc_auc_mean'], best_xgb['roc_auc_mean']]
stds = [best_rf['roc_auc_std'], best_xgb['roc_auc_std']]

bars = ax.bar(models, roc_aucs, yerr=stds, capsize=10,
              color=[colors['RF'], colors['XGBoost']], edgecolor='black', linewidth=1.5)

ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('Miglior Configurazione per Modello', fontsize=14, fontweight='bold')
ax.set_ylim(0.8, 0.95)

# Add config labels
ax.text(0, roc_aucs[0] + stds[0] + 0.01,
        f'K={int(best_rf["k_features"])}, W={int(best_rf["window_size"])}\nThr={best_rf["threshold"]}',
        ha='center', fontsize=9)
ax.text(1, roc_aucs[1] + stds[1] + 0.01,
        f'K={int(best_xgb["k_features"])}, W={int(best_xgb["window_size"])}\nThr={best_xgb["threshold"]}, D={int(best_xgb["xgb_max_depth"])}',
        ha='center', fontsize=9)

# Add value labels
for i, (bar, val, std) in enumerate(zip(bars, roc_aucs, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.02,
            f'{val:.3f}±{std:.3f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '5_rf_vs_xgboost.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 5_rf_vs_xgboost.png")
plt.close()

# ============================================================================
# FIGURA 6: Summary - Percorso verso la configurazione ottimale
# ============================================================================
print("[6/6] Generating: Optimization Path Summary...")

fig = plt.figure(figsize=(16, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: Baseline vs Optimized
ax1 = fig.add_subplot(gs[0, 0])
configs = ['Baseline\n(K=60, W=3)', 'Optimized\n(K=70, W=7)']
baseline_auc = 0.808  # From original run
optimized_auc = best_rf['roc_auc_mean']
bars = ax1.bar(configs, [baseline_auc, optimized_auc],
               color=['#95a5a6', '#2ecc71'], edgecolor='black', linewidth=2)
ax1.set_ylabel('ROC-AUC (CV)', fontsize=11)
ax1.set_title('A) Miglioramento Complessivo', fontsize=12, fontweight='bold')
ax1.set_ylim(0.75, 0.95)
ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target 0.90')

# Arrow showing improvement
ax1.annotate('', xy=(1, optimized_auc), xytext=(0, baseline_auc),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(0.5, (baseline_auc + optimized_auc)/2, f'+{100*(optimized_auc-baseline_auc)/baseline_auc:.1f}%',
         ha='center', fontsize=12, fontweight='bold', color='red')

# Panel B: Window size progression
ax2 = fig.add_subplot(gs[0, 1])
window_progression = rf_df.groupby('window_size')['roc_auc_mean'].max()
ax2.plot([3, 5, 7], window_progression.values, 'o-', color='#2ecc71',
         linewidth=3, markersize=12)
ax2.fill_between([3, 5, 7], window_progression.values, alpha=0.3, color='#2ecc71')
ax2.set_xlabel('Window Size', fontsize=11)
ax2.set_ylabel('Best ROC-AUC', fontsize=11)
ax2.set_title('B) Effetto Window Size', fontsize=12, fontweight='bold')
ax2.set_xticks([3, 5, 7])
ax2.set_xticklabels(['3\n(1.5 min)', '5\n(2.5 min)', '7\n(3.5 min)'])

# Panel C: K features (at W=7)
ax3 = fig.add_subplot(gs[0, 2])
k_progression = rf_df[rf_df['window_size'] == 7].groupby('k_features')['roc_auc_mean'].max()
ax3.plot(k_progression.index, k_progression.values, 's-', color='#3498db',
         linewidth=3, markersize=12)
ax3.axvline(x=70, color='red', linestyle='--', alpha=0.7)
ax3.set_xlabel('K Features', fontsize=11)
ax3.set_ylabel('Best ROC-AUC', fontsize=11)
ax3.set_title('C) Effetto K (Window=7)', fontsize=12, fontweight='bold')

# Panel D: Stability improvement
ax4 = fig.add_subplot(gs[1, 0])
baseline_std = 0.068
optimized_std = best_rf['roc_auc_std']
bars = ax4.bar(['Baseline', 'Optimized'], [baseline_std, optimized_std],
               color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2)
ax4.set_ylabel('Std Deviation (CV)', fontsize=11)
ax4.set_title('D) Stabilità del Modello', fontsize=12, fontweight='bold')
ax4.text(1, optimized_std + 0.005, f'-{100*(1-optimized_std/baseline_std):.0f}%',
         ha='center', fontsize=12, fontweight='bold', color='green')

# Panel E: Top 10 configurations
ax5 = fig.add_subplot(gs[1, 1:])
top10 = df.nlargest(10, 'roc_auc_mean')
y_pos = np.arange(len(top10))
colors_top10 = ['#2ecc71' if m == 'RF' else '#e74c3c' for m in top10['model']]

bars = ax5.barh(y_pos, top10['roc_auc_mean'], xerr=top10['roc_auc_std'],
                color=colors_top10, edgecolor='black', capsize=3)
ax5.set_yticks(y_pos)
ax5.set_yticklabels([f"{row['model']} K={int(row['k_features'])} W={int(row['window_size'])} T={row['threshold']}"
                     for _, row in top10.iterrows()], fontsize=9)
ax5.set_xlabel('ROC-AUC', fontsize=11)
ax5.set_title('E) Top 10 Configurazioni', fontsize=12, fontweight='bold')
ax5.set_xlim(0.85, 0.92)
ax5.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Random Forest'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='XGBoost')]
ax5.legend(handles=legend_elements, loc='lower right')

plt.suptitle('Hyperparameter Tuning: Percorso verso la Configurazione Ottimale\n'
             'Dataset con Etichette Manuali - Sleep Apnea Detection',
             fontsize=14, fontweight='bold', y=1.02)

plt.savefig(FIGURES_DIR / '6_optimization_summary.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 6_optimization_summary.png")
plt.close()

# ============================================================================
# FIGURA 7: 3D Surface (bonus)
# ============================================================================
print("[Bonus] Generating: 3D Surface Plot...")

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# RF 3D surface
ax1 = fig.add_subplot(121, projection='3d')
rf_pivot_3d = rf_df.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().reset_index()
X = rf_pivot_3d['k_features'].values.reshape(5, 3)
Y = rf_pivot_3d['window_size'].values.reshape(5, 3)
Z = rf_pivot_3d['roc_auc_mean'].values.reshape(5, 3)

surf = ax1.plot_surface(X, Y, Z, cmap='Greens', edgecolor='black', alpha=0.8, linewidth=0.5)
ax1.set_xlabel('K Features')
ax1.set_ylabel('Window Size')
ax1.set_zlabel('ROC-AUC')
ax1.set_title('Random Forest', fontsize=12, fontweight='bold')
ax1.view_init(elev=25, azim=45)

# Mark optimal point
best_rf_row = rf_df.loc[rf_df['roc_auc_mean'].idxmax()]
ax1.scatter([best_rf_row['k_features']], [best_rf_row['window_size']],
            [best_rf_row['roc_auc_mean']], color='red', s=100, marker='*')

# XGBoost 3D surface
ax2 = fig.add_subplot(122, projection='3d')
xgb_pivot_3d = xgb_df.groupby(['k_features', 'window_size'])['roc_auc_mean'].max().reset_index()
X2 = xgb_pivot_3d['k_features'].values.reshape(5, 3)
Y2 = xgb_pivot_3d['window_size'].values.reshape(5, 3)
Z2 = xgb_pivot_3d['roc_auc_mean'].values.reshape(5, 3)

surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='Reds', edgecolor='black', alpha=0.8, linewidth=0.5)
ax2.set_xlabel('K Features')
ax2.set_ylabel('Window Size')
ax2.set_zlabel('ROC-AUC')
ax2.set_title('XGBoost', fontsize=12, fontweight='bold')
ax2.view_init(elev=25, azim=45)

plt.suptitle('Superficie di Ottimizzazione degli Hyperparameter', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '7_3d_surface.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 7_3d_surface.png")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("VISUALIZATIONS COMPLETED")
print("=" * 60)
print(f"\nGenerated {len(list(FIGURES_DIR.glob('*.png')))} figures in: {FIGURES_DIR}/")
print("\nFiles:")
for f in sorted(FIGURES_DIR.glob('*.png')):
    print(f"  - {f.name}")

print("\n" + "-" * 60)
print("KEY INSIGHTS FROM TUNING:")
print("-" * 60)
print(f"""
1. WINDOW SIZE è il fattore più importante:
   - Window=3 (1.5 min): ROC-AUC max ~0.81
   - Window=5 (2.5 min): ROC-AUC max ~0.85
   - Window=7 (3.5 min): ROC-AUC max ~0.90 ✓

2. K FEATURES ha effetto moderato:
   - K=40-60: performance simile
   - K=70: leggero miglioramento ✓
   - K=80: inizia a calare (overfitting)

3. THRESHOLD trade-off:
   - 0.25: massimizza Recall (84.6%) ✓
   - 0.30: bilanciato
   - 0.35+: più Precision ma meno Recall

4. RF supera XGBoost:
   - RF best:  0.901 ± 0.034
   - XGB best: 0.873 ± 0.031

5. Stabilità migliorata:
   - Baseline std: 0.068
   - Optimized std: 0.034 (-50%)
""")
