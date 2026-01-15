#!/usr/bin/env python3
"""
Visualizza il layout corretto della matrice 4×10 e l'importanza dei canali.
Basato sul documento firmware che mostra la disposizione reale dei sensori.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# ========= CONFIG =========
from pathlib import Path

DATASET_NAME = "dataset_apnea_windows_30s_features_mat_acc.csv"
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

# Search paths for dataset (needs per-channel features, not zone-aggregated)
SEARCH_PATHS = [
    BASE_DIR / ".." / "data" / DATASET_NAME,
    BASE_DIR / ".." / DATASET_NAME,
    BASE_DIR / "preprocessing_output" / "dataset_windows_30s_features_mat_acc.csv",
    Path(".") / DATASET_NAME,
]

def find_dataset():
    for path in SEARCH_PATHS:
        if path.exists():
            return str(path)
    return None

INPUT_CSV = find_dataset()
TEST_SUBJECTS = ['GZ01', 'FDR01', 'AM01', 'FC01']

# Layout CORRETTO della matrice 4×10 (dal PDF firmware)
# D1: 1-10 (riga superiore - torace/spalle)
# D2: 11-20 (seconda riga - torace medio)
# D3: 21-30 (terza riga - addome/lombare)
# D4: 31-40 (riga inferiore - bacino/glutei)

MATRIX_SHAPE = (4, 10)  # 4 righe × 10 colonne

def channel_to_position(ch):
    """Converte numero canale (1-40) in posizione (riga, colonna) nella matrice 4×10."""
    row = (ch - 1) // 10
    col = (ch - 1) % 10
    return row, col

def position_to_channel(row, col):
    """Converte posizione (riga, colonna) in numero canale."""
    return row * 10 + col + 1

def main():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Dataset shape: {df.shape}")

    # Split train/test
    train_mask = ~df['Subject'].isin(TEST_SUBJECTS)
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()

    print(f"\nTrain: {len(df_train)} samples")
    print(f"Test: {len(df_test)} samples")

    # Prepare features: solo canali della matrice
    channel_cols = []
    for ch in range(1, 41):
        for stat in ['mean', 'std', 'diff_std']:
            col_name = f'ch{ch}_{stat}'
            if col_name in df.columns:
                channel_cols.append(col_name)

    print(f"\nTotal channel features: {len(channel_cols)}")

    X_train = df_train[channel_cols].values
    y_train = df_train['label_apnea'].values
    X_test = df_test[channel_cols].values
    y_test = df_test['label_apnea'].values

    # Oversample minority class
    X_pos = X_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    y_pos = y_train[y_train == 1]
    y_neg = y_train[y_train == 0]

    X_pos_resampled, y_pos_resampled = resample(
        X_pos, y_pos,
        n_samples=len(X_neg),
        random_state=42,
        replace=True
    )

    X_train_bal = np.vstack([X_neg, X_pos_resampled])
    y_train_bal = np.hstack([y_neg, y_pos_resampled])

    print(f"\nBalanced training set: {len(y_train_bal)} samples")

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': channel_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 most important features:")
    print(feature_importance.head(20).to_string(index=False))

    # Aggregate importance by channel
    channel_importance = {}
    for ch in range(1, 41):
        importance_sum = 0
        for stat in ['mean', 'std', 'diff_std']:
            col_name = f'ch{ch}_{stat}'
            if col_name in feature_importance['feature'].values:
                importance_sum += feature_importance[
                    feature_importance['feature'] == col_name
                ]['importance'].values[0]
        channel_importance[ch] = importance_sum

    # Create importance matrix 4×10
    importance_matrix = np.zeros(MATRIX_SHAPE)
    for ch, imp in channel_importance.items():
        row, col = channel_to_position(ch)
        importance_matrix[row, col] = imp

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1. Heatmap with channel numbers
    ax1 = axes[0]
    sns.heatmap(
        importance_matrix,
        annot=np.arange(1, 41).reshape(MATRIX_SHAPE),
        fmt='d',
        cmap='YlOrRd',
        cbar_kws={'label': 'Total Importance'},
        linewidths=1,
        linecolor='gray',
        ax=ax1
    )
    ax1.set_title('Channel Importance on 4×10 Matrix\n(Numbers = Channel ID)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column (A1-A10)', fontsize=12)
    ax1.set_ylabel('Row (D1-D4)', fontsize=12)
    ax1.set_yticklabels(['D1 (Upper)', 'D2 (Mid-Upper)', 'D3 (Mid-Lower)', 'D4 (Lower)'], rotation=0)

    # Add anatomical labels
    ax1.text(11.5, 0.5, '← Shoulders/Upper Torso', fontsize=10, va='center')
    ax1.text(11.5, 1.5, '← Mid Torso', fontsize=10, va='center')
    ax1.text(11.5, 2.5, '← Abdomen/Lumbar', fontsize=10, va='center')
    ax1.text(11.5, 3.5, '← Pelvis/Gluteal', fontsize=10, va='center')

    # 2. Top 10 channels bar plot
    ax2 = axes[1]
    top_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    ch_ids = [ch for ch, _ in top_channels]
    ch_imps = [imp for _, imp in top_channels]

    # Create labels with position info
    labels = []
    for ch in ch_ids:
        row, col = channel_to_position(ch)
        row_name = ['D1', 'D2', 'D3', 'D4'][row]
        labels.append(f'Ch{ch}\n({row_name}, A{col+1})')

    bars = ax2.barh(range(len(ch_ids)), ch_imps, color='steelblue')
    ax2.set_yticks(range(len(ch_ids)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Total Importance', fontsize=12)
    ax2.set_title('Top 10 Most Important Channels', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # Add importance values on bars
    for i, (bar, imp) in enumerate(zip(bars, ch_imps)):
        ax2.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('results_analysis/matrix_4x10_channel_importance.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results_analysis/matrix_4x10_channel_importance.png")

    # Print analysis
    print("\n" + "="*70)
    print("ANALYSIS OF CHANNEL DISTRIBUTION")
    print("="*70)

    # Count top channels by row
    row_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for ch, _ in top_channels:
        row, _ = channel_to_position(ch)
        row_counts[row] += 1

    print("\nTop 10 channels by row:")
    print(f"  D1 (Upper - Shoulders/Torso):  {row_counts[0]} channels")
    print(f"  D2 (Mid-Upper - Torso):        {row_counts[1]} channels")
    print(f"  D3 (Mid-Lower - Abdomen):      {row_counts[2]} channels")
    print(f"  D4 (Lower - Pelvis/Gluteal):   {row_counts[3]} channels")

    # Count by side (left vs right)
    left_count = sum(1 for ch, _ in top_channels if channel_to_position(ch)[1] < 5)
    right_count = sum(1 for ch, _ in top_channels if channel_to_position(ch)[1] >= 5)

    print(f"\nTop 10 channels by side:")
    print(f"  Left (A1-A5):  {left_count} channels")
    print(f"  Right (A6-A10): {right_count} channels")

    # Save channel importance to CSV
    channel_df = pd.DataFrame([
        {
            'channel': ch,
            'row': channel_to_position(ch)[0] + 1,
            'col': channel_to_position(ch)[1] + 1,
            'row_name': ['D1', 'D2', 'D3', 'D4'][channel_to_position(ch)[0]],
            'importance': imp
        }
        for ch, imp in sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    ])
    channel_df.to_csv('results_analysis/channel_importance_4x10.csv', index=False)
    print("\nSaved: results_analysis/channel_importance_4x10.csv")

    print("\n" + "="*70)
    print("RECOMMENDED ZONE DEFINITIONS")
    print("="*70)

    # Proponi definizioni zone corrette
    print("\nOption 1: Simple 4 zones (2×5 each):")
    print("  zone_UL (Upper Left):  D1-D2, A1-A5  →", [position_to_channel(r, c) for r in [0, 1] for c in range(5)])
    print("  zone_UR (Upper Right): D1-D2, A6-A10 →", [position_to_channel(r, c) for r in [0, 1] for c in range(5, 10)])
    print("  zone_LL (Lower Left):  D3-D4, A1-A5  →", [position_to_channel(r, c) for r in [2, 3] for c in range(5)])
    print("  zone_LR (Lower Right): D3-D4, A6-A10 →", [position_to_channel(r, c) for r in [2, 3] for c in range(5, 10)])

    print("\nOption 2: Row-based 4 zones:")
    print("  zone_upper (D1):       ", list(range(1, 11)))
    print("  zone_mid_upper (D2):   ", list(range(11, 21)))
    print("  zone_mid_lower (D3):   ", list(range(21, 31)))
    print("  zone_lower (D4):       ", list(range(31, 41)))

if __name__ == "__main__":
    main()
