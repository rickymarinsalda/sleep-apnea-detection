#!/usr/bin/env python3
"""
SLEEP APNEA DETECTION - Dataset Preparation Pipeline
VERSIONE PER DATASET CON ETICHETTE MANUALI (più accurate)

Questo dataset ha etichette apnea/non-apnea verificate manualmente,
con ~72% più eventi apnea rispetto al dataset originale.

Pipeline steps:
1. Load raw pressure mat data (MAT) and accelerometer data (ACC)
2. Create 30-second windows with labels
3. Extract statistical features per channel
4. Extract accelerometer features
5. Merge MAT + ACC features
6. Aggregate channels into anatomical zones (4x10 matrix -> 4 zones)

Input files (in DATA_DIR):
- dataset_apnea_ricky_MAT_filt_indiciApneaManuale.csv
- dataset_apnea_ricky_ACC_indiciApneaManuale.csv

Output files (in OUTPUT_DIR):
- dataset_windows_30s_flat.csv
- dataset_windows_30s_features.csv
- dataset_windows_30s_features_mat_acc.csv
- dataset_windows_30s_features_zones_MANUAL.csv (FINAL)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")
DATA_DIR = BASE_DIR / ".." / "dataset_a_mano"
OUTPUT_DIR = BASE_DIR / "preprocessing_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Raw data files (MANUAL LABELS VERSION)
MAT_RAW_CSV = DATA_DIR / "dataset_apnea_ricky_MAT_filt_indiciApneaManuale.csv"
ACC_RAW_CSV = DATA_DIR / "dataset_apnea_ricky_ACC_indiciApneaManuale.csv"

# Output files
WINDOWS_FLAT_CSV = OUTPUT_DIR / "dataset_windows_30s_flat.csv"
FEATURES_MAT_CSV = OUTPUT_DIR / "dataset_windows_30s_features.csv"
ACC_FEATURES_CSV = OUTPUT_DIR / "dataset_windows_30s_acc_features.csv"
FEATURES_MAT_ACC_CSV = OUTPUT_DIR / "dataset_windows_30s_features_mat_acc.csv"
FEATURES_ZONES_CSV = OUTPUT_DIR / "dataset_windows_30s_features_zones_MANUAL.csv"

# Sensor parameters
FS = 8.0              # Sampling frequency (Hz)
WINDOW_SEC = 30       # Window duration (seconds)
WIN_SIZE = int(FS * WINDOW_SEC)  # Samples per window (240)
N_CHANNELS = 40       # Number of pressure mat channels

# Labeling parameters
MIN_FRAC_APNEA = 0.5
MIN_FRAC_NONAPNEA = 0.5

# Accelerometer columns
ACC_TIME_COL = "Time_acc1"
ACC_SUBJ_COL = "Subject"
ACC_SIGNAL_COLS = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]

# Zone definitions (CORRECTED 4x10 layout)
ZONE_DEFS_CORRECTED = {
    "zone_UL": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15],
    "zone_UR": [6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    "zone_LL": [21, 22, 23, 24, 25, 31, 32, 33, 34, 35],
    "zone_LR": [26, 27, 28, 29, 30, 36, 37, 38, 39, 40],
}

META_COLS = [
    "Subject", "start_time", "end_time", "Position_mode",
    "majority_status", "label_apnea", "frac_apnea", "frac_altro", "frac_respiro",
]
GLOBAL_COLS = ["global_mean", "global_std", "global_min", "global_max"]
CHANNEL_STAT_TYPES = ["mean", "std", "diff_std"]


# ============================================================================
# STEP 1: CREATE 30-SECOND WINDOWS
# ============================================================================

def step1_create_windows():
    print("\n" + "=" * 80)
    print("STEP 1: Creating 30-second windows from raw MAT data (MANUAL LABELS)")
    print("=" * 80)

    if not MAT_RAW_CSV.exists():
        print(f"ERROR: Raw MAT file not found: {MAT_RAW_CSV}")
        return None

    print(f"\nLoading {MAT_RAW_CSV} ...")
    df = pd.read_csv(MAT_RAW_CSV)
    print(f"Raw data shape: {df.shape}")

    # Count apnea samples in raw data
    status_counts = df['Status'].value_counts()
    print(f"\nRaw data Status distribution:")
    for status, count in sorted(status_counts.items()):
        pct = 100 * count / len(df)
        label = {0: 'normal', 1: 'fast', 2: 'slow', 3: 'APNEA', 4: 'other'}.get(status, '?')
        print(f"  Status {status} ({label}): {count} ({pct:.1f}%)")

    channel_cols = [c for c in df.columns if c.startswith("ch")]
    print(f"\nFound {len(channel_cols)} channels")

    # Drop rows where all channels are zero
    mask_all_zero = (df[channel_cols] == 0).all(axis=1)
    n_all_zero = mask_all_zero.sum()
    if n_all_zero > 0:
        print(f"Dropping {n_all_zero} rows where all channels are zero.")
        df = df[~mask_all_zero].copy()

    df = df.sort_values(["Subject", "Time"]).reset_index(drop=True)

    print(f"\nWindow parameters:")
    print(f"  - Window size: {WIN_SIZE} samples ({WINDOW_SEC} seconds)")
    print(f"  - Min apnea fraction: {MIN_FRAC_APNEA}")

    windows = []
    skipped_windows = 0

    for subj, sub_df in df.groupby("Subject"):
        sub_df = sub_df.sort_values("Time")
        n = len(sub_df)
        num_full = n // WIN_SIZE
        print(f"  Subject {subj}: {n} samples -> {num_full} windows", end="")

        subj_windows = 0
        subj_apnea = 0
        for w in range(num_full):
            start_idx = w * WIN_SIZE
            end_idx = start_idx + WIN_SIZE
            chunk = sub_df.iloc[start_idx:end_idx]

            if len(chunk) < WIN_SIZE:
                continue

            statuses = chunk["Status"].to_numpy()
            vals, counts = np.unique(statuses, return_counts=True)
            majority_status = int(vals[counts.argmax()])

            frac_apnea = np.mean(statuses == 3)
            frac_altro = np.mean(statuses == 4)
            frac_respiro = np.mean(np.isin(statuses, [0, 1, 2]))

            label_apnea = None
            if majority_status == 3 and frac_apnea >= MIN_FRAC_APNEA:
                label_apnea = 1
            elif majority_status in (0, 1, 2) and frac_respiro >= MIN_FRAC_NONAPNEA:
                label_apnea = 0
            else:
                skipped_windows += 1
                continue

            pos_mode_series = chunk["Position"].dropna().mode()
            pos_mode = pos_mode_series.iloc[0] if not pos_mode_series.empty else np.nan

            start_time = float(chunk["Time"].iloc[0])
            end_time = float(chunk["Time"].iloc[-1])

            signals = chunk[channel_cols].to_numpy()
            flat = signals.flatten()

            row = {
                "Subject": subj,
                "start_time": start_time,
                "end_time": end_time,
                "Position_mode": pos_mode,
                "majority_status": majority_status,
                "label_apnea": label_apnea,
                "frac_apnea": float(frac_apnea),
                "frac_altro": float(frac_altro),
                "frac_respiro": float(frac_respiro),
            }

            for j, value in enumerate(flat):
                row[f"x{j}"] = float(value)

            windows.append(row)
            subj_windows += 1
            if label_apnea == 1:
                subj_apnea += 1

        print(f" -> kept {subj_windows} ({subj_apnea} apnea)")

    df_win = pd.DataFrame(windows)
    print(f"\n✓ Total windows created: {len(df_win)}")
    print(f"  Skipped (ambiguous): {skipped_windows}")

    n_apnea = (df_win['label_apnea'] == 1).sum()
    n_normal = (df_win['label_apnea'] == 0).sum()
    print(f"\nClass distribution:")
    print(f"  Non-apnea (0): {n_normal} ({100*n_normal/len(df_win):.1f}%)")
    print(f"  Apnea (1): {n_apnea} ({100*n_apnea/len(df_win):.1f}%)")

    df_win.to_csv(WINDOWS_FLAT_CSV, index=False)
    print(f"\n✓ Saved to: {WINDOWS_FLAT_CSV}")

    return df_win


# ============================================================================
# STEP 2: EXTRACT PRESSURE MAT FEATURES
# ============================================================================

def step2_extract_mat_features(df_win=None):
    print("\n" + "=" * 80)
    print("STEP 2: Extracting pressure mat features")
    print("=" * 80)

    if df_win is None:
        if not WINDOWS_FLAT_CSV.exists():
            print(f"ERROR: Windows file not found: {WINDOWS_FLAT_CSV}")
            return None
        print(f"\nLoading {WINDOWS_FLAT_CSV} ...")
        df_win = pd.read_csv(WINDOWS_FLAT_CSV)

    feature_cols = [c for c in df_win.columns if c.startswith("x")]
    expected_cols = WIN_SIZE * N_CHANNELS

    print(f"Processing {len(df_win)} windows...")

    rows = []
    for idx, row in df_win.iterrows():
        flat = row[feature_cols].to_numpy(dtype=np.float32)
        sig = flat.reshape(WIN_SIZE, N_CHANNELS)

        ch_mean = sig.mean(axis=0)
        ch_std = sig.std(axis=0)
        diff_sig = np.diff(sig, axis=0)
        ch_diff_std = diff_sig.std(axis=0)

        global_mean = sig.mean()
        global_std = sig.std()
        global_min = sig.min()
        global_max = sig.max()

        feat = {
            "Subject": row["Subject"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "Position_mode": row["Position_mode"],
            "majority_status": row["majority_status"],
            "label_apnea": row["label_apnea"],
            "frac_apnea": row["frac_apnea"],
            "frac_altro": row["frac_altro"],
            "frac_respiro": row["frac_respiro"],
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "global_min": float(global_min),
            "global_max": float(global_max),
        }

        for ch in range(N_CHANNELS):
            ch_name = f"ch{ch + 1}"
            feat[f"{ch_name}_mean"] = float(ch_mean[ch])
            feat[f"{ch_name}_std"] = float(ch_std[ch])
            feat[f"{ch_name}_diff_std"] = float(ch_diff_std[ch])

        rows.append(feat)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_win)} windows...")

    df_feat = pd.DataFrame(rows)
    print(f"\n✓ Feature dataset shape: {df_feat.shape}")

    df_feat.to_csv(FEATURES_MAT_CSV, index=False)
    print(f"\n✓ Saved to: {FEATURES_MAT_CSV}")

    return df_feat


# ============================================================================
# STEP 3: EXTRACT ACCELEROMETER FEATURES
# ============================================================================

def compute_acc_features(acc_segment):
    data = acc_segment[ACC_SIGNAL_COLS].to_numpy(dtype=np.float32)

    if data.size == 0:
        return None

    global_mean = float(data.mean())
    global_std = float(data.std())
    global_min = float(data.min())
    global_max = float(data.max())

    ch_mean = data.mean(axis=0)
    ch_std = data.std(axis=0)

    diff = np.diff(data, axis=0)
    ch_diff_std = diff.std(axis=0) if diff.size > 0 else np.zeros_like(ch_mean)

    feats = {
        "acc_global_mean": global_mean,
        "acc_global_std": global_std,
        "acc_global_min": global_min,
        "acc_global_max": global_max,
    }

    for ch_name, m, s, ds in zip(ACC_SIGNAL_COLS, ch_mean, ch_std, ch_diff_std):
        feats[f"acc_{ch_name}_mean"] = float(m)
        feats[f"acc_{ch_name}_std"] = float(s)
        feats[f"acc_{ch_name}_diff_std"] = float(ds)

    return feats


def step3_extract_acc_features(df_win=None):
    print("\n" + "=" * 80)
    print("STEP 3: Extracting accelerometer features")
    print("=" * 80)

    if not ACC_RAW_CSV.exists():
        print(f"WARNING: ACC file not found: {ACC_RAW_CSV}")
        print("Skipping accelerometer features.")
        return None

    if df_win is None:
        if not WINDOWS_FLAT_CSV.exists():
            print(f"ERROR: Windows file not found: {WINDOWS_FLAT_CSV}")
            return None
        print(f"\nLoading windows from {WINDOWS_FLAT_CSV} ...")
        df_win = pd.read_csv(WINDOWS_FLAT_CSV)

    print(f"Loading ACC raw data from {ACC_RAW_CSV} ...")
    df_acc = pd.read_csv(ACC_RAW_CSV)
    print(f"ACC data shape: {df_acc.shape}")

    df_acc_grouped = dict(tuple(df_acc.groupby(ACC_SUBJ_COL)))

    rows = []
    n_missing = 0

    for idx, row in df_win.iterrows():
        subj = row["Subject"]
        start = row["start_time"]
        end = row["end_time"]

        acc_subj = df_acc_grouped.get(subj)
        if acc_subj is None:
            n_missing += 1
            continue

        mask = (acc_subj[ACC_TIME_COL] >= start) & (acc_subj[ACC_TIME_COL] < end)
        segment = acc_subj.loc[mask]

        feats = compute_acc_features(segment)
        if feats is None:
            n_missing += 1
            continue

        out = {
            "Subject": subj,
            "start_time": start,
            "end_time": end,
        }
        out.update(feats)
        rows.append(out)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_win)} windows...")

    df_acc_feat = pd.DataFrame(rows)
    print(f"\n✓ ACC feature dataset shape: {df_acc_feat.shape}")
    print(f"  Windows without ACC data: {n_missing}")

    df_acc_feat.to_csv(ACC_FEATURES_CSV, index=False)
    print(f"\n✓ Saved to: {ACC_FEATURES_CSV}")

    return df_acc_feat


# ============================================================================
# STEP 4: MERGE MAT + ACC FEATURES
# ============================================================================

def step4_merge_features(df_mat_feat=None, df_acc_feat=None):
    print("\n" + "=" * 80)
    print("STEP 4: Merging MAT + ACC features")
    print("=" * 80)

    if df_mat_feat is None:
        if not FEATURES_MAT_CSV.exists():
            print(f"ERROR: MAT features not found: {FEATURES_MAT_CSV}")
            return None
        print(f"\nLoading MAT features from {FEATURES_MAT_CSV} ...")
        df_mat_feat = pd.read_csv(FEATURES_MAT_CSV)

    if df_acc_feat is None:
        if not ACC_FEATURES_CSV.exists():
            print(f"WARNING: ACC features not found, proceeding with MAT only.")
            df_mat_feat.to_csv(FEATURES_MAT_ACC_CSV, index=False)
            return df_mat_feat
        print(f"Loading ACC features from {ACC_FEATURES_CSV} ...")
        df_acc_feat = pd.read_csv(ACC_FEATURES_CSV)

    print(f"\nMAT features shape: {df_mat_feat.shape}")
    print(f"ACC features shape: {df_acc_feat.shape}")

    print("\nMerging on Subject, start_time, end_time...")
    df_merged = pd.merge(
        df_mat_feat,
        df_acc_feat,
        on=["Subject", "start_time", "end_time"],
        how="inner",
    )

    print(f"\n✓ Merged dataset shape: {df_merged.shape}")

    df_merged.to_csv(FEATURES_MAT_ACC_CSV, index=False)
    print(f"\n✓ Saved to: {FEATURES_MAT_ACC_CSV}")

    return df_merged


# ============================================================================
# STEP 5: AGGREGATE INTO ANATOMICAL ZONES
# ============================================================================

def step5_create_zone_features(df_merged=None):
    print("\n" + "=" * 80)
    print("STEP 5: Aggregating channels into anatomical zones (CORRECTED)")
    print("=" * 80)

    if df_merged is None:
        if not FEATURES_MAT_ACC_CSV.exists():
            print(f"ERROR: Merged features not found: {FEATURES_MAT_ACC_CSV}")
            return None
        print(f"\nLoading merged features from {FEATURES_MAT_ACC_CSV} ...")
        df_merged = pd.read_csv(FEATURES_MAT_ACC_CSV)

    print("\nZone definitions (4x10 matrix layout):")
    for zone_name, ch_list in ZONE_DEFS_CORRECTED.items():
        print(f"  {zone_name}: {len(ch_list)} channels")

    acc_cols = [c for c in df_merged.columns if c.startswith("acc_")]
    print(f"\nFound {len(acc_cols)} ACC feature columns.")

    keep_cols = [c for c in META_COLS if c in df_merged.columns]
    keep_cols += [c for c in GLOBAL_COLS if c in df_merged.columns]
    keep_cols += acc_cols

    df_out = df_merged[keep_cols].copy()

    available_ch_stats = {}
    for ch_idx in range(1, 41):
        for stat in CHANNEL_STAT_TYPES:
            col_name = f"ch{ch_idx}_{stat}"
            if col_name in df_merged.columns:
                available_ch_stats[(ch_idx, stat)] = col_name

    print("\nGenerating zone features:")
    for zone_name, ch_list in ZONE_DEFS_CORRECTED.items():
        for stat in CHANNEL_STAT_TYPES:
            cols = [
                available_ch_stats[(ch, stat)]
                for ch in ch_list
                if (ch, stat) in available_ch_stats
            ]

            if not cols:
                continue

            zone_mat = df_merged[cols]
            df_out[f"{zone_name}_{stat}_mean"] = zone_mat.mean(axis=1)
            df_out[f"{zone_name}_{stat}_std"] = zone_mat.std(axis=1)

        print(f"  ✓ {zone_name}: 6 features created")

    zone_features = [c for c in df_out.columns if c.startswith('zone_')]
    print(f"\n✓ Output dataset shape: {df_out.shape}")
    print(f"  - Zone features: {len(zone_features)}")

    df_out.to_csv(FEATURES_ZONES_CSV, index=False)
    print(f"\n✓ Saved to: {FEATURES_ZONES_CSV}")

    return df_out


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(" " * 10 + "SLEEP APNEA DETECTION - DATASET PREPARATION")
    print(" " * 10 + "          (MANUAL LABELS VERSION)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Output directory: {OUTPUT_DIR}")

    print(f"\nChecking input files:")
    print(f"  - MAT raw: {MAT_RAW_CSV} {'✓' if MAT_RAW_CSV.exists() else '✗ NOT FOUND'}")
    print(f"  - ACC raw: {ACC_RAW_CSV} {'✓' if ACC_RAW_CSV.exists() else '✗ NOT FOUND'}")

    if not MAT_RAW_CSV.exists():
        print("\nERROR: Raw MAT data not found!")
        return

    # Run pipeline
    df_win = step1_create_windows()
    if df_win is None:
        return

    df_mat_feat = step2_extract_mat_features(df_win)
    if df_mat_feat is None:
        return

    df_acc_feat = step3_extract_acc_features(df_win)

    df_merged = step4_merge_features(df_mat_feat, df_acc_feat)
    if df_merged is None:
        return

    df_final = step5_create_zone_features(df_merged)
    if df_final is None:
        return

    # Final summary
    print("\n" + "=" * 80)
    print(" " * 20 + "PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nFinal dataset: {FEATURES_ZONES_CSV}")
    print(f"  Shape: {df_final.shape}")
    print(f"  Windows: {len(df_final)}")
    print(f"  Subjects: {df_final['Subject'].nunique()}")

    n_apnea = (df_final['label_apnea'] == 1).sum()
    n_normal = (df_final['label_apnea'] == 0).sum()
    print(f"\nClass distribution:")
    print(f"  Non-apnea (0): {n_normal} ({100*n_normal/len(df_final):.1f}%)")
    print(f"  Apnea (1): {n_apnea} ({100*n_apnea/len(df_final):.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
