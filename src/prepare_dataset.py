#!/usr/bin/env python3
"""
SLEEP APNEA DETECTION - Dataset Preparation Pipeline

This script prepares the dataset from raw sensor data to the final feature dataset.

Pipeline steps:
1. Load raw pressure mat data (MAT) and accelerometer data (ACC)
2. Create 30-second windows with labels
3. Extract statistical features per channel
4. Extract accelerometer features
5. Merge MAT + ACC features
6. Aggregate channels into anatomical zones (4x10 matrix -> 4 zones)

Input files (in DATA_DIR):
- dataset_apnea_ricky_MAT.csv: Raw pressure mat data (40 channels @ 8Hz)
- dataset_apnea_ricky_ACC.csv: Raw accelerometer data (6 channels)

Output files (in OUTPUT_DIR):
- dataset_windows_30s_flat.csv: Windowed raw data (intermediate)
- dataset_windows_30s_features.csv: Per-channel features (intermediate)
- dataset_windows_30s_features_mat_acc.csv: MAT + ACC features merged
- dataset_windows_30s_features_zones_CORRECTED.csv: Final dataset with zone aggregation

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory (where this script is located)
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path(".")

# Paths - search multiple locations for data
DATA_SEARCH_PATHS = [
    BASE_DIR / ".." / "data",           # ../data/ (GitHub structure)
    BASE_DIR / ".." / ".." / "data",    # ../../data/ (if running from src/)
    Path("data"),                        # ./data/
    Path("."),                           # current directory
]

def find_data_dir():
    """Find directory containing raw data files."""
    for path in DATA_SEARCH_PATHS:
        mat_file = path / "dataset_apnea_ricky_MAT.csv"
        if mat_file.exists():
            return path
    return None

DATA_DIR = find_data_dir()
OUTPUT_DIR = BASE_DIR / "preprocessing_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Raw data files
if DATA_DIR is not None:
    MAT_RAW_CSV = DATA_DIR / "dataset_apnea_ricky_MAT.csv"
    ACC_RAW_CSV = DATA_DIR / "dataset_apnea_ricky_ACC.csv"
else:
    MAT_RAW_CSV = Path("dataset_apnea_ricky_MAT.csv")
    ACC_RAW_CSV = Path("dataset_apnea_ricky_ACC.csv")

# Output files
WINDOWS_FLAT_CSV = OUTPUT_DIR / "dataset_windows_30s_flat.csv"
FEATURES_MAT_CSV = OUTPUT_DIR / "dataset_windows_30s_features.csv"
ACC_FEATURES_CSV = OUTPUT_DIR / "dataset_windows_30s_acc_features.csv"
FEATURES_MAT_ACC_CSV = OUTPUT_DIR / "dataset_windows_30s_features_mat_acc.csv"
FEATURES_ZONES_CSV = OUTPUT_DIR / "dataset_windows_30s_features_zones_CORRECTED.csv"

# Sensor parameters
FS = 8.0              # Sampling frequency (Hz)
WINDOW_SEC = 30       # Window duration (seconds)
WIN_SIZE = int(FS * WINDOW_SEC)  # Samples per window (240)
N_CHANNELS = 40       # Number of pressure mat channels

# Labeling parameters
MIN_FRAC_APNEA = 0.5      # Minimum fraction of apnea samples to label window as apnea
MIN_FRAC_NONAPNEA = 0.5   # Minimum fraction of normal samples to label as non-apnea

# Accelerometer columns
ACC_TIME_COL = "Time_acc1"
ACC_SUBJ_COL = "Subject"
ACC_SIGNAL_COLS = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]

# Zone definitions (CORRECTED - based on actual 4x10 matrix layout)
# Pressure mat layout:
#   D1 (row 1): channels 1-10  (shoulders/upper torso)
#   D2 (row 2): channels 11-20 (mid torso)
#   D3 (row 3): channels 21-30 (abdomen/lumbar)
#   D4 (row 4): channels 31-40 (pelvis/gluteal)
ZONE_DEFS_CORRECTED = {
    # Upper Left: D1-D2, columns A1-A5 (left chest)
    "zone_UL": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15],
    # Upper Right: D1-D2, columns A6-A10 (right chest)
    "zone_UR": [6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    # Lower Left: D3-D4, columns A1-A5 (left abdomen/pelvis)
    "zone_LL": [21, 22, 23, 24, 25, 31, 32, 33, 34, 35],
    # Lower Right: D3-D4, columns A6-A10 (right abdomen/pelvis)
    "zone_LR": [26, 27, 28, 29, 30, 36, 37, 38, 39, 40],
}

# Meta columns (not features)
META_COLS = [
    "Subject", "start_time", "end_time", "Position_mode",
    "majority_status", "label_apnea", "frac_apnea", "frac_altro", "frac_respiro",
]

# Global pressure features to keep
GLOBAL_COLS = ["global_mean", "global_std", "global_min", "global_max"]

# Channel statistics types
CHANNEL_STAT_TYPES = ["mean", "std", "diff_std"]


# ============================================================================
# STEP 1: CREATE 30-SECOND WINDOWS
# ============================================================================

def step1_create_windows():
    """
    Create 30-second windows from raw pressure mat data.
    Each window is labeled as apnea (1) or non-apnea (0).
    """
    print("\n" + "=" * 80)
    print("STEP 1: Creating 30-second windows from raw MAT data")
    print("=" * 80)

    if not MAT_RAW_CSV.exists():
        print(f"ERROR: Raw MAT file not found: {MAT_RAW_CSV}")
        print("Please ensure the raw data files are in the data/ directory.")
        return None

    print(f"\nLoading {MAT_RAW_CSV} ...")
    df = pd.read_csv(MAT_RAW_CSV)
    print(f"Raw data shape: {df.shape}")

    # Identify pressure channels
    channel_cols = [c for c in df.columns if c.startswith("ch")]
    print(f"Found {len(channel_cols)} channels: {channel_cols[0]} ... {channel_cols[-1]}")

    # Drop rows where all channels are zero (sensor off / out of bed)
    mask_all_zero = (df[channel_cols] == 0).all(axis=1)
    n_all_zero = mask_all_zero.sum()
    if n_all_zero > 0:
        print(f"Dropping {n_all_zero} rows where all channels are zero.")
        df = df[~mask_all_zero].copy()

    # Sort by Subject and Time
    df = df.sort_values(["Subject", "Time"]).reset_index(drop=True)

    print(f"Window parameters:")
    print(f"  - Window size: {WIN_SIZE} samples ({WINDOW_SEC} seconds)")
    print(f"  - Min apnea fraction: {MIN_FRAC_APNEA}")
    print(f"  - Min non-apnea fraction: {MIN_FRAC_NONAPNEA}")

    windows = []
    skipped_windows = 0

    # Process each subject
    for subj, sub_df in df.groupby("Subject"):
        sub_df = sub_df.sort_values("Time")
        n = len(sub_df)
        num_full = n // WIN_SIZE
        print(f"  Subject {subj}: {n} samples -> {num_full} windows", end="")

        subj_windows = 0
        for w in range(num_full):
            start_idx = w * WIN_SIZE
            end_idx = start_idx + WIN_SIZE
            chunk = sub_df.iloc[start_idx:end_idx]

            if len(chunk) < WIN_SIZE:
                continue

            # Label handling (Status column)
            statuses = chunk["Status"].to_numpy()
            vals, counts = np.unique(statuses, return_counts=True)
            majority_status = int(vals[counts.argmax()])

            # Fractions of different status types
            frac_apnea = np.mean(statuses == 3)      # apnea
            frac_altro = np.mean(statuses == 4)      # other/noise
            frac_respiro = np.mean(np.isin(statuses, [0, 1, 2]))  # normal breathing

            # Determine binary label
            label_apnea = None
            if majority_status == 3 and frac_apnea >= MIN_FRAC_APNEA:
                label_apnea = 1  # Apnea
            elif majority_status in (0, 1, 2) and frac_respiro >= MIN_FRAC_NONAPNEA:
                label_apnea = 0  # Non-apnea
            else:
                skipped_windows += 1
                continue  # Skip ambiguous windows

            # Position mode
            pos_mode_series = chunk["Position"].dropna().mode()
            pos_mode = pos_mode_series.iloc[0] if not pos_mode_series.empty else np.nan

            # Time range
            start_time = float(chunk["Time"].iloc[0])
            end_time = float(chunk["Time"].iloc[-1])

            # Flatten signals
            signals = chunk[channel_cols].to_numpy()
            flat = signals.flatten()

            # Build output row
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

            # Add flattened pressure values
            for j, value in enumerate(flat):
                row[f"x{j}"] = float(value)

            windows.append(row)
            subj_windows += 1

        print(f" -> kept {subj_windows}")

    df_win = pd.DataFrame(windows)
    print(f"\n✓ Total windows created: {len(df_win)}")
    print(f"  Skipped (ambiguous): {skipped_windows}")

    # Class balance
    print(f"\nClass distribution:")
    print(f"  Non-apnea (0): {(df_win['label_apnea'] == 0).sum()}")
    print(f"  Apnea (1): {(df_win['label_apnea'] == 1).sum()}")

    df_win.to_csv(WINDOWS_FLAT_CSV, index=False)
    print(f"\n✓ Saved to: {WINDOWS_FLAT_CSV}")

    return df_win


# ============================================================================
# STEP 2: EXTRACT PRESSURE MAT FEATURES
# ============================================================================

def step2_extract_mat_features(df_win=None):
    """
    Extract statistical features from each window's pressure mat data.
    Features: per-channel mean, std, diff_std + global statistics.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Extracting pressure mat features")
    print("=" * 80)

    if df_win is None:
        if not WINDOWS_FLAT_CSV.exists():
            print(f"ERROR: Windows file not found: {WINDOWS_FLAT_CSV}")
            return None
        print(f"\nLoading {WINDOWS_FLAT_CSV} ...")
        df_win = pd.read_csv(WINDOWS_FLAT_CSV)

    # Identify flattened signal columns
    feature_cols = [c for c in df_win.columns if c.startswith("x")]
    expected_cols = WIN_SIZE * N_CHANNELS
    assert len(feature_cols) == expected_cols, \
        f"Expected {expected_cols} feature cols, found {len(feature_cols)}"

    print(f"Processing {len(df_win)} windows...")
    print(f"  WIN_SIZE={WIN_SIZE}, N_CHANNELS={N_CHANNELS}")

    rows = []
    for idx, row in df_win.iterrows():
        flat = row[feature_cols].to_numpy(dtype=np.float32)
        # Reshape to (time x channel): (240, 40)
        sig = flat.reshape(WIN_SIZE, N_CHANNELS)

        # Per-channel statistics
        ch_mean = sig.mean(axis=0)  # (40,)
        ch_std = sig.std(axis=0)    # (40,)

        # Temporal variability (std of differences)
        diff_sig = np.diff(sig, axis=0)  # (239, 40)
        ch_diff_std = diff_sig.std(axis=0)

        # Global statistics
        global_mean = sig.mean()
        global_std = sig.std()
        global_min = sig.min()
        global_max = sig.max()

        # Build feature dictionary
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

        # Add per-channel features
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
    print(f"  - {len(META_COLS)} meta columns")
    print(f"  - {len(GLOBAL_COLS)} global features")
    print(f"  - {N_CHANNELS * 3} channel features ({N_CHANNELS} channels × 3 stats)")

    df_feat.to_csv(FEATURES_MAT_CSV, index=False)
    print(f"\n✓ Saved to: {FEATURES_MAT_CSV}")

    return df_feat


# ============================================================================
# STEP 3: EXTRACT ACCELEROMETER FEATURES
# ============================================================================

def compute_acc_features(acc_segment):
    """Compute features from accelerometer segment."""
    data = acc_segment[ACC_SIGNAL_COLS].to_numpy(dtype=np.float32)

    if data.size == 0:
        return None

    # Global stats
    global_mean = float(data.mean())
    global_std = float(data.std())
    global_min = float(data.min())
    global_max = float(data.max())

    # Per-channel stats
    ch_mean = data.mean(axis=0)
    ch_std = data.std(axis=0)

    # Temporal variability
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
    """
    Extract accelerometer features for each window.
    """
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

    # Group by subject for faster lookup
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
    """
    Merge pressure mat features with accelerometer features.
    """
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
            print(f"WARNING: ACC features not found: {ACC_FEATURES_CSV}")
            print("Proceeding with MAT features only.")
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
    """
    Aggregate per-channel features into anatomical zones.
    Uses CORRECTED zone definitions based on actual 4x10 matrix layout.
    """
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
        print(f"  {zone_name}: channels {ch_list} ({len(ch_list)} channels)")

    # Identify accelerometer columns
    acc_cols = [c for c in df_merged.columns if c.startswith("acc_")]
    print(f"\nFound {len(acc_cols)} ACC feature columns.")

    # Start output dataframe with meta + global + acc columns
    keep_cols = [c for c in META_COLS if c in df_merged.columns]
    keep_cols += [c for c in GLOBAL_COLS if c in df_merged.columns]
    keep_cols += acc_cols

    df_out = df_merged[keep_cols].copy()

    # Build mapping from (channel_index, stat_type) -> column_name
    available_ch_stats = {}
    for ch_idx in range(1, 41):
        for stat in CHANNEL_STAT_TYPES:
            col_name = f"ch{ch_idx}_{stat}"
            if col_name in df_merged.columns:
                available_ch_stats[(ch_idx, stat)] = col_name

    # Calculate zone features
    print("\nGenerating zone features:")
    for zone_name, ch_list in ZONE_DEFS_CORRECTED.items():
        for stat in CHANNEL_STAT_TYPES:
            # Get all columns for this zone and stat
            cols = [
                available_ch_stats[(ch, stat)]
                for ch in ch_list
                if (ch, stat) in available_ch_stats
            ]

            if not cols:
                print(f"  WARNING: no columns found for {zone_name}, stat={stat}")
                continue

            zone_mat = df_merged[cols]

            # Aggregate: mean and std across channels in the zone
            df_out[f"{zone_name}_{stat}_mean"] = zone_mat.mean(axis=1)
            df_out[f"{zone_name}_{stat}_std"] = zone_mat.std(axis=1)

        print(f"  ✓ {zone_name}: 6 features created")

    # Summary
    zone_features = [c for c in df_out.columns if c.startswith('zone_')]
    print(f"\n✓ Output dataset shape: {df_out.shape}")
    print(f"  - Meta columns: {len([c for c in META_COLS if c in df_out.columns])}")
    print(f"  - Global features: {len([c for c in GLOBAL_COLS if c in df_out.columns])}")
    print(f"  - Accelerometer features: {len(acc_cols)}")
    print(f"  - Zone features: {len(zone_features)}")
    print(f"  - TOTAL: {df_out.shape[1]} columns")

    df_out.to_csv(FEATURES_ZONES_CSV, index=False)
    print(f"\n✓ Saved to: {FEATURES_ZONES_CSV}")

    # Print zone features
    print(f"\nZone features created ({len(zone_features)}):")
    for feat in sorted(zone_features):
        print(f"  - {feat}")

    return df_out


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(" " * 15 + "SLEEP APNEA DETECTION - DATASET PREPARATION")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  - Data directory: {DATA_DIR}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print(f"  - Sampling frequency: {FS} Hz")
    print(f"  - Window duration: {WINDOW_SEC} seconds")
    print(f"  - Channels: {N_CHANNELS}")

    # Check if raw data exists
    print(f"\nChecking input files:")
    print(f"  - MAT raw: {MAT_RAW_CSV} {'✓' if MAT_RAW_CSV.exists() else '✗ NOT FOUND'}")
    print(f"  - ACC raw: {ACC_RAW_CSV} {'✓' if ACC_RAW_CSV.exists() else '✗ NOT FOUND'}")

    if not MAT_RAW_CSV.exists():
        print("\nERROR: Raw MAT data not found!")
        print("Please ensure the raw data files are in the data/ directory.")
        print("\nExpected files:")
        print(f"  {MAT_RAW_CSV}")
        print(f"  {ACC_RAW_CSV}")
        return

    # Run pipeline
    print("\n" + "=" * 80)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 80)

    # Step 1: Create windows
    df_win = step1_create_windows()
    if df_win is None:
        return

    # Step 2: Extract MAT features
    df_mat_feat = step2_extract_mat_features(df_win)
    if df_mat_feat is None:
        return

    # Step 3: Extract ACC features
    df_acc_feat = step3_extract_acc_features(df_win)

    # Step 4: Merge features
    df_merged = step4_merge_features(df_mat_feat, df_acc_feat)
    if df_merged is None:
        return

    # Step 5: Create zone features
    df_final = step5_create_zone_features(df_merged)
    if df_final is None:
        return

    # Final summary
    print("\n" + "=" * 80)
    print(" " * 25 + "PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nFinal dataset: {FEATURES_ZONES_CSV}")
    print(f"  Shape: {df_final.shape}")
    print(f"  Windows: {len(df_final)}")
    print(f"  Subjects: {df_final['Subject'].nunique()}")

    print(f"\nClass distribution:")
    print(f"  Non-apnea (0): {(df_final['label_apnea'] == 0).sum()} ({100*(df_final['label_apnea']==0).sum()/len(df_final):.1f}%)")
    print(f"  Apnea (1): {(df_final['label_apnea'] == 1).sum()} ({100*(df_final['label_apnea']==1).sum()/len(df_final):.1f}%)")

    print(f"\nOutput files in {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("You can now run 'python run_complete_analysis.py' for the full analysis.")
    print("=" * 80)


if __name__ == "__main__":
    main()
