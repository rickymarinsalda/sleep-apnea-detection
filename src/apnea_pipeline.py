#!/usr/bin/env python3
"""Reusable training and inference utilities for sleep apnea detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample


FS = 8.0
WINDOW_SEC = 30
WIN_SIZE = int(FS * WINDOW_SEC)
N_CHANNELS = 40
RANDOM_STATE = 42

ACC_TIME_COL = "Time_acc1"
ACC_SUBJ_COL = "Subject"
ACC_SIGNAL_COLS = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]

META_COLS = [
    "Subject",
    "start_time",
    "end_time",
    "Position_mode",
    "majority_status",
    "label_apnea",
    "frac_apnea",
    "frac_altro",
    "frac_respiro",
]
GLOBAL_COLS = ["global_mean", "global_std", "global_min", "global_max"]
EXCLUDE_COLS = set(META_COLS)

ZONE_DEFS_CORRECTED = {
    "zone_UL": [1, 2, 3, 4, 5, 11, 12, 13, 14, 15],
    "zone_UR": [6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    "zone_LL": [21, 22, 23, 24, 25, 31, 32, 33, 34, 35],
    "zone_LR": [26, 27, 28, 29, 30, 36, 37, 38, 39, 40],
}

TEMPORAL_BASE_FEATURES = [
    "global_mean",
    "global_std",
    "global_max",
    "zone_UL_mean_mean",
    "zone_UR_mean_mean",
    "zone_LL_mean_mean",
    "zone_LR_mean_mean",
    "zone_UL_diff_std_mean",
    "zone_UR_diff_std_mean",
    "zone_LL_diff_std_mean",
    "zone_LR_diff_std_mean",
    "acc_global_mean",
    "acc_global_std",
]


def add_temporal_features(df: pd.DataFrame, window_size: int = 7) -> pd.DataFrame:
    """Add backward-looking temporal features per subject."""
    base_feats = [feat for feat in TEMPORAL_BASE_FEATURES if feat in df.columns]
    temporal = pd.DataFrame(index=df.index)

    for feat in base_feats:
        grouped = df.groupby("Subject")[feat]
        temporal[f"delta_{feat}"] = grouped.diff().fillna(0)
        temporal[f"roll{window_size}_mean_{feat}"] = grouped.transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        temporal[f"roll{window_size}_std_{feat}"] = grouped.transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        ).fillna(0)
        temporal[f"trend_{feat}"] = df[feat] - temporal[f"roll{window_size}_mean_{feat}"]

    return temporal


def baseline_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in EXCLUDE_COLS]


def _balanced_training_set(
    X: pd.DataFrame, y: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    X_neg = X[y == 0]
    X_pos = X[y == 1]
    y_neg = y[y == 0]
    y_pos = y[y == 1]

    if len(X_pos) == 0 or len(X_neg) == 0:
        raise ValueError("Training requires both classes: label_apnea=0 and label_apnea=1.")

    X_pos_up, y_pos_up = resample(
        X_pos,
        y_pos,
        replace=True,
        n_samples=len(X_neg),
        random_state=random_state,
    )

    X_bal = pd.concat([X_neg, X_pos_up], axis=0)
    y_bal = pd.concat([y_neg, y_pos_up], axis=0)
    return X_bal, y_bal


def train_final_rf_temporal(
    df: pd.DataFrame,
    dataset_path: str,
    k_features: int = 70,
    rolling_window: int = 7,
    threshold: float = 0.25,
    n_estimators: int = 400,
    min_samples_leaf: int = 3,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Train the final RF+temporal model on all labelled windows."""
    if "label_apnea" not in df.columns:
        raise ValueError("Dataset must contain label_apnea for training.")

    df = df.sort_values(["Subject", "start_time"]).reset_index(drop=True)
    feature_cols = baseline_feature_columns(df)
    y = df["label_apnea"].astype(int)

    temporal = add_temporal_features(df, window_size=rolling_window)
    X_full = pd.concat([df[feature_cols].reset_index(drop=True), temporal], axis=1)

    X_bal, y_bal = _balanced_training_set(X_full, y, random_state=random_state)

    k_used = min(k_features, X_bal.shape[1])
    selector = SelectKBest(f_classif, k=k_used)
    X_selected = selector.fit_transform(X_bal, y_bal)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_selected, y_bal)

    selected_features = X_full.columns[selector.get_support()].tolist()
    class_counts = y.value_counts().sort_index().to_dict()

    return {
        "pipeline_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": dataset_path,
        "model_type": "rf_temporal",
        "model": model,
        "selector": selector,
        "threshold": threshold,
        "k_features": k_used,
        "rolling_window": rolling_window,
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
        "fs": FS,
        "window_sec": WINDOW_SEC,
        "win_size": WIN_SIZE,
        "baseline_features": feature_cols,
        "all_model_features": X_full.columns.tolist(),
        "selected_features": selected_features,
        "temporal_base_features": [f for f in TEMPORAL_BASE_FEATURES if f in df.columns],
        "training_summary": {
            "n_windows": int(len(df)),
            "n_subjects": int(df["Subject"].nunique()),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "n_features_before_selection": int(X_full.shape[1]),
            "n_features_after_selection": int(k_used),
        },
        "versions": {
            "python_package_sklearn": sklearn.__version__,
            "python_package_pandas": pd.__version__,
            "python_package_numpy": np.__version__,
        },
    }


def _mode_or_nan(series: pd.Series) -> Any:
    mode = series.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def _label_fields(chunk: pd.DataFrame) -> dict[str, Any]:
    if "Status" not in chunk.columns:
        return {
            "majority_status": np.nan,
            "label_apnea": np.nan,
            "frac_apnea": np.nan,
            "frac_altro": np.nan,
            "frac_respiro": np.nan,
        }

    statuses = chunk["Status"].to_numpy()
    vals, counts = np.unique(statuses, return_counts=True)
    majority_status = vals[counts.argmax()] if len(vals) else np.nan

    frac_apnea = float(np.mean(statuses == 3))
    frac_altro = float(np.mean(statuses == 4))
    frac_respiro = float(np.mean(np.isin(statuses, [0, 1, 2])))

    label = np.nan
    if majority_status == 3 and frac_apnea >= 0.5:
        label = 1
    elif majority_status in (0, 1, 2) and frac_respiro >= 0.5:
        label = 0

    return {
        "majority_status": majority_status,
        "label_apnea": label,
        "frac_apnea": frac_apnea,
        "frac_altro": frac_altro,
        "frac_respiro": frac_respiro,
    }


def _compute_mat_features(chunk: pd.DataFrame, channel_cols: list[str]) -> dict[str, float]:
    sig = chunk[channel_cols].to_numpy(dtype=np.float32)
    ch_mean = sig.mean(axis=0)
    ch_std = sig.std(axis=0)
    ch_diff_std = np.diff(sig, axis=0).std(axis=0)

    feats: dict[str, float] = {
        "global_mean": float(sig.mean()),
        "global_std": float(sig.std()),
        "global_min": float(sig.min()),
        "global_max": float(sig.max()),
    }

    channel_stats = {
        "mean": ch_mean,
        "std": ch_std,
        "diff_std": ch_diff_std,
    }
    for zone_name, channels in ZONE_DEFS_CORRECTED.items():
        idx = [ch - 1 for ch in channels]
        for stat_name, values in channel_stats.items():
            zone_values = values[idx]
            feats[f"{zone_name}_{stat_name}_mean"] = float(np.mean(zone_values))
            feats[f"{zone_name}_{stat_name}_std"] = float(np.std(zone_values, ddof=1))

    return feats


def _compute_acc_features(acc_segment: pd.DataFrame) -> dict[str, float] | None:
    if acc_segment.empty:
        return None

    missing_cols = [col for col in ACC_SIGNAL_COLS if col not in acc_segment.columns]
    if missing_cols:
        raise ValueError(f"ACC file is missing columns: {missing_cols}")

    data = acc_segment[ACC_SIGNAL_COLS].to_numpy(dtype=np.float32)
    diff = np.diff(data, axis=0)
    ch_mean = data.mean(axis=0)
    ch_std = data.std(axis=0)
    ch_diff_std = diff.std(axis=0) if diff.size else np.zeros_like(ch_mean)

    feats = {
        "acc_global_mean": float(data.mean()),
        "acc_global_std": float(data.std()),
        "acc_global_min": float(data.min()),
        "acc_global_max": float(data.max()),
    }
    for col, mean, std, diff_std in zip(ACC_SIGNAL_COLS, ch_mean, ch_std, ch_diff_std):
        feats[f"acc_{col}_mean"] = float(mean)
        feats[f"acc_{col}_std"] = float(std)
        feats[f"acc_{col}_diff_std"] = float(diff_std)
    return feats


def build_feature_dataset_from_raw(
    mat_csv: Path,
    acc_csv: Path,
    fs: float = FS,
    window_sec: int = WINDOW_SEC,
    drop_all_zero: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create model-ready window features from raw MAT and ACC CSV files."""
    mat_csv = Path(mat_csv)
    acc_csv = Path(acc_csv)
    if not mat_csv.exists():
        raise FileNotFoundError(f"MAT raw CSV not found: {mat_csv}")
    if not acc_csv.exists():
        raise FileNotFoundError(f"ACC raw CSV not found: {acc_csv}")

    win_size = int(fs * window_sec)
    channel_cols = [f"ch{i}" for i in range(1, N_CHANNELS + 1)]

    df_mat = pd.read_csv(mat_csv)
    missing_mat_cols = [col for col in ["Time", "Subject", *channel_cols] if col not in df_mat.columns]
    if missing_mat_cols:
        raise ValueError(f"MAT file is missing columns: {missing_mat_cols}")

    if drop_all_zero:
        mask_all_zero = (df_mat[channel_cols] == 0).all(axis=1)
        df_mat = df_mat.loc[~mask_all_zero].copy()

    df_mat = df_mat.sort_values(["Subject", "Time"]).reset_index(drop=True)

    df_acc = pd.read_csv(acc_csv)
    missing_acc_cols = [col for col in [ACC_TIME_COL, ACC_SUBJ_COL, *ACC_SIGNAL_COLS] if col not in df_acc.columns]
    if missing_acc_cols:
        raise ValueError(f"ACC file is missing columns: {missing_acc_cols}")
    acc_by_subject = {subj: sub_df.sort_values(ACC_TIME_COL) for subj, sub_df in df_acc.groupby(ACC_SUBJ_COL)}

    rows: list[dict[str, Any]] = []
    stats = {
        "mat_rows_after_zero_drop": int(len(df_mat)),
        "candidate_windows": 0,
        "kept_windows": 0,
        "skipped_missing_acc": 0,
    }

    for subj, sub_df in df_mat.groupby("Subject", sort=False):
        sub_df = sub_df.sort_values("Time")
        n_windows = len(sub_df) // win_size
        stats["candidate_windows"] += int(n_windows)
        acc_subj = acc_by_subject.get(subj)

        for window_idx in range(n_windows):
            start_idx = window_idx * win_size
            chunk = sub_df.iloc[start_idx : start_idx + win_size]
            start_time = float(chunk["Time"].iloc[0])
            end_time = float(chunk["Time"].iloc[-1])

            if acc_subj is None:
                stats["skipped_missing_acc"] += 1
                continue

            acc_segment = acc_subj.loc[
                (acc_subj[ACC_TIME_COL] >= start_time) & (acc_subj[ACC_TIME_COL] < end_time)
            ]
            acc_feats = _compute_acc_features(acc_segment)
            if acc_feats is None:
                stats["skipped_missing_acc"] += 1
                continue

            row: dict[str, Any] = {
                "Subject": subj,
                "start_time": start_time,
                "end_time": end_time,
                "Position_mode": _mode_or_nan(chunk["Position"]) if "Position" in chunk.columns else np.nan,
            }
            row.update(_label_fields(chunk))
            row.update(_compute_mat_features(chunk, channel_cols))
            row.update(acc_feats)
            rows.append(row)

    features = pd.DataFrame(rows)
    if not features.empty:
        ordered = [col for col in META_COLS + GLOBAL_COLS if col in features.columns]
        acc_cols = [col for col in features.columns if col.startswith("acc_")]
        zone_cols = [col for col in features.columns if col.startswith("zone_")]
        features = features[ordered + acc_cols + zone_cols]
        features = features.sort_values(["Subject", "start_time"]).reset_index(drop=True)

    stats["kept_windows"] = int(len(features))
    return features, stats


def predict_feature_dataset(
    df_features: pd.DataFrame,
    artifact: dict[str, Any],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Predict apnea probabilities and labels from precomputed window features."""
    if df_features.empty:
        raise ValueError("No windows available for prediction.")

    df_features = df_features.sort_values(["Subject", "start_time"]).reset_index(drop=True)
    baseline_features = artifact["baseline_features"]
    all_model_features = artifact["all_model_features"]

    missing_baseline = [col for col in baseline_features if col not in df_features.columns]
    if missing_baseline:
        raise ValueError(f"Feature dataset is missing required columns: {missing_baseline}")

    rolling_window = int(artifact["rolling_window"])
    temporal = add_temporal_features(df_features, window_size=rolling_window)
    X_full = pd.concat([df_features[baseline_features].reset_index(drop=True), temporal], axis=1)

    missing_model_features = [col for col in all_model_features if col not in X_full.columns]
    if missing_model_features:
        raise ValueError(f"Model feature matrix is missing columns: {missing_model_features}")
    X_full = X_full[all_model_features]

    X_selected = artifact["selector"].transform(X_full)
    probabilities = artifact["model"].predict_proba(X_selected)[:, 1]
    decision_threshold = float(artifact["threshold"] if threshold is None else threshold)
    labels = (probabilities >= decision_threshold).astype(int)

    output_cols = [
        col
        for col in [
            "Subject",
            "start_time",
            "end_time",
            "Position_mode",
            "majority_status",
            "label_apnea",
            "frac_apnea",
            "frac_altro",
            "frac_respiro",
        ]
        if col in df_features.columns
    ]
    out = df_features[output_cols].copy()
    out["apnea_probability"] = probabilities
    out["predicted_label"] = labels
    out["prediction_threshold"] = decision_threshold
    return out


def evaluation_summary(predictions: pd.DataFrame) -> dict[str, Any] | None:
    """Return metrics when ground-truth labels are present in predictions."""
    if "label_apnea" not in predictions.columns:
        return None

    valid = predictions["label_apnea"].notna()
    valid &= predictions["label_apnea"].isin([0, 1])
    if valid.sum() == 0:
        return None

    y_true = predictions.loc[valid, "label_apnea"].astype(int)
    y_pred = predictions.loc[valid, "predicted_label"].astype(int)
    y_prob = predictions.loc[valid, "apnea_probability"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    summary: dict[str, Any] = {
        "n_evaluated": int(valid.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    summary["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() == 2 else None
    return summary
