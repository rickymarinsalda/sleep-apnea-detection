#!/usr/bin/env python3
"""Run apnea inference on new raw MAT/ACC CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from apnea_pipeline import (
    build_feature_dataset_from_raw,
    evaluation_summary,
    predict_feature_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/rf_temporal_manual.joblib"))
    parser.add_argument("--mat", type=Path, required=True, help="Raw pressure mat CSV.")
    parser.add_argument("--acc", type=Path, required=True, help="Raw accelerometer CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions/raw_predictions.csv"),
        help="Output predictions CSV.",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=None,
        help="Optional output path for generated window features.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override saved threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model)

    print(f"Loaded model: {args.model}")
    print(f"Building windows/features from:\n  MAT: {args.mat}\n  ACC: {args.acc}")
    features, preprocessing_stats = build_feature_dataset_from_raw(
        args.mat,
        args.acc,
        fs=float(artifact["fs"]),
        window_sec=int(artifact["window_sec"]),
    )

    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(args.features_output, index=False)
        print(f"Saved generated features: {args.features_output}")

    predictions = predict_feature_dataset(features, artifact, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    n_windows = len(predictions)
    n_apnea = int(predictions["predicted_label"].sum())
    print(f"\nSaved predictions: {args.output}")
    print(f"Windows predicted: {n_windows}")
    print(f"Predicted apnea windows: {n_apnea} ({100 * n_apnea / n_windows:.1f}%)")
    print(f"Preprocessing: {json.dumps(preprocessing_stats, sort_keys=True)}")

    per_subject = (
        predictions.groupby("Subject")
        .agg(
            windows=("predicted_label", "size"),
            predicted_apnea=("predicted_label", "sum"),
            mean_probability=("apnea_probability", "mean"),
            max_probability=("apnea_probability", "max"),
        )
        .reset_index()
    )
    print("\nPer-subject summary:")
    print(per_subject.to_string(index=False))

    metrics = evaluation_summary(predictions)
    if metrics is not None:
        print("\nEvaluation against labels found in raw Status:")
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
