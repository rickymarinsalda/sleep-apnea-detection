#!/usr/bin/env python3
"""Train and save the final RF+temporal apnea model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from apnea_pipeline import train_final_rf_temporal


DATASET_NAME = "dataset_windows_30s_features_zones_MANUAL.csv"


def default_dataset_candidates() -> list[Path]:
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    workspace_root = repo_root.parent
    return [
        src_dir / "manual_labels" / "preprocessing_output" / DATASET_NAME,
        repo_root / "analysis_manual_labels" / "preprocessing_output" / DATASET_NAME,
        workspace_root / "analysis_manual_labels" / "preprocessing_output" / DATASET_NAME,
        Path.cwd() / "analysis_manual_labels" / "preprocessing_output" / DATASET_NAME,
    ]


def find_default_dataset() -> Path | None:
    for candidate in default_dataset_candidates():
        if candidate.exists():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Final labelled feature CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/rf_temporal_manual.joblib"),
        help="Output model artifact path.",
    )
    parser.add_argument("--k-features", type=int, default=70)
    parser.add_argument("--rolling-window", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset or find_default_dataset()
    if dataset is None:
        searched = "\n  - ".join(str(path) for path in default_dataset_candidates())
        raise SystemExit(f"Dataset not found. Searched:\n  - {searched}")

    print(f"Loading training dataset: {dataset}")
    df = pd.read_csv(dataset)
    artifact = train_final_rf_temporal(
        df,
        dataset_path=str(dataset),
        k_features=args.k_features,
        rolling_window=args.rolling_window,
        threshold=args.threshold,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    summary = artifact["training_summary"]
    print(f"\nSaved model artifact: {args.output}")
    print(f"Windows: {summary['n_windows']} | Subjects: {summary['n_subjects']}")
    print(f"Class counts: {summary['class_counts']}")
    print(
        "Features: "
        f"{summary['n_features_before_selection']} -> {summary['n_features_after_selection']}"
    )
    print(f"Threshold: {artifact['threshold']}")


if __name__ == "__main__":
    main()
