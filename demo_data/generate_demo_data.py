"""Generate Zero-Setup Demo Data and Pre-Trained Weights.

Creates sample 12-lead ECG signals and precomputed feature sets covering
all 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP) for immediate testing.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ecg_xai.dataset import generate_synthetic_ecg_12lead, DIAGNOSTIC_CLASSES_5, LEADS_12
from ecg_xai.feature_extraction import extract_all_features_12lead
from ecg_xai.feature_selection import MutualInformationSelector
from ecg_xai.models import build_xgboost_classifier, build_rf_classifier

def main():
    demo_dir = BASE / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic 12-lead ECG demo dataset...")
    records = []
    features_list = []
    targets_list = []

    # Generate 50 sample patient recordings across the 5 conditions
    conditions = ["NORM", "MI", "STTC", "CD", "HYP"]
    patient_id = 1

    for cond_idx, cond in enumerate(conditions):
        for rep in range(10):
            hr = 60.0 + (rep * 3.5)
            # 10s at 500Hz = 5000 samples
            sig = generate_synthetic_ecg_12lead(
                duration_sec=10.0,
                fs=500.0,
                condition=cond,
                heart_rate_bpm=hr,
                noise_level=0.03,
                random_state=42 + patient_id,
            )

            # Store preview downsampled to 100Hz (1000 samples) for lightweight web rendering
            sig_preview = sig[::5, :]

            rec_info = {
                "patient_id": f"PTB_{patient_id:04d}",
                "condition": cond,
                "heart_rate": float(round(hr, 1)),
                "duration_sec": 10.0,
                "fs": 500,
                "leads": LEADS_12,
                "waveform": sig_preview.tolist(),
            }
            records.append(rec_info)

            # Extract 780 features
            feats = extract_all_features_12lead(sig, fs=500)
            feats["patient_id"] = rec_info["patient_id"]
            features_list.append(feats)

            # One-hot target vector
            target_vec = [1 if c == cond else 0 for c in DIAGNOSTIC_CLASSES_5]
            targets_list.append(target_vec)

            patient_id += 1

    # Save records JSON
    records_path = demo_dir / "sample_12lead_records.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} sample ECG patient waveforms to {records_path}")

    # Save features DataFrame
    df_features = pd.DataFrame(features_list).set_index("patient_id")
    df_targets = pd.DataFrame(targets_list, columns=DIAGNOSTIC_CLASSES_5, index=df_features.index)

    features_path = demo_dir / "sample_features.csv"
    df_features.to_csv(features_path)
    print(f"Saved 780-feature matrix of shape {df_features.shape} to {features_path}")

    targets_path = demo_dir / "sample_targets.csv"
    df_targets.to_csv(targets_path)
    print(f"Saved target labels to {targets_path}")

    # Feature Selection: Top 40 features
    print("Running Mutual Information feature selection for top 40 features...")
    selector = MutualInformationSelector(top_k=40, random_state=42)
    X_top40 = selector.fit_transform(df_features, df_targets)

    schema_path = demo_dir / "selected_features_schema.json"
    selector.save_schema(schema_path)
    print(f"Saved top-40 feature schema to {schema_path}")

    # Fit a fast classifier and save weights
    print("Training baseline model for zero-setup dashboard...")
    model = build_xgboost_classifier(n_estimators=100)
    model.fit(X_top40.to_numpy(), df_targets.to_numpy())

    model_path = demo_dir / "pretrained_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved pre-trained classifier to {model_path}")
    print("Zero-setup demo artifacts generated successfully!")

if __name__ == "__main__":
    main()
