"""Command-Line Inference Tool for 12-Lead ECG Records.

Loads a 12-lead ECG signal, performs preprocessing, extracts clinical features,
runs model prediction, and outputs lead-wise anatomical SHAP attributions.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ecg_xai.pipeline import ECGDiagnosisPipeline
from ecg_xai.dataset import generate_synthetic_ecg_12lead


def main():
    parser = argparse.ArgumentParser(description="Run ECG Diagnostic Pipeline Inference")
    parser.add_argument("--file", type=str, default=None, help="Path to .npy or .csv containing (T, 12) signal")
    parser.add_argument("--condition", type=str, default="MI", help="Simulated condition for demo: NORM, MI, STTC, CD, HYP")
    parser.add_argument("--fs", type=float, default=500.0)
    parser.add_argument("--output", type=str, default=None, help="Output path for JSON report")
    args = parser.parse_args()

    if args.file is not None:
        p = Path(args.file)
        if p.suffix == ".npy":
            sig = np.load(p)
        else:
            sig = pd.read_csv(p).to_numpy()
        print(f"Loaded ECG signal from {p} with shape: {sig.shape}")
    else:
        print(f"Simulating test 12-lead ECG for condition: {args.condition}...")
        sig = generate_synthetic_ecg_12lead(
            duration_sec=10.0,
            fs=args.fs,
            condition=args.condition,
            heart_rate_bpm=78.0,
            noise_level=0.04,
            random_state=101,
        )

    pipeline = ECGDiagnosisPipeline(fs=args.fs, top_k=40)
    print("Running full diagnostic pipeline (Preprocessing -> 780 Features -> Top 40 -> Inference -> SHAP)...")
    result = pipeline.predict_from_signal(sig)

    print("\n================ DIAGNOSTIC REPORT ================")
    print("Detected Conditions:", result["predicted_conditions"])
    print("Class Probabilities:")
    for cls, prob in result["class_probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {cls:<6} : {prob*100:5.1f}% [{bar:<30}]")

    print("\nSignal Quality & Preprocessing Metrics:")
    print(f"  SNR: {result['preprocessing_metrics']['snr_db']:.2f} dB")
    print(f"  Extracted Features: {result['extracted_features_count']}")
    print(f"  Selected Model Features: {result['selected_features_count']}")

    print("\nLead-Wise Anatomical Importance (Section 4.7.3 & Figure 14):")
    for row in result["lead_importance"][:6]:
        print(f"  Lead {row['lead']:<4} ({row['anatomical_region']:<25}): Total Impact={row['abs_shap']:6.3f} | Top: {row['top_features']}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_serial = dict(result)
        res_serial["cleaned_signal_shape"] = list(res_serial["cleaned_signal_shape"])
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res_serial, f, indent=2)
        print(f"\nSaved diagnostic report to: {out_p}")


if __name__ == "__main__":
    main()
