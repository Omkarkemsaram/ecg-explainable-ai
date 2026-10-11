"""Comparative Evaluation Script (40 vs 70 features across all models).

Reproduces Figures 3 and 4 of the RM6 paper:
- Micro F1, Macro F1, Micro AUC, Macro AUC, Hamming Loss
- Comparison across Random Forest, XGBoost, LightGBM, and PyTorch MLP
- Comparison across 40 selected features and 70 selected features
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Evaluate 40 vs 70 feature subsets")
    parser.add_argument("--features", type=str, default=str(BASE / "demo_data" / "sample_features.csv"))
    parser.add_argument("--targets", type=str, default=str(BASE / "demo_data" / "sample_targets.csv"))
    args = parser.parse_args()

    feat_path = Path(args.features)
    targ_path = Path(args.targets)

    print("=======================================================================")
    print("  RUNNING BENCHMARK EVALUATION: 40 FEATURES vs 70 FEATURES (SECTION 5)  ")
    print("=======================================================================")

    res_40 = run_training(feat_path, targ_path, top_k=40, output_dir=BASE / "artifacts" / "k40")
    res_70 = run_training(feat_path, targ_path, top_k=70, output_dir=BASE / "artifacts" / "k70")

    df_40 = pd.DataFrame(res_40).T
    df_40["Feature_Set"] = "40 Features"

    df_70 = pd.DataFrame(res_70).T
    df_70["Feature_Set"] = "70 Features"

    combined = pd.concat([df_40, df_70])
    combined_path = BASE / "artifacts" / "comparison_40_vs_70_features.csv"
    combined.to_csv(combined_path)

    print("
==================== FINAL COMPARATIVE SUMMARY ====================")
    print(combined.to_string())
    print(f"Saved final comparison table to {combined_path}")


if __name__ == "__main__":
    main()
