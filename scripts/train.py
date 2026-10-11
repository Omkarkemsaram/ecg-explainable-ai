"""Training and Evaluation Script for Multilabel ECG Disease Classification.

Reproduces Section 5 benchmarks from the RM6 paper across:
- Random Forest (350 trees, max depth 20)
- XGBoost (300 estimators, lr=0.05, max depth 6)
- LightGBM (400 estimators, lr=0.05, leaves 31)
- PyTorch MLP ([40 -> 128 -> 64 -> 32 -> 5] with BatchNorm & Dropout 0.3)
- Feature sets: 40 features vs 70 features
- Class imbalance: SMOTE on training split (70% train, 10% val, 20% test)
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from ecg_xai.dataset import PTBXLDataset, DIAGNOSTIC_CLASSES_5
from ecg_xai.feature_selection import MutualInformationSelector
from ecg_xai.models import (
    build_rf_classifier,
    build_xgboost_classifier,
    build_lightgbm_classifier,
    PyTorchMLP,
    ModelEvaluator,
)


def run_training(
    features_csv: Path,
    targets_csv: Path,
    top_k: int = 40,
    use_smote: bool = True,
    output_dir: Path = BASE / "artifacts",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ECG Multi-Label Disease Classification Training (Top {top_k} Features) ===")
    print(f"Loading features from: {features_csv}")
    print(f"Loading targets from:  {targets_csv}")

    df_feats = pd.read_csv(features_csv, index_col=0)
    df_targets = pd.read_csv(targets_csv, index_col=0)

    # 1. Stratified 70/10/20 Splitting (Section 4.6.1)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = PTBXLDataset.split_data(
        df_feats, df_targets,
        train_ratio=0.70, val_ratio=0.10, test_ratio=0.20,
        random_state=42,
    )
    print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. SMOTE on Training Split (Section 4.6.1)
    if use_smote:
        print("Applying SMOTE oversampling to training set to handle class imbalance...")
        X_train_res, Y_train_res = PTBXLDataset.apply_smote(X_train, Y_train, random_state=42)
        print(f"Post-SMOTE training samples: {len(X_train_res)}")
    else:
        X_train_res, Y_train_res = X_train, Y_train

    # 3. Mutual Information Feature Selection (Section 4.5)
    print(f"Selecting top {top_k} features via Mutual Information...")
    selector = MutualInformationSelector(top_k=top_k, random_state=42)
    X_train_sel = selector.fit_transform(X_train_res, Y_train_res)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    # Save schema
    schema_path = output_dir / f"selected_features_k{top_k}.json"
    selector.save_schema(schema_path)
    print(f"Saved feature schema to {schema_path}")

    # 4. Standard Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(np.nan_to_num(X_train_sel.to_numpy(), nan=0.0))
    X_va_s = scaler.transform(np.nan_to_num(X_val_sel.to_numpy(), nan=0.0))
    X_te_s = scaler.transform(np.nan_to_num(X_test_sel.to_numpy(), nan=0.0))

    scaler_path = output_dir / f"scaler_k{top_k}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    results = {}

    # --- Model 1: Random Forest ---
    print("
--- Training Model 1: Random Forest (350 trees) ---")
    rf = build_rf_classifier(n_estimators=350, max_depth=20, min_samples_leaf=5)
    rf.fit(X_tr_s, Y_train_res.to_numpy())
    rf_pred = rf.predict(X_te_s)
    rf_prob = np.array([est.predict_proba(X_te_s)[:, 1] if est.predict_proba(X_te_s).shape[1] > 1 else est.predict_proba(X_te_s)[:, 0] for est in rf.estimators_]).T
    rf_metrics = ModelEvaluator.evaluate(Y_test.to_numpy(), rf_pred, rf_prob)
    results["Random Forest"] = rf_metrics
    with open(output_dir / f"rf_model_k{top_k}.pkl", "wb") as f:
        pickle.dump(rf, f)

    # --- Model 2: XGBoost ---
    print("
--- Training Model 2: XGBoost (300 estimators, lr=0.05) ---")
    xgb = build_xgboost_classifier(n_estimators=300, learning_rate=0.05, max_depth=6)
    xgb.fit(X_tr_s, Y_train_res.to_numpy())
    xgb_pred = xgb.predict(X_te_s)
    xgb_prob = np.array([est.predict_proba(X_te_s)[:, 1] if est.predict_proba(X_te_s).shape[1] > 1 else est.predict_proba(X_te_s)[:, 0] for est in xgb.estimators_]).T
    xgb_metrics = ModelEvaluator.evaluate(Y_test.to_numpy(), xgb_pred, xgb_prob)
    results["XGBoost"] = xgb_metrics
    with open(output_dir / f"xgb_model_k{top_k}.pkl", "wb") as f:
        pickle.dump(xgb, f)

    # --- Model 3: LightGBM ---
    print("
--- Training Model 3: LightGBM (400 estimators, lr=0.05) ---")
    lgb = build_lightgbm_classifier(n_estimators=400, learning_rate=0.05, num_leaves=31)
    lgb.fit(X_tr_s, Y_train_res.to_numpy())
    lgb_pred = lgb.predict(X_te_s)
    lgb_prob = np.array([est.predict_proba(X_te_s)[:, 1] if est.predict_proba(X_te_s).shape[1] > 1 else est.predict_proba(X_te_s)[:, 0] for est in lgb.estimators_]).T
    lgb_metrics = ModelEvaluator.evaluate(Y_test.to_numpy(), lgb_pred, lgb_prob)
    results["LightGBM"] = lgb_metrics
    with open(output_dir / f"lgb_model_k{top_k}.pkl", "wb") as f:
        pickle.dump(lgb, f)

    # --- Model 4: PyTorch MLP ---
    print("
--- Training Model 4: Multi-Layer Perceptron (MLP) ---")
    mlp = PyTorchMLP(in_dim=top_k, n_classes=5, dropout=0.3)
    mlp.fit_model(X_tr_s, Y_train_res.to_numpy(), X_va_s, Y_val.to_numpy(), epochs=50)
    mlp_prob = mlp.predict_proba(X_te_s)
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    mlp_metrics = ModelEvaluator.evaluate(Y_test.to_numpy(), mlp_pred, mlp_prob)
    results["MLP"] = mlp_metrics

    # Print Comparative Results Table (matching Section 5 of paper)
    print("
================== BENCHMARK RESULTS (Section 5) ==================")
    df_res = pd.DataFrame(results).T
    print(df_res.to_string())
    print("===================================================================")

    df_res.to_csv(output_dir / f"benchmark_results_k{top_k}.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG Disease Classification Models")
    parser.add_argument("--features", type=str, default=str(BASE / "demo_data" / "sample_features.csv"))
    parser.add_argument("--targets", type=str, default=str(BASE / "demo_data" / "sample_targets.csv"))
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-smote", action="store_true")
    args = parser.parse_args()

    run_training(
        features_csv=Path(args.features),
        targets_csv=Path(args.targets),
        top_k=args.top_k,
        use_smote=not args.no_smote,
    )
