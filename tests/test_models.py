import pytest
import numpy as np
import pandas as pd
from ecg_xai.models import build_rf_classifier, PyTorchMLP, ModelEvaluator
from ecg_xai.explainability import ECGExplainer, aggregate_shap_by_lead

def test_model_training_and_eval():
    X = np.random.randn(30, 40).astype(np.float32)
    Y = (np.random.rand(30, 5) > 0.6).astype(int)

    rf = build_rf_classifier(n_estimators=10)
    rf.fit(X, Y)
    preds = rf.predict(X)
    metrics = ModelEvaluator.evaluate(Y, preds)
    assert 0.0 <= metrics["micro_f1"] <= 1.0

def test_lead_wise_shap_aggregation():
    feature_names = ["I_qrs_duration_ms", "I_st_segment_level", "V1_r_amp", "V2_st_dev", "II_mean_rr_ms"]
    sv = np.array([0.25, -0.15, 0.40, 0.30, 0.05])
    df_leads = aggregate_shap_by_lead(feature_names, sv)
    assert not df_leads.empty
    assert "lead" in df_leads.columns
    assert "abs_shap" in df_leads.columns
