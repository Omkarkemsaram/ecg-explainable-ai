"""ECG-XAI: Interpretable ECG Signal Analysis using Feature-Based Machine Learning and SHAP Explainability.

An end-to-end clinical decision-support pipeline for 12-lead ECG signals:
- Multi-stage signal denoising (DC, baseline wander, notch, bandpass, wavelet)
- High-dimensional feature engineering (780 clinical & HRV features across 12 leads)
- Mutual Information feature selection (top 40 / 70 features)
- Multilabel classification (Random Forest, XGBoost, LightGBM, MLP)
- TreeSHAP & lead-wise anatomical explainability
- Interactive expert clinical dashboard
"""

__version__ = "1.0.0"
__author__ = "Kemsaram Omkar"

from .preprocessing import (
    preprocess_12lead_ecg,
    remove_dc_offset,
    correct_baseline_wander,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_wavelet_denoising,
    check_signal_quality,
)
from .feature_extraction import (
    extract_all_features_12lead,
    extract_lead_features,
    pan_tompkins_qrs_detector,
    LEAD_NAMES_12,
)
from .feature_selection import (
    MutualInformationSelector,
    select_top_features,
)
from .dataset import (
    PTBXLDataset,
    DIAGNOSTIC_CLASSES_5,
)
from .models import (
    build_rf_classifier,
    build_xgboost_classifier,
    build_lightgbm_classifier,
    PyTorchMLP,
    ModelEvaluator,
)
from .explainability import (
    ECGExplainer,
    aggregate_shap_by_lead,
)
from .pipeline import ECGDiagnosisPipeline

__all__ = [
    "preprocess_12lead_ecg",
    "remove_dc_offset",
    "correct_baseline_wander",
    "apply_notch_filter",
    "apply_bandpass_filter",
    "apply_wavelet_denoising",
    "check_signal_quality",
    "extract_all_features_12lead",
    "extract_lead_features",
    "pan_tompkins_qrs_detector",
    "LEAD_NAMES_12",
    "MutualInformationSelector",
    "select_top_features",
    "PTBXLDataset",
    "DIAGNOSTIC_CLASSES_5",
    "build_rf_classifier",
    "build_xgboost_classifier",
    "build_lightgbm_classifier",
    "PyTorchMLP",
    "ModelEvaluator",
    "ECGExplainer",
    "aggregate_shap_by_lead",
    "ECGDiagnosisPipeline",
]
