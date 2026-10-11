import pytest
import numpy as np
from ecg_xai.feature_extraction import (
    pan_tompkins_qrs_detector,
    extract_lead_features,
    extract_all_features_12lead,
    LEAD_NAMES_12,
)
from ecg_xai.dataset import generate_synthetic_ecg_12lead

def test_pan_tompkins_peak_detection():
    # 2 seconds at 500 Hz with 2 sharp peaks
    x = np.zeros(1000, dtype=np.float32)
    x[250] = 1.5
    x[750] = 1.5
    peaks = pan_tompkins_qrs_detector(x, fs=500)
    assert len(peaks) >= 1

def test_feature_extraction_count_780():
    sig_12 = generate_synthetic_ecg_12lead(duration_sec=3.0, fs=500.0, condition="NORM", random_state=42)
    feats = extract_all_features_12lead(sig_12, fs=500.0)
    # 65 features * 12 leads = 780 features
    assert len(feats) == 780
    assert "I_mean" in feats
    assert "V1_qrs_duration_ms" in feats
    assert "II_spectral_entropy" in feats
