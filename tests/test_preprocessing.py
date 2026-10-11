import pytest
import numpy as np
from ecg_xai.preprocessing import (
    remove_dc_offset,
    correct_baseline_wander,
    apply_notch_filter,
    apply_bandpass_filter,
    apply_wavelet_denoising,
    check_signal_quality,
    preprocess_12lead_ecg,
)

def test_remove_dc_offset():
    x = np.ones((500, 12)) * 3.5
    cleaned = remove_dc_offset(x)
    assert np.allclose(np.mean(cleaned, axis=0), 0.0, atol=1e-5)

def test_signal_quality_checks():
    x_valid = np.random.randn(1000, 12).astype(np.float32)
    q = check_signal_quality(x_valid)
    assert q["is_usable"] is True
    assert q["has_nan"] is False

    x_flat = np.zeros((1000, 12), dtype=np.float32)
    q_flat = check_signal_quality(x_flat)
    assert len(q_flat["flatline_leads"]) == 12
    assert q_flat["is_usable"] is False

def test_complete_preprocessing_pipeline():
    fs = 500
    t = np.arange(fs * 2) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)[:, np.newaxis] + 0.2 * np.sin(2 * np.pi * 50 * t)[:, np.newaxis]
    sig_12 = np.repeat(sig, 12, axis=1)

    cleaned, metrics = preprocess_12lead_ecg(sig_12, fs=fs, notch_f0=50.0)
    assert cleaned.shape == sig_12.shape
    assert metrics["snr_db"] > 0
    assert "quality_raw" in metrics
