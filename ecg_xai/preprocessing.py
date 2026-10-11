"""Multi-stage ECG Signal Preprocessing and Denoising Pipeline.

Strictly follows Algorithm 1 and Section 4.3 of the RM6 paper:
1. DC Offset Removal (Equation 1)
2. Baseline Wander Correction (Equations 2-3: 4th-order Butterworth HPF fc=0.5 Hz)
3. Notch Filtering (Equation 4: 2nd-order IIR notch Q=30 at 50/60 Hz)
4. Bandpass Filtering (Equation 5: 4th-order Butterworth BPF 0.5-40 Hz)
5. Wavelet Denoising (Equations 6-9: db4 DWT, depth J=5, universal soft-thresholding)
6. Comprehensive Signal Quality Assessment
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

DEFAULT_FS = 500  # Sampling frequency in Hz for clinical 12-lead ECG (PTB-XL)


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """Stage 2.1: Remove DC offset by subtracting the sample mean (Equation 1)."""
    x_clean = np.nan_to_num(x, nan=0.0)
    return x_clean - np.mean(x_clean, axis=0, keepdims=True)


def correct_baseline_wander(x: np.ndarray, fs: float = DEFAULT_FS, fc: float = 0.5, order: int = 4) -> np.ndarray:
    """Stage 2.2: Baseline Wander Correction using zero-phase Butterworth HPF (Equations 2-3)."""
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    # Zero-phase bidirectional filtering
    return filtfilt(b, a, x, axis=0)


def apply_notch_filter(x: np.ndarray, fs: float = DEFAULT_FS, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Stage 2.3: Second-order IIR notch filter for Power Line Interference removal (Equation 4)."""
    nyq = 0.5 * fs
    w0 = f0 / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, x, axis=0)


def apply_bandpass_filter(x: np.ndarray, fs: float = DEFAULT_FS, fl: float = 0.5, fh: float = 40.0, order: int = 4) -> np.ndarray:
    """Stage 2.4: 4th-order Butterworth bandpass filter (0.5-40 Hz) (Equation 5)."""
    nyq = 0.5 * fs
    low = fl / nyq
    high = min(fh / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band", analog=False)
    return filtfilt(b, a, x, axis=0)


def apply_wavelet_denoising(x: np.ndarray, wavelet: str = "db4", level: int = 5) -> np.ndarray:
    """Stage 2.5: Wavelet based denoising with soft-thresholding (Equations 6-9).

    Uses Daubechies-4 wavelet (db4) with J=5 decomposition levels.
    Threshold: lambda_j = sigma_j * sqrt(2 * ln(N))
    Noise sigma_j = median(|d_{j,k}|) / 0.6745 (Median Absolute Deviation)
    """
    if not HAS_PYWT:
        # Fallback if PyWavelets is not installed: apply mild Gaussian smoothing
        from scipy.signal.windows import gaussian as _gauss_win; b_sm = _gauss_win(15, std=2)
        b_sm /= np.sum(b_sm)
        if x.ndim == 1:
            return signal.convolve(x, b_sm, mode="same")
        return np.apply_along_axis(lambda col: signal.convolve(col, b_sm, mode="same"), 0, x)

    def denoise_1d(sig: np.ndarray) -> np.ndarray:
        n_samples = len(sig)
        max_level = pywt.dwt_max_level(n_samples, pywt.Wavelet(wavelet).dec_len)
        actual_level = min(level, max(1, max_level))
        coeffs = pywt.wavedec(sig, wavelet, level=actual_level)
        cA = coeffs[0]
        cDs = coeffs[1:]

        thresholded_cDs = []
        for d in cDs:
            mad = np.median(np.abs(d))
            sigma_j = mad / 0.6745 if mad > 1e-12 else 1e-6
            lambda_j = sigma_j * np.sqrt(2.0 * np.log(max(n_samples, 2)))
            # Soft-thresholding: sign(d) * max(|d| - lambda, 0)
            d_thresh = pywt.threshold(d, value=lambda_j, mode="soft")
            thresholded_cDs.append(d_thresh)

        reconstructed = pywt.waverec([cA] + thresholded_cDs, wavelet)
        return reconstructed[:n_samples]

    if x.ndim == 1:
        return denoise_1d(x)
    
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = denoise_1d(x[:, c])
    return out


def calculate_snr(raw: np.ndarray, denoised: np.ndarray) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB between raw and cleaned signal."""
    noise = raw - denoised
    power_signal = np.mean(denoised ** 2)
    power_noise = np.mean(noise ** 2)
    if power_noise < 1e-12:
        return 60.0
    return float(10.0 * np.log10(max(power_signal / power_noise, 1e-12)))


def check_signal_quality(x: np.ndarray) -> Dict[str, Any]:
    """Evaluates ECG signal quality flags for each lead."""
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, np.newaxis]

    n_samples, n_channels = x_arr.shape
    has_nan = bool(np.isnan(x_arr).any() or np.isinf(x_arr).any())

    std_per_channel = np.nanstd(x_arr, axis=0)
    maxabs_per_channel = np.nanmax(np.abs(x_arr), axis=0)

    flatline_leads = [int(i) for i in np.where(std_per_channel < 1e-4)[0]]
    extreme_leads = [int(i) for i in np.where(maxabs_per_channel > 20.0)[0]]

    is_usable = (not has_nan) and (len(flatline_leads) == 0) and (len(extreme_leads) <= 2)

    return {
        "has_nan": has_nan,
        "flatline_leads": flatline_leads,
        "extreme_amp_leads": extreme_leads,
        "is_usable": is_usable,
        "samples": n_samples,
        "channels": n_channels,
    }


def preprocess_12lead_ecg(
    x: np.ndarray,
    fs: float = DEFAULT_FS,
    notch_f0: float = 50.0,
    stages: Optional[Dict[str, bool]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Runs the complete configurable multi-stage ECG preprocessing pipeline (Algorithm 1).

    Parameters
    ----------
    x : np.ndarray
        Raw ECG array of shape (T, 12) or (T,)
    fs : float
        Sampling frequency in Hz (default: 500)
    notch_f0 : float
        Powerline noise frequency in Hz (50 or 60)
    stages : dict, optional
        Interactive toggle configuration:
        {'dc_offset': True, 'baseline_wander': True, 'notch': True, 'bandpass': True, 'wavelet': True}

    Returns
    -------
    x_denoised : np.ndarray
        Cleaned ECG array
    metrics : dict
        Execution statistics, SNR before/after, and quality flags.
    """
    x_orig = np.asarray(x, dtype=np.float32)
    q_raw = check_signal_quality(x_orig)

    config = {
        "dc_offset": True,
        "baseline_wander": True,
        "notch": True,
        "bandpass": True,
        "wavelet": True,
    }
    if stages:
        config.update(stages)

    curr = np.nan_to_num(x_orig, nan=0.0)

    # Stage 1: DC offset removal
    if config.get("dc_offset", True):
        curr = remove_dc_offset(curr)

    # Stage 2: Baseline wander correction
    if config.get("baseline_wander", True) and curr.shape[0] > int(fs * 2):
        curr = correct_baseline_wander(curr, fs=fs, fc=0.5, order=4)

    # Stage 3: Notch filtering
    if config.get("notch", True) and curr.shape[0] > int(fs * 2):
        curr = apply_notch_filter(curr, fs=fs, f0=notch_f0, Q=30.0)

    # Stage 4: Bandpass filtering
    if config.get("bandpass", True) and curr.shape[0] > int(fs * 2):
        curr = apply_bandpass_filter(curr, fs=fs, fl=0.5, fh=40.0, order=4)

    # Stage 5: Wavelet denoising
    if config.get("wavelet", True):
        curr = apply_wavelet_denoising(curr, wavelet="db4", level=5)

    q_clean = check_signal_quality(curr)
    snr_val = calculate_snr(x_orig, curr)

    metrics = {
        "quality_raw": q_raw,
        "quality_clean": q_clean,
        "snr_db": snr_val,
        "stages_applied": config,
    }
    return curr, metrics
