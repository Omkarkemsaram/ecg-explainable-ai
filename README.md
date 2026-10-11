# ECG-XAI: Interpretable ECG Signal Analysis using Feature-Based Machine Learning and SHAP Explainability

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: PTB-XL](https://img.shields.io/badge/Dataset-PTB--XL-green.svg)](https://physionet.org/content/ptb-xl/1.0.3/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![XAI: SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://github.com/shap/shap)

**End-to-End Clinical Decision Support Pipeline for 12-Lead ECG Disease Classification with Explainable AI**

> *Project by: IIT203057 – Kemsaram Omkar | IIT2023011 – Harsha Vardhan | IIT203059 – Chetan Tirkey*

---

## Project Overview

Cardiovascular diseases are among the leading causes of death worldwide. This project implements a **production-grade, interpretable machine learning pipeline** for automated 12-lead ECG signal analysis that:

- Classifies cardiac conditions across **5 PTB-XL diagnostic superclasses** (NORM, MI, STTC, CD, HYP)
- Provides **transparent, clinically meaningful explanations** via TreeSHAP with anatomical lead-wise aggregation
- Delivers an **interactive Clinical Decision Support Dashboard** for real-time cardiologist-facing prediction

---

## Architecture

```
Raw 12-Lead ECG (10s @ 500 Hz)
        │
        ▼
┌──────────────────────────────────────────┐
│   Stage 1: Multi-Stage Signal Denoising  │
│  • DC Offset Removal (Equation 1)        │
│  • Butterworth HPF – Baseline Wander     │
│  • IIR Notch Filter – 50/60 Hz           │
│  • Butterworth BPF – 0.5–40 Hz           │
│  • Daubechies-4 Wavelet (db4, J=5)       │
└───────────────────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 2: Pan-Tompkins QRS Detection     │
│   + 780-Feature Engineering              │
│   (65 features × 12 leads)              │
│  • Time-domain moments                   │
│  • Clinical intervals (QRS, PR, QT, QTc) │
│  • HRV: SDNN, RMSSD, pNN50, SampEn, ApEn│
│  • Frequency: Welch PSD (ULF/VLF/LF/HF) │
└───────────────────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 3: Mutual Information Selection   │
│   Top K=40 or K=70 discriminative feats  │
└───────────────────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 4: Multilabel Classification      │
│   with SMOTE class-imbalance handling   │
│  • Random Forest (350 trees)             │
│  • XGBoost (300 est., lr=0.05)           │
│  • LightGBM (400 est., lr=0.05)          │
│  • PyTorch MLP [D→128→64→32→5]           │
└───────────────────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 5: TreeSHAP + Anatomical          │
│   Lead-Wise SHAP Aggregation             │
│  • Anterior Wall: V1, V2, V3, V4        │
│  • Lateral Wall:  I, aVL, V5, V6        │
│  • Inferior Wall: II, III, aVF           │
└───────────────────┬──────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Stage 6: Clinical Decision Dashboard    │
│   Interactive Streamlit UI               │
└──────────────────────────────────────────┘
```

---

## Five Diagnostic Classes (PTB-XL)

| Code | Condition | Clinical Meaning |
|------|-----------|-----------------|
| **NORM** | Normal ECG | No significant abnormality |
| **MI** | Myocardial Infarction | ST elevation, abnormal Q-waves → heart attack |
| **STTC** | ST/T-wave Change | Ischaemic changes or early repolarisation |
| **CD** | Conduction Disturbance | Prolonged QRS or PR interval → electrical delay |
| **HYP** | Hypertrophy | High-voltage waveforms → ventricular enlargement |

---

## Project Structure

```
ecg-explainable-ai/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── ecg_xai/                       # Core Python Package
│   ├── __init__.py
│   ├── preprocessing.py           # 5-Stage ECG denoising pipeline (Equations 1–9)
│   ├── feature_extraction.py      # Pan-Tompkins + 780 clinical features
│   ├── feature_selection.py       # Mutual Information feature ranking
│   ├── dataset.py                 # PTB-XL loader, synthetic ECG generator, SMOTE
│   ├── models.py                  # RF, XGBoost, LightGBM, PyTorch MLP + evaluation
│   ├── explainability.py          # TreeSHAP, lead-wise anatomical aggregation
│   └── pipeline.py                # End-to-end inference engine
│
├── app/
│   └── dashboard.py               # Interactive Streamlit Clinical Dashboard
│
├── scripts/
│   ├── train.py                   # Model training script (Section 5 benchmarks)
│   ├── evaluate.py                # 40 vs 70 feature comparative benchmark
│   └── run_inference.py           # CLI inference tool
│
├── demo_data/                     # Zero-setup demo artifacts
│   ├── generate_demo_data.py
│   ├── sample_12lead_records.json # 50 sample patient ECGs (5 conditions)
│   ├── sample_features.csv        # Precomputed 780-feature matrix
│   ├── sample_targets.csv         # Multilabel targets
│   ├── selected_features_schema.json
│   └── pretrained_classifier.pkl
│
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_models.py
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Omkarkemsaram/ecg-explainable-ai.git
cd ecg-explainable-ai
pip install -r requirements.txt
```

### 2. Generate Demo Data (no PTB-XL download needed)

```bash
python demo_data/generate_demo_data.py
```

This generates:
- 50 synthetic patient ECG recordings (10 per condition)
- Precomputed 780-feature matrix
- Pre-trained XGBoost classifier
- Top-40 MI feature schema

### 3. Launch the Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

### 4. Run CLI Inference

```bash
# Simulate a Myocardial Infarction ECG
python scripts/run_inference.py --condition MI --output reports/mi_report.json

# Load your own .npy signal (shape: T x 12)
python scripts/run_inference.py --file my_ecg.npy
```

### 5. Train All 4 Models (with SMOTE + MI feature selection)

```bash
python scripts/train.py --top-k 40
# Or compare 40 vs 70 features across all models
python scripts/evaluate.py
```

### 6. Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Key Features

### Signal Preprocessing (5 Stages)
- **DC Offset Removal**: $x_{dc}(t) = x(t) - \frac{1}{T}\sum_{t=1}^{T}x(t)$
- **Baseline Wander**: 4th-order Butterworth HPF at $f_c=0.5\text{ Hz}$
- **Notch Filter**: IIR notch at 50/60 Hz with $Q=30$
- **Bandpass**: Butterworth BPF [0.5–40 Hz]
- **Wavelet**: db4 DWT, $J=5$ levels, soft universal threshold $\lambda_j = \sigma_j\sqrt{2\ln N}$

### Feature Engineering (780 Features)
- **Pan-Tompkins QRS detection** for R-peak localization
- **65 features per lead** across 12 leads:
  - Statistical moments (mean, variance, skewness, kurtosis)
  - Clinical intervals (QRS width, PR, QT, QTc Bazett, ST deviation)
  - HRV metrics (SDNN, RMSSD, pNN50, SampEn, ApEn, Poincaré SD1/SD2)
  - Spectral features via Welch PSD (ULF/VLF/LF/HF band power, LF/HF ratio, spectral entropy)

### Model Performance (Section 5)

| Model | Micro F1 | Macro F1 | Micro AUC | Hamming Loss |
|-------|----------|----------|-----------|--------------|
| Random Forest (40 feat.) | ~0.82 | ~0.79 | ~0.91 | ~0.08 |
| XGBoost (40 feat.)       | ~0.84 | ~0.81 | ~0.93 | ~0.07 |
| LightGBM (40 feat.)      | ~0.85 | ~0.82 | ~0.93 | ~0.06 |
| MLP PyTorch (40 feat.)   | ~0.83 | ~0.80 | ~0.92 | ~0.07 |

> *Run `python scripts/evaluate.py` to reproduce full benchmark results.*

### SHAP Explainability
- **TreeSHAP** for exact Shapley value computation on tree ensembles
- **Lead-wise anatomical aggregation**: Groups features by cardiac territory
  - Anterior MI → Precordial leads V1–V4 (highest SHAP)
  - Inferior MI → Inferior leads II, III, aVF
  - LV Hypertrophy → Lateral leads I, aVL, V5–V6

---

## Dataset

**PTB-XL ECG Dataset** (PhysioNet)
- 21,837 12-lead ECG recordings from 18,885 patients
- 10-second recordings at 500 Hz sampling rate
- Multi-label annotations via SCP codes
- Stratified 10-fold cross-validation splits

Download: https://physionet.org/content/ptb-xl/1.0.3/

Place data at: `data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`

---

## Authors

| Name | Roll Number |
|------|-------------|
| Harsha Vardhan | IIT2023011 |
| Kemsaram Omkar | IIT203057  |
| Chetan Tirkey  | IIT203059  |

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you use this project in your research, please cite the PTB-XL dataset:

```bibtex
@article{wagner2020ptbxl,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dierk and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  year={2020},
  publisher={Nature Publishing Group}
}
```
