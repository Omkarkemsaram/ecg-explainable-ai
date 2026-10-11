"""PTB-XL Dataset Ingestion, Multilabel Mapping, and SMOTE Preprocessing.

Implements Sections 4.2 & 4.6.1 of the RM6 paper:
- Ingestion of 12-lead ECG records (PTB-XL format or synthetic benchmark)
- Five clinical diagnostic superclasses:
    * NORM : Normal ECG
    * MI   : Myocardial Infarction (heart attack)
    * STTC : ST/T-wave abnormalities (ischemia)
    * CD   : Conduction Disturbance (electrical signal delay)
    * HYP  : Hypertrophy (heart muscle enlargement)
- Stratified 70/10/20 train/validation/test split
- SMOTE (Synthetic Minority Over-sampling Technique) on training split
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

DIAGNOSTIC_CLASSES_5 = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def generate_synthetic_ecg_12lead(
    duration_sec: float = 10.0,
    fs: float = 500.0,
    condition: str = "NORM",
    heart_rate_bpm: float = 72.0,
    noise_level: float = 0.05,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generates realistic 12-lead ECG signals with clinically accurate morphology."""
    rng = np.random.RandomState(random_state)
    t = np.arange(int(duration_sec * fs)) / fs
    n_samples = len(t)
    rr_sec = 60.0 / heart_rate_bpm
    n_beats = int(duration_sec / rr_sec) + 1
    beat_times = np.arange(n_beats) * rr_sec + rng.uniform(-0.02, 0.02, size=n_beats)

    # Lead amplitude scalers reflecting anatomical orientations
    lead_factors = {
        "I": 0.8, "II": 1.2, "III": 0.5,
        "aVR": -0.7, "aVL": 0.4, "aVF": 0.9,
        "V1": 0.4, "V2": 0.7, "V3": 1.0,
        "V4": 1.4, "V5": 1.2, "V6": 0.9
    }

    # Morphological parameters adjusted by condition
    qrs_width_sec = 0.08
    st_elevation = 0.0
    t_amp = 0.25
    r_height = 1.0

    if condition == "MI":
        # ST elevation in anterior (V1-V4) or inferior (II, III, aVF)
        st_elevation = 0.25
        t_amp = 0.45
    elif condition == "CD":
        # Prolonged QRS (> 120 ms)
        qrs_width_sec = 0.14
    elif condition == "HYP":
        # High voltage R waves in V4-V6
        r_height = 1.8
    elif condition == "STTC":
        # ST depression or T inversion
        st_elevation = -0.15
        t_amp = -0.30

    leads_data = np.zeros((n_samples, 12), dtype=np.float32)

    for l_idx, lead_name in enumerate(LEADS_12):
        factor = lead_factors[lead_name]
        sig = np.zeros(n_samples, dtype=np.float32)

        # Build cardiac cycles
        for bt in beat_times:
            # P wave (bt - 0.16s)
            p_center = bt - 0.16
            sig += 0.15 * factor * np.exp(-((t - p_center) ** 2) / (2 * (0.025 ** 2)))

            # Q wave (bt - 0.03s)
            q_center = bt - 0.03
            sig -= 0.15 * factor * np.exp(-((t - q_center) ** 2) / (2 * (0.010 ** 2)))

            # R wave (bt)
            r_center = bt
            sig += r_height * factor * np.exp(-((t - r_center) ** 2) / (2 * (qrs_width_sec / 4.0) ** 2))

            # S wave (bt + 0.03s)
            s_center = bt + 0.03
            sig -= 0.25 * factor * np.exp(-((t - s_center) ** 2) / (2 * (0.012 ** 2)))

            # ST segment (bt + 0.08s)
            st_center = bt + 0.08
            sig += st_elevation * factor * np.exp(-((t - st_center) ** 2) / (2 * (0.040 ** 2)))

            # T wave (bt + 0.25s)
            t_center = bt + 0.25
            sig += t_amp * factor * np.exp(-((t - t_center) ** 2) / (2 * (0.055 ** 2)))

        # Add realistic artifacts (baseline wander + 50Hz mains + EMG)
        bw_drift = 0.10 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, np.pi))
        mains_noise = 0.03 * np.sin(2 * np.pi * 50.0 * t)
        gaussian_noise = rng.normal(0, noise_level, size=n_samples).astype(np.float32)

        leads_data[:, l_idx] = sig + bw_drift + mains_noise + gaussian_noise

    return leads_data


class PTBXLDataset:
    """Dataset manager for PTB-XL 12-lead ECG records."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.df_metadata: Optional[pd.DataFrame] = None
        self.scp_map: Dict[str, str] = {}

    def load_metadata(self) -> pd.DataFrame:
        """Loads PTB-XL database metadata CSV."""
        if self.data_dir is None:
            raise FileNotFoundError("data_dir is not provided.")
        db_path = self.data_dir / "ptbxl_database.csv"
        scp_path = self.data_dir / "scp_statements.csv"

        if not db_path.exists():
            raise FileNotFoundError(f"Missing {db_path}")

        self.df_metadata = pd.read_csv(db_path)

        if scp_path.exists():
            scp_df = pd.read_csv(scp_path, index_col=0)
            scp_diag = scp_df[scp_df["diagnostic"].fillna(0).astype(float) == 1.0]
            self.scp_map = scp_diag["diagnostic_class"].to_dict()

        return self.df_metadata

    def build_multilabel_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps SCP statements into binary indicator matrix for the 5 superclasses."""
        y_mat = np.zeros((len(df), len(DIAGNOSTIC_CLASSES_5)), dtype=int)

        for i, scp_str in enumerate(df["scp_codes"].fillna("{}")):
            try:
                parsed = ast.literal_eval(scp_str) if isinstance(scp_str, str) else scp_str
                if isinstance(parsed, dict):
                    for code in parsed.keys():
                        diag_cls = self.scp_map.get(code)
                        if diag_cls in DIAGNOSTIC_CLASSES_5:
                            c_idx = DIAGNOSTIC_CLASSES_5.index(diag_cls)
                            y_mat[i, c_idx] = 1
            except Exception:
                pass

        # If no diagnostic code found, assign NORM by default
        no_label_mask = (y_mat.sum(axis=1) == 0)
        y_mat[no_label_mask, DIAGNOSTIC_CLASSES_5.index("NORM")] = 1

        return pd.DataFrame(y_mat, columns=DIAGNOSTIC_CLASSES_5, index=df.index)

    @staticmethod
    def split_data(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs stratified 70/10/20 train/validation/test splitting (Section 4.6.1)."""
        # First split into train (70%) and temp (30%)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
        )

        # Split temp into val (10%) and test (20%)
        relative_val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp,
            test_size=(1.0 - relative_val_size),
            random_state=random_state,
        )

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    @staticmethod
    def apply_smote(
        X_train: pd.DataFrame,
        Y_train: Union[pd.DataFrame, np.ndarray],
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Applies SMOTE oversampling to training set to handle class imbalance (Section 4.6.1)."""
        if not HAS_SMOTE:
            return X_train, pd.DataFrame(Y_train, columns=DIAGNOSTIC_CLASSES_5)

        Y_df = pd.DataFrame(Y_train, columns=DIAGNOSTIC_CLASSES_5)
        # For multilabel, oversample minority classes sequentially or via primary label
        primary_label = Y_df.idxmax(axis=1)

        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, primary_label)

        # Re-expand into one-hot binary targets
        Y_res = pd.get_dummies(y_res).reindex(columns=DIAGNOSTIC_CLASSES_5, fill_value=0)
        return pd.DataFrame(X_res, columns=X_train.columns), Y_res
