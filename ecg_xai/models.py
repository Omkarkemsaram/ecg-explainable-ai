"""Machine Learning and Deep Learning Models for Multilabel ECG Classification.

Implements Section 4.6 and Table 1 of the RM6 research paper:
- Random Forest (350 trees, max depth 20, min samples leaf 5)
- XGBoost (300 estimators, lr=0.05, max depth 6, subsample 0.8, colsample 0.8)
- LightGBM (400 estimators, lr=0.05, max leaves 31, min child samples 20)
- Multi-Layer Perceptron (PyTorch [in_dim -> 128 -> 64 -> 32 -> 5] with BatchNorm & Dropout 0.3)
- Multi-label Evaluation Metrics: Micro F1, Macro F1, Micro AUC, Macro AUC, Hamming Loss (Section 5)
"""

from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------------------------------------------
# 1. Model Builders (Table 1 Hyperparameters)
# -------------------------------------------------------------
def build_rf_classifier(n_estimators: int = 350, max_depth: int = 20, min_samples_leaf: int = 5, random_state: int = 42) -> MultiOutputClassifier:
    """Model 1: Random Forest Classifier (Section 4.6.2)."""
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    return MultiOutputClassifier(base)


def build_xgboost_classifier(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> MultiOutputClassifier:
    """Model 2: XGBoost Classifier (Section 4.6.3)."""
    if not HAS_XGB:
        # Fallback to Random Forest if XGBoost not installed
        return build_rf_classifier(random_state=random_state)

    base = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return MultiOutputClassifier(base)


def build_lightgbm_classifier(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    random_state: int = 42,
) -> MultiOutputClassifier:
    """Model 3: LightGBM Classifier (Section 4.6.4)."""
    if not HAS_LGB:
        return build_rf_classifier(random_state=random_state)

    base = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1,
    )
    return MultiOutputClassifier(base)


# -------------------------------------------------------------
# 2. PyTorch Multi-Layer Perceptron (Section 4.6.5, Equation 19)
# -------------------------------------------------------------
class PyTorchMLP(nn.Module):
    """Model 4: Fully Connected Neural Network with BatchNorm & Dropout."""

    def __init__(self, in_dim: int = 40, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        # Architecture: [in_dim -> 128 -> 64 -> 32 -> n_classes] (Section 4.6.5)
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_layer = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        return self.out_layer(h)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts sigmoid probabilities for multilabel classification."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.forward(x_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary labels based on threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def fit_model(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> Dict[str, list]:
        """Trains MLP with early stopping on validation loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        X_tr = torch.tensor(X_train, dtype=torch.float32)
        Y_tr = torch.tensor(Y_train, dtype=torch.float32)
        X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_va = torch.tensor(Y_val, dtype=torch.float32).to(device)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        no_improve = 0

        n_samples = len(X_tr)

        for ep in range(epochs):
            self.train()
            perm = np.random.permutation(n_samples)
            epoch_loss = 0.0
            steps = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = perm[start:end]

                xb = X_tr[batch_idx].to(device)
                yb = Y_tr[batch_idx].to(device)

                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1

            train_loss = epoch_loss / max(1, steps)

            # Validation
            self.eval()
            with torch.no_grad():
                val_logits = self.forward(X_va)
                val_loss = criterion(val_logits, Y_va).item()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        return history


# -------------------------------------------------------------
# 3. Model Evaluation Suite (Section 5 Metrics)
# -------------------------------------------------------------
class ModelEvaluator:
    """Calculates full performance metrics matching Section 5 of the paper."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculates Micro F1, Macro F1, Micro AUC, Macro AUC, and Hamming Loss."""
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_pred, dtype=int)

        micro_f1 = float(f1_score(y_t, y_p, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
        h_loss = float(hamming_loss(y_t, y_p))

        micro_auc = 0.0
        macro_auc = 0.0

        if y_prob is not None:
            try:
                micro_auc = float(roc_auc_score(y_t, y_prob, average="micro"))
                macro_auc = float(roc_auc_score(y_t, y_prob, average="macro"))
            except Exception:
                pass

        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "micro_auc": micro_auc,
            "macro_auc": macro_auc,
            "hamming_loss": h_loss,
        }
