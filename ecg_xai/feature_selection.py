"""Mutual Information Feature Selection Engine (Section 4.5).

Implements Equations 14-15 of the RM6 paper:
- Mutual Information (MI) criterion using k-NN non-parametric estimator
- Average MI across C multilabel classes: MI(X_j) = 1/C * sum(MI(X_j, y_i))
- Ranking and retention of Top K=40 and Top K=70 discriminative features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MutualInformationSelector:
    """Ranks and selects top K features based on Average Mutual Information."""

    def __init__(self, top_k: int = 40, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, Y: Union[pd.DataFrame, np.ndarray]) -> "MutualInformationSelector":
        """Fits the MI selector on feature matrix X and multilabel targets Y."""
        feature_names = list(X.columns)
        X_mat = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0)

        # Target matrix
        if isinstance(Y, pd.DataFrame):
            Y_mat = Y.to_numpy()
        else:
            Y_mat = np.asarray(Y)

        if Y_mat.ndim == 1:
            Y_mat = Y_mat[:, np.newaxis]

        n_samples, n_features = X_mat.shape
        n_classes = Y_mat.shape[1]

        mi_scores = np.zeros(n_features, dtype=np.float64)

        # Average MI across classes (Equation 14)
        for c in range(n_classes):
            y_col = Y_mat[:, c]
            # Ensure at least 2 unique labels in column
            if len(np.unique(y_col)) > 1:
                score_c = mutual_info_classif(
                    X_mat,
                    y_col,
                    n_neighbors=3,
                    random_state=self.random_state,
                )
                mi_scores += score_c

        mi_scores /= max(1, n_classes)

        # Rank features descending
        self.feature_scores_ = {name: float(score) for name, score in zip(feature_names, mi_scores)}
        sorted_feats = sorted(self.feature_scores_.items(), key=lambda item: item[1], reverse=True)
        self.selected_features_ = [item[0] for item in sorted_feats[: self.top_k]]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms feature dataframe X to retain only selected features."""
        if not self.selected_features_:
            raise ValueError("Selector has not been fitted yet.")
        return X.reindex(columns=self.selected_features_, fill_value=0.0)

    def fit_transform(self, X: pd.DataFrame, Y: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Fits and transforms X in a single call."""
        self.fit(X, Y)
        return self.transform(X)

    def save_schema(self, path: Union[str, Path]) -> None:
        """Saves selected feature schema and scores to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "top_k": self.top_k,
            "selected_features": self.selected_features_,
            "scores": self.feature_scores_,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_schema(cls, path: Union[str, Path]) -> "MutualInformationSelector":
        """Loads a saved selector schema from file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        sel = cls(top_k=data.get("top_k", 40))
        sel.selected_features_ = data.get("selected_features", [])
        sel.feature_scores_ = data.get("scores", {})
        return sel


def select_top_features(
    X: pd.DataFrame,
    Y: Union[pd.DataFrame, np.ndarray],
    k: int = 40,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Convenience function for top-K MI feature selection."""
    sel = MutualInformationSelector(top_k=k, random_state=random_state)
    X_sel = sel.fit_transform(X, Y)
    return X_sel, sel.selected_features_, sel.feature_scores_
