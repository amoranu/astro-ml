"""Stacking meta-learner: combine classifier + survival predictions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def train_stacking_ensemble(
    classifier_probs_train: np.ndarray,
    survival_hazard_train: np.ndarray,
    y_train: np.ndarray,
    classifier_probs_val: np.ndarray,
    survival_hazard_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Stack classifier class probabilities with survival hazard score.

    Args:
        classifier_probs_train/val: (N, 3) class probabilities from LightGBM
        survival_hazard_train/val: (N,) partial hazard from CoxPH
        y_train/y_val: class labels (0, 1, 2)

    Returns:
        Dict with: model, f1_macro, y_pred
    """
    X_meta_train = np.column_stack([
        classifier_probs_train, survival_hazard_train.reshape(-1, 1)
    ])
    X_meta_val = np.column_stack([
        classifier_probs_val, survival_hazard_val.reshape(-1, 1)
    ])

    meta = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    meta.fit(X_meta_train, y_train)
    y_pred = meta.predict(X_meta_val)
    f1 = f1_score(y_val, y_pred, average='macro')

    return {'model': meta, 'f1_macro': f1, 'y_pred': y_pred}
