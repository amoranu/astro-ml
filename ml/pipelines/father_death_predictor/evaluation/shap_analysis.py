"""SHAP analysis for model interpretability."""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def compute_shap_values(model, X_val: pd.DataFrame):
    """Compute SHAP values using TreeExplainer.

    Returns:
        shap_values: list of 3 arrays (one per class), each (N, F)
        explainer: TreeExplainer instance
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_val)

    # Normalize shape: newer shap may return (N, F, C) instead of list of (N, F)
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        # (N, F, C) -> list of 3 (N, F)
        shap_values = [raw[:, :, i] for i in range(raw.shape[2])]
    elif isinstance(raw, list):
        shap_values = raw
    else:
        shap_values = [raw]

    return shap_values, explainer


def plot_global_importance(shap_values, X_val: pd.DataFrame,
                           output_dir: str):
    """Save global feature importance bar plot."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_global_importance.png'), dpi=150)
    plt.close()


def plot_per_class(shap_values, X_val: pd.DataFrame, output_dir: str):
    """Save per-class SHAP summary plots."""
    os.makedirs(output_dir, exist_ok=True)
    class_names = ['alpayu', 'madhyayu', 'purnayu']
    for i, name in enumerate(class_names):
        if i >= len(shap_values):
            break
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], X_val, show=False)
        plt.title(f'SHAP \u2014 {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_{name}.png'), dpi=150)
        plt.close()


def get_top_features_per_class(shap_values, feature_names: list,
                               top_n: int = 10) -> dict:
    """Return top N features by mean |SHAP| for each class.

    Returns dict of class_name -> list of (feature_name, mean_abs_shap).
    """
    class_names = ['alpayu', 'madhyayu', 'purnayu']
    result = {}
    for i, name in enumerate(class_names):
        if i >= len(shap_values):
            break
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        result[name] = [
            (feature_names[j], float(mean_abs[j])) for j in top_idx
        ]
    return result
