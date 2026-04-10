"""Evaluation metrics: F1, Kappa, AUC, C-index, McNemar, bootstrap CI."""

import numpy as np
from sklearn.metrics import (
    f1_score, cohen_kappa_score, classification_report,
    roc_auc_score, confusion_matrix
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: np.ndarray = None) -> dict:
    """Compute full classification metric suite.

    Returns dict with: f1_macro, kappa, per_class_report, confusion_matrix,
                       auc_ovr (if y_prob provided)
    """
    results = {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=['alpayu', 'madhyayu', 'purnayu'],
            output_dict=True,
        ),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_prob is not None:
        try:
            results['auc_ovr'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
        except ValueError:
            results['auc_ovr'] = None

    return results


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """Bootstrap 95% confidence interval for macro F1.

    Vectorized index generation + parallel F1 computation.
    Returns dict with: mean, ci_lower, ci_upper
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.RandomState(42)
    n = len(y_true)
    # Pre-generate all bootstrap index arrays at once (vectorized)
    all_idx = rng.choice(n, size=(n_boot, n), replace=True)

    def _f1_for_boot(idx):
        return f1_score(y_true[idx], y_pred[idx], average='macro')

    # Compute F1 scores in parallel threads
    with ThreadPoolExecutor() as pool:
        boot_f1s = np.array(list(pool.map(_f1_for_boot, all_idx)))

    return {
        'mean': float(boot_f1s.mean()),
        'ci_lower': float(np.percentile(boot_f1s, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(boot_f1s, 100 * (1 - alpha / 2))),
    }


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """McNemar's test comparing two classifiers.

    Returns dict with: chi2, p_value
    """
    from scipy.stats import chi2 as chi2_dist

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    if b + c == 0:
        return {'chi2': 0.0, 'p_value': 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return {'chi2': float(chi2), 'p_value': float(p_value)}
