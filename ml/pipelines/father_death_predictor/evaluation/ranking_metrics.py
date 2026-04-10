"""Ranking evaluation: top-k accuracy, tolerance, significance, error analysis."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chisquare


RANDOM_TOPK = {1: 1/24, 3: 3/24, 5: 5/24}
RANDOM_TOPK_TOL1 = {1: 3/24, 3: 7/24, 5: 11/24}


def evaluate_topk(df: pd.DataFrame, scores: np.ndarray,
                  k_values=(1, 3, 5)) -> dict:
    """Compute strict top-k accuracy.

    Args:
        df: DataFrame with group_id, month_idx, label columns
        scores: model prediction scores (one per row)
        k_values: k values to evaluate

    Returns:
        dict of {f'top_{k}': accuracy} for each k
    """
    df = df.copy()
    df['score'] = scores

    results = {k: 0 for k in k_values}
    n_groups = 0

    for _, group_df in df.groupby('group_id', sort=False):
        ranked = group_df.sort_values('score', ascending=False)
        correct_rank = ranked['label'].values.tolist().index(1) + 1

        for k in k_values:
            if correct_rank <= k:
                results[k] += 1
        n_groups += 1

    return {f'top_{k}': results[k] / n_groups if n_groups > 0 else 0
            for k in k_values}


def evaluate_topk_tolerant(df: pd.DataFrame, scores: np.ndarray,
                           k_values=(1, 3, 5), tolerance: int = 1) -> dict:
    """Top-k with adjacency tolerance (±tolerance months).

    A prediction is correct if ANY of the top-k months is within
    ±tolerance months of the true death month.
    """
    df = df.copy()
    df['score'] = scores

    results = {k: 0 for k in k_values}
    n_groups = 0

    for _, group_df in df.groupby('group_id', sort=False):
        ranked = group_df.sort_values('score', ascending=False)
        correct_idx = int(
            ranked[ranked['label'] == 1].iloc[0]['month_idx'])

        for k in k_values:
            top_k_idxs = ranked.head(k)['month_idx'].values.astype(int)
            if any(abs(m - correct_idx) <= tolerance for m in top_k_idxs):
                results[k] += 1
        n_groups += 1

    return {f'top_{k}_tol{tolerance}': results[k] / n_groups
            if n_groups > 0 else 0
            for k in k_values}


def significance_test(topk_acc: float, n_charts: int, k: int) -> float:
    """Binomial test: is top-k accuracy > k/24?

    Returns p-value (one-sided, greater).
    """
    successes = int(round(topk_acc * n_charts))
    result = binomtest(successes, n_charts, k / 24, alternative='greater')
    return result.pvalue


def error_analysis(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Month-error distribution analysis.

    Returns dict with mean/median error in months, within-1, within-3.
    """
    df = df.copy()
    df['score'] = scores
    errors = []

    for _, group_df in df.groupby('group_id', sort=False):
        ranked = group_df.sort_values('score', ascending=False)
        predicted_idx = int(ranked.iloc[0]['month_idx'])
        correct_idx = int(
            ranked[ranked['label'] == 1].iloc[0]['month_idx'])
        errors.append(abs(predicted_idx - correct_idx))

    errors = np.array(errors)
    return {
        'mean_error_months': float(errors.mean()),
        'median_error_months': float(np.median(errors)),
        'within_1_month': float(np.mean(errors <= 1)),
        'within_3_months': float(np.mean(errors <= 3)),
    }


def check_window_uniformity(df: pd.DataFrame) -> dict:
    """Gate G0: verify correct_index is uniformly distributed.

    Returns chi-square test result.
    """
    correct_positions = []
    for _, group_df in df.groupby('group_id', sort=False):
        correct_row = group_df[group_df['label'] == 1]
        if len(correct_row) > 0:
            correct_positions.append(int(correct_row.iloc[0]['month_idx']))

    # Bin into 20 bins (positions 2-21)
    bins = np.bincount(correct_positions, minlength=24)
    # Only count valid positions (2-21)
    valid_bins = bins[2:22]
    expected = len(correct_positions) / 20.0

    stat, p_value = chisquare(valid_bins, f_exp=[expected] * 20)
    return {
        'chi2': float(stat),
        'p_value': float(p_value),
        'uniform': p_value > 0.05,
        'position_counts': valid_bins.tolist(),
    }


def full_evaluation(df: pd.DataFrame, scores: np.ndarray,
                    model_name: str = 'model') -> dict:
    """Run full evaluation suite for a single model.

    Returns dict with all metrics + formatted report string.
    """
    n_charts = df['group_id'].nunique()

    topk = evaluate_topk(df, scores)
    topk_tol = evaluate_topk_tolerant(df, scores, tolerance=1)
    err = error_analysis(df, scores)

    # Significance tests
    p_values = {}
    for k in [1, 3, 5]:
        p_values[f'p_top_{k}'] = significance_test(
            topk[f'top_{k}'], n_charts, k)

    # Format report
    lines = [
        f"Model: {model_name}",
        f"Charts: {n_charts}",
        "",
        "Strict Top-K:",
        f"  top-1:  {topk['top_1']:.1%}  (random: {RANDOM_TOPK[1]:.1%}, "
        f"p={p_values['p_top_1']:.4f})",
        f"  top-3:  {topk['top_3']:.1%}  (random: {RANDOM_TOPK[3]:.1%}, "
        f"p={p_values['p_top_3']:.4f})",
        f"  top-5:  {topk['top_5']:.1%}  (random: {RANDOM_TOPK[5]:.1%}, "
        f"p={p_values['p_top_5']:.4f})",
        "",
        "Tolerant Top-K (±1 month):",
        f"  top-1:  {topk_tol['top_1_tol1']:.1%}  "
        f"(random: {RANDOM_TOPK_TOL1[1]:.1%})",
        f"  top-3:  {topk_tol['top_3_tol1']:.1%}  "
        f"(random: {RANDOM_TOPK_TOL1[3]:.1%})",
        f"  top-5:  {topk_tol['top_5_tol1']:.1%}  "
        f"(random: {RANDOM_TOPK_TOL1[5]:.1%})",
        "",
        "Error Distribution:",
        f"  mean:   {err['mean_error_months']:.1f} months",
        f"  median: {err['median_error_months']:.1f} months",
        f"  within 1 month: {err['within_1_month']:.1%}",
        f"  within 3 months: {err['within_3_months']:.1%}",
    ]

    return {
        'name': model_name,
        'n_charts': n_charts,
        **topk, **topk_tol, **p_values, **err,
        'report': '\n'.join(lines),
    }
