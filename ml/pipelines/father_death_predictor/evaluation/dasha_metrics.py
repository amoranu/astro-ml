"""Duration-aware evaluation for dasha-native timing.

Handles variable group sizes and unequal period durations.
Metrics: strict containment, ±15/30 day accuracy, calendar coverage.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def evaluate_dasha_topk(df: pd.DataFrame, scores: np.ndarray,
                        k_values=(1, 3, 5)) -> dict:
    """Full evaluation in both dasha-native and calendar-equivalent terms."""
    df = df.copy()
    df['score'] = scores

    results = {
        'strict': {k: 0 for k in k_values},
        'within_30d': {k: 0 for k in k_values},
        'within_15d': {k: 0 for k in k_values},
        'coverage_days': {k: [] for k in k_values},
        'correct_period_duration': [],
        'error_days': [],
    }
    n_groups = 0

    for _, group_df in df.groupby('group_id', sort=False):
        ranked = group_df.sort_values('score', ascending=False)
        correct_row = ranked[ranked['label'] == 1]
        if len(correct_row) == 0:
            continue
        correct_row = correct_row.iloc[0]

        death_jd = (correct_row['period_start_jd']
                     + correct_row['period_end_jd']) / 2
        correct_dur = (correct_row['period_end_jd']
                       - correct_row['period_start_jd'])
        results['correct_period_duration'].append(correct_dur)

        # Top-1 error in days
        top1 = ranked.iloc[0]
        top1_mid = (top1['period_start_jd'] + top1['period_end_jd']) / 2
        results['error_days'].append(abs(top1_mid - death_jd))

        for k in k_values:
            top_k = ranked.head(k)

            # Strict containment
            if correct_row.name in top_k.index:
                results['strict'][k] += 1

            # Calendar coverage
            total_covered = sum(
                r['period_end_jd'] - r['period_start_jd']
                for _, r in top_k.iterrows())
            results['coverage_days'][k].append(total_covered)

            # ±30 days
            for _, r in top_k.iterrows():
                r_mid = (r['period_start_jd'] + r['period_end_jd']) / 2
                if abs(r_mid - death_jd) <= 30:
                    results['within_30d'][k] += 1
                    break

            # ±15 days
            for _, r in top_k.iterrows():
                r_mid = (r['period_start_jd'] + r['period_end_jd']) / 2
                if abs(r_mid - death_jd) <= 15:
                    results['within_15d'][k] += 1
                    break

        n_groups += 1

    if n_groups == 0:
        return {'n_charts': 0}

    report = {'n_charts': n_groups}
    for k in k_values:
        avg_cov = np.mean(results['coverage_days'][k])
        cov_frac = avg_cov / 730.0

        strict_acc = results['strict'][k] / n_groups
        # Significance: per-chart random baseline = k / avg_group_size
        avg_group_size = len(df) / n_groups
        p_val = binomtest(
            results['strict'][k], n_groups,
            min(1.0, k / avg_group_size),
            alternative='greater').pvalue

        report[f'top_{k}'] = {
            'strict': strict_acc,
            'within_30d': results['within_30d'][k] / n_groups,
            'within_15d': results['within_15d'][k] / n_groups,
            'coverage_days': avg_cov,
            'coverage_fraction': cov_frac,
            'random_baseline': min(1.0, k / avg_group_size),
            'p_value': p_val,
        }

    errors = np.array(results['error_days'])
    report['error'] = {
        'mean_days': float(errors.mean()),
        'median_days': float(np.median(errors)),
    }
    report['avg_period_duration'] = float(
        np.mean(results['correct_period_duration']))

    return report


def format_dasha_report(report: dict, model_name: str = 'model') -> str:
    """Format evaluation report as readable text."""
    lines = [
        f"Model: {model_name}",
        f"Charts: {report['n_charts']}",
        f"Avg correct period duration: "
        f"{report.get('avg_period_duration', 0):.1f} days",
        "",
    ]

    for k in [1, 3, 5]:
        key = f'top_{k}'
        if key not in report:
            continue
        r = report[key]
        lines.append(f"Top-{k}:")
        lines.append(f"  Strict containment: {r['strict']:.1%}  "
                     f"(random: {r['random_baseline']:.1%}, "
                     f"p={r['p_value']:.4f})")
        lines.append(f"  Within +/-30 days:  {r['within_30d']:.1%}")
        lines.append(f"  Within +/-15 days:  {r['within_15d']:.1%}")
        lines.append(f"  Calendar coverage:  {r['coverage_days']:.0f} days "
                     f"({r['coverage_fraction']:.1%} of window)")
        lines.append("")

    if 'error' in report:
        lines.append("Error (top-1 midpoint vs death):")
        lines.append(f"  Mean:   {report['error']['mean_days']:.0f} days")
        lines.append(f"  Median: {report['error']['median_days']:.0f} days")

    return '\n'.join(lines)


def duration_bias_baseline(df: pd.DataFrame, k_values=(1, 3, 5)) -> dict:
    """Baseline: always pick the k longest periods.

    Tests whether model learns anything beyond 'longer = more likely'.
    """
    dur_col = None
    for c in ['lord_period_duration_days', 'period_duration_days']:
        if c in df.columns:
            dur_col = c
            break
    if dur_col is None:
        return {}

    scores = df[dur_col].values
    return evaluate_dasha_topk(df, scores, k_values)


def random_baseline_simulation(df: pd.DataFrame, k_values=(1, 3, 5),
                               n_sim: int = 5000) -> dict:
    """Monte Carlo random baseline accounting for variable group sizes."""
    rng = np.random.RandomState(42)
    results = {k: 0 for k in k_values}
    n_total = 0

    for _, group_df in df.groupby('group_id', sort=False):
        n_cand = len(group_df)
        correct_pos = group_df['label'].values.tolist().index(1)

        for _ in range(n_sim):
            perm = rng.permutation(n_cand)
            rank = np.where(perm == correct_pos)[0][0] + 1
            for k in k_values:
                if rank <= k:
                    results[k] += 1
        n_total += n_sim

    n_groups = df['group_id'].nunique()
    return {
        f'random_top_{k}': results[k] / n_total
        for k in k_values
    }
