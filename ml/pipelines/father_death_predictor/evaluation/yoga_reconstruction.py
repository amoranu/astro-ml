"""Yoga reconstruction: map model decisions to classical Parashari yogas."""

import numpy as np
import pandas as pd

# Classical yogas for father's death/suffering
CLASSICAL_YOGAS = {
    'sun_saturn_mars_conjunction': {
        'condition': lambda f: f['sun_saturn_angle'] < 30 and f['sun_mars_angle'] < 30,
        'text': 'BPHS: Sun aspected by Saturn and Mars -> early death of father',
    },
    'malefics_in_9th_and_4th': {
        'condition': lambda f: f['h9_malefic_count'] >= 2 and f['h4_malefic_count'] >= 2,
        'text': 'BPHS: Malefics in 9th and 4th -> grief to father',
    },
    'ninth_lord_in_dusthana': {
        'condition': lambda f: f['h9_lord_in_dusthana'] == 1.0,
        'text': 'Parashari: 9th lord in 6/8/12 -> father suffers',
    },
    'sun_bav_zero_in_9th': {
        'condition': lambda f: f['h9_sun_bav'] == 0,
        'text': 'Chandrakala Nadi: 0 bindus in 9th in Sun BAV -> father misfortune',
    },
    'h8_lord_in_9th': {
        'condition': lambda f: f['h8_lord_in_9th'] == 1.0,
        'text': 'BPHS: 8th lord in 9th -> loss affecting father',
    },
    'sun_in_dusthana': {
        'condition': lambda f: f['sun_in_dusthana'] == 1.0,
        'text': 'Parashari: Sun in 6/8/12 -> weak Pitru Karaka',
    },
    'sun_debilitated_enemy': {
        'condition': lambda f: f['sun_sign_dignity'] <= 0.25,
        'text': 'Parashari: Sun in enemy/debilitated sign -> father weakened',
    },
    'maraka_dasha_early': {
        'condition': lambda f: (
            f.get('age10_maha_is_maraka', 0) == 1.0
            or f.get('age15_maha_is_maraka', 0) == 1.0
        ),
        'text': 'Parashari: Father-maraka Mahadasha running in youth',
    },
}


def check_yogas(features_row: dict) -> list:
    """Check which classical yogas are present for a single record.

    Returns list of yoga names that matched.
    """
    matched = []
    for name, yoga in CLASSICAL_YOGAS.items():
        try:
            if yoga['condition'](features_row):
                matched.append(name)
        except (KeyError, TypeError):
            continue
    return matched


def reconstruct_yogas(X_val: pd.DataFrame, y_true: np.ndarray,
                      y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """For high-confidence alpayu predictions, check classical yoga presence.

    Returns:
        dict with:
        - yoga_counts: {yoga_name: count in correct alpayu predictions}
        - coverage: fraction of correct alpayu preds with at least one known yoga
        - novel_patterns: count of correct alpayu preds with NO known yoga
    """
    # High-confidence correct alpayu predictions
    alpayu_mask = (y_pred == 0) & (y_true == 0) & (y_prob[:, 0] > 0.6)
    indices = np.where(alpayu_mask)[0]

    yoga_counts = {name: 0 for name in CLASSICAL_YOGAS}
    has_yoga = 0

    for idx in indices:
        row = X_val.iloc[idx].to_dict()
        matched = check_yogas(row)
        if matched:
            has_yoga += 1
        for name in matched:
            yoga_counts[name] += 1

    total = len(indices)
    return {
        'yoga_counts': yoga_counts,
        'coverage': has_yoga / total if total > 0 else 0.0,
        'novel_patterns': total - has_yoga,
        'total_alpayu_confident': total,
    }
