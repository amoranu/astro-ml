"""Transit timing baselines.

1. Random — uniform score for all candidates
2. Center bias — always pick middle months (leak detection)
3. Season match — score by proximity to birth month
4. Single feature — only tr_sat_house (1 feature)
"""

import numpy as np
import pandas as pd


def random_baseline(df: pd.DataFrame) -> np.ndarray:
    """Uniform random scores. Breaks ties randomly but reproducibly."""
    rng = np.random.RandomState(42)
    return rng.random(len(df))


def center_bias_baseline(df: pd.DataFrame) -> np.ndarray:
    """Score by closeness to center of window (months 10-13).
    If this beats random, the window construction leaks position."""
    center = 11.5
    return -np.abs(df['month_idx'].values - center)


def season_match_baseline(df: pd.DataFrame,
                          birth_months: dict) -> np.ndarray:
    """Score by closeness to birth month (seasonal death pattern check).

    Args:
        birth_months: {group_id: birth_month (1-12)}
    """
    scores = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        bm = birth_months.get(row['group_id'], 6)
        # Circular month distance (1-12)
        cand_m = row['month']
        dist = min(abs(cand_m - bm), 12 - abs(cand_m - bm))
        scores[i] = -dist  # higher = closer
    return scores


def single_feature_baseline(df: pd.DataFrame,
                            feature_col: str = 'tr_sat_house') -> np.ndarray:
    """Use a single feature as the score.

    Default: Saturn's transit house position. If the full model doesn't
    meaningfully beat this, the other features are noise.
    """
    if feature_col in df.columns:
        return df[feature_col].values
    return np.zeros(len(df))
