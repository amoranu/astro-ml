"""Five untapped information sources — all additive features.

1. Sookshma maraka density (depth-4 on depth-3 candidates)
2. Within-group relative features
3. Dasha sandhi (junction proximity)
4. Candidate lord identity + lagna-conditioned priors
5. Transit on lord's natal degree (per-candidate)
"""

import numpy as np
import swisseph as swe

from ..astro_engine.dasha import DASHA_YEARS, DASHA_SEQUENCE, TOTAL_YEARS
from ..astro_engine.houses import get_house_lord, get_sign
from ..astro_engine.dignity import uchcha_bala

swe.set_sid_mode(swe.SIDM_LAHIRI)

PLANET_TO_IDX = {
    'Sun': 0, 'Moon': 1, 'Mars': 2, 'Mercury': 3, 'Jupiter': 4,
    'Venus': 5, 'Saturn': 6, 'Rahu': 7, 'Ketu': 8,
}
NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}


# ── 1. Sookshma Maraka Density ──────────────────────────────────────────

def extract_sookshma_features(candidate: dict, natal_asc: float) -> dict:
    """Sookshma-level maraka density for a pratyantardasha (~10 features)."""
    h10_lord = get_house_lord(10, natal_asc)
    h3_lord = get_house_lord(3, natal_asc)
    h4_lord = get_house_lord(4, natal_asc)
    h8_lord = get_house_lord(8, natal_asc)
    primary = {h10_lord, h3_lord}
    danger = {h10_lord, h3_lord, h4_lord, h8_lord}

    pl = candidate['lords'][-1]
    pl_idx = DASHA_SEQUENCE.index(pl)
    sk_seq = DASHA_SEQUENCE[pl_idx:] + DASHA_SEQUENCE[:pl_idx]

    parent_days = candidate['duration_days']

    n_primary = 0
    n_danger = 0
    n_malefic = 0
    longest_streak = 0
    current_streak = 0

    midpoint_offset = parent_days / 2
    cumulative = 0.0
    mid_lord = None
    first_maraka_frac = 1.0  # default: no maraka found
    maraka_time_days = 0.0

    for idx, sk_lord in enumerate(sk_seq):
        sk_days = parent_days * DASHA_YEARS[sk_lord] / TOTAL_YEARS

        if sk_lord in primary:
            n_primary += 1
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
            maraka_time_days += sk_days
            if first_maraka_frac == 1.0:
                first_maraka_frac = cumulative / max(parent_days, 0.1)
        else:
            current_streak = 0

        if sk_lord in danger:
            n_danger += 1
        if sk_lord in NATURAL_MALEFICS:
            n_malefic += 1

        if mid_lord is None and cumulative + sk_days > midpoint_offset:
            mid_lord = sk_lord
        cumulative += sk_days

    ml = candidate['lords'][0]
    al = candidate['lords'][1]

    # Danger score for midpoint lord (tier-based)
    _TIER_SCORES = {True: 1.0, False: 0.0}

    return {
        'sk_primary_count': n_primary,
        'sk_primary_frac': n_primary / 9.0,
        'sk_danger_count': n_danger,
        'sk_danger_frac': n_danger / 9.0,
        'sk_malefic_frac': n_malefic / 9.0,
        'sk_longest_streak': longest_streak,
        'sk_mid_is_primary': 1.0 if mid_lord in primary else 0.0,
        'sk_mid_is_danger': 1.0 if mid_lord in danger else 0.0,
        'sk_quad_depth': sum([
            1 if ml in primary else 0,
            1 if al in primary else 0,
            1 if pl in primary else 0,
            n_primary / 9.0,
        ]),
        # Phase 3: Sookshma activation timing
        'sk_first_maraka_frac': first_maraka_frac,
        'sk_maraka_time_frac': maraka_time_days / max(parent_days, 0.1),
        'sk_peak_at_start': 1.0 if sk_seq[0] in primary else 0.0,
        'sk_peak_at_end': 1.0 if sk_seq[-1] in primary else 0.0,
        'sk_mid_maraka_score': (
            1.0 if mid_lord in primary else
            0.5 if mid_lord in danger else 0.0
        ),
    }


# ── 2. Within-Group Relative Features ───────────────────────────────────

KEY_REL_FEATURES = [
    # Original 5
    'lord_pl_uchcha', 'lord_pl_sign_dignity',
    'lord_pl_jup_aspect', 'lord_pl_sat_aspect',
    'ref_maraka_agreement_count',
    # Continuous transit aspect strengths (Prong 1b)
    'gc_sat_asp_str_sun', 'gc_sat_asp_str_h9',
    'gc_jup_asp_str_sun', 'gc_jup_asp_str_h9',
    'gc_mars_asp_str_sun',
    # Degree distances (high no-dur importance)
    'gc_merc_dist_sun', 'gc_mars_dist_sun', 'gc_mars_dist_moon',
    'gc_jup_dist_sun', 'gc_jup_dist_moon',
    'ec_rahu_dist_sun', 'ec_rahu_dist_h9', 'gc_sat_dist_sun',
]


def add_relative_features(df, group_col='group_id'):
    """Add within-group rank/z-score/min/max features (~24 features).

    Must be called AFTER all per-candidate features are computed.
    Modifies df in place and returns it.
    """
    df = df.copy()

    available = [f for f in KEY_REL_FEATURES if f in df.columns]

    for feat in available:
        df[f'{feat}_rank'] = df.groupby(group_col)[feat].rank(
            ascending=False, method='min')

        g_mean = df.groupby(group_col)[feat].transform('mean')
        g_std = df.groupby(group_col)[feat].transform('std').clip(lower=1e-6)
        df[f'{feat}_zscore'] = (df[feat] - g_mean) / g_std

        g_min = df.groupby(group_col)[feat].transform('min')
        g_max = df.groupby(group_col)[feat].transform('max')
        df[f'{feat}_is_min'] = (df[feat] == g_min).astype(float)
        df[f'{feat}_is_max'] = (df[feat] == g_max).astype(float)

    # Group-level stats (use a stable column, not loop variable)
    df['group_n_candidates'] = df.groupby(group_col)[group_col].transform('count')

    if 'lord_pl_maraka_score' in df.columns:
        df['group_n_marakas'] = df.groupby(group_col)[
            'lord_pl_maraka_score'].transform(lambda x: (x > 0).sum())

    return df


# ── 3. Dasha Sandhi (Junction Proximity) ────────────────────────────────

def extract_sandhi_features(candidate: dict,
                            all_periods_depth2: list,
                            all_periods_depth1: list) -> dict:
    """Junction proximity features (~9 features).

    Args:
        candidate: pratyantardasha (depth 3)
        all_periods_depth2: antardasha-level periods (depth=2)
        all_periods_depth1: mahadasha-level periods (depth=1)
    """
    cand_mid = (candidate['start_jd'] + candidate['end_jd']) / 2
    f = {}

    # Parent antardasha
    parent_antar = None
    for p in all_periods_depth2:
        if p['start_jd'] <= cand_mid < p['end_jd']:
            parent_antar = p
            break

    if parent_antar:
        dur = parent_antar['end_jd'] - parent_antar['start_jd']
        if dur > 0:
            elapsed = (cand_mid - parent_antar['start_jd']) / dur
            f['antar_elapsed'] = elapsed
            f['antar_remaining'] = 1.0 - elapsed
            f['antar_sandhi'] = min(elapsed, 1.0 - elapsed)
            f['antar_early'] = 1.0 if elapsed < 0.2 else 0.0
            f['antar_late'] = 1.0 if elapsed > 0.8 else 0.0
            f['days_to_antar_end'] = parent_antar['end_jd'] - cand_mid
        else:
            f.update(_sandhi_defaults_antar())
    else:
        f.update(_sandhi_defaults_antar())

    # Parent mahadasha
    parent_maha = None
    for p in all_periods_depth1:
        if p['start_jd'] <= cand_mid < p['end_jd']:
            parent_maha = p
            break

    if parent_maha:
        dur = parent_maha['end_jd'] - parent_maha['start_jd']
        if dur > 0:
            elapsed = (cand_mid - parent_maha['start_jd']) / dur
            f['maha_elapsed'] = elapsed
            f['maha_sandhi'] = min(elapsed, 1.0 - elapsed)
            f['maha_late'] = 1.0 if elapsed > 0.85 else 0.0
        else:
            f.update(_sandhi_defaults_maha())
    else:
        f.update(_sandhi_defaults_maha())

    return f


def _sandhi_defaults_antar():
    return {
        'antar_elapsed': 0.5, 'antar_remaining': 0.5,
        'antar_sandhi': 0.5, 'antar_early': 0.0,
        'antar_late': 0.0, 'days_to_antar_end': 365.0,
    }


def _sandhi_defaults_maha():
    return {'maha_elapsed': 0.5, 'maha_sandhi': 0.5, 'maha_late': 0.0}


# ── 4. Lord Identity + Lagna-Conditioned ────────────────────────────────

def extract_identity_features(candidate: dict, natal_asc: float) -> dict:
    """Planet identity and lagna features (~6 features)."""
    pl = candidate['lords'][-1]
    al = candidate['lords'][1]

    lagna = get_sign(natal_asc)
    pl_idx = PLANET_TO_IDX.get(pl, -1)

    return {
        'pl_planet_idx': pl_idx,
        'al_planet_idx': PLANET_TO_IDX.get(al, -1),
        'lagna_sign': lagna,
        'lagna_planet_combo': lagna * 9 + pl_idx,
        'pl_is_saturn': 1.0 if pl == 'Saturn' else 0.0,
        'pl_is_node': 1.0 if pl in ('Rahu', 'Ketu') else 0.0,
    }


# ── 5. Transit on Lord's Natal Degree ───────────────────────────────────

def extract_lord_transit_features(candidate: dict,
                                  natal_chart: dict) -> dict:
    """Tight transit on this candidate's lord's natal degree (~7 features)."""
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2

    pl = candidate['lords'][-1]
    pl_deg = natal_chart[pl]['longitude']
    al = candidate['lords'][1]
    al_deg = natal_chart[al]['longitude']

    t_sat = swe.calc_ut(midpoint_jd, swe.SATURN, swe.FLG_SIDEREAL)[0][0]
    t_jup = swe.calc_ut(midpoint_jd, swe.JUPITER, swe.FLG_SIDEREAL)[0][0]
    t_rahu = swe.calc_ut(midpoint_jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)[0][0]

    ORB = 5.0

    def _dist(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    sat_pl = _dist(t_sat, pl_deg)
    jup_pl = _dist(t_jup, pl_deg)
    rahu_pl = _dist(t_rahu, pl_deg)
    sat_al = _dist(t_sat, al_deg)

    is_maraka = candidate.get('maraka_type', 'none') != 'none'

    return {
        'lt_sat_on_pl': 1.0 if sat_pl <= ORB else 0.0,
        'lt_sat_pl_dist': sat_pl,
        'lt_jup_on_pl': 1.0 if jup_pl <= ORB else 0.0,
        'lt_rahu_on_pl': 1.0 if rahu_pl <= ORB else 0.0,
        'lt_sat_on_al': 1.0 if sat_al <= ORB else 0.0,
        'lt_malefic_on_pl': max(
            1.0 if sat_pl <= ORB else 0.0,
            1.0 if rahu_pl <= ORB else 0.0),
        'lt_sat_on_maraka_lord': (
            (1.0 if sat_pl <= ORB else 0.0)
            * (1.0 if is_maraka else 0.0)),
    }
