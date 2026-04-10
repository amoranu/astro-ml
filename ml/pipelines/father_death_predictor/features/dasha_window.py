"""Dasha-native window construction and transit at period midpoints.

Replaces calendar-month windows with variable-length dasha period windows.
"""

import random
import swisseph as swe

from ..astro_engine.ephemeris import compute_jd
from ..astro_engine.houses import get_sign
from ..astro_engine.aspects import aspect_strength

swe.set_sid_mode(swe.SIDM_LAHIRI)

_TRANSIT_PLANETS = {
    'Saturn': swe.SATURN,
    'Jupiter': swe.JUPITER,
}

AVG_DAYS_PER_MONTH = 30.44


def date_to_jd(date_str: str) -> float:
    """Convert 'YYYY-MM-DD' to Julian day (noon UT)."""
    return compute_jd(date_str, '12:00')


def construct_dasha_window(death_date_str: str, all_periods: list,
                           window_months: int = 24, seed=None):
    """Build a window of dasha periods around the death date.

    Args:
        death_date_str: 'YYYY-MM-DD' father's death date
        all_periods: list of dasha period dicts (from compute_full_dasha)
        window_months: total window size in months
        seed: RNG seed for reproducible death offset

    Returns:
        candidates: list of period dicts with clipped_* keys added
        correct_idx: index of period containing the death (or None)
        window_start_jd: JD start of window
        window_end_jd: JD end of window
    """
    rng = random.Random(seed)
    death_jd = date_to_jd(death_date_str)

    # Random offset: death positioned randomly within window
    window_days = window_months * AVG_DAYS_PER_MONTH
    offset_days = rng.randint(0, int(window_days))
    window_start_jd = death_jd - offset_days
    window_end_jd = window_start_jd + window_days

    candidates = []
    correct_idx = None

    for period in all_periods:
        if period['end_jd'] <= window_start_jd:
            continue
        if period['start_jd'] >= window_end_jd:
            break

        # Clip to window bounds
        clipped_start = max(period['start_jd'], window_start_jd)
        clipped_end = min(period['end_jd'], window_end_jd)

        candidate = {
            **period,
            'clipped_start_jd': clipped_start,
            'clipped_end_jd': clipped_end,
            'clipped_duration_days': clipped_end - clipped_start,
        }
        candidates.append(candidate)

        # Does this period contain the death?
        if period['start_jd'] <= death_jd <= period['end_jd']:
            correct_idx = len(candidates) - 1

    return candidates, correct_idx, window_start_jd, window_end_jd


def compute_transit_at_jd(jd: float) -> dict:
    """Compute sidereal positions of Saturn and Jupiter at a given JD."""
    positions = {}
    for name, planet_id in _TRANSIT_PLANETS.items():
        pos = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
        positions[name] = pos[0][0]
    return positions


def extract_transit_for_period(candidate: dict, natal_chart: dict,
                               natal_asc: float,
                               natal_sat_bav: list, natal_sav: list) -> dict:
    """Transit features at the midpoint of a dasha period (~10 features)."""
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
    transit_pos = compute_transit_at_jd(midpoint_jd)

    t_sat = transit_pos['Saturn']
    t_jup = transit_pos['Jupiter']
    n_sun = natal_chart['Sun']['longitude']
    h9_mid = (natal_asc + 8 * 30 + 15) % 360
    h4_mid = (natal_asc + 3 * 30 + 15) % 360

    sat_asp_h9 = aspect_strength('Saturn', 'point', t_sat, h9_mid)
    sat_asp_h4 = aspect_strength('Saturn', 'point', t_sat, h4_mid)
    sat_asp_sun = aspect_strength('Saturn', 'Sun', t_sat, n_sun)
    jup_asp_h9 = aspect_strength('Jupiter', 'point', t_jup, h9_mid)
    jup_asp_h4 = aspect_strength('Jupiter', 'point', t_jup, h4_mid)

    # Transit on pratyantar lord's natal position
    pl = candidate['lords'][-1]
    pl_long = natal_chart[pl]['longitude']
    from .transit_features import fold_angle
    sat_on_pl = 1.0 if fold_angle(t_sat, pl_long) < 15 else 0.0
    jup_on_pl = 1.0 if fold_angle(t_jup, pl_long) < 15 else 0.0

    return {
        'tr_sat_asp_h9': sat_asp_h9,
        'tr_sat_asp_h4': sat_asp_h4,
        'tr_sat_asp_sun': sat_asp_sun,
        'tr_sat_bav': natal_sat_bav[get_sign(t_sat)],
        'tr_jup_asp_h9': jup_asp_h9,
        'tr_jup_asp_h4': jup_asp_h4,
        'double_transit_h9': min(sat_asp_h9, jup_asp_h9),
        'double_transit_h4': min(sat_asp_h4, jup_asp_h4),
        'tr_sat_on_pl': sat_on_pl,
        'tr_jup_on_pl': jup_on_pl,
    }
