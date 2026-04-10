"""Mahadasha/Antardasha lord context features per Laghu Parashari.

The 42 slokas of Laghu Parashari are entirely about how MD, AD, and PD
lords interact to produce results. This module extracts the framing
context that current features completely miss.

~14 features per candidate.
"""

from ..astro_engine.houses import get_sign, get_sign_lord, get_house_lord, get_house_number
from ..astro_engine.dignity import uchcha_bala, sign_dignity
from .friendship import get_friendship

ALL_PLANETS = [
    'Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter',
    'Venus', 'Saturn', 'Rahu', 'Ketu',
]


def _temporal_friendship(p1: str, p2: str, chart: dict) -> float:
    """Temporal friendship: +1 if within 2-12 houses, -1 if within 1/others.

    Per BPHS: planets in 2nd, 3rd, 4th, 10th, 11th, 12th from each other
    are temporal friends; those in 1st, 5th, 6th, 7th, 8th, 9th are enemies.
    """
    if p1 == p2:
        return 0.0
    if p1 in ('Rahu', 'Ketu') or p2 in ('Rahu', 'Ketu'):
        return 0.0  # nodes don't have temporal friendships

    s1 = int(chart[p1]['longitude'] / 30) % 12
    s2 = int(chart[p2]['longitude'] / 30) % 12
    dist = (s2 - s1) % 12  # 0-11

    # Temporal friends: houses 2,3,4,10,11,12 (dist 1,2,3,9,10,11)
    if dist in (1, 2, 3, 9, 10, 11):
        return 1.0
    return -1.0


def precompute_hierarchy_context(natal_asc: float) -> dict:
    """Precompute chart-level maraka/dusthana lord sets (once per chart)."""
    asc_sign = get_sign(natal_asc)
    h9_sign = (asc_sign + 8) % 12

    # Father's maraka lords (2nd and 7th from 9th)
    h10_lord = get_sign_lord((h9_sign + 1) % 12)  # father's 2nd
    h3_lord = get_sign_lord((h9_sign + 6) % 12)    # father's 7th
    father_marakas = {h10_lord, h3_lord}

    # Father's dusthana lords (6th, 8th, 12th from 9th)
    h2_lord = get_sign_lord((h9_sign + 5) % 12)    # father's 6th = natal 2nd
    h4_lord = get_sign_lord((h9_sign + 7) % 12)    # father's 8th = natal 4th
    h8_lord = get_sign_lord((h9_sign + 11) % 12)   # father's 12th = natal 8th
    father_dusthana_lords = {h2_lord, h4_lord, h8_lord}

    # Extended danger set: maraka + dusthana lords
    father_danger = father_marakas | father_dusthana_lords

    return {
        'father_marakas': father_marakas,
        'father_dusthana_lords': father_dusthana_lords,
        'father_danger': father_danger,
    }


def extract_hierarchy_features(candidate: dict,
                               natal_chart: dict,
                               hier_ctx: dict) -> dict:
    """Extract MD/AD lord context features for a pratyantardasha.

    Args:
        candidate: pratyantardasha period dict with 'lords' key
        natal_chart: {planet: {longitude, ...}} from compute_chart()
        hier_ctx: from precompute_hierarchy_context()

    Returns:
        ~14 features dict.
    """
    lords = candidate['lords']
    md_lord = lords[0]
    ad_lord = lords[1]
    pd_lord = lords[-1]  # pratyantardasha lord

    father_mk = hier_ctx['father_marakas']
    father_danger = hier_ctx['father_danger']

    f = {}

    # --- Mahadasha lord context ---
    f['hi_md_is_maraka'] = 1.0 if md_lord in father_mk else 0.0
    f['hi_md_is_danger'] = 1.0 if md_lord in father_danger else 0.0

    # MD lord dignity (strong MD = stronger period effects)
    if md_lord in natal_chart:
        f['hi_md_dignity'] = sign_dignity(md_lord, natal_chart[md_lord]['longitude'])
    else:
        f['hi_md_dignity'] = 0.5

    # --- Antardasha lord context ---
    f['hi_ad_is_maraka'] = 1.0 if ad_lord in father_mk else 0.0
    f['hi_ad_is_danger'] = 1.0 if ad_lord in father_danger else 0.0

    # --- Laghu Parashari: lord relationships ---
    # Natural friendship between MD↔PD and AD↔PD
    f['hi_md_pd_natural'] = get_friendship(md_lord, pd_lord)
    f['hi_ad_pd_natural'] = get_friendship(ad_lord, pd_lord)

    # Temporal friendship (based on natal positions)
    f['hi_md_pd_temporal'] = _temporal_friendship(md_lord, pd_lord, natal_chart)
    f['hi_ad_pd_temporal'] = _temporal_friendship(ad_lord, pd_lord, natal_chart)

    # Combined (panchadha maitri): natural + temporal
    # +2 = best friend, +1 = friend, 0 = neutral, -1 = enemy, -2 = bitter enemy
    f['hi_md_pd_combined'] = f['hi_md_pd_natural'] + f['hi_md_pd_temporal']
    f['hi_ad_pd_combined'] = f['hi_ad_pd_natural'] + f['hi_ad_pd_temporal']

    # --- Maraka cascade (THE key Laghu Parashari principle) ---
    maraka_count = sum([
        md_lord in father_mk,
        ad_lord in father_mk,
        pd_lord in father_mk,
    ])
    f['hi_maraka_cascade'] = maraka_count  # 0, 1, 2, or 3

    # Both MD and AD are maraka → entire sub-period is death-prone
    f['hi_md_ad_both_maraka'] = 1.0 if (
        md_lord in father_mk and ad_lord in father_mk
    ) else 0.0

    # Danger cascade (broader: includes dusthana lords)
    danger_count = sum([
        md_lord in father_danger,
        ad_lord in father_danger,
        pd_lord in father_danger,
    ])
    f['hi_danger_cascade'] = danger_count

    return f
