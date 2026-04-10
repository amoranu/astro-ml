"""Multi-reference agreement + lord discrimination features.

For each Moon-dasha pratyantardasha candidate, check what lords are
running in Sun-dasha and H9-dasha at the same time. Extract agreement
features + lord strength features for Stage 2 discrimination.
"""

import numpy as np
from ..astro_engine.houses import get_house_number, get_house_lord
from ..astro_engine.dignity import uchcha_bala, sign_dignity
from ..astro_engine.aspects import aspect_strength
from ..astro_engine.multiref_dasha import find_period_at_jd

_PLANET_IDX = {
    'Sun': 0, 'Moon': 1, 'Mars': 2, 'Mercury': 3, 'Jupiter': 4,
    'Venus': 5, 'Saturn': 6, 'Rahu': 7, 'Ketu': 8,
}


def _maraka_set(natal_asc: float) -> set:
    """Primary + secondary maraka planets from 9th house."""
    return {
        get_house_lord(10, natal_asc),  # 2nd from 9th
        get_house_lord(3, natal_asc),   # 7th from 9th
        get_house_lord(4, natal_asc),   # 8th from 9th
    }


def classify_candidate(candidate: dict, natal_asc: float) -> str:
    """Classify as 'primary', 'secondary', or 'none'."""
    pl = candidate['lords'][-1]
    h10 = get_house_lord(10, natal_asc)
    h3 = get_house_lord(3, natal_asc)
    h4 = get_house_lord(4, natal_asc)
    h8 = get_house_lord(8, natal_asc)
    h9 = get_house_lord(9, natal_asc)

    if pl in (h10, h3):
        return 'primary'
    if pl in (h4, h8, h9):
        return 'secondary'
    return 'none'


def extract_multiref_features(candidate: dict, multi_ref: dict,
                              natal_chart: dict, natal_asc: float) -> dict:
    """Extract ~22 multi-reference agreement features."""
    midpoint = (candidate['start_jd'] + candidate['end_jd']) / 2
    moon_pl = candidate['lords'][-1]
    mk_set = _maraka_set(natal_asc)

    f = {}

    # Moon-dasha lord
    f['moon_is_maraka'] = 1.0 if moon_pl in mk_set else 0.0

    # Sun-dasha lord at this time (pratyantar level)
    sun_p = find_period_at_jd(multi_ref['sun'], midpoint, target_depth=3)
    if sun_p:
        sun_pl = sun_p['lords'][-1]
        f['sun_lord_idx'] = _PLANET_IDX.get(sun_pl, -1)
        f['sun_is_maraka'] = 1.0 if sun_pl in mk_set else 0.0
        f['sun_lord_uchcha'] = uchcha_bala(sun_pl, natal_chart[sun_pl]['longitude'])
        f['sun_lord_dignity'] = sign_dignity(sun_pl, natal_chart[sun_pl]['longitude'])
    else:
        f['sun_lord_idx'] = -1
        f['sun_is_maraka'] = 0.0
        f['sun_lord_uchcha'] = 0.5
        f['sun_lord_dignity'] = 0.5

    # H9-cusp dasha lord at this time
    h9_p = find_period_at_jd(multi_ref['h9_cusp'], midpoint, target_depth=3)
    if h9_p:
        h9_pl = h9_p['lords'][-1]
        f['h9_lord_idx'] = _PLANET_IDX.get(h9_pl, -1)
        f['h9_is_maraka'] = 1.0 if h9_pl in mk_set else 0.0
        f['h9_lord_uchcha'] = uchcha_bala(h9_pl, natal_chart[h9_pl]['longitude'])
        f['h9_lord_dignity'] = sign_dignity(h9_pl, natal_chart[h9_pl]['longitude'])
    else:
        f['h9_lord_idx'] = -1
        f['h9_is_maraka'] = 0.0
        f['h9_lord_uchcha'] = 0.5
        f['h9_lord_dignity'] = 0.5

    # Agreement features
    f['maraka_agreement_count'] = (
        f['moon_is_maraka'] + f['sun_is_maraka'] + f['h9_is_maraka'])
    f['triple_maraka'] = 1.0 if f['maraka_agreement_count'] == 3.0 else 0.0
    f['double_maraka'] = 1.0 if f['maraka_agreement_count'] >= 2.0 else 0.0

    # Same lord across references
    sun_pl_name = sun_p['lords'][-1] if sun_p else ''
    h9_pl_name = h9_p['lords'][-1] if h9_p else ''
    f['moon_sun_same_lord'] = 1.0 if moon_pl == sun_pl_name else 0.0
    f['moon_h9_same_lord'] = 1.0 if moon_pl == h9_pl_name else 0.0
    f['all_same_lord'] = (
        1.0 if moon_pl == sun_pl_name == h9_pl_name and sun_pl_name else 0.0)

    # Antardasha-level agreement
    sun_a = find_period_at_jd(multi_ref['sun'], midpoint, target_depth=2)
    h9_a = find_period_at_jd(multi_ref['h9_cusp'], midpoint, target_depth=2)
    moon_al = candidate['lords'][1]

    f['sun_antar_is_maraka'] = (
        1.0 if sun_a and sun_a['lords'][-1] in mk_set else 0.0)
    f['h9_antar_is_maraka'] = (
        1.0 if h9_a and h9_a['lords'][-1] in mk_set else 0.0)
    f['antar_agreement_count'] = (
        (1.0 if moon_al in mk_set else 0.0)
        + f['sun_antar_is_maraka']
        + f['h9_antar_is_maraka'])

    # --- Mahadasha-level agreement (Phase 3) ---
    sun_md = find_period_at_jd(multi_ref['sun'], midpoint, target_depth=1)
    h9_md = find_period_at_jd(multi_ref['h9_cusp'], midpoint, target_depth=1)
    moon_ml = candidate['lords'][0]

    moon_md_maraka = 1.0 if moon_ml in mk_set else 0.0
    f['sun_md_is_maraka'] = (
        1.0 if sun_md and sun_md['lords'][-1] in mk_set else 0.0)
    f['h9_md_is_maraka'] = (
        1.0 if h9_md and h9_md['lords'][-1] in mk_set else 0.0)
    f['md_agreement_count'] = (
        moon_md_maraka + f['sun_md_is_maraka'] + f['h9_md_is_maraka'])

    # Full-depth consensus: all 3 systems agree at MD AND AD AND PD
    f['full_depth_consensus'] = 1.0 if (
        f['md_agreement_count'] == 3.0
        and f['antar_agreement_count'] == 3.0
        and f['maraka_agreement_count'] == 3.0
    ) else 0.0

    # Weighted cascade consensus across all depths
    f['cascade_consensus'] = (
        f['md_agreement_count'] * 3
        + f['antar_agreement_count'] * 2
        + f['maraka_agreement_count'] * 1
    ) / 18.0  # max = (3*3 + 3*2 + 3*1) / 18 = 1.0

    return f


def extract_lord_discrimination(candidate: dict, natal_chart: dict,
                                natal_asc: float) -> dict:
    """Extract ~11 lord discrimination features for Stage 2."""
    pl = candidate['lords'][-1]
    al = candidate['lords'][1]
    ml = candidate['lords'][0]

    h10 = get_house_lord(10, natal_asc)
    h3 = get_house_lord(3, natal_asc)
    mk = {h10, h3}

    pl_long = natal_chart[pl]['longitude']
    f = {}

    f['pl_uchcha'] = uchcha_bala(pl, pl_long)
    f['pl_dignity'] = sign_dignity(pl, pl_long)

    pl_house = get_house_number(pl_long, natal_asc)
    f['pl_in_dusthana'] = 1.0 if pl_house in (6, 8, 12) else 0.0
    f['pl_in_kendra'] = 1.0 if pl_house in (1, 4, 7, 10) else 0.0

    f['pl_jup_aspect'] = aspect_strength(
        'Jupiter', pl, natal_chart['Jupiter']['longitude'], pl_long)
    f['pl_sat_aspect'] = aspect_strength(
        'Saturn', pl, natal_chart['Saturn']['longitude'], pl_long)

    f['al_is_maraka'] = 1.0 if al in mk else 0.0
    f['ml_is_maraka'] = 1.0 if ml in mk else 0.0
    f['hierarchy_maraka_count'] = f['al_is_maraka'] + f['ml_is_maraka']

    f['pl_is_primary_maraka'] = 1.0 if pl in mk else 0.0
    f['pl_natural_malefic'] = (
        1.0 if pl in ('Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu') else 0.0)

    return f
