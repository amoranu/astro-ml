"""Extended Maraka Hierarchy per Laghu Parashari.

6-tier death-capable planet classification from the 9th house perspective.
Broadens recall from ~53% to ~65-72%.
"""

from ..astro_engine.houses import get_house_number, get_house_lord
from ..astro_engine.dignity import uchcha_bala
from ..astro_engine.aspects import aspect_strength
from .transit_features import fold_angle

ALL_PLANETS = [
    'Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter',
    'Venus', 'Saturn', 'Rahu', 'Ketu',
]
ALL_PLANETS_NO_NODES = [
    'Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn',
]
NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}


def _get_tier_for_planet(planet: str, natal_chart: dict, natal_asc: float,
                         precomputed: dict) -> int:
    """Return the tier (1-6) for a planet, or 0 if non-danger."""
    if planet in precomputed['primary_lords']:
        return 1
    if planet in precomputed['malefic_occupants']:
        return 2
    if planet in precomputed['associated_malefics']:
        return 3
    if planet in precomputed['secondary_lords']:
        return 4
    if planet == 'Saturn' and precomputed['saturn_associated']:
        return 5
    if planet in precomputed['general_lords'] or planet == precomputed['weakest']:
        return 6
    return 0


def _tier_to_score(tier: int) -> float:
    """Map tier to danger score."""
    return {1: 1.0, 2: 0.85, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.35}.get(tier, 0.0)


def precompute_maraka_sets(natal_chart: dict, natal_asc: float) -> dict:
    """Precompute all maraka sets for a chart (once per chart)."""
    h10_lord = get_house_lord(10, natal_asc)
    h3_lord = get_house_lord(3, natal_asc)
    primary = {h10_lord, h3_lord}

    h4_lord = get_house_lord(4, natal_asc)
    h8_lord = get_house_lord(8, natal_asc)
    h2_lord = get_house_lord(2, natal_asc)
    secondary = {h4_lord, h8_lord, h2_lord}

    # Tier 2: malefics occupying maraka houses (10th, 3rd from lagna)
    occ_10 = [p for p in ALL_PLANETS
              if get_house_number(natal_chart[p]['longitude'], natal_asc) == 10]
    occ_3 = [p for p in ALL_PLANETS
             if get_house_number(natal_chart[p]['longitude'], natal_asc) == 3]
    malefic_occ = {p for p in occ_10 + occ_3 if p in NATURAL_MALEFICS}

    # Tier 3: malefics conjunct/aspecting primary maraka lords
    associated = set()
    for mk in primary:
        mk_long = natal_chart[mk]['longitude']
        for p in ALL_PLANETS:
            if p == mk or p not in NATURAL_MALEFICS:
                continue
            p_long = natal_chart[p]['longitude']
            if fold_angle(p_long, mk_long) < 10:
                associated.add(p)
            elif aspect_strength(p, mk, p_long, mk_long) > 0.3:
                associated.add(p)

    # Tier 5: Saturn associated with any danger planet
    sat_assoc = False
    sat_long = natal_chart['Saturn']['longitude']
    for mk in primary | secondary | malefic_occ:
        mk_long = natal_chart[mk]['longitude']
        if fold_angle(sat_long, mk_long) < 15:
            sat_assoc = True
            break
        if aspect_strength('Saturn', mk, sat_long, mk_long) > 0.3:
            sat_assoc = True
            break

    # Tier 6: 6th/8th lords + weakest planet
    h6_lord = get_house_lord(6, natal_asc)
    general = {h6_lord, h8_lord}

    weakest = min(
        ALL_PLANETS_NO_NODES,
        key=lambda p: uchcha_bala(p, natal_chart[p]['longitude']))

    return {
        'primary_lords': primary,
        'malefic_occupants': malefic_occ,
        'associated_malefics': associated,
        'secondary_lords': secondary,
        'saturn_associated': sat_assoc,
        'general_lords': general,
        'weakest': weakest,
    }


def classify_candidate_extended(candidate: dict, natal_chart: dict,
                                natal_asc: float,
                                precomputed: dict) -> tuple:
    """Classify a candidate with extended hierarchy.

    Returns (tier, danger_score).
    """
    pl = candidate['lords'][-1]
    tier = _get_tier_for_planet(pl, natal_chart, natal_asc, precomputed)
    return tier, _tier_to_score(tier)


def extract_tier_features(candidate: dict, natal_chart: dict,
                          natal_asc: float, precomputed: dict,
                          multi_ref: dict) -> dict:
    """Extract tier-aware features (~15 features)."""
    from ..astro_engine.multiref_dasha import find_period_at_jd

    pl = candidate['lords'][-1]
    al = candidate['lords'][1]
    tier = candidate.get('tier', 0)
    midpoint = (candidate['start_jd'] + candidate['end_jd']) / 2

    f = {}
    f['tier'] = tier
    f['danger_score'] = candidate.get('danger_score', 0.0)
    f['is_tier1'] = 1.0 if tier == 1 else 0.0
    f['is_tier2'] = 1.0 if tier == 2 else 0.0
    f['is_tier3'] = 1.0 if tier == 3 else 0.0
    f['is_tier4'] = 1.0 if tier == 4 else 0.0
    f['is_tier5'] = 1.0 if tier == 5 else 0.0

    # Multi-ref agreement using extended tiers
    sun_p = find_period_at_jd(multi_ref['sun'], midpoint, target_depth=3)
    h9_p = find_period_at_jd(multi_ref['h9_cusp'], midpoint, target_depth=3)

    sun_tier = 0
    if sun_p:
        sun_tier = _get_tier_for_planet(
            sun_p['lords'][-1], natal_chart, natal_asc, precomputed)
    h9_tier = 0
    if h9_p:
        h9_tier = _get_tier_for_planet(
            h9_p['lords'][-1], natal_chart, natal_asc, precomputed)

    f['sun_ref_tier'] = sun_tier
    f['sun_ref_is_danger'] = 1.0 if sun_tier > 0 else 0.0
    f['h9_ref_tier'] = h9_tier
    f['h9_ref_is_danger'] = 1.0 if h9_tier > 0 else 0.0

    f['danger_agreement'] = (
        (1.0 if tier > 0 else 0.0)
        + f['sun_ref_is_danger']
        + f['h9_ref_is_danger'])
    f['triple_danger'] = 1.0 if f['danger_agreement'] == 3.0 else 0.0

    # Min tier among active references (lower = more dangerous)
    active = [t for t in [tier, sun_tier, h9_tier] if t > 0]
    f['min_active_tier'] = min(active) if active else 7

    # Antardasha lord tier
    al_tier = _get_tier_for_planet(al, natal_chart, natal_asc, precomputed)
    f['antar_lord_tier'] = al_tier
    f['hierarchy_danger'] = (
        f['danger_score'] * (al_tier / 6.0 if al_tier > 0 else 0.0))

    return f
