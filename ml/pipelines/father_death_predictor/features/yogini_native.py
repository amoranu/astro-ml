"""Yogini-native features for Yogini Dasha pipeline.

Features based on the 36-year Yogini cycle: 8 Yoginis mapped to planets.
These are TRADITION-SPECIFIC — no Vimshottari features mixed in.

Supports depth=2 (AD) and depth=3 (PD) candidates.
~35 features per candidate at depth=3.
"""

import swisseph as swe

from ..astro_engine.yogini_dasha import (
    YOGINI_PLANETS, YOGINI_NAMES, YOGINI_YEARS, YOGINI_TOTAL)
from ..astro_engine.houses import (
    get_house_lord, get_house_number, get_sign, get_sign_lord)
from ..astro_engine.dignity import uchcha_bala, sign_dignity

swe.set_sid_mode(swe.SIDM_LAHIRI)

NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}

# Yogini nature score: inherent danger quality per Yogini
# Sankata (Rahu) = trouble, Ulka (Saturn) = destruction,
# Bhramari (Mars) = aggression, Mangala (Moon) = auspicious
_YOGINI_NATURE = {
    'Mangala': 1.0, 'Pingala': -0.1, 'Dhanya': 0.5, 'Bhramari': -0.3,
    'Bhadrika': 0.3, 'Ulka': -0.5, 'Siddha': 0.7, 'Sankata': -1.0,
}

# Planetary friendship (simplified Parashari)
_FRIENDS = {
    'Sun': {'Moon', 'Mars', 'Jupiter'},
    'Moon': {'Sun', 'Mercury'},
    'Mars': {'Sun', 'Moon', 'Jupiter'},
    'Mercury': {'Sun', 'Venus'},
    'Jupiter': {'Sun', 'Moon', 'Mars'},
    'Venus': {'Mercury', 'Saturn'},
    'Saturn': {'Mercury', 'Venus'},
    'Rahu': {'Saturn', 'Venus', 'Mercury'},
}

_SWE_IDS = {
    'Sun': swe.SUN, 'Moon': swe.MOON, 'Mars': swe.MARS,
    'Mercury': swe.MERCURY, 'Jupiter': swe.JUPITER, 'Venus': swe.VENUS,
    'Saturn': swe.SATURN, 'Rahu': swe.MEAN_NODE,
}


def precompute_yogini_context(natal_asc, natal_chart,
                              moon_long=None, birth_jd=None):
    """Precompute chart-level data for Yogini features."""
    asc_sign = get_sign(natal_asc)
    h9_sign = (asc_sign + 8) % 12

    # Father's maraka lords (2nd and 7th from 9th house)
    father_marakas = {
        get_house_lord(10, natal_asc),  # 2nd from 9th
        get_house_lord(3, natal_asc),   # 7th from 9th
    }
    # Father's dusthana lords (6th, 8th, 12th from 9th)
    father_dusthana = {
        get_house_lord(2, natal_asc),   # 6th from 9th = 2nd house
        get_house_lord(4, natal_asc),   # 8th from 9th = 4th house
        get_house_lord(8, natal_asc),   # 12th from 9th = 8th house
    }
    father_danger = father_marakas | father_dusthana

    # Father's maraka signs (for transit checks)
    father_maraka_signs = {
        (h9_sign + 1) % 12,   # father's 2nd = natal 10th
        (h9_sign + 6) % 12,   # father's 7th = natal 3rd
    }
    father_danger_signs = father_maraka_signs | {
        (h9_sign + 7) % 12,   # father's 8th = natal 4th
        (h9_sign + 11) % 12,  # father's 12th = natal 8th
    }

    # Birth Yogini index (for Mrityu Yoga features)
    birth_yogini_idx = None
    if moon_long is not None:
        nakshatra_span = 360.0 / 27.0
        nakshatra_index = int(moon_long / nakshatra_span)
        if nakshatra_index >= 27:
            nakshatra_index = 26
        birth_yogini_idx = (nakshatra_index + 3) % 8

    ctx = {
        'father_marakas': father_marakas,
        'father_danger': father_danger,
        'father_maraka_signs': father_maraka_signs,
        'father_danger_signs': father_danger_signs,
        'natal_chart': natal_chart,
        'natal_asc': natal_asc,
    }
    if birth_yogini_idx is not None:
        ctx['birth_yogini_idx'] = birth_yogini_idx
    if birth_jd is not None:
        ctx['birth_jd'] = birth_jd
    return ctx


def extract_yogini_native_features(candidate, yg_ctx):
    """Yogini-native features for a Yogini candidate (depth=2 or depth=3).

    Args:
        candidate: Yogini period dict with 'lords', 'planets', 'start_jd', etc.
        yg_ctx: from precompute_yogini_context()

    Returns: ~35 features dict at depth=3, ~15 at depth=2.
    """
    father_mk = yg_ctx['father_marakas']
    father_danger = yg_ctx['father_danger']
    chart = yg_ctx['natal_chart']
    natal_asc = yg_ctx['natal_asc']

    yogini_names = candidate['lords']
    planets = candidate['planets']
    depth = len(planets)

    md_planet = planets[0]
    # At depth=2: planets=[MD, AD], at depth=3: planets=[MD, AD, PD]
    ad_planet = planets[1] if depth >= 2 else md_planet
    pd_planet = planets[2] if depth >= 3 else planets[-1]
    deepest = planets[-1]

    f = {}

    # --- Lord identity (deepest level) ---
    f['yg_ad_planet_idx'] = list(YOGINI_PLANETS.values()).index(deepest)

    # --- MD-level features ---
    f['yg_md_is_maraka'] = 1.0 if md_planet in father_mk else 0.0
    f['yg_md_is_danger'] = 1.0 if md_planet in father_danger else 0.0

    # --- AD-level features ---
    f['yg_ad_is_maraka'] = 1.0 if ad_planet in father_mk else 0.0
    f['yg_ad_is_danger'] = 1.0 if ad_planet in father_danger else 0.0
    f['yg_ad_is_malefic'] = 1.0 if ad_planet in NATURAL_MALEFICS else 0.0

    # --- AD natal strength ---
    if ad_planet in chart and ad_planet not in ('Rahu', 'Ketu'):
        ad_lon = chart[ad_planet]['longitude']
        f['yg_ad_uchcha'] = uchcha_bala(ad_planet, ad_lon)
        f['yg_ad_dignity'] = sign_dignity(ad_planet, ad_lon)
    else:
        f['yg_ad_uchcha'] = 0.5
        f['yg_ad_dignity'] = 0.5

    # --- PD-level features (depth=3 only) ---
    if depth >= 3:
        f['yg_pd_is_maraka'] = 1.0 if pd_planet in father_mk else 0.0
        f['yg_pd_is_danger'] = 1.0 if pd_planet in father_danger else 0.0
        f['yg_pd_is_malefic'] = 1.0 if pd_planet in NATURAL_MALEFICS else 0.0

        # PD natal strength
        if pd_planet in chart and pd_planet not in ('Rahu', 'Ketu'):
            pd_lon = chart[pd_planet]['longitude']
            f['yg_pd_uchcha'] = uchcha_bala(pd_planet, pd_lon)
            f['yg_pd_dignity'] = sign_dignity(pd_planet, pd_lon)
            # PD lord house placement
            pd_house = get_house_number(pd_lon, natal_asc)
            f['yg_pd_in_dusthana'] = 1.0 if pd_house in (6, 8, 12) else 0.0
        else:
            f['yg_pd_uchcha'] = 0.5
            f['yg_pd_dignity'] = 0.5
            f['yg_pd_in_dusthana'] = 0.0

    # --- Cascade: how many levels point to maraka ---
    mk_levels = [md_planet in father_mk, ad_planet in father_mk]
    dg_levels = [md_planet in father_danger, ad_planet in father_danger]
    if depth >= 3:
        mk_levels.append(pd_planet in father_mk)
        dg_levels.append(pd_planet in father_danger)

    f['yg_maraka_cascade'] = sum(mk_levels[:2])
    f['yg_danger_cascade'] = sum(dg_levels[:2])
    f['yg_both_maraka'] = 1.0 if mk_levels[0] and mk_levels[1] else 0.0

    if depth >= 3:
        f['yg_maraka_cascade_3'] = sum(mk_levels)
        f['yg_danger_cascade_3'] = sum(dg_levels)
        f['yg_all_three_maraka'] = 1.0 if all(mk_levels) else 0.0

    # --- Lord relationships (friendly/enemy) ---
    md_friends = _FRIENDS.get(md_planet, set())
    f['yg_md_ad_friendly'] = 1.0 if ad_planet in md_friends else 0.0
    f['yg_md_ad_enemy'] = 1.0 if (
        ad_planet not in md_friends and ad_planet != md_planet
    ) else 0.0

    if depth >= 3:
        ad_friends = _FRIENDS.get(ad_planet, set())
        f['yg_ad_pd_friendly'] = 1.0 if pd_planet in ad_friends else 0.0
        f['yg_ad_pd_enemy'] = 1.0 if (
            pd_planet not in ad_friends and pd_planet != ad_planet
        ) else 0.0
        f['yg_md_pd_friendly'] = 1.0 if pd_planet in md_friends else 0.0
        f['yg_md_pd_enemy'] = 1.0 if (
            pd_planet not in md_friends and pd_planet != md_planet
        ) else 0.0
        f['yg_any_pair_enemy'] = 1.0 if (
            f['yg_md_ad_enemy'] or f['yg_ad_pd_enemy'] or f['yg_md_pd_enemy']
        ) else 0.0

    # --- Yogini nature scores ---
    md_yogini = yogini_names[0]
    ad_yogini = yogini_names[1] if depth >= 2 else md_yogini
    pd_yogini = yogini_names[2] if depth >= 3 else yogini_names[-1]

    f['yg_md_nature'] = _YOGINI_NATURE.get(md_yogini, 0.0)
    f['yg_ad_nature'] = _YOGINI_NATURE.get(ad_yogini, 0.0)
    if depth >= 3:
        f['yg_pd_nature'] = _YOGINI_NATURE.get(pd_yogini, 0.0)
        f['yg_nature_sum'] = f['yg_md_nature'] + f['yg_ad_nature'] + f['yg_pd_nature']
        f['yg_nature_product'] = f['yg_md_nature'] * f['yg_ad_nature'] * f['yg_pd_nature']

    # --- PD lord transit position at midpoint ---
    if depth >= 3 and pd_planet in _SWE_IDS:
        midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
        pd_swe = _SWE_IDS[pd_planet]
        pd_transit_lon = swe.calc_ut(midpoint_jd, pd_swe, swe.FLG_SIDEREAL)[0][0]
        pd_transit_sign = int(pd_transit_lon / 30) % 12

        mk_signs = yg_ctx['father_maraka_signs']
        dg_signs = yg_ctx['father_danger_signs']
        f['yg_pd_transit_on_maraka'] = 1.0 if pd_transit_sign in mk_signs else 0.0
        f['yg_pd_transit_on_danger'] = 1.0 if pd_transit_sign in dg_signs else 0.0
    elif depth >= 3:
        f['yg_pd_transit_on_maraka'] = 0.0
        f['yg_pd_transit_on_danger'] = 0.0

    # --- Self-return ---
    f['yg_self_return'] = 1.0 if md_planet == deepest else 0.0
    if depth >= 3:
        f['yg_ad_pd_same'] = 1.0 if ad_planet == pd_planet else 0.0

    # --- Yogini Mrityu Yoga (classical death combinations) ---
    # Per BPHS: specific Yogini distances from birth Yogini are inauspicious
    # 5th Yogini from birth = Nidhan (death), 1st = Janma (self-harm)
    if 'birth_yogini_idx' in yg_ctx:
        birth_idx = yg_ctx['birth_yogini_idx']
        pd_idx = YOGINI_NAMES.index(pd_yogini)
        ad_idx = YOGINI_NAMES.index(ad_yogini)
        # Distance from birth Yogini (0-7)
        pd_dist = (pd_idx - birth_idx) % 8
        ad_dist = (ad_idx - birth_idx) % 8
        f['yg_pd_from_birth'] = pd_dist
        f['yg_pd_is_janma'] = 1.0 if pd_dist == 0 else 0.0  # 1st = Janma
        f['yg_pd_is_vipat'] = 1.0 if pd_dist == 2 else 0.0  # 3rd = Vipat
        f['yg_pd_is_pratyari'] = 1.0 if pd_dist == 4 else 0.0  # 5th = Pratyari
        f['yg_pd_is_nidhan'] = 1.0 if pd_dist == 6 else 0.0  # 7th = Nidhan (death)
        f['yg_pd_inauspicious'] = 1.0 if pd_dist in (0, 2, 4, 6) else 0.0
        f['yg_ad_is_nidhan'] = 1.0 if ad_dist == 6 else 0.0
        f['yg_ad_pd_both_inauspicious'] = 1.0 if (
            pd_dist in (0, 2, 4, 6) and ad_dist in (0, 2, 4, 6)
        ) else 0.0
        # Combined inauspicious count across all 3 levels
        md_idx = YOGINI_NAMES.index(md_yogini)
        md_dist = (md_idx - birth_idx) % 8
        f['yg_inauspicious_count'] = sum(
            d in (0, 2, 4, 6)
            for d in [md_dist, ad_dist, pd_dist])

    # --- Yogini 36-year cycle position ---
    if 'birth_jd' in yg_ctx:
        mid_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
        years_elapsed = (mid_jd - yg_ctx['birth_jd']) / 365.25
        cycle_pos = (years_elapsed % 36.0) / 36.0  # 0-1 within cycle
        f['yg_cycle_pos'] = cycle_pos
        # Second half of cycle (years 18-36) is classically more malefic
        f['yg_cycle_second_half'] = 1.0 if cycle_pos > 0.5 else 0.0
        # Quadrant within cycle
        f['yg_cycle_quadrant'] = int(cycle_pos * 4)
        # Cycle position x maraka interactions
        mk_cascade = f.get('yg_maraka_cascade_3', f.get('yg_maraka_cascade', 0))
        f['yg_cycle_x_maraka'] = cycle_pos * mk_cascade
        f['yg_cycle_x_danger'] = cycle_pos * f.get('yg_danger_cascade_3',
                                                     f.get('yg_danger_cascade', 0))
        # Age at event (continuous)
        f['yg_age_years'] = years_elapsed

    return f


def extract_yogini_subperiod_features(candidate, yg_ctx):
    """Yogini depth=4 sub-period density for a depth=3 PD candidate.

    Analogous to Vimshottari sookshma density: count how many of the 8
    depth=4 sub-periods have lords that are father marakas.

    Returns: ~10 features dict.
    """
    father_mk = yg_ctx['father_marakas']
    father_danger = yg_ctx['father_danger']

    yogini_names = candidate['lords']
    if len(yogini_names) < 3:
        return {}

    pd_yogini = yogini_names[2]
    pd_idx = YOGINI_NAMES.index(pd_yogini)
    sub_seq = YOGINI_NAMES[pd_idx:] + YOGINI_NAMES[:pd_idx]

    parent_days = candidate['duration_days']

    n_primary = 0
    n_danger = 0
    n_malefic = 0
    longest_streak = 0
    current_streak = 0
    midpoint_offset = parent_days / 2
    cumulative = 0.0
    mid_planet = None
    first_maraka_frac = 1.0
    maraka_time_days = 0.0

    for sub_yog in sub_seq:
        sub_planet = YOGINI_PLANETS[sub_yog]
        sub_frac = YOGINI_YEARS[sub_yog] / YOGINI_TOTAL
        sub_days = parent_days * sub_frac

        if sub_planet in father_mk:
            n_primary += 1
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
            maraka_time_days += sub_days
            if first_maraka_frac == 1.0:
                first_maraka_frac = cumulative / max(parent_days, 0.1)
        else:
            current_streak = 0

        if sub_planet in father_danger:
            n_danger += 1
        if sub_planet in NATURAL_MALEFICS:
            n_malefic += 1

        if mid_planet is None and cumulative + sub_days > midpoint_offset:
            mid_planet = sub_planet
        cumulative += sub_days

    return {
        'ysk_primary_count': n_primary,
        'ysk_primary_frac': n_primary / 8.0,
        'ysk_danger_count': n_danger,
        'ysk_danger_frac': n_danger / 8.0,
        'ysk_malefic_frac': n_malefic / 8.0,
        'ysk_longest_streak': longest_streak,
        'ysk_mid_is_primary': 1.0 if mid_planet in father_mk else 0.0,
        'ysk_mid_is_danger': 1.0 if mid_planet in father_danger else 0.0,
        'ysk_first_maraka_frac': first_maraka_frac,
        'ysk_maraka_time_frac': maraka_time_days / max(parent_days, 0.1),
    }
