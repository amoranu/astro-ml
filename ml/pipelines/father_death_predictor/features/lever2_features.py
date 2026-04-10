"""Lever 2: Enhanced transit sampling + cross-interaction features.

1. Multi-point transit: sample 4 points within each pratyantardasha
   to capture sign changes during the period (~12 features)
2. Jupiter BAV filter: currently only Saturn BAV is used (~3 features)
3. Cross-interactions: Gochar x Hierarchy combinations that encode
   the Parashari "trigger + context" principle (~8 features)

~23 features total.
"""

import swisseph as swe

from ..astro_engine.houses import get_sign
from ..astro_engine.ashtakavarga import compute_bav, compute_sav, BAV_TABLES

swe.set_sid_mode(swe.SIDM_LAHIRI)


def _transit_sign(jd, planet_id):
    return int(swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)[0][0] / 30) % 12


def _vedic_aspect_signs(planet, sign):
    signs = {sign, (sign + 6) % 12}
    if planet == 'Saturn':
        signs.add((sign + 2) % 12)
        signs.add((sign + 9) % 12)
    elif planet == 'Jupiter':
        signs.add((sign + 4) % 12)
        signs.add((sign + 8) % 12)
    elif planet == 'Mars':
        signs.add((sign + 3) % 12)
        signs.add((sign + 7) % 12)
    return signs


# ── 1. Multi-Point Transit Sampling ────────────────────────────────────

def extract_multipoint_features(candidate, gochar_ctx):
    """Sample transit at 4 points within the pratyantardasha.

    Points: 12.5%, 37.5%, 62.5%, 87.5% through the period.
    Captures sign changes that happen mid-period.
    """
    start = candidate['start_jd']
    dur = candidate['duration_days']
    fracs = [0.125, 0.375, 0.625, 0.875]
    sample_jds = [start + dur * fr for fr in fracs]

    father_mk = gochar_ctx['father_maraka_signs']
    danger = gochar_ctx['danger_signs']

    # Collect binary hits at each sample point
    sat_mk_hits = 0
    jup_mk_hits = 0
    double_hits = 0
    mars_mk_hits = 0
    merc_mk_hits = 0
    venus_mk_hits = 0
    sat_signs_seen = set()
    jup_signs_seen = set()
    mars_signs_seen = set()
    merc_signs_seen = set()
    venus_signs_seen = set()

    for jd in sample_jds:
        sat_s = _transit_sign(jd, swe.SATURN)
        jup_s = _transit_sign(jd, swe.JUPITER)
        mars_s = _transit_sign(jd, swe.MARS)
        merc_s = _transit_sign(jd, swe.MERCURY)
        venus_s = _transit_sign(jd, swe.VENUS)

        sat_asp = _vedic_aspect_signs('Saturn', sat_s)
        jup_asp = _vedic_aspect_signs('Jupiter', jup_s)
        mars_asp = _vedic_aspect_signs('Mars', mars_s)
        merc_asp = _vedic_aspect_signs('Mercury', merc_s)
        venus_asp = _vedic_aspect_signs('Venus', venus_s)

        sat_on_mk = bool(sat_asp & father_mk)
        jup_on_mk = bool(jup_asp & father_mk)

        if sat_on_mk:
            sat_mk_hits += 1
        if jup_on_mk:
            jup_mk_hits += 1
        if sat_on_mk and jup_on_mk:
            double_hits += 1
        if bool(mars_asp & danger):
            mars_mk_hits += 1
        if bool(merc_asp & danger):
            merc_mk_hits += 1
        if bool(venus_asp & danger):
            venus_mk_hits += 1

        sat_signs_seen.add(sat_s)
        jup_signs_seen.add(jup_s)
        mars_signs_seen.add(mars_s)
        merc_signs_seen.add(merc_s)
        venus_signs_seen.add(venus_s)

    f = {}
    # Fraction of sample points where transit is active
    f['mp_sat_maraka_frac'] = sat_mk_hits / 4.0
    f['mp_jup_maraka_frac'] = jup_mk_hits / 4.0
    f['mp_double_frac'] = double_hits / 4.0
    f['mp_mars_danger_frac'] = mars_mk_hits / 4.0
    f['mp_merc_danger_frac'] = merc_mk_hits / 4.0
    f['mp_venus_danger_frac'] = venus_mk_hits / 4.0

    # Any/all patterns
    f['mp_double_any'] = 1.0 if double_hits > 0 else 0.0
    f['mp_double_all'] = 1.0 if double_hits == 4 else 0.0

    # Sign changes during period (indicates transit ingress)
    f['mp_sat_sign_change'] = 1.0 if len(sat_signs_seen) > 1 else 0.0
    f['mp_jup_sign_change'] = 1.0 if len(jup_signs_seen) > 1 else 0.0
    f['mp_mars_sign_change'] = 1.0 if len(mars_signs_seen) > 1 else 0.0
    f['mp_merc_sign_change'] = 1.0 if len(merc_signs_seen) > 1 else 0.0
    f['mp_venus_sign_change'] = 1.0 if len(venus_signs_seen) > 1 else 0.0

    # Consistency: same transit pattern at all 4 points = stable trigger
    f['mp_sat_stable'] = 1.0 if sat_mk_hits in (0, 4) else 0.0
    f['mp_jup_stable'] = 1.0 if jup_mk_hits in (0, 4) else 0.0

    # Activation count (how many of 5 channels are active at majority)
    active_channels = sum([
        sat_mk_hits >= 2,
        jup_mk_hits >= 2,
        mars_mk_hits >= 2,
        merc_mk_hits >= 2,
        venus_mk_hits >= 2,
    ])
    f['mp_active_channels'] = active_channels
    f['mp_full_activation'] = 1.0 if active_channels >= 3 else 0.0

    return f


# ── 2. Jupiter BAV Filter ──────────────────────────────────────────────

def precompute_jup_bav(natal_chart, natal_asc):
    """Compute Jupiter BAV (once per chart)."""
    return compute_bav(BAV_TABLES['Jupiter'], natal_chart, natal_asc)


def extract_jup_bav_features(candidate, jup_bav):
    """Jupiter BAV at its transit sign — low = reduced protection."""
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
    jup_sign = _transit_sign(midpoint_jd, swe.JUPITER)

    f = {}
    f['jb_jup_bav'] = jup_bav[jup_sign]
    f['jb_jup_bav_low'] = 1.0 if jup_bav[jup_sign] <= 2 else 0.0
    # Jupiter in high-BAV sign = protective (death less likely)
    f['jb_jup_bav_high'] = 1.0 if jup_bav[jup_sign] >= 5 else 0.0

    return f


# ── 3. Cross-Interaction Features ──────────────────────────────────────

def extract_cross_features(gc_feats, hi_feats):
    """Gochar x Hierarchy interactions: the Parashari trigger+context principle.

    These encode: "a dangerous dasha period WHERE the transit also triggers"
    is far worse than either alone.
    """
    f = {}

    # Double transit during maraka cascade
    f['cx_double_x_cascade2'] = 1.0 if (
        gc_feats.get('gc_double_transit_maraka', 0) == 1.0
        and hi_feats.get('hi_maraka_cascade', 0) >= 2
    ) else 0.0

    # Saturn on danger AND MD is maraka
    f['cx_sat_danger_x_md_maraka'] = 1.0 if (
        gc_feats.get('gc_sat_asp_danger', 0) == 1.0
        and hi_feats.get('hi_md_is_maraka', 0) == 1.0
    ) else 0.0

    # Triple activation AND any maraka cascade
    f['cx_triple_x_cascade'] = 1.0 if (
        gc_feats.get('gc_triple_activation', 0) == 1.0
        and hi_feats.get('hi_maraka_cascade', 0) >= 1
    ) else 0.0

    # Saturn low BAV AND danger cascade >= 2
    f['cx_sat_low_bav_x_danger2'] = 1.0 if (
        gc_feats.get('gc_sat_bav_low', 0) == 1.0
        and hi_feats.get('hi_danger_cascade', 0) >= 2
    ) else 0.0

    # MD+AD both maraka AND double transit (strongest possible combination)
    f['cx_md_ad_maraka_x_double'] = 1.0 if (
        hi_feats.get('hi_md_ad_both_maraka', 0) == 1.0
        and gc_feats.get('gc_double_transit_maraka', 0) == 1.0
    ) else 0.0

    # Enemy relationship (MD-PD) AND Saturn on maraka (hostile period + transit)
    f['cx_md_pd_enemy_x_sat'] = 1.0 if (
        hi_feats.get('hi_md_pd_combined', 0) <= -1
        and gc_feats.get('gc_sat_asp_maraka', 0) == 1.0
    ) else 0.0

    # Maraka cascade count * double transit (continuous interaction)
    f['cx_cascade_x_double_score'] = (
        hi_feats.get('hi_maraka_cascade', 0)
        * gc_feats.get('gc_double_transit_maraka', 0)
    )

    # Danger cascade * Saturn BAV inverse (more danger + weaker transit sign)
    sat_bav = gc_feats.get('gc_sat_bav', 4)
    f['cx_danger_x_sat_weakness'] = (
        hi_feats.get('hi_danger_cascade', 0)
        * max(0, 4 - sat_bav) / 4.0  # 0 when BAV>=4, 1 when BAV=0
    )

    # --- Phase 2a: Additional convergence interactions ---
    # Full maraka cascade (3) AND Saturn on danger
    f['cx_cascade3_x_sat_danger'] = 1.0 if (
        hi_feats.get('hi_maraka_cascade', 0) >= 3
        and gc_feats.get('gc_sat_on_danger', 0) == 1.0
    ) else 0.0

    # Double transit AND triple maraka agreement (multiref)
    f['cx_double_transit_x_triple_maraka'] = 1.0 if (
        gc_feats.get('gc_double_transit_maraka', 0) == 1.0
        and hi_feats.get('hi_maraka_cascade', 0) >= 3
    ) else 0.0

    return f
