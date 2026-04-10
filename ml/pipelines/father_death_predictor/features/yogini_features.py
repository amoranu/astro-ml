"""Yogini Dasha agreement features — independent 36-year cycle witness.

Yogini uses a completely different cycle (36 yrs vs 120 yrs) with
different planet assignments. Agreement between Vimshottari and Yogini
pointing to the same candidate is a strong independent signal.

~8 features per candidate.
"""

from ..astro_engine.yogini_dasha import compute_yogini_dasha, YOGINI_PLANETS
from ..astro_engine.houses import get_sign, get_sign_lord


def precompute_yogini(moon_longitude, birth_jd, father_marakas):
    """Compute Yogini dasha periods once per chart.

    Returns depth-3 periods with maraka flags pre-computed.
    """
    periods = compute_yogini_dasha(moon_longitude, birth_jd, max_depth=3)
    p3 = [p for p in periods if p['depth'] == 3]
    return p3


def _find_yogini_at_jd(yogini_p3, jd):
    """Find the Yogini pratyantardasha containing a given JD."""
    for p in yogini_p3:
        if p['start_jd'] <= jd < p['end_jd']:
            return p
    return None


def extract_yogini_features(candidate, yogini_p3, father_marakas):
    """Yogini dasha agreement features for a Vimshottari candidate.

    Checks whether Yogini's concurrent period also points to death.
    """
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
    yp = _find_yogini_at_jd(yogini_p3, midpoint_jd)

    f = {}

    if yp is None:
        f['yg_lord_is_maraka'] = 0.0
        f['yg_md_is_maraka'] = 0.0
        f['yg_ad_is_maraka'] = 0.0
        f['yg_maraka_count'] = 0
        f['yg_agrees_vim'] = 0.0
        f['yg_same_planet'] = 0.0
        f['yg_lord_is_malefic'] = 0.0
        f['yg_cascade_agree'] = 0.0
        return f

    # Yogini planets at depth 3: [MD_planet, AD_planet, PD_planet]
    yg_planets = yp['planets']
    yg_pd = yg_planets[-1]  # pratyantardasha planet
    yg_md = yg_planets[0]
    yg_ad = yg_planets[1] if len(yg_planets) > 1 else yg_md

    # Vimshottari candidate's PD lord
    vim_pd = candidate['lords'][-1]

    # Maraka checks
    f['yg_lord_is_maraka'] = 1.0 if yg_pd in father_marakas else 0.0
    f['yg_md_is_maraka'] = 1.0 if yg_md in father_marakas else 0.0
    f['yg_ad_is_maraka'] = 1.0 if yg_ad in father_marakas else 0.0

    # Count of Yogini lords that are maraka (0-3)
    f['yg_maraka_count'] = sum([
        yg_md in father_marakas,
        yg_ad in father_marakas,
        yg_pd in father_marakas,
    ])

    # Cross-system agreement: both Vim and Yogini PD lords are maraka
    vim_pd_maraka = vim_pd in father_marakas
    f['yg_agrees_vim'] = 1.0 if (
        f['yg_lord_is_maraka'] and vim_pd_maraka
    ) else 0.0

    # Same planet in both systems (strongest agreement)
    f['yg_same_planet'] = 1.0 if yg_pd == vim_pd else 0.0

    # Natural malefic
    f['yg_lord_is_malefic'] = 1.0 if yg_pd in (
        'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu') else 0.0

    # Both systems have maraka cascade (MD or AD is maraka)
    vim_md_maraka = candidate['lords'][0] in father_marakas
    f['yg_cascade_agree'] = 1.0 if (
        (f['yg_md_is_maraka'] or f['yg_ad_is_maraka'])
        and vim_md_maraka
    ) else 0.0

    # --- Phase 3: Cross-system features ---
    # Both MD lords are maraka (top-level period dangerous in both systems)
    f['yg_dual_md_maraka'] = 1.0 if (
        f['yg_md_is_maraka'] and vim_md_maraka
    ) else 0.0

    # Total maraka count across both systems (0-6: 3 yogini + 3 vimshottari)
    vim_maraka_count = sum([
        vim_md_maraka,
        candidate['lords'][1] in father_marakas,
        vim_pd in father_marakas,
    ])
    f['yg_combined_maraka_count'] = f['yg_maraka_count'] + vim_maraka_count

    # Both systems have 2+ maraka lords in their hierarchy
    f['yg_both_cascade_strong'] = 1.0 if (
        f['yg_maraka_count'] >= 2 and vim_maraka_count >= 2
    ) else 0.0

    # --- Phase 2b: Quad-system agreement ---
    # All 4 positions (Vim MD, AD, PD + Yogini PD) are maraka
    vim_ad_maraka = candidate['lords'][1] in father_marakas
    f['yg_quad_agree'] = 1.0 if (
        vim_md_maraka and vim_ad_maraka and vim_pd_maraka
        and f['yg_lord_is_maraka']
    ) else 0.0

    # Combined danger score (continuous 0-6)
    f['yg_dual_system_danger_score'] = (
        f['yg_maraka_count'] + vim_maraka_count
    ) / 6.0  # normalized to 0-1

    # Triple system PD: Moon-Vim PD + Sun-Vim PD + Yogini PD all maraka
    # (Moon/Sun refs are in multiref features, but we can check Yogini + Vim)
    f['yg_vim_yogini_both_strong'] = 1.0 if (
        f['yg_maraka_count'] >= 2 and vim_maraka_count >= 2
        and f['yg_lord_is_maraka'] and vim_pd_maraka
    ) else 0.0

    return f
