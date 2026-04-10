"""Rahu-Ketu axis transit features.

The eclipse axis (Rahu-Ketu) transiting key natal points is a
classic Parashari death trigger. Rahu/Ketu on natal Sun (Pitru Karaka),
9th cusp (father's lagna), or Moon = amplified danger.

~6 features per candidate.
"""

import swisseph as swe

from ..astro_engine.houses import get_sign

swe.set_sid_mode(swe.SIDM_LAHIRI)


def extract_eclipse_features(candidate, natal_chart, natal_asc):
    """Rahu-Ketu axis transit on key natal points.

    Uses sign-based aspects (conjunction + opposition = axis).
    """
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2

    # Transit Rahu position
    t_rahu_lon = swe.calc_ut(midpoint_jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)[0][0]
    rahu_sign = int(t_rahu_lon / 30) % 12
    ketu_sign = (rahu_sign + 6) % 12  # always opposite
    axis = {rahu_sign, ketu_sign}

    # Natal key signs
    sun_sign = int(natal_chart['Sun']['longitude'] / 30) % 12
    moon_sign = int(natal_chart['Moon']['longitude'] / 30) % 12
    asc_sign = get_sign(natal_asc)
    h9_sign = (asc_sign + 8) % 12  # father's lagna

    # Father's maraka house signs
    father_maraka_signs = {
        (h9_sign + 1) % 12,   # father's 2nd
        (h9_sign + 6) % 12,   # father's 7th
    }

    f = {}

    # Rahu-Ketu axis on natal Sun (Pitru Karaka — father indicator)
    f['ec_axis_on_sun'] = 1.0 if sun_sign in axis else 0.0

    # Axis on natal Moon (emotional/health impact)
    f['ec_axis_on_moon'] = 1.0 if moon_sign in axis else 0.0

    # Axis on 9th cusp sign (father's lagna)
    f['ec_axis_on_9th'] = 1.0 if h9_sign in axis else 0.0

    # Axis on father's maraka houses
    f['ec_axis_on_maraka'] = 1.0 if bool(axis & father_maraka_signs) else 0.0

    # Any key point afflicted
    f['ec_any_affliction'] = 1.0 if (
        f['ec_axis_on_sun'] or f['ec_axis_on_9th'] or f['ec_axis_on_maraka']
    ) else 0.0

    # Affliction count (how many key points the axis touches)
    f['ec_affliction_count'] = sum([
        sun_sign in axis,
        moon_sign in axis,
        h9_sign in axis,
        bool(axis & father_maraka_signs),
    ])

    # --- Degree-level Rahu distances (Phase 1a) ---
    # Continuous features that vary smoothly across PD candidates
    natal_sun_deg = natal_chart['Sun']['longitude']
    natal_moon_deg = natal_chart['Moon']['longitude']
    h9_cusp_deg = (natal_asc + 8 * 30) % 360  # 9th house cusp (equal house)

    def _arc(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    f['ec_rahu_dist_sun'] = _arc(t_rahu_lon, natal_sun_deg)
    f['ec_rahu_dist_h9'] = _arc(t_rahu_lon, h9_cusp_deg)
    f['ec_rahu_dist_moon'] = _arc(t_rahu_lon, natal_moon_deg)
    f['ec_rahu_tight_sun'] = 1.0 if _arc(t_rahu_lon, natal_sun_deg) <= 5 else 0.0

    return f
