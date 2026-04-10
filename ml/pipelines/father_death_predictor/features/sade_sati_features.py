"""Sade Sati (Saturn's 7.5-year transit over Moon) features.

Classic Parashari death/hardship indicator. Saturn transiting the
12th, 1st, or 2nd from natal Moon sign = Sade Sati active.
Phase determines severity: peak (same sign) is worst.

Also: Saturn on 4th/8th from Moon = Dhaiyya (2.5-year affliction).

~6 features per candidate.
"""

import swisseph as swe

swe.set_sid_mode(swe.SIDM_LAHIRI)


def extract_sade_sati_features(candidate, natal_chart):
    """Sade Sati and Dhaiyya features at candidate midpoint.

    Args:
        candidate: pratyantardasha period dict
        natal_chart: from compute_chart()

    Returns:
        ~6 features dict.
    """
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2

    moon_sign = int(natal_chart['Moon']['longitude'] / 30) % 12
    t_sat_lon = swe.calc_ut(midpoint_jd, swe.SATURN, swe.FLG_SIDEREAL)[0][0]
    sat_sign = int(t_sat_lon / 30) % 12

    # Distance from Moon sign (0 = same sign)
    dist = (sat_sign - moon_sign) % 12

    f = {}

    # Sade Sati phases
    f['ss_active'] = 1.0 if dist in (11, 0, 1) else 0.0
    f['ss_peak'] = 1.0 if dist == 0 else 0.0       # Saturn ON Moon sign
    f['ss_rising'] = 1.0 if dist == 11 else 0.0     # 12th from Moon (entering)
    f['ss_setting'] = 1.0 if dist == 1 else 0.0     # 2nd from Moon (leaving)

    # Dhaiyya (small Sade Sati): Saturn on 4th or 8th from Moon
    f['ss_dhaiyya'] = 1.0 if dist in (3, 7) else 0.0

    # Combined affliction score
    # Peak=3, rising/setting=2, dhaiyya=1, none=0
    if dist == 0:
        f['ss_severity'] = 3.0
    elif dist in (11, 1):
        f['ss_severity'] = 2.0
    elif dist in (3, 7):
        f['ss_severity'] = 1.0
    else:
        f['ss_severity'] = 0.0

    # --- Degree-level Saturn-Moon distance (Phase 1b) ---
    natal_moon_deg = natal_chart['Moon']['longitude']
    d = abs(t_sat_lon - natal_moon_deg) % 360
    sat_moon_deg = min(d, 360 - d)
    f['ss_sat_moon_deg'] = sat_moon_deg
    f['ss_sat_moon_tight'] = 1.0 if sat_moon_deg <= 5 else 0.0
    f['ss_sat_moon_orb15'] = 1.0 if sat_moon_deg <= 15 else 0.0

    return f
