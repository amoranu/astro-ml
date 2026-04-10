"""Combustion features — planets too close to the Sun lose power.

Per BPHS, a planet conjunct the Sun within a specific orb is 'combust'
and its ability to produce results is severely weakened.

~5 features per candidate.
"""

# Combustion orbs per BPHS (degrees)
_COMBUSTION_ORBS = {
    'Moon': 12.0,
    'Mars': 17.0,
    'Mercury': 12.0,  # 14 when retrograde, but using 12 as default
    'Jupiter': 11.0,
    'Venus': 10.0,
    'Saturn': 15.0,
}


def _angular_distance(lon1: float, lon2: float) -> float:
    """Shortest angular distance between two longitudes."""
    d = abs(lon1 - lon2) % 360
    return min(d, 360 - d)


def extract_combustion_features(candidate: dict, natal_chart: dict) -> dict:
    """Check if dasha lords are combust (too close to natal Sun).

    A combust planet is weakened — if it's a maraka lord, combustion
    reduces its potency to kill; if it's a benefic protector, combustion
    removes the protection.
    """
    sun_lon = natal_chart['Sun']['longitude']
    lords = candidate['lords']

    f = {}

    # SD lord (deepest) combustion
    sd_lord = lords[-1]
    if sd_lord in _COMBUSTION_ORBS:
        sd_dist = _angular_distance(
            natal_chart[sd_lord]['longitude'], sun_lon)
        orb = _COMBUSTION_ORBS[sd_lord]
        f['comb_sd_combust'] = 1.0 if sd_dist <= orb else 0.0
        f['comb_sd_sun_dist'] = sd_dist / 180.0  # normalized
    else:
        f['comb_sd_combust'] = 0.0
        f['comb_sd_sun_dist'] = 1.0

    # Count of combust lords in the hierarchy
    combust_count = 0
    for lord in lords:
        if lord in _COMBUSTION_ORBS:
            dist = _angular_distance(
                natal_chart[lord]['longitude'], sun_lon)
            if dist <= _COMBUSTION_ORBS[lord]:
                combust_count += 1
    f['comb_count'] = combust_count
    f['comb_any'] = 1.0 if combust_count > 0 else 0.0

    # PD lord combustion (if different from SD lord and depth >= 3)
    if len(lords) >= 3:
        pd_lord = lords[2]  # depth 3 = PD
        if pd_lord in _COMBUSTION_ORBS:
            pd_dist = _angular_distance(
                natal_chart[pd_lord]['longitude'], sun_lon)
            f['comb_pd_combust'] = 1.0 if pd_dist <= _COMBUSTION_ORBS[pd_lord] else 0.0
        else:
            f['comb_pd_combust'] = 0.0
    else:
        f['comb_pd_combust'] = 0.0

    return f
