"""Bhinnashtakavarga (BAV) and Sarvashtakavarga (SAV) computation.

Full BAV tables from BPHS Chapter 66.
"""

from .houses import get_sign

# BAV benefic-position tables: from each reference point, the planet
# contributes a bindu at these house offsets (1-indexed from reference).
# Reference points: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Asc

SUN_BAV = {
    'Sun':     [1, 2, 4, 7, 8, 9, 10, 11],
    'Moon':    [3, 6, 10, 11],
    'Mars':    [1, 2, 4, 7, 8, 9, 10, 11],
    'Mercury': [3, 5, 6, 9, 10, 11, 12],
    'Jupiter': [5, 6, 9, 11],
    'Venus':   [6, 7, 12],
    'Saturn':  [1, 2, 4, 7, 8, 9, 10, 11],
    'Asc':     [3, 4, 6, 10, 11, 12],
}

MOON_BAV = {
    'Sun':     [3, 6, 7, 8, 10, 11],
    'Moon':    [1, 3, 6, 7, 10, 11],
    'Mars':    [2, 3, 5, 6, 9, 10, 11],
    'Mercury': [1, 3, 4, 5, 7, 8, 10, 11],
    'Jupiter': [1, 4, 7, 8, 10, 11, 12],
    'Venus':   [3, 4, 5, 7, 9, 10, 11],
    'Saturn':  [3, 5, 6, 11],
    'Asc':     [3, 6, 10, 11],
}

MARS_BAV = {
    'Sun':     [3, 5, 6, 10, 11],
    'Moon':    [3, 6, 11],
    'Mars':    [1, 2, 4, 7, 8, 10, 11],
    'Mercury': [3, 5, 6, 11],
    'Jupiter': [6, 10, 11, 12],
    'Venus':   [6, 8, 11, 12],
    'Saturn':  [1, 4, 7, 8, 9, 10, 11],
    'Asc':     [1, 3, 6, 10, 11],
}

MERCURY_BAV = {
    'Sun':     [5, 6, 9, 11, 12],
    'Moon':    [2, 4, 6, 8, 10, 11],
    'Mars':    [1, 2, 4, 7, 8, 9, 10, 11],
    'Mercury': [1, 3, 5, 6, 9, 10, 11, 12],
    'Jupiter': [6, 8, 11, 12],
    'Venus':   [1, 2, 3, 4, 5, 8, 9, 11],
    'Saturn':  [1, 2, 4, 7, 8, 9, 10, 11],
    'Asc':     [1, 2, 4, 6, 8, 10, 11],
}

JUPITER_BAV = {
    'Sun':     [1, 2, 3, 4, 7, 8, 9, 10, 11],
    'Moon':    [2, 5, 7, 9, 11],
    'Mars':    [1, 2, 4, 7, 8, 10, 11],
    'Mercury': [1, 2, 4, 5, 6, 9, 10, 11],
    'Jupiter': [1, 2, 3, 4, 7, 8, 10, 11],
    'Venus':   [2, 5, 6, 9, 10, 11],
    'Saturn':  [3, 5, 6, 12],
    'Asc':     [1, 2, 4, 5, 6, 7, 9, 10, 11],
}

VENUS_BAV = {
    'Sun':     [8, 11, 12],
    'Moon':    [1, 2, 3, 4, 5, 8, 9, 11, 12],
    'Mars':    [3, 5, 6, 9, 11, 12],
    'Mercury': [3, 5, 6, 9, 11],
    'Jupiter': [5, 8, 9, 10, 11],
    'Venus':   [1, 2, 3, 4, 5, 8, 9, 10, 11],
    'Saturn':  [3, 4, 5, 8, 9, 10, 11],
    'Asc':     [1, 2, 3, 4, 5, 8, 9, 11],
}

SATURN_BAV = {
    'Sun':     [1, 2, 4, 7, 8, 9, 10, 11],
    'Moon':    [3, 6, 11],
    'Mars':    [3, 5, 6, 10, 11, 12],
    'Mercury': [6, 8, 9, 10, 11, 12],
    'Jupiter': [5, 6, 11, 12],
    'Venus':   [6, 11, 12],
    'Saturn':  [3, 5, 6, 11],
    'Asc':     [1, 3, 4, 6, 10, 11],
}

BAV_TABLES = {
    'Sun': SUN_BAV,
    'Moon': MOON_BAV,
    'Mars': MARS_BAV,
    'Mercury': MERCURY_BAV,
    'Jupiter': JUPITER_BAV,
    'Venus': VENUS_BAV,
    'Saturn': SATURN_BAV,
}


def compute_bav(planet_table: dict, chart_positions: dict, asc_long: float) -> list:
    """Compute Bhinnashtakavarga for one planet.

    Args:
        planet_table: BAV table dict for the planet
        chart_positions: {planet_name: {'longitude': float, ...}}
        asc_long: ascendant longitude in degrees

    Returns:
        List of 12 integers (bindus per sign, index 0 = Aries)
    """
    bindus = [0] * 12

    references = {}
    for p in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']:
        references[p] = get_sign(chart_positions[p]['longitude'])
    references['Asc'] = get_sign(asc_long)

    for ref_name, ref_sign in references.items():
        if ref_name in planet_table:
            for house_offset in planet_table[ref_name]:
                target_sign = (ref_sign + house_offset - 1) % 12
                bindus[target_sign] += 1

    return bindus


def compute_sav(chart_positions: dict, asc_long: float) -> list:
    """Compute Sarvashtakavarga — sum of all 7 planet BAVs.

    Returns:
        List of 12 integers (total bindus per sign)
    """
    sav = [0] * 12
    for planet_name, table in BAV_TABLES.items():
        bav = compute_bav(table, chart_positions, asc_long)
        for i in range(12):
            sav[i] += bav[i]
    return sav
