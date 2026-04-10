"""Swiss Ephemeris wrapper for sidereal chart computation."""

import swisseph as swe

# Use Lahiri ayanamsha exclusively
swe.set_sid_mode(swe.SIDM_LAHIRI)

PLANETS = {
    'Sun': swe.SUN,
    'Moon': swe.MOON,
    'Mars': swe.MARS,
    'Mercury': swe.MERCURY,
    'Jupiter': swe.JUPITER,
    'Venus': swe.VENUS,
    'Saturn': swe.SATURN,
    'Rahu': swe.MEAN_NODE,  # Mean Node per Parashari convention
}


def compute_jd(birth_date_str: str, birth_time_str: str) -> float:
    """Convert date + time strings to Julian Day.

    Args:
        birth_date_str: 'YYYY-MM-DD'
        birth_time_str: 'HH:MM' or 'HH:MM:SS'
    """
    parts = birth_date_str.split('-')
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

    time_parts = birth_time_str.split(':')
    hour = int(time_parts[0])
    minute = int(time_parts[1])
    second = int(time_parts[2]) if len(time_parts) > 2 else 0

    ut_hour = hour + minute / 60.0 + second / 3600.0
    return swe.julday(year, month, day, ut_hour)


def compute_chart(birth_date: str, birth_time: str, lat: float, lon: float):
    """Compute sidereal planetary positions and ascendant.

    Args:
        birth_date: 'YYYY-MM-DD'
        birth_time: 'HH:MM' or 'HH:MM:SS'
        lat: geographic latitude
        lon: geographic longitude

    Returns:
        (chart_dict, ascendant_degree)
        chart_dict: {planet_name: {'longitude': float, 'latitude': float, 'speed': float}}
        ascendant_degree: sidereal ascendant in degrees
    """
    jd = compute_jd(birth_date, birth_time)

    chart = {}
    for name, planet_id in PLANETS.items():
        pos = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)
        chart[name] = {
            'longitude': pos[0][0],
            'latitude': pos[0][1],
            'speed': pos[0][3],
        }

    # Ketu = Rahu + 180°
    chart['Ketu'] = {
        'longitude': (chart['Rahu']['longitude'] + 180) % 360,
        'latitude': -chart['Rahu']['latitude'],
        'speed': chart['Rahu']['speed'],
    }

    # Ascendant — Equal House system
    cusps, asc_mc = swe.houses_ex(jd, lat, lon, b'E', swe.FLG_SIDEREAL)
    asc = asc_mc[0]

    return chart, asc
