"""Exaltation, debilitation, sign dignity calculations."""

# Exact exaltation degrees (sidereal)
EXALTATION_DEGREES = {
    'Sun': 10.0,      # Aries 10°
    'Moon': 33.0,     # Taurus 3°
    'Mars': 298.0,    # Capricorn 28°
    'Mercury': 165.0, # Virgo 15°
    'Jupiter': 95.0,  # Cancer 5°
    'Venus': 357.0,   # Pisces 27°
    'Saturn': 200.0,  # Libra 20°
    'Rahu': 50.0,     # Taurus 20°
    'Ketu': 230.0,    # Scorpio 20°
}

# Own signs per planet (sign indices 0-11)
OWN_SIGNS = {
    'Sun': [4],          # Leo
    'Moon': [3],         # Cancer
    'Mars': [0, 7],      # Aries, Scorpio
    'Mercury': [2, 5],   # Gemini, Virgo
    'Jupiter': [8, 11],  # Sagittarius, Pisces
    'Venus': [1, 6],     # Taurus, Libra
    'Saturn': [9, 10],   # Capricorn, Aquarius
    'Rahu': [10],        # Aquarius
    'Ketu': [7],         # Scorpio
}

# Friendly signs (includes own + friends' signs)
FRIEND_SIGNS = {
    'Sun': [0, 3, 4, 7, 8, 11],
    'Moon': [0, 1, 3, 4, 7, 8, 11],
    'Mars': [0, 3, 4, 7, 8, 11],
    'Mercury': [1, 2, 4, 5, 6],
    'Jupiter': [0, 3, 4, 7, 8, 11],
    'Venus': [1, 2, 5, 6, 9, 10],
    'Saturn': [1, 2, 5, 6, 9, 10],
    'Rahu': [1, 2, 5, 6, 9, 10],
    'Ketu': [0, 3, 4, 7, 8, 11],
}


def uchcha_bala(planet: str, longitude: float) -> float:
    """Exaltation strength. 1.0 at exact exaltation, 0.0 at debilitation."""
    if planet not in EXALTATION_DEGREES:
        return 0.5
    exalt = EXALTATION_DEGREES[planet]
    dist = abs(longitude - exalt) % 360
    if dist > 180:
        dist = 360 - dist
    return (180 - dist) / 180.0


def sign_dignity(planet: str, longitude: float) -> float:
    """Dignity score: own=1.0, friend=0.75, neutral=0.5, enemy=0.25, debilitated=0.0."""
    sign = int(longitude / 30) % 12

    if sign in OWN_SIGNS.get(planet, []):
        return 1.0
    if sign in FRIEND_SIGNS.get(planet, []):
        return 0.75

    # Check debilitation (opposite of exaltation sign)
    if planet in EXALTATION_DEGREES:
        exalt_sign = int(EXALTATION_DEGREES[planet] / 30) % 12
        debil_sign = (exalt_sign + 6) % 12
        if sign == debil_sign:
            return 0.0

    return 0.5  # neutral
