"""Natural friendship/enmity between planets per Parashari."""

NATURAL_FRIENDS = {
    'Sun':     {'Moon', 'Mars', 'Jupiter'},
    'Moon':    {'Sun', 'Mercury'},
    'Mars':    {'Sun', 'Moon', 'Jupiter'},
    'Mercury': {'Sun', 'Venus'},
    'Jupiter': {'Sun', 'Moon', 'Mars'},
    'Venus':   {'Mercury', 'Saturn'},
    'Saturn':  {'Mercury', 'Venus'},
    'Rahu':    {'Saturn', 'Mercury', 'Venus'},
    'Ketu':    {'Mars', 'Jupiter'},
}

NATURAL_ENEMIES = {
    'Sun':     {'Venus', 'Saturn'},
    'Moon':    {'Rahu', 'Ketu'},
    'Mars':    {'Mercury'},
    'Mercury': {'Moon'},
    'Jupiter': {'Mercury', 'Venus'},
    'Venus':   {'Sun', 'Moon'},
    'Saturn':  {'Sun', 'Moon', 'Mars'},
    'Rahu':    {'Sun', 'Moon', 'Mars'},
    'Ketu':    {'Moon', 'Rahu'},
}


def get_friendship(planet1: str, planet2: str) -> float:
    """Returns +1 (friends), 0 (neutral), -1 (enemies)."""
    if planet2 in NATURAL_FRIENDS.get(planet1, set()):
        return 1.0
    if planet2 in NATURAL_ENEMIES.get(planet1, set()):
        return -1.0
    return 0.0
