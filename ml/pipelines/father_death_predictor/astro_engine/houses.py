"""House cusp and sign-lord calculations (Equal House, Parashari)."""


def get_house_position(planet_long: float, asc_long: float) -> float:
    """Continuous house position [0, 12). 0.0 = exactly on ascendant."""
    return ((planet_long - asc_long) % 360) / 30.0


def get_house_number(planet_long: float, asc_long: float) -> int:
    """House number 1-12."""
    return int(get_house_position(planet_long, asc_long)) + 1


def get_sign(longitude: float) -> int:
    """Sign number 0-11 (0=Aries, 1=Taurus, ...)."""
    return int(longitude / 30) % 12


# Standard Parashari sign lordship
_SIGN_LORDS = [
    'Mars', 'Venus', 'Mercury', 'Moon', 'Sun', 'Mercury',
    'Venus', 'Mars', 'Jupiter', 'Saturn', 'Saturn', 'Jupiter'
]


def get_sign_lord(sign_num: int) -> str:
    """Lord of sign (0-11)."""
    return _SIGN_LORDS[sign_num % 12]


def get_house_lord(house_num: int, asc_long: float) -> str:
    """Planet that lords the given house (1-12)."""
    house_sign = (get_sign(asc_long) + house_num - 1) % 12
    return get_sign_lord(house_sign)
