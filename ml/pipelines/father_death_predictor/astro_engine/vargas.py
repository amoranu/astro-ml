"""Divisional chart computations (D9 Navamsha, D12 Dwadashamsha)."""


def navamsha_sign(longitude: float) -> int:
    """D9: Divide each sign into 9 parts of 3°20'. Returns D9 sign (0-11)."""
    total_navamsha = int(longitude / (30.0 / 9.0))
    return total_navamsha % 12


def dwadashamsha_sign(longitude: float) -> int:
    """D12: Divide each sign into 12 parts of 2°30'. Returns D12 sign (0-11)."""
    sign = int(longitude / 30) % 12
    amsha = int((longitude % 30) / 2.5)
    return (sign + amsha) % 12
