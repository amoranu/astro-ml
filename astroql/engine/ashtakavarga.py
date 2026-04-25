"""Bhinnashtakavarga (BAV) and Sarvashtakavarga (SAV) computation.

Per BPHS Ch. 66. The BAV tables encode, for each of the 8 contributors
(Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Asc), which house
offsets that contributor donates a bindu to a given subject planet's
sign placement. SAV = sum of all 7 planet BAVs across each sign.

The neuro-symbolic engine uses BAV/SAV to **gate transit rules**: a
malefic transit through a sign with low BAV bindus has classical
death-timing force; through a sign with high BAV bindus it does not.

Public API:
    compute_bav(planet, sign_by_planet, lagna_sign) -> List[int]
    compute_sav(sign_by_planet, lagna_sign) -> List[int]
    bav_grid(sign_by_planet, lagna_sign) -> Dict[str, Dict[str, int]]

`bav_grid` returns the structure used in `EpochState.ashtakavarga` —
keyed by planet then sign name (e.g., grid["Saturn"]["Aries"] = 4).
This shape lets the JSON DSL evaluator reach BAV scores via dot-paths
("ashtakavarga.Saturn.Aries"), unblocking transit-gating rules:

    {"all": [
        {"path": "planets.Saturn.transit_sign", "op": "==", "value": "Aries"},
        {"path": "ashtakavarga.Saturn.Aries", "op": "<", "value": 4},
    ]}
"""
from __future__ import annotations

from typing import Dict, List


# BPHS Ch. 66 BAV tables. Each entry maps a contributor → list of
# 1-indexed house offsets where that contributor donates a bindu to
# the SUBJECT planet's sign.
SUN_BAV = {
    "Sun":     [1, 2, 4, 7, 8, 9, 10, 11],
    "Moon":    [3, 6, 10, 11],
    "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
    "Mercury": [3, 5, 6, 9, 10, 11, 12],
    "Jupiter": [5, 6, 9, 11],
    "Venus":   [6, 7, 12],
    "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    "Asc":     [3, 4, 6, 10, 11, 12],
}

MOON_BAV = {
    "Sun":     [3, 6, 7, 8, 10, 11],
    "Moon":    [1, 3, 6, 7, 10, 11],
    "Mars":    [2, 3, 5, 6, 9, 10, 11],
    "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
    "Jupiter": [1, 4, 7, 8, 10, 11, 12],
    "Venus":   [3, 4, 5, 7, 9, 10, 11],
    "Saturn":  [3, 5, 6, 11],
    "Asc":     [3, 6, 10, 11],
}

MARS_BAV = {
    "Sun":     [3, 5, 6, 10, 11],
    "Moon":    [3, 6, 11],
    "Mars":    [1, 2, 4, 7, 8, 10, 11],
    "Mercury": [3, 5, 6, 11],
    "Jupiter": [6, 10, 11, 12],
    "Venus":   [6, 8, 11, 12],
    "Saturn":  [1, 4, 7, 8, 9, 10, 11],
    "Asc":     [1, 3, 6, 10, 11],
}

MERCURY_BAV = {
    "Sun":     [5, 6, 9, 11, 12],
    "Moon":    [2, 4, 6, 8, 10, 11],
    "Mars":    [1, 2, 4, 7, 8, 9, 10, 11],
    "Mercury": [1, 3, 5, 6, 9, 10, 11, 12],
    "Jupiter": [6, 8, 11, 12],
    "Venus":   [1, 2, 3, 4, 5, 8, 9, 11],
    "Saturn":  [1, 2, 4, 7, 8, 9, 10, 11],
    "Asc":     [1, 2, 4, 6, 8, 10, 11],
}

JUPITER_BAV = {
    "Sun":     [1, 2, 3, 4, 7, 8, 9, 10, 11],
    "Moon":    [2, 5, 7, 9, 11],
    "Mars":    [1, 2, 4, 7, 8, 10, 11],
    "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
    "Jupiter": [1, 2, 3, 4, 7, 8, 10, 11],
    "Venus":   [2, 5, 6, 9, 10, 11],
    "Saturn":  [3, 5, 6, 12],
    "Asc":     [1, 2, 4, 5, 6, 7, 9, 10, 11],
}

VENUS_BAV = {
    "Sun":     [8, 11, 12],
    "Moon":    [1, 2, 3, 4, 5, 8, 9, 11, 12],
    "Mars":    [3, 5, 6, 9, 11, 12],
    "Mercury": [3, 5, 6, 9, 11],
    "Jupiter": [5, 8, 9, 10, 11],
    "Venus":   [1, 2, 3, 4, 5, 8, 9, 10, 11],
    "Saturn":  [3, 4, 5, 8, 9, 10, 11],
    "Asc":     [1, 2, 3, 4, 5, 8, 9, 11],
}

SATURN_BAV = {
    "Sun":     [1, 2, 4, 7, 8, 10, 11],
    "Moon":    [3, 6, 11],
    "Mars":    [3, 5, 6, 10, 11, 12],
    "Mercury": [6, 8, 9, 10, 11, 12],
    "Jupiter": [5, 6, 11, 12],
    "Venus":   [6, 11, 12],
    "Saturn":  [3, 5, 6, 11],
    "Asc":     [1, 3, 4, 6, 10, 11],
}

BAV_TABLES = {
    "Sun": SUN_BAV,
    "Moon": MOON_BAV,
    "Mars": MARS_BAV,
    "Mercury": MERCURY_BAV,
    "Jupiter": JUPITER_BAV,
    "Venus": VENUS_BAV,
    "Saturn": SATURN_BAV,
}

_SIGN_ORDER = (
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
)


def _sign_index(sign: str) -> int:
    """Return 0-indexed sign number (Aries=0, ..., Pisces=11)."""
    try:
        return _SIGN_ORDER.index(sign)
    except ValueError:
        raise ValueError(
            f"unknown sign {sign!r}; expected one of {_SIGN_ORDER}"
        )


def compute_bav(
    subject_planet: str,
    sign_by_planet: Dict[str, str],
    lagna_sign: str,
) -> List[int]:
    """Compute the Bhinnashtakavarga (12-element bindu list) for one
    subject planet in a chart.

    Args:
        subject_planet: the planet whose BAV is being computed
            (must be in BAV_TABLES — Sun..Saturn).
        sign_by_planet: mapping of contributor planet name → sign name.
            Must include all of {Sun, Moon, Mars, Mercury, Jupiter,
            Venus, Saturn}.
        lagna_sign: ascendant sign name.

    Returns:
        List[int] of length 12, where index 0=Aries..11=Pisces is the
        bindu count contributed across all 8 reference points.
    """
    if subject_planet not in BAV_TABLES:
        raise ValueError(
            f"compute_bav: subject_planet must be one of "
            f"{list(BAV_TABLES.keys())}, got {subject_planet!r}"
        )
    table = BAV_TABLES[subject_planet]
    bindus = [0] * 12
    refs = {}
    for p in ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"):
        if p in sign_by_planet:
            refs[p] = _sign_index(sign_by_planet[p])
    refs["Asc"] = _sign_index(lagna_sign)
    for ref_name, ref_sign in refs.items():
        if ref_name not in table:
            continue
        for house_offset in table[ref_name]:
            target_sign = (ref_sign + house_offset - 1) % 12
            bindus[target_sign] += 1
    return bindus


def compute_sav(
    sign_by_planet: Dict[str, str], lagna_sign: str,
) -> List[int]:
    """Sarvashtakavarga (SAV) = sum of all 7 planet BAVs per sign.

    Returns a 12-element list; index k is the total bindus for sign k.
    """
    sav = [0] * 12
    for planet in BAV_TABLES:
        bav = compute_bav(planet, sign_by_planet, lagna_sign)
        for i in range(12):
            sav[i] += bav[i]
    return sav


def bav_grid(
    sign_by_planet: Dict[str, str], lagna_sign: str,
) -> Dict[str, Dict[str, int]]:
    """Materialize the BAV grid as `Dict[planet, Dict[sign, bindus]]`.

    This is the shape consumed by `EpochState.ashtakavarga` and
    reachable via DSL paths like "ashtakavarga.Saturn.Aries".
    The outer key "SAV" carries the Sarvashtakavarga totals.
    """
    out: Dict[str, Dict[str, int]] = {}
    for planet in BAV_TABLES:
        bavs = compute_bav(planet, sign_by_planet, lagna_sign)
        out[planet] = {
            _SIGN_ORDER[i]: bavs[i] for i in range(12)
        }
    sav = compute_sav(sign_by_planet, lagna_sign)
    out["SAV"] = {_SIGN_ORDER[i]: sav[i] for i in range(12)}
    return out
