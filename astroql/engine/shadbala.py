"""Shadbala normalization (NEUROSYMBOLIC_ENGINE_DESIGN.md §3.B step 5).

Classical Parashara per-planet required-strength thresholds (virupas):

    Sun      390
    Moon     360
    Mars     300
    Mercury  420
    Jupiter  390
    Venus    330
    Saturn   300

Formula for the classical seven: μ = min(virupas / required, 1.0)

Rahu/Ketu: no classical shadbala. Parashari dispositor rule —
  μ_node = μ_of(sign_lord(node's sign)).
"""
from __future__ import annotations

from typing import Dict, Optional


REQUIRED_VIRUPAS: Dict[str, float] = {
    "Sun": 390.0,
    "Moon": 360.0,
    "Mars": 300.0,
    "Mercury": 420.0,
    "Jupiter": 390.0,
    "Venus": 330.0,
    "Saturn": 300.0,
}


# Sign → sign-lord (classical Parashari dispositorship, sidereal).
_SIGN_LORD: Dict[str, str] = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}


_NODES = ("Rahu", "Ketu")

# The 9-planet classical Parashari set. Outer planets (Uranus/Neptune/
# Pluto) that astro-prod may also compute are NOT part of Parashari and
# must be filtered out before normalization.
PARASHARI_PLANETS = frozenset(REQUIRED_VIRUPAS.keys()) | frozenset(_NODES)


def classical_mu(planet: str, virupas: float) -> float:
    """Normalize classical-seven shadbala to μ ∈ [0, 1].

    Raises ValueError if called for a node (Rahu/Ketu) — use
    `node_mu_via_dispositor` for those.
    """
    if planet in _NODES:
        raise ValueError(
            f"{planet} has no classical shadbala; use "
            f"node_mu_via_dispositor instead."
        )
    required = REQUIRED_VIRUPAS.get(planet)
    if required is None:
        raise ValueError(f"Unknown planet {planet!r} for shadbala")
    if virupas < 0:
        virupas = 0.0
    return min(virupas / required, 1.0)


def sign_lord(sign: str) -> str:
    lord = _SIGN_LORD.get(sign)
    if lord is None:
        raise ValueError(f"Unknown sidereal sign {sign!r}")
    return lord


def node_mu_via_dispositor(
    node: str,
    node_sign: str,
    classical_mu_by_planet: Dict[str, float],
) -> float:
    """μ for Rahu/Ketu inherited from their dispositor (sign lord).

    Defensive fallback: if dispositor is itself a node (should not occur
    classically — nodes don't own signs), return 0.5 as neutral default.
    """
    if node not in _NODES:
        raise ValueError(
            f"{node!r} is not a node; use classical_mu instead."
        )
    lord = sign_lord(node_sign)
    if lord in _NODES:
        return 0.5
    mu = classical_mu_by_planet.get(lord)
    if mu is None:
        raise ValueError(
            f"Dispositor {lord} of {node} not in provided mu table"
        )
    return mu


def normalize_all(
    shadbala_by_planet: Dict[str, float],
    sign_by_planet: Dict[str, str],
) -> Dict[str, float]:
    """Normalize a full natal shadbala table to μ coefficients.

    shadbala_by_planet: {planet: raw_virupas}. Rahu/Ketu may be absent
      or zero; they're resolved via dispositor.
    sign_by_planet: {planet: sidereal sign name}. Required for nodes.

    Returns: {planet: μ in [0, 1]}. Includes all planets in input plus
      nodes if they're in sign_by_planet and a dispositor is resolvable.
    """
    classical: Dict[str, float] = {}
    for planet, virupas in shadbala_by_planet.items():
        if planet in _NODES:
            continue
        classical[planet] = classical_mu(planet, virupas)
    # Resolve nodes via their dispositor.
    out: Dict[str, float] = dict(classical)
    for node in _NODES:
        if node in sign_by_planet:
            out[node] = node_mu_via_dispositor(
                node, sign_by_planet[node], classical,
            )
    return out
