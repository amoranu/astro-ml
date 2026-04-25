"""Parashari aspects (drishti) — sign-based.

Classical 7th-house aspect for all planets, plus special aspects:
  - Mars:    4th and 8th from self
  - Jupiter: 5th and 9th from self
  - Saturn:  3rd and 10th from self
  - Nodes (Rahu/Ketu): 5th, 7th, 9th — follow Jupiter convention.

Signs are 1..12 with Aries=1. The k-th sign from sign s is
  ((s - 1 + (k - 1)) % 12) + 1.
"""
from __future__ import annotations

from typing import Dict, List


_ASPECT_OFFSETS: Dict[str, tuple] = {
    # 7th = offset 6 (1-indexed "nth from": 7th from sign s means
    # (s - 1 + 6) mod 12 + 1 = s + 6 wrapping within 1..12).
    "Sun": (6,),
    "Moon": (6,),
    "Mercury": (6,),
    "Venus": (6,),
    "Mars": (3, 6, 7),       # 4th, 7th, 8th
    "Jupiter": (4, 6, 8),    # 5th, 7th, 9th
    "Saturn": (2, 6, 9),     # 3rd, 7th, 10th
    "Rahu": (4, 6, 8),       # Jupiter convention
    "Ketu": (4, 6, 8),       # Jupiter convention
}


def aspected_signs(planet: str, sign_num: int) -> List[int]:
    """Return the 1-indexed sign numbers that `planet` aspects from
    `sign_num` (1=Aries .. 12=Pisces).
    """
    if not (1 <= sign_num <= 12):
        raise ValueError(f"sign_num must be 1..12, got {sign_num}")
    offsets = _ASPECT_OFFSETS.get(planet)
    if offsets is None:
        return []
    return [((sign_num - 1 + off) % 12) + 1 for off in offsets]


def aspects_receiving(
    target_planet: str,
    target_sign_num: int,
    all_planet_signs: Dict[str, int],
) -> List[str]:
    """Return the list of *other* planets whose aspect reaches
    target_planet's sign.

    Target itself is excluded from its own aspects.
    """
    receivers: List[str] = []
    for other, other_sign in all_planet_signs.items():
        if other == target_planet:
            continue
        if target_sign_num in aspected_signs(other, other_sign):
            receivers.append(other)
    return receivers
