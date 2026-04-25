"""MYCIN certainty-factor arithmetic (NEUROSYMBOLIC_ENGINE_DESIGN.md §3.B).

Strict-open invariant: all inputs to `combine`/`aggregate` must satisfy
-1 < cf < 1 (not inclusive). The engine enforces this by:
  * requiring non-veto rules to declare `base_cf` in that open interval
    (loader-enforced)
  * shadbala modulation μ ∈ [0, 1] keeps μ * base_cf in the same interval
  * vetoes are handled OUTSIDE CF math (short-circuit to ±1.0)

Post-combine clip to `[-1 + EPSILON, 1 - EPSILON]` handles floating-point
drift so the invariant holds through chains of many combinations.

Associativity + commutativity hold under the invariant, so aggregation
order does not affect the result.
"""
from __future__ import annotations

from typing import Iterable


EPSILON = 1e-9


class CFInvariantError(ValueError):
    """Raised when a CF input violates the strict-open (-1, 1) invariant."""


def _check(cf: float, name: str = "cf") -> float:
    if not (-1.0 < cf < 1.0):
        raise CFInvariantError(
            f"{name}={cf} violates strict-open (-1, 1) invariant. "
            f"Vetoes (|cf|=1) must be handled by short-circuit, not "
            f"passed into combine/aggregate."
        )
    return cf


def _clip(cf: float) -> float:
    """Clip to [-1 + EPSILON, 1 - EPSILON] to preserve the invariant
    through floating-point drift.
    """
    if cf >= 1.0 - EPSILON:
        return 1.0 - EPSILON
    if cf <= -1.0 + EPSILON:
        return -1.0 + EPSILON
    return cf


def combine(cf1: float, cf2: float) -> float:
    """MYCIN combination of two certainty factors.

    Both positive: cf1 + cf2 * (1 - cf1)
    Both negative: cf1 + cf2 * (1 + cf1)
    Mixed signs:   (cf1 + cf2) / (1 - min(|cf1|, |cf2|))

    Inputs must lie in the strict-open interval (-1, 1) (see module
    docstring). Returns a value in the same interval.
    """
    _check(cf1, "cf1")
    _check(cf2, "cf2")
    if cf1 == 0.0:
        return _clip(cf2)
    if cf2 == 0.0:
        return _clip(cf1)
    if cf1 > 0 and cf2 > 0:
        out = cf1 + cf2 * (1.0 - cf1)
    elif cf1 < 0 and cf2 < 0:
        out = cf1 + cf2 * (1.0 + cf1)
    else:
        # Mixed signs. Denominator is strictly positive under the
        # invariant (min(|cf1|, |cf2|) < 1), so no divide-by-zero.
        denom = 1.0 - min(abs(cf1), abs(cf2))
        out = (cf1 + cf2) / denom
    return _clip(out)


def aggregate(cfs: Iterable[float]) -> float:
    """Reduce an iterable of certainty factors with `combine`.

    Returns 0.0 for an empty input (neutral element). Under the strict
    invariant, `combine` is associative and commutative, so order does
    not matter.
    """
    result = 0.0
    for cf in cfs:
        result = combine(result, cf)
    return result
