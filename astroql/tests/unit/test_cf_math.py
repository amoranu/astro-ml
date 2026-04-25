"""MYCIN certainty-factor math tests."""
from __future__ import annotations

import itertools
import random

import pytest

from astroql.engine.cf_math import (
    CFInvariantError,
    EPSILON,
    aggregate,
    combine,
)


# ── Spec formulas (spot-checked against design §3.B step 6) ─────────

def test_both_positive_formula():
    # cf1 + cf2 * (1 - cf1)
    assert combine(0.5, 0.5) == pytest.approx(0.75)


def test_both_negative_formula():
    # cf1 + cf2 * (1 + cf1)
    assert combine(-0.5, -0.5) == pytest.approx(-0.75)


def test_mixed_signs_formula():
    # (cf1 + cf2) / (1 - min(|cf1|, |cf2|))
    assert combine(0.8, -0.3) == pytest.approx(0.5 / 0.7)
    assert combine(0.3, -0.8) == pytest.approx(-0.5 / 0.7)


def test_mixed_signs_zero_sum():
    assert combine(0.5, -0.5) == pytest.approx(0.0)


def test_neutral_element():
    assert combine(0.0, 0.7) == pytest.approx(0.7)
    assert combine(-0.6, 0.0) == pytest.approx(-0.6)


# ── Algebraic properties ────────────────────────────────────────────

def test_commutative_same_sign():
    for a, b in [(0.3, 0.8), (-0.2, -0.9), (0.1, 0.1)]:
        assert combine(a, b) == pytest.approx(combine(b, a))


def test_commutative_mixed_sign():
    for a, b in [(0.8, -0.3), (0.1, -0.9), (0.5, -0.5)]:
        assert combine(a, b) == pytest.approx(combine(b, a))


def test_associative_same_sign():
    a, b, c = 0.3, 0.5, 0.7
    left = combine(combine(a, b), c)
    right = combine(a, combine(b, c))
    assert left == pytest.approx(right)


def test_associative_mixed_sign():
    # Well-known MYCIN property: combine is associative under the
    # strict (-1, 1) invariant.
    a, b, c = 0.8, 0.5, -0.3
    left = combine(combine(a, b), c)
    right = combine(a, combine(b, c))
    assert left == pytest.approx(right)


def test_aggregate_order_invariant_random():
    rng = random.Random(1337)
    vals = [rng.uniform(-0.95, 0.95) for _ in range(12)]
    baseline = aggregate(vals)
    # Try 20 random shuffles — all must produce the same aggregate.
    for _ in range(20):
        shuffled = list(vals)
        rng.shuffle(shuffled)
        assert aggregate(shuffled) == pytest.approx(baseline, abs=1e-9)


def test_aggregate_empty_is_zero():
    assert aggregate([]) == 0.0


def test_aggregate_single_element():
    assert aggregate([0.42]) == pytest.approx(0.42)


# ── Invariant enforcement ──────────────────────────────────────────

def test_combine_rejects_cf1_at_plus_one():
    with pytest.raises(CFInvariantError, match="cf1="):
        combine(1.0, 0.5)


def test_combine_rejects_cf1_at_minus_one():
    with pytest.raises(CFInvariantError, match="cf1="):
        combine(-1.0, 0.5)


def test_combine_rejects_cf2_at_plus_one():
    with pytest.raises(CFInvariantError, match="cf2="):
        combine(0.5, 1.0)


def test_combine_rejects_cf2_beyond_one():
    with pytest.raises(CFInvariantError):
        combine(0.5, 1.0001)


# ── Chain stability: long aggregations stay within invariant ────────

def test_long_chain_stays_bounded():
    # Aggregating many strong same-sign CFs asymptotes to ±(1 - ε),
    # never hits ±1, so subsequent combines keep working.
    vals = [0.9] * 100
    result = aggregate(vals)
    assert result < 1.0
    assert result >= 1.0 - 10 * EPSILON  # Close to 1 but clipped
    # And we can still combine more CFs with it.
    combine(result, -0.3)  # No raise.


def test_long_mixed_chain_bounded():
    # Aggregating many alternating-sign CFs — ensure no overflow.
    vals = [0.7 if i % 2 == 0 else -0.6 for i in range(50)]
    result = aggregate(vals)
    assert -1.0 + EPSILON <= result <= 1.0 - EPSILON
