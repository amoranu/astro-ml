"""Unit tests for BAV/SAV computation + DSL gating round-trip."""
from __future__ import annotations

import pytest

from astroql.engine.ashtakavarga import (
    BAV_TABLES, bav_grid, compute_bav, compute_sav,
)
from astroql.engine.dsl_evaluator import evaluate
from astroql.schemas.epoch_state import (
    DashaStack, EpochState, PlanetEpochState,
)
from datetime import datetime, timezone


# ── Sample chart positions (a synthetic, deterministic chart) ────

@pytest.fixture
def sample_signs():
    """Sample chart for a Cancer-lagna with planets in distinct signs.
    Used for shape + sanity tests, not for canonical BPHS values.
    """
    return {
        "Sun":     "Leo",
        "Moon":    "Pisces",
        "Mars":    "Capricorn",
        "Mercury": "Virgo",
        "Jupiter": "Sagittarius",
        "Venus":   "Libra",
        "Saturn":  "Aquarius",
    }


@pytest.fixture
def sample_lagna() -> str:
    return "Cancer"


# ── compute_bav / compute_sav ────────────────────────────────────

def test_compute_bav_returns_12_ints(sample_signs, sample_lagna):
    bav = compute_bav("Saturn", sample_signs, sample_lagna)
    assert isinstance(bav, list)
    assert len(bav) == 12
    assert all(isinstance(v, int) and v >= 0 for v in bav)


def test_compute_bav_total_matches_table_count(
    sample_signs, sample_lagna,
):
    """Each contributor in SATURN_BAV donates len(offset_list) bindus
    across the 12 signs. Total Saturn-BAV bindus = sum of all
    contributor list lengths.
    """
    bav = compute_bav("Saturn", sample_signs, sample_lagna)
    expected_total = sum(
        len(offsets) for offsets in BAV_TABLES["Saturn"].values()
    )
    assert sum(bav) == expected_total


def test_compute_bav_unknown_planet_raises(sample_signs, sample_lagna):
    with pytest.raises(ValueError, match="subject_planet must be"):
        compute_bav("Pluto", sample_signs, sample_lagna)


def test_compute_sav_sums_seven_planet_bavs(sample_signs, sample_lagna):
    sav = compute_sav(sample_signs, sample_lagna)
    assert len(sav) == 12
    expected = [0] * 12
    for planet in BAV_TABLES:
        bav = compute_bav(planet, sample_signs, sample_lagna)
        for i in range(12):
            expected[i] += bav[i]
    assert sav == expected


def test_sav_total_is_337_per_classical_total(
    sample_signs, sample_lagna,
):
    """Classical SAV total = 337 across all 12 signs (sum of all
    contributor offsets across all 7 subject planets).
    """
    sav = compute_sav(sample_signs, sample_lagna)
    assert sum(sav) == 337


# ── bav_grid (DSL-friendly shape) ────────────────────────────────

def test_bav_grid_shape(sample_signs, sample_lagna):
    grid = bav_grid(sample_signs, sample_lagna)
    # 7 planets + 1 SAV row.
    assert set(grid.keys()) == {
        "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn",
        "SAV",
    }
    # Each row carries all 12 sign keys.
    for row in grid.values():
        assert set(row.keys()) == {
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn",
            "Aquarius", "Pisces",
        }


def test_bav_grid_values_match_compute_bav(sample_signs, sample_lagna):
    grid = bav_grid(sample_signs, sample_lagna)
    saturn_bav = compute_bav("Saturn", sample_signs, sample_lagna)
    sign_order = ("Aries", "Taurus", "Gemini", "Cancer", "Leo",
                  "Virgo", "Libra", "Scorpio", "Sagittarius",
                  "Capricorn", "Aquarius", "Pisces")
    for i, sign in enumerate(sign_order):
        assert grid["Saturn"][sign] == saturn_bav[i]


# ── DSL transit-gating round-trip ────────────────────────────────

def test_dsl_can_gate_a_rule_via_bav(sample_signs, sample_lagna):
    """The transit-gating pattern: combine a transit-position clause
    with a BAV-bindu clause via `all`. This is the LLM-autonomous
    use case the reviewer flagged.
    """
    grid = bav_grid(sample_signs, sample_lagna)
    saturn_aries = grid["Saturn"]["Aries"]

    state = EpochState(
        epoch_id="test",
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2020, 1, 8, tzinfo=timezone.utc),
        dashas=DashaStack(maha="Saturn", antar="Saturn",
                          pratyantar="Saturn", sookshma="Saturn"),
        planets={
            "Saturn": PlanetEpochState(
                transit_sign="Aries", transit_house=10,
                natal_house=8, shadbala_coefficient=0.5,
                is_retrograde=False, natal_sign="Aquarius",
            ),
        },
        natal_lagna_sign="Cancer",
        ashtakavarga=grid,
    )

    # "Saturn transits Aries AND Aries has fewer than (saturn_aries+1)
    # Saturn-BAV bindus" → fires (the actual count is < count+1).
    cond = {"all": [
        {"path": "planets.Saturn.transit_sign", "op": "==",
         "value": "Aries"},
        {"path": "ashtakavarga.Saturn.Aries", "op": "<",
         "value": saturn_aries + 1},
    ]}
    assert evaluate(cond, state) is True

    # Same rule but require strictly fewer than the actual count:
    # the BAV gate suppresses the rule.
    blocked = {"all": [
        {"path": "planets.Saturn.transit_sign", "op": "==",
         "value": "Aries"},
        {"path": "ashtakavarga.Saturn.Aries", "op": "<",
         "value": saturn_aries},
    ]}
    assert evaluate(blocked, state) is False


def test_sav_path_resolves(sample_signs, sample_lagna):
    """SAV totals are reachable via "ashtakavarga.SAV.<sign>"."""
    grid = bav_grid(sample_signs, sample_lagna)
    state = EpochState(
        epoch_id="test",
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2020, 1, 8, tzinfo=timezone.utc),
        dashas=DashaStack(maha="X", antar="X", pratyantar="X",
                          sookshma="X"),
        planets={},
        ashtakavarga=grid,
    )
    cond = {"path": "ashtakavarga.SAV.Aries", "op": "==",
            "value": grid["SAV"]["Aries"]}
    assert evaluate(cond, state) is True
