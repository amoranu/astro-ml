"""Epoch state emitter + shadbala normalization + aspects tests."""
from __future__ import annotations

import datetime as _dt
from datetime import date

import pytest
from zoneinfo import ZoneInfo

from astroql.engine import aspects as _aspects
from astroql.engine import shadbala as _sb
from astroql.engine.epoch_emitter import (
    EpochEmissionError,
    emit_epochs,
)
from astroql.schemas.birth import BirthDetails
from astroql.schemas.epoch_state import EpochState


# ── Shadbala normalization ──────────────────────────────────────────

def test_classical_mu_at_required_threshold_is_one():
    # Sun required = 390 virupas
    assert _sb.classical_mu("Sun", 390.0) == pytest.approx(1.0)


def test_classical_mu_half_of_required_is_half():
    assert _sb.classical_mu("Mars", 150.0) == pytest.approx(0.5)


def test_classical_mu_caps_at_one_for_overshoot():
    assert _sb.classical_mu("Saturn", 1000.0) == 1.0


def test_classical_mu_negative_treated_as_zero():
    assert _sb.classical_mu("Moon", -50.0) == 0.0


def test_classical_mu_for_node_raises():
    with pytest.raises(ValueError, match="classical shadbala"):
        _sb.classical_mu("Rahu", 100.0)


def test_node_mu_inherits_from_dispositor():
    # Rahu in Leo → dispositor Sun
    classical = {"Sun": 0.8, "Moon": 0.5, "Mars": 0.3}
    mu = _sb.node_mu_via_dispositor("Rahu", "Leo", classical)
    assert mu == 0.8


def test_node_mu_unknown_sign_raises():
    with pytest.raises(ValueError):
        _sb.node_mu_via_dispositor("Ketu", "NotARealSign", {})


def test_normalize_all_full_table():
    raw = {
        "Sun": 390.0, "Moon": 180.0, "Mars": 300.0,
        "Mercury": 210.0, "Jupiter": 390.0, "Venus": 165.0,
        "Saturn": 150.0, "Rahu": 0.0, "Ketu": 0.0,
    }
    signs = {
        "Sun": "Leo", "Moon": "Cancer", "Mars": "Aries",
        "Mercury": "Gemini", "Jupiter": "Sagittarius",
        "Venus": "Taurus", "Saturn": "Capricorn",
        "Rahu": "Gemini",     # dispositor Mercury
        "Ketu": "Sagittarius",  # dispositor Jupiter
    }
    mu = _sb.normalize_all(raw, signs)
    assert mu["Sun"] == pytest.approx(1.0)
    assert mu["Moon"] == pytest.approx(0.5)
    assert mu["Mercury"] == pytest.approx(0.5)
    # Rahu inherits Mercury's μ (0.5)
    assert mu["Rahu"] == pytest.approx(0.5)
    # Ketu inherits Jupiter's μ (1.0)
    assert mu["Ketu"] == pytest.approx(1.0)


# ── Parashari aspects ───────────────────────────────────────────────

def test_sun_aspects_only_7th():
    # Sun in Aries (1) → 7th from Aries = Libra (7)
    assert _aspects.aspected_signs("Sun", 1) == [7]


def test_mars_aspects_4_7_8():
    # Mars in Aries (1) → 4th=Cancer(4), 7th=Libra(7), 8th=Scorpio(8)
    assert sorted(_aspects.aspected_signs("Mars", 1)) == [4, 7, 8]


def test_jupiter_aspects_5_7_9():
    # Jupiter in Capricorn (10) → 5th=Taurus(2), 7th=Cancer(4),
    # 9th=Virgo(6)
    assert sorted(_aspects.aspected_signs("Jupiter", 10)) == [2, 4, 6]


def test_saturn_aspects_3_7_10():
    # Saturn in Libra (7) → 3rd=Sagittarius(9), 7th=Aries(1),
    # 10th=Cancer(4)
    assert sorted(_aspects.aspected_signs("Saturn", 7)) == [1, 4, 9]


def test_aspect_wraps_around_zodiac():
    # Jupiter in Pisces (12) → 5th=Cancer(4), 7th=Virgo(6), 9th=Scorpio(8)
    assert sorted(_aspects.aspected_signs("Jupiter", 12)) == [4, 6, 8]


def test_aspects_receiving_excludes_self():
    # If Sun is in Aries and Mars is in Aries, and we ask what aspects
    # Sun's sign (Aries), Mars itself is skipped if Mars is the target,
    # but Mars CAN aspect Sun from the same sign via its 7th aspect...
    # actually Mars in Aries aspects 4th/7th/8th = Cancer/Libra/Scorpio,
    # not Aries. So no aspects on Sun from Mars when they're co-located.
    signs = {"Sun": 1, "Mars": 1}  # both in Aries
    assert _aspects.aspects_receiving("Sun", 1, signs) == []


def test_aspects_receiving_basic_case():
    # Saturn in Sagittarius (9); Sun in Gemini (3). Saturn's 7th aspect
    # from Sag is Gemini. So Sun receives Saturn.
    signs = {"Sun": 3, "Saturn": 9}
    assert _aspects.aspects_receiving("Sun", 3, signs) == ["Saturn"]


def test_invalid_sign_num_raises():
    with pytest.raises(ValueError):
        _aspects.aspected_signs("Sun", 13)
    with pytest.raises(ValueError):
        _aspects.aspected_signs("Sun", 0)


# ── Epoch emitter input contract ────────────────────────────────────

def _birth() -> BirthDetails:
    # Arbitrary real birth — Delhi.
    return BirthDetails(
        date=date(1980, 6, 15),
        time="12:30:00",
        tz="Asia/Kolkata",
        lat=28.6139,
        lon=77.2090,
    )


def test_missing_query_start_rejected():
    with pytest.raises(EpochEmissionError, match="required"):
        emit_epochs(_birth(), None, _dt.datetime(2010, 1, 1))  # type: ignore[arg-type]


def test_missing_query_end_rejected():
    with pytest.raises(EpochEmissionError, match="required"):
        emit_epochs(_birth(), _dt.datetime(2010, 1, 1), None)  # type: ignore[arg-type]


def test_inverted_window_rejected():
    tz = ZoneInfo("Asia/Kolkata")
    a = _dt.datetime(2020, 1, 1, tzinfo=tz)
    b = _dt.datetime(2019, 1, 1, tzinfo=tz)
    with pytest.raises(EpochEmissionError, match="must be strictly after"):
        emit_epochs(_birth(), a, b)


def test_zero_window_rejected():
    tz = ZoneInfo("Asia/Kolkata")
    t = _dt.datetime(2020, 1, 1, tzinfo=tz)
    with pytest.raises(EpochEmissionError, match="must be strictly after"):
        emit_epochs(_birth(), t, t)


def test_window_over_max_rejected():
    tz = ZoneInfo("Asia/Kolkata")
    a = _dt.datetime(2010, 1, 1, tzinfo=tz)
    b = _dt.datetime(2025, 1, 1, tzinfo=tz)  # 15 years
    with pytest.raises(EpochEmissionError, match="max_window_years"):
        emit_epochs(_birth(), a, b, max_window_years=10.0)


def test_window_over_default_allowed_with_opt_in():
    tz = ZoneInfo("Asia/Kolkata")
    a = _dt.datetime(2010, 1, 1, tzinfo=tz)
    b = _dt.datetime(2021, 1, 1, tzinfo=tz)  # 11 years
    # No emission check — just ensure it does not reject on contract.
    # Actual emission hits astro-prod; tested separately.
    epochs = emit_epochs(_birth(), a, b, max_window_years=12.0)
    assert len(epochs) > 0


# ── Smoke test: real emission against astro-prod ────────────────────

def test_emit_real_small_window_smoke():
    tz = ZoneInfo("Asia/Kolkata")
    a = _dt.datetime(2020, 1, 1, tzinfo=tz)
    b = _dt.datetime(2020, 2, 1, tzinfo=tz)
    epochs = emit_epochs(_birth(), a, b)
    assert len(epochs) > 0
    first = epochs[0]
    assert isinstance(first, EpochState)
    # Every epoch lies within the query window (after clipping).
    for e in epochs:
        assert e.start_time >= a
        assert e.end_time <= b
        assert e.end_time > e.start_time
    # Dasha stack populated at all four levels.
    assert first.dashas.maha
    assert first.dashas.antar
    assert first.dashas.pratyantar
    assert first.dashas.sookshma
    # All 9 planets present (including Rahu/Ketu).
    expected = {"Sun", "Moon", "Mars", "Mercury", "Jupiter",
                "Venus", "Saturn", "Rahu", "Ketu"}
    assert set(first.planets.keys()) == expected
    # Shadbala μ in [0, 1].
    for p, ps in first.planets.items():
        assert 0.0 <= ps.shadbala_coefficient <= 1.0, (
            f"{p} μ={ps.shadbala_coefficient}"
        )
        assert 1 <= ps.transit_house <= 12
        assert 1 <= ps.natal_house <= 12


def test_emit_real_json_roundtrip():
    tz = ZoneInfo("Asia/Kolkata")
    a = _dt.datetime(2020, 1, 1, tzinfo=tz)
    b = _dt.datetime(2020, 1, 10, tzinfo=tz)
    epochs = emit_epochs(_birth(), a, b)
    assert len(epochs) > 0
    serialized = [e.to_dict() for e in epochs]
    restored = [EpochState.from_dict(d) for d in serialized]
    assert len(restored) == len(epochs)
    # Field-level equality on a spot-checked epoch.
    e0 = epochs[0]
    r0 = restored[0]
    assert r0.epoch_id == e0.epoch_id
    assert r0.start_time == e0.start_time
    assert r0.end_time == e0.end_time
    assert r0.dashas.sookshma == e0.dashas.sookshma
    p_name = next(iter(e0.planets.keys()))
    assert (
        r0.planets[p_name].shadbala_coefficient
        == e0.planets[p_name].shadbala_coefficient
    )
