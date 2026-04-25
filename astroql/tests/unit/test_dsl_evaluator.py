"""Unit tests for the DSL evaluator."""
from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Dict, List

from astroql.engine.dsl_evaluator import (
    DSLEvalError,
    evaluate,
    evaluate_modifier_indices,
    resolve_path,
)


# ── Fixtures (mini EpochState-shaped object) ──────────────────────

@dataclass
class _Planet:
    transit_house: int
    transit_sign: str
    is_retrograde: bool
    natal_house: int
    aspects_receiving: List[str] = field(default_factory=list)


@dataclass
class _Dashas:
    maha: str
    antar: str
    pratyantar: str
    sookshma: str


@dataclass
class _State:
    dashas: _Dashas
    planets: Dict[str, _Planet]
    derived_lords: Dict[str, str]
    natal_lagna_sign: str = "Scorpio"


@pytest.fixture
def state() -> _State:
    return _State(
        dashas=_Dashas(maha="Saturn", antar="Moon",
                       pratyantar="Saturn", sookshma="Rahu"),
        planets={
            "Mars": _Planet(transit_house=11, transit_sign="Virgo",
                            is_retrograde=True, natal_house=9),
            "Saturn": _Planet(transit_house=6, transit_sign="Aries",
                              is_retrograde=False, natal_house=2,
                              aspects_receiving=["Rahu"]),
            "Jupiter": _Planet(transit_house=4, transit_sign="Pisces",
                               is_retrograde=False, natal_house=11),
        },
        derived_lords={"ninth_lord": "Moon", "father_8L": "Saturn"},
    )


# ── resolve_path ──────────────────────────────────────────────────

def test_path_attribute_access(state):
    assert resolve_path(state, "dashas.maha") == "Saturn"
    assert resolve_path(state, "natal_lagna_sign") == "Scorpio"


def test_path_mapping_access(state):
    assert resolve_path(state, "planets.Mars.transit_house") == 11
    assert resolve_path(state, "planets.Saturn.is_retrograde") is False
    assert resolve_path(state, "derived_lords.ninth_lord") == "Moon"


def test_path_mixed_attr_then_dict(state):
    assert resolve_path(state, "planets.Jupiter.natal_house") == 11


def test_path_missing_segment_raises(state):
    with pytest.raises(DSLEvalError, match="not in mapping"):
        resolve_path(state, "planets.Pluto.transit_house")
    with pytest.raises(DSLEvalError, match="unreachable"):
        resolve_path(state, "dashas.unknown_field")


def test_path_empty_raises(state):
    with pytest.raises(DSLEvalError):
        resolve_path(state, "")


# ── evaluate: binary ops ──────────────────────────────────────────

def test_eq_op(state):
    assert evaluate({"path": "dashas.maha", "op": "==",
                     "value": "Saturn"}, state) is True
    assert evaluate({"path": "dashas.maha", "op": "==",
                     "value": "Mercury"}, state) is False


def test_ne_op(state):
    assert evaluate({"path": "dashas.antar", "op": "!=",
                     "value": "Saturn"}, state) is True


def test_lt_le_gt_ge(state):
    assert evaluate({"path": "planets.Mars.transit_house", "op": "<",
                     "value": 12}, state) is True
    assert evaluate({"path": "planets.Mars.transit_house", "op": "<=",
                     "value": 11}, state) is True
    assert evaluate({"path": "planets.Mars.transit_house", "op": ">",
                     "value": 11}, state) is False
    assert evaluate({"path": "planets.Mars.transit_house", "op": ">=",
                     "value": 11}, state) is True


def test_in_op(state):
    # Sun-in-dusthana style: transit_house in [6, 8, 12]
    assert evaluate({"path": "planets.Saturn.transit_house", "op": "in",
                     "value": [6, 8, 12]}, state) is True
    assert evaluate({"path": "planets.Mars.transit_house", "op": "in",
                     "value": [6, 8, 12]}, state) is False


def test_not_in_op(state):
    assert evaluate({"path": "dashas.maha", "op": "not_in",
                     "value": ["Mercury", "Venus"]}, state) is True


def test_contains_op(state):
    # Saturn's aspects_receiving contains "Rahu"
    assert evaluate({"path": "planets.Saturn.aspects_receiving",
                     "op": "contains", "value": "Rahu"}, state) is True
    assert evaluate({"path": "planets.Saturn.aspects_receiving",
                     "op": "contains", "value": "Mars"}, state) is False


# ── evaluate: unary ops ──────────────────────────────────────────

def test_truthy_falsy(state):
    assert evaluate({"path": "planets.Mars.is_retrograde",
                     "op": "truthy"}, state) is True
    assert evaluate({"path": "planets.Saturn.is_retrograde",
                     "op": "falsy"}, state) is True


def test_unary_op_must_not_have_value(state):
    with pytest.raises(DSLEvalError, match="must not carry"):
        evaluate({"path": "planets.Mars.is_retrograde", "op": "truthy",
                  "value": True}, state)


def test_binary_op_requires_value(state):
    with pytest.raises(DSLEvalError, match="requires a 'value'"):
        evaluate({"path": "dashas.maha", "op": "=="}, state)


# ── evaluate: combinators ─────────────────────────────────────────

def test_all_combinator(state):
    # Saturn is in dusthana 6 AND aspecting Rahu — true
    cond = {"all": [
        {"path": "planets.Saturn.transit_house", "op": "in",
         "value": [6, 8, 12]},
        {"path": "planets.Saturn.aspects_receiving",
         "op": "contains", "value": "Rahu"},
    ]}
    assert evaluate(cond, state) is True


def test_all_short_circuits_false(state):
    cond = {"all": [
        {"path": "planets.Saturn.transit_house", "op": "in",
         "value": [6, 8, 12]},
        {"path": "dashas.maha", "op": "==", "value": "Mercury"},
    ]}
    assert evaluate(cond, state) is False


def test_any_combinator(state):
    cond = {"any": [
        {"path": "dashas.maha", "op": "==", "value": "Mercury"},
        {"path": "dashas.maha", "op": "==", "value": "Saturn"},
    ]}
    assert evaluate(cond, state) is True


def test_not_combinator(state):
    cond = {"not": {"path": "planets.Mars.is_retrograde",
                    "op": "==", "value": False}}
    assert evaluate(cond, state) is True


def test_nested_combinators(state):
    # (MD = Saturn OR MD = Mars) AND Mars is retrograde
    cond = {"all": [
        {"any": [
            {"path": "dashas.maha", "op": "==", "value": "Saturn"},
            {"path": "dashas.maha", "op": "==", "value": "Mars"},
        ]},
        {"path": "planets.Mars.is_retrograde", "op": "truthy"},
    ]}
    assert evaluate(cond, state) is True


# ── Error paths ───────────────────────────────────────────────────

def test_unknown_operator_raises(state):
    with pytest.raises(DSLEvalError, match="unknown operator"):
        evaluate({"path": "dashas.maha", "op": "??",
                  "value": "Saturn"}, state)


def test_leaf_missing_keys_raises(state):
    with pytest.raises(DSLEvalError, match="must have 'path' and 'op'"):
        evaluate({"path": "dashas.maha"}, state)


def test_non_dict_condition_raises(state):
    with pytest.raises(DSLEvalError, match="must be a dict"):
        evaluate("not a dict", state)


def test_empty_condition_returns_true(state):
    # Vacuous true — used as legacy-passthrough sentinel.
    assert evaluate({}, state) is True


# ── evaluate_modifier_indices ─────────────────────────────────────

def test_modifier_indices_picks_only_fired(state):
    # idx=0 fires (Saturn in dusthana), idx=1 does not (MD != Mer),
    # idx=2 is empty (legacy passthrough — should NOT fire from DSL).
    conditions = [
        {"path": "planets.Saturn.transit_house", "op": "in",
         "value": [6, 8, 12]},
        {"path": "dashas.maha", "op": "==", "value": "Mercury"},
        {},
    ]
    assert evaluate_modifier_indices(conditions, state) == [0]


def test_modifier_indices_skips_malformed(state):
    conditions = [
        {"path": "planets.Saturn.transit_house", "op": "in",
         "value": [6]},
        {"path": "no.such.path", "op": "==", "value": 1},  # malformed
    ]
    # idx=0 fires (Saturn H6 ∈ [6]); idx=1 raises path-error and is
    # silently skipped (must not crash the prediction loop).
    assert evaluate_modifier_indices(conditions, state) == [0]


# ── derived_lords access (the LLM autonomy story) ─────────────────

def test_derived_lords_access(state):
    cond = {"path": "derived_lords.ninth_lord", "op": "==",
            "value": "Moon"}
    assert evaluate(cond, state) is True


def test_father_lord_check(state):
    # "AD-lord is father's 8L (= chart's 4L)" via DSL
    # state.dashas.antar == state.derived_lords.father_8L?
    # Saturn vs Saturn — but we can't compare via path-vs-path with the
    # current grammar. This test documents the limitation: a path-vs-
    # path comparison would require a 'path_eq' op or a setter.
    # We can still express it via two clauses + the actual concrete
    # planet name from a precomputed convenience field.
    cond = {"path": "derived_lords.father_8L", "op": "==",
            "value": "Saturn"}
    assert evaluate(cond, state) is True
