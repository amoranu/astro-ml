"""Round-trip test: a rule whose `fires_when` and modifier conditions
are pure JSON produces the same fired indices and final CF as an
equivalent rule using Python lambdas.

This is the regression guard for the LLM-autonomy story: an LLM-
emitted rule (JSON) must execute identically to a human-written
rule (lambdas). If anyone removes the DSL path or the modifier-
union step in `cf_engine`, this test fails loudly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pytest

from astroql.engine import cf_engine
from astroql.engine.cf_predict import _resolve_predicate
from astroql.engine.dsl_evaluator import evaluate_modifier_indices
from astroql.schemas.epoch_state import (
    DashaStack, EpochState, PlanetEpochState,
)
from astroql.schemas.rules import CFModifier, FiredRule, Rule
from astroql.schemas.enums import School


def _build_state() -> EpochState:
    """Minimal hand-built EpochState — enough to exercise rule logic."""
    return EpochState(
        epoch_id="test-epoch",
        start_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2020, 1, 8, tzinfo=timezone.utc),
        dashas=DashaStack(
            maha="Saturn", antar="Rahu",
            pratyantar="Mars", sookshma="Venus",
        ),
        planets={
            "Sun": PlanetEpochState(
                transit_sign="Sagittarius", transit_house=8,
                natal_house=2, shadbala_coefficient=0.5,
                is_retrograde=False, natal_sign="Cancer",
            ),
            "Saturn": PlanetEpochState(
                transit_sign="Capricorn", transit_house=9,
                natal_house=4, shadbala_coefficient=0.7,
                is_retrograde=False, natal_sign="Aquarius",
            ),
            "Mars": PlanetEpochState(
                transit_sign="Aries", transit_house=12,
                natal_house=11, shadbala_coefficient=0.4,
                is_retrograde=True, natal_sign="Virgo",
            ),
        },
        natal_lagna_sign="Taurus",
        derived_lords={
            "lagna_lord": "Venus",
            "ninth_lord": "Saturn",
            "eighth_lord": "Jupiter",
            "father_lagna_sign": "Capricorn",
            "father_8L": "Sun",
        },
    )


def _build_rule_with_modifiers(modifiers):
    return Rule(
        rule_id="test.dsl_round_trip.cf1",
        school=School.PARASHARI,
        source="test",
        base_cf=-0.40,
        primary_planet="Sun",
        modifiers=list(modifiers),
        applicable_to={"effects": ["event_negative"]},
    )


# ── Test cases ─────────────────────────────────────────────────────

def test_fires_when_json_matches_lambda():
    """A JSON `fires_when` and a Python lambda with the same logic
    return the same boolean for the same EpochState.
    """
    state = _build_state()

    # "Sun is in dusthana (6/8/12)"
    py_pred = lambda ep: ep.planets["Sun"].transit_house in (6, 8, 12)
    json_pred = {"path": "planets.Sun.transit_house", "op": "in",
                 "value": [6, 8, 12]}

    assert _resolve_predicate(py_pred, state) is True
    assert _resolve_predicate(json_pred, state) is True


def test_modifier_json_round_trip_via_cf_engine():
    """A modifier whose condition is JSON fires through cf_engine's
    DSL path and adjusts CF identically to a pre-evaluated index.
    """
    state = _build_state()

    # Modifier: "MD lord is Saturn" (-0.10 intensifier)
    json_mod = CFModifier(
        condition={"path": "dashas.maha", "op": "==", "value": "Saturn"},
        effect_cf=-0.10,
        explanation="MD = Saturn (JSON)",
    )
    rule = _build_rule_with_modifiers([json_mod])
    fired = FiredRule(rule=rule)  # no fired_modifier_indices populated

    score, trace = cf_engine.infer_cf(
        [fired], mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
        epoch_state=state,
    )
    assert len(trace.rules_fired) == 1
    rt = trace.rules_fired[0]
    assert rt.modifiers_applied == ["MD = Saturn (JSON)"]
    # combine(-0.40, -0.10) = -0.46; * mu(1.0) = -0.46
    assert rt.final_cf == pytest.approx(-0.46)
    assert score == pytest.approx(-0.46)


def test_legacy_indices_and_json_combined():
    """Mix legacy pre-evaluated indices and a JSON-condition modifier
    in the same rule. Both fire, no double-counting.
    """
    state = _build_state()

    rule = _build_rule_with_modifiers([
        CFModifier(condition={}, effect_cf=-0.10,  # legacy
                   explanation="legacy lambda fired this"),
        CFModifier(condition={"path": "planets.Mars.is_retrograde",
                              "op": "truthy"},
                   effect_cf=-0.05, explanation="Mars retrograde (JSON)"),
    ])
    fired = FiredRule(rule=rule, fired_modifier_indices=[0])

    score, trace = cf_engine.infer_cf(
        [fired], mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
        epoch_state=state,
    )
    rt = trace.rules_fired[0]
    # Both modifiers should be applied: combine(-0.4, -0.1) = -0.46;
    # combine(-0.46, -0.05) ≈ -0.487
    assert "legacy lambda fired this" in rt.modifiers_applied
    assert "Mars retrograde (JSON)" in rt.modifiers_applied
    assert rt.final_cf < -0.48  # both intensifiers applied


def test_json_modifier_skipped_without_epoch_state():
    """If `epoch_state` isn't passed, JSON-condition modifiers cannot
    fire via the engine — only pre-evaluated indices count. This
    preserves backward compatibility for callers that already do
    their own evaluation.
    """
    state = _build_state()

    rule = _build_rule_with_modifiers([
        CFModifier(condition={"path": "dashas.maha", "op": "==",
                              "value": "Saturn"},
                   effect_cf=-0.10, explanation="MD = Saturn (JSON)"),
    ])
    fired = FiredRule(rule=rule)  # no indices, no epoch_state

    score, trace = cf_engine.infer_cf(
        [fired], mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
        # no epoch_state passed
    )
    rt = trace.rules_fired[0]
    assert rt.modifiers_applied == []  # JSON path inactive
    assert rt.final_cf == pytest.approx(-0.40)  # base only


def test_derived_lords_path_works():
    """A rule referencing `derived_lords.father_8L` resolves via DSL.
    This is the LLM-autonomy story — no Python predicate needed.
    """
    state = _build_state()

    rule = _build_rule_with_modifiers([
        CFModifier(
            condition={"path": "derived_lords.father_8L", "op": "==",
                       "value": "Sun"},
            effect_cf=-0.15,
            explanation="AD lord = Sun, which is F8L for this chart",
        ),
    ])
    fired = FiredRule(rule=rule)

    _, trace = cf_engine.infer_cf(
        [fired], mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
        epoch_state=state,
    )
    assert len(trace.rules_fired[0].modifiers_applied) == 1
