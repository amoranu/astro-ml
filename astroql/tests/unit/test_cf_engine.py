"""CF inference engine tests (Module B)."""
from __future__ import annotations

from typing import List

import pytest

from astroql.engine.cf_engine import CFEngineError, infer_cf
from astroql.schemas.enums import School
from astroql.schemas.rules import FiredRule, Rule


# ── Builders ────────────────────────────────────────────────────────

def _rule(
    rule_id: str,
    base_cf: float,
    primary_planet: str = "Sun",
    is_veto: bool = False,
    subsumes_rules: List[str] = (),
) -> Rule:
    return Rule(
        rule_id=rule_id,
        school=School.PARASHARI,
        source="test",
        is_veto=is_veto,
        base_cf=base_cf,
        primary_planet=primary_planet if not is_veto else None,
        subsumes_rules=list(subsumes_rules),
    )


def _fire(rule: Rule) -> FiredRule:
    return FiredRule(rule=rule)


_MU = {
    "Sun": 0.8, "Moon": 0.5, "Mars": 0.6, "Mercury": 0.7,
    "Jupiter": 0.9, "Venus": 0.4, "Saturn": 0.3,
    "Rahu": 0.5, "Ketu": 0.5,
}


# ── Empty / neutral ─────────────────────────────────────────────────

def test_empty_fired_list_returns_zero():
    score, trace = infer_cf([], _MU, "longevity")
    assert score == 0.0
    assert trace.rules_fired == []
    assert trace.rules_subsumed == []
    assert trace.veto_fired is None


def test_non_cf_native_rules_filtered_out():
    # Legacy rule: no base_cf, no is_veto. Should be ignored entirely.
    legacy = Rule(
        rule_id="legacy.x",
        school=School.PARASHARI,
        source="test",
        consequent={"polarity": "negative", "strength": 0.5},
    )
    score, trace = infer_cf([_fire(legacy)], _MU, "longevity")
    assert score == 0.0
    assert len(trace.rules_fired) == 0


# ── Modulation + aggregation ────────────────────────────────────────

def test_single_rule_modulated_by_mu():
    r = _rule("r1", base_cf=-0.5, primary_planet="Sun")  # μ = 0.8
    score, trace = infer_cf([_fire(r)], _MU, "longevity")
    assert score == pytest.approx(-0.4)
    assert len(trace.rules_fired) == 1
    ft = trace.rules_fired[0]
    assert ft.rule_id == "r1"
    assert ft.initial_cf == -0.5
    assert ft.strength_multiplier == 0.8
    assert ft.final_cf == pytest.approx(-0.4)


def test_two_negative_rules_aggregate_mycin():
    # base=-0.5 each, μ_Sun=0.8, μ_Saturn=0.3
    # final_cfs: -0.4 and -0.15
    # MYCIN same-sign: a + b*(1+a) = -0.4 + -0.15*(1-0.4) = -0.4 - 0.09 = -0.49
    r1 = _rule("r1", base_cf=-0.5, primary_planet="Sun")
    r2 = _rule("r2", base_cf=-0.5, primary_planet="Saturn")
    score, _ = infer_cf([_fire(r1), _fire(r2)], _MU, "longevity")
    assert score == pytest.approx(-0.49)


def test_order_independent_aggregation():
    r1 = _rule("r1", base_cf=-0.4, primary_planet="Sun")
    r2 = _rule("r2", base_cf=0.6, primary_planet="Jupiter")
    r3 = _rule("r3", base_cf=-0.2, primary_planet="Saturn")
    s1, _ = infer_cf([_fire(r1), _fire(r2), _fire(r3)], _MU, "x")
    s2, _ = infer_cf([_fire(r3), _fire(r1), _fire(r2)], _MU, "x")
    s3, _ = infer_cf([_fire(r2), _fire(r3), _fire(r1)], _MU, "x")
    assert s1 == pytest.approx(s2)
    assert s2 == pytest.approx(s3)


# ── Yoga-bhanga pruning ─────────────────────────────────────────────

def test_subsumed_rule_dropped():
    a = _rule("a", base_cf=-0.5, primary_planet="Sun")
    b = _rule("b", base_cf=-0.7, primary_planet="Mars",
              subsumes_rules=["a"])
    score, trace = infer_cf([_fire(a), _fire(b)], _MU, "longevity")
    # Only b contributes: -0.7 * 0.6 = -0.42
    assert score == pytest.approx(-0.42)
    assert "a" in trace.rules_subsumed
    assert len(trace.rules_fired) == 1
    assert trace.rules_fired[0].rule_id == "b"


def test_subsumed_rule_not_active_doesnt_prune():
    # B subsumes A, but A isn't even fired. B should still contribute.
    a = _rule("a", base_cf=-0.5, primary_planet="Sun")
    b = _rule("b", base_cf=-0.7, primary_planet="Mars",
              subsumes_rules=["a"])
    _ = a  # A intentionally not in fired list
    score, trace = infer_cf([_fire(b)], _MU, "longevity")
    assert score == pytest.approx(-0.42)
    assert trace.rules_subsumed == []


# ── Veto short-circuit ──────────────────────────────────────────────

def test_negative_veto_short_circuits():
    v = _rule("v", base_cf=-1.0, is_veto=True)
    normal = _rule("r", base_cf=-0.3, primary_planet="Mars")
    score, trace = infer_cf([_fire(v), _fire(normal)], _MU, "longevity")
    assert score == -1.0
    assert trace.veto_fired == "v"
    # Normal rule not in rules_fired (veto short-circuits before CF
    # modulation loop).
    assert {r.rule_id for r in trace.rules_fired} == {"v"}


def test_positive_veto_short_circuits():
    v = _rule("v", base_cf=1.0, is_veto=True)
    normal = _rule("r", base_cf=-0.3, primary_planet="Mars")
    score, trace = infer_cf([_fire(v), _fire(normal)], _MU, "longevity")
    assert score == 1.0
    assert trace.veto_fired == "v"


def test_subsumed_veto_does_not_fire():
    # Mahamrityunjaya veto subsumes durmarana veto. Both fire
    # initially; only the subsuming (positive) one survives.
    durmarana = _rule("durmarana", base_cf=-1.0, is_veto=True)
    mahamrityunjaya = _rule(
        "mahamrityunjaya", base_cf=1.0, is_veto=True,
        subsumes_rules=["durmarana"],
    )
    score, trace = infer_cf(
        [_fire(durmarana), _fire(mahamrityunjaya)], _MU, "longevity",
    )
    assert score == 1.0
    assert trace.veto_fired == "mahamrityunjaya"
    assert "durmarana" in trace.rules_subsumed


def test_conflicting_unsubsumed_vetoes_raise():
    pos = _rule("pos", base_cf=1.0, is_veto=True)
    neg = _rule("neg", base_cf=-1.0, is_veto=True)
    with pytest.raises(CFEngineError, match="Conflicting vetoes"):
        infer_cf([_fire(pos), _fire(neg)], _MU, "longevity")


# ── Trace content ───────────────────────────────────────────────────

def test_trace_carries_query_id_and_target():
    r = _rule("r", base_cf=-0.4, primary_planet="Sun")
    _, trace = infer_cf(
        [_fire(r)], _MU, "longevity", query_id="q42",
    )
    assert trace.query_id == "q42"
    assert trace.target_aspect == "longevity"


def test_trace_json_roundtrip():
    r = _rule("r", base_cf=-0.4, primary_planet="Sun")
    _, trace = infer_cf([_fire(r)], _MU, "longevity", query_id="q")
    from astroql.schemas.trace import ExecutionTrace
    restored = ExecutionTrace.from_dict(trace.to_dict())
    assert restored.query_id == trace.query_id
    assert restored.final_score == trace.final_score
    assert restored.rules_fired[0].rule_id == "r"
    assert restored.rules_fired[0].final_cf == pytest.approx(-0.32)


# ── Edge cases ──────────────────────────────────────────────────────

def test_missing_mu_for_planet_raises():
    r = _rule("r", base_cf=-0.4, primary_planet="Sun")
    with pytest.raises(CFEngineError, match="primary_planet"):
        infer_cf([_fire(r)], {"Moon": 0.5}, "longevity")


def test_mu_zero_zeros_rule():
    # μ=0 means the primary planet is powerless → the rule contributes
    # nothing to aggregation even though its conditions matched.
    r = _rule("r", base_cf=-0.4, primary_planet="Sun")
    score, trace = infer_cf([_fire(r)], {"Sun": 0.0}, "longevity")
    assert score == 0.0
    # Skipped from trace — it produced zero signal.
    assert trace.rules_fired == []
