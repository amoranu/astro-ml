"""Tests for correlation-group max-pooling in cf_engine.

Two rules with the same `correlation_group` should NOT both contribute
to the MYCIN aggregate — only the one with the larger |final_cf|.
Independent groups still combine via MYCIN.
"""
from __future__ import annotations

import pytest

from astroql.engine import cf_engine, cf_math
from astroql.schemas.enums import School
from astroql.schemas.rules import FiredRule, Rule


def _make_rule(rid: str, base_cf: float,
               group=None, primary="Sun") -> Rule:
    return Rule(
        rule_id=rid,
        school=School.PARASHARI,
        source="test",
        base_cf=base_cf,
        primary_planet=primary,
        correlation_group=group,
        applicable_to={"effects": ["event_negative"]},
    )


# ── Sanity: legacy independent rules unaffected ──────────────────

def test_legacy_no_groups_aggregates_via_mycin():
    """When no rule has correlation_group, behavior matches the
    pre-refactor MYCIN aggregate exactly.
    """
    r1 = _make_rule("a.cf1", -0.4)
    r2 = _make_rule("b.cf1", -0.3)
    fired = [FiredRule(rule=r1), FiredRule(rule=r2)]
    score, trace = cf_engine.infer_cf(
        fired, mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
    )
    expected = cf_math.aggregate([-0.4, -0.3])
    assert score == pytest.approx(expected)
    # Neither rule was suppressed.
    for rt in trace.rules_fired:
        assert rt.suppressed_by_group is None


# ── Max-pool: same group, both fire, only larger contributes ─────

def test_same_group_max_pools_to_larger():
    """Two rules in the same group: the engine keeps the one with
    larger |final_cf|, the other gets `suppressed_by_group` set.
    Final score = MYCIN of just the surviving rule (here, just it).
    """
    r1 = _make_rule("saturn.transit_9h.cf1", -0.40,
                    group="saturn_to_father_bhava")
    r2 = _make_rule("saturn.aspect_9L.cf1", -0.30,
                    group="saturn_to_father_bhava")
    fired = [FiredRule(rule=r1), FiredRule(rule=r2)]
    score, trace = cf_engine.infer_cf(
        fired, mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
    )
    # Only r1 (-0.40) survives → score = -0.40 * 1.0
    assert score == pytest.approx(-0.40)
    # Both still appear in rules_fired (audit), but r2 is suppressed.
    by_id = {rt.rule_id: rt for rt in trace.rules_fired}
    assert by_id["saturn.transit_9h.cf1"].suppressed_by_group is None
    assert (
        by_id["saturn.aspect_9L.cf1"].suppressed_by_group
        == "saturn_to_father_bhava"
    )


def test_independent_groups_still_combine():
    """Two distinct groups → both contribute, MYCIN-aggregated."""
    r1 = _make_rule("a.cf1", -0.4, group="grp_A")
    r2 = _make_rule("b.cf1", -0.3, group="grp_B")
    fired = [FiredRule(rule=r1), FiredRule(rule=r2)]
    score, trace = cf_engine.infer_cf(
        fired, mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
    )
    expected = cf_math.aggregate([-0.4, -0.3])
    assert score == pytest.approx(expected)
    for rt in trace.rules_fired:
        assert rt.suppressed_by_group is None


def test_three_in_group_max_pool_picks_largest():
    """Three rules in one group: only the strongest survives.
    Tests that the iteration retains the running max, not the first.
    """
    r1 = _make_rule("rule.cf1", -0.20, group="grp")
    r2 = _make_rule("rule.cf2", -0.45, group="grp")  # strongest
    r3 = _make_rule("rule.cf3", -0.30, group="grp")
    fired = [FiredRule(rule=r) for r in (r1, r2, r3)]
    score, _ = cf_engine.infer_cf(
        fired, mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
    )
    assert score == pytest.approx(-0.45)


def test_mixed_grouped_and_independent():
    """One group with two rules + one independent rule. Group is
    max-pooled, then MYCIN combines the survivor with the indep rule.
    """
    r1 = _make_rule("g1.cf1", -0.40, group="grp_A")
    r2 = _make_rule("g1.cf2", -0.30, group="grp_A")  # suppressed
    r3 = _make_rule("indep.cf1", -0.20)              # independent
    fired = [FiredRule(rule=r) for r in (r1, r2, r3)]
    score, _ = cf_engine.infer_cf(
        fired, mu_by_planet={"Sun": 1.0}, target_aspect="longevity",
    )
    expected = cf_math.aggregate([-0.40, -0.20])
    assert score == pytest.approx(expected)


def test_max_pool_uses_modulated_cf_with_mu():
    """Max-pool happens on POST-μ values, not pre-μ base_cf. A rule
    with a large base but a low μ can lose to a smaller-base rule
    backed by a strong planet.
    """
    r_strong_base = _make_rule(
        "weak_planet.cf1", -0.50, group="grp", primary="Mars",
    )
    r_weaker_base = _make_rule(
        "strong_planet.cf1", -0.30, group="grp", primary="Saturn",
    )
    fired = [FiredRule(rule=r_strong_base),
             FiredRule(rule=r_weaker_base)]
    # Mars μ = 0.2 (very weak), Saturn μ = 1.0 (full strength).
    # Effective: Mars rule = -0.10, Saturn rule = -0.30.
    # Saturn rule wins; final = -0.30.
    score, trace = cf_engine.infer_cf(
        fired, mu_by_planet={"Mars": 0.2, "Saturn": 1.0},
        target_aspect="longevity",
    )
    assert score == pytest.approx(-0.30)
    by_id = {rt.rule_id: rt for rt in trace.rules_fired}
    assert by_id["weak_planet.cf1"].suppressed_by_group == "grp"
    assert by_id["strong_planet.cf1"].suppressed_by_group is None
