"""LLM critic tests (Module D). Uses mocked RAG + synth — the critic's
value is orchestration + schema enforcement, which is what we verify.
Father-scoped per v1 project memory.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import pytest

from astroql.engine.llm_critic import (
    CriticError,
    ProposedRule,
    _default_query_gen,
    _default_rule_synth,
    propose_rule,
)
from astroql.rules.loader import RuleLoadError
from astroql.schemas.trace import ExecutionTrace, FiredRuleTrace


# ── Fixtures ────────────────────────────────────────────────────────

def _trace(query_id: str = "q1", veto: bool = False) -> ExecutionTrace:
    t = ExecutionTrace(
        query_id=query_id,
        target_aspect="longevity",
        final_score=-0.4,
        rules_fired=[
            FiredRuleTrace(
                rule_id="parashari.longevity.rotated_lord_dusthana.cf01",
                initial_cf=-0.5,
                strength_multiplier=0.8,
                final_cf=-0.4,
            ),
        ],
    )
    if veto:
        t.veto_fired = "parashari.longevity.durmarana_confluence.veto01"
    return t


def _natal_context() -> Dict[str, Any]:
    return {
        "planets": {
            "Sun": {"rashi": "Cancer", "house": 4},
            "Moon": {"rashi": "Pisces", "house": 12},
        },
        "dasha": {"md": "Saturn", "ad": "Rahu"},
        "lagna": {"sign": "Aries"},
    }


def _fake_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "text": "When the lord of the primary bhava is placed in "
                    "dusthana and further afflicted, the native suffers "
                    "early parental loss.",
            "source": "bphs_ch46_father_longevity.txt",
            "score": 0.12,
        },
        {
            "text": "Sun as the natural karaka of the father must be "
                    "examined from both natal and progressed positions.",
            "source": "phaladeepika_father_karaka.txt",
            "score": 0.18,
        },
    ]


def _good_rule_synth(
    trace, chunks, natal_ctx, target_relationship, target_life_area,
) -> Dict[str, Any]:
    return {
        "rule_id": f"parashari.longevity.critic_test.{trace.query_id}",
        "school": "parashari",
        "source": "llm_critic",
        "applicable_to": {
            "relationships": [target_relationship],
            "life_areas": [target_life_area],
            "effects": ["event_negative"],
        },
        "antecedent": [{
            "feature": "karaka_data.Sun.house",
            "op": "in",
            "value": [6, 8, 12],
        }],
        "base_cf": -0.45,
        "primary_planet": "Sun",
        "priority_tier": 2,
        "provenance": {
            "author": "llm_critic",
            "confidence": 0.5,
            "citations": [
                {
                    "source_id": chunks[0]["source"],
                    "text_chunk": chunks[0]["text"],
                },
            ],
        },
    }


# ── Orchestration paths ─────────────────────────────────────────────

def test_propose_happy_path():
    def fake_rag(**_):
        return _fake_chunks()

    proposed = propose_rule(
        _trace("q42"), _natal_context(),
        rag_fn=fake_rag,
        rule_synth=_good_rule_synth,
    )
    assert isinstance(proposed, ProposedRule)
    assert proposed.rule.rule_id == (
        "parashari.longevity.critic_test.q42"
    )
    assert proposed.rule.base_cf == -0.45
    assert proposed.rule.primary_planet == "Sun"
    assert proposed.rule.provenance is not None
    assert proposed.rule.provenance.author == "llm_critic"
    assert len(proposed.rule.provenance.citations) == 1


def test_default_query_gen_returns_three_queries():
    qs = _default_query_gen(
        _trace(), _natal_context(), "father", "longevity",
    )
    assert len(qs) == 3
    for q in qs:
        assert "father" in q or "longevity" in q


def test_default_query_gen_mentions_veto_in_bhanga_query():
    qs = _default_query_gen(
        _trace(veto=True), _natal_context(), "father", "longevity",
    )
    # The veto-triggered third query should reference yoga-bhanga.
    assert any("bhanga" in q.lower() or "counter" in q.lower()
               for q in qs)


def test_chunks_deduplicated_across_queries():
    calls = {"n": 0}

    def fake_rag(**_):
        calls["n"] += 1
        # Same chunk returned from every query — critic should dedupe.
        return [_fake_chunks()[0]]

    proposed = propose_rule(
        _trace(), _natal_context(),
        rag_fn=fake_rag,
        rule_synth=_good_rule_synth,
    )
    # Three queries issued.
    assert calls["n"] == 3
    # Only one unique chunk surfaced.
    assert len(proposed.chunks) == 1


# ── Schema enforcement ─────────────────────────────────────────────

def test_synth_output_not_dict_rejected():
    def bad_synth(*_args, **_kwargs):
        return "not a dict"

    with pytest.raises(CriticError, match="must return dict"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=bad_synth,
        )


def test_synth_missing_primary_planet_rejected():
    # CF-native non-veto without primary_planet — loader rejects.
    def bad_synth(trace, chunks, natal_ctx, rel, la):
        rule = _good_rule_synth(trace, chunks, natal_ctx, rel, la)
        rule.pop("primary_planet")
        return rule

    with pytest.raises(CriticError, match="loader validation"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=bad_synth,
        )


def test_synth_illegal_base_cf_rejected():
    def bad_synth(trace, chunks, natal_ctx, rel, la):
        rule = _good_rule_synth(trace, chunks, natal_ctx, rel, la)
        rule["base_cf"] = 1.0  # non-veto at ±1.0 illegal
        return rule

    with pytest.raises(CriticError, match="loader validation"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=bad_synth,
        )


def test_synth_without_llm_critic_author_rejected():
    # The critic must tag proposals with author='llm_critic' — even a
    # schema-valid rule with author='human' is refused.
    def sneaky_synth(trace, chunks, natal_ctx, rel, la):
        rule = _good_rule_synth(trace, chunks, natal_ctx, rel, la)
        rule["provenance"]["author"] = "human"
        return rule

    with pytest.raises(CriticError, match="llm_critic"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=sneaky_synth,
        )


def test_synth_citation_missing_text_chunk_rejected():
    def bad_synth(trace, chunks, natal_ctx, rel, la):
        rule = _good_rule_synth(trace, chunks, natal_ctx, rel, la)
        rule["provenance"]["citations"][0].pop("text_chunk")
        return rule

    with pytest.raises(CriticError, match="loader validation"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=bad_synth,
        )


# ── No-RAG corner cases ────────────────────────────────────────────

def test_default_synth_requires_chunks():
    with pytest.raises(CriticError, match="RAG chunk"):
        propose_rule(
            _trace(), _natal_context(),
            rag_fn=lambda **_: [],  # empty retrieval
            rule_synth=_default_rule_synth,
        )


def test_default_synth_round_trips_with_chunks():
    # The default (placeholder) synth must produce a loader-valid rule
    # when chunks are available — it's the air-gapped fallback.
    proposed = propose_rule(
        _trace("deterministic-id"), _natal_context(),
        rag_fn=lambda **_: _fake_chunks(),
        # uses _default_rule_synth
    )
    assert proposed.rule.provenance.author == "llm_critic"
    assert proposed.rule.base_cf == -0.3
    # Rule ID encodes the trace id.
    assert "deterministic-id"[:12] in proposed.rule.rule_id


def test_empty_queries_from_query_gen_rejected():
    with pytest.raises(CriticError, match="no queries"):
        propose_rule(
            _trace(), _natal_context(),
            query_gen=lambda *_args, **_kwargs: [],
            rag_fn=lambda **_: _fake_chunks(),
            rule_synth=_good_rule_synth,
        )


# ── Wiring: critic output feeds regression gate ────────────────────

def _gt_records() -> List[Dict[str, Any]]:
    from datetime import date, timedelta
    return [
        {"id": f"chart{i}",
         "father_death_date": (
             date(2020, 1, 1) + timedelta(days=i * 60)
         ).isoformat(),
        }
        for i in range(20)
    ]


def test_critic_gate_cycle_commits_when_augmented_improves():
    from datetime import date
    from astroql.engine.llm_critic import (
        CycleResult, critic_gate_cycle,
    )
    from astroql.engine.regression import PredictedEvent

    recs = _gt_records()

    def baseline_predict(rec):
        # Always 9 months off from truth — miss with 6mo window.
        from datetime import timedelta
        truth = date.fromisoformat(rec["father_death_date"])
        return PredictedEvent(truth + timedelta(days=270), cf=-0.2)

    def augmented_factory(proposed_rule):
        # The "augmented" prediction function lands exactly on truth,
        # i.e. the proposed rule is assumed to fix the miss.
        def pf(rec):
            return PredictedEvent(
                date.fromisoformat(rec["father_death_date"]), cf=-0.7,
            )
        return pf

    result = critic_gate_cycle(
        failing_trace=_trace(),
        natal_context=_natal_context(),
        holdout=recs,
        baseline_predict_fn=baseline_predict,
        augmented_predict_fn_factory=augmented_factory,
        critic_kwargs={
            "rag_fn": lambda **_: _fake_chunks(),
            "rule_synth": _good_rule_synth,
        },
    )
    assert isinstance(result, CycleResult)
    assert result.committed, result.reason
    assert result.proposed is not None
    assert result.baseline_metrics.hits == 0
    assert result.with_rule_metrics.hits == len(recs)


def test_critic_gate_cycle_rejects_non_improvement():
    from datetime import date, timedelta
    from astroql.engine.llm_critic import critic_gate_cycle
    from astroql.engine.regression import PredictedEvent

    recs = _gt_records()

    def identical_predict(rec):
        truth = date.fromisoformat(rec["father_death_date"])
        return PredictedEvent(truth + timedelta(days=270), cf=-0.2)

    def augmented_factory(_proposed_rule):
        return identical_predict  # no improvement

    result = critic_gate_cycle(
        failing_trace=_trace(),
        natal_context=_natal_context(),
        holdout=recs,
        baseline_predict_fn=identical_predict,
        augmented_predict_fn_factory=augmented_factory,
        critic_kwargs={
            "rag_fn": lambda **_: _fake_chunks(),
            "rule_synth": _good_rule_synth,
        },
    )
    assert not result.committed
    assert "did not improve" in result.reason


def test_critic_gate_cycle_bubbles_up_synth_failure():
    from astroql.engine.llm_critic import critic_gate_cycle

    def bad_synth(*_a, **_k):
        return "not a dict"

    def never_called_factory(_r):
        raise AssertionError("factory should not be called on failure")

    result = critic_gate_cycle(
        failing_trace=_trace(),
        natal_context=_natal_context(),
        holdout=_gt_records(),
        baseline_predict_fn=lambda _: None,
        augmented_predict_fn_factory=never_called_factory,
        critic_kwargs={
            "rag_fn": lambda **_: _fake_chunks(),
            "rule_synth": bad_synth,
        },
    )
    assert not result.committed
    assert "proposal failed" in result.reason


def test_proposed_rule_raw_yaml_reloadable():
    """The raw_yaml dict must be dump-able to the rule library and
    reload cleanly (so the regression harness can persist commits).
    """
    import yaml as _yaml

    from astroql.rules.loader import _validate_rule
    import astroql.engine.llm_critic as critic_mod

    proposed = propose_rule(
        _trace(), _natal_context(),
        rag_fn=lambda **_: _fake_chunks(),
        rule_synth=_good_rule_synth,
    )
    dumped = _yaml.safe_dump(proposed.raw_yaml)
    reloaded = _yaml.safe_load(dumped)
    schema = critic_mod._load_features_schema()
    rule2 = _validate_rule(reloaded, schema)
    assert rule2.rule_id == proposed.rule.rule_id
    assert rule2.base_cf == proposed.rule.base_cf
