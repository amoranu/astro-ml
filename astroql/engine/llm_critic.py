"""LLM critic loop (Module D of NEUROSYMBOLIC_ENGINE_DESIGN.md).

Scope v1 (locked 2026-04-24): **father's longevity only**. The critic
consumes a failing execution trace, pulls relevant Parashari classical
text via astro-prod's RAG, and proposes a new CF-native rule that
addresses the gap.

Three pluggable callables:
  * `query_gen(trace, natal_context) -> List[str]`  — 3 search queries
  * `rag_fn(query, tradition, ...) -> List[{text, source, score}]`
  * `rule_synth(trace, chunks, natal_context) -> dict` — the raw rule

Defaults wire to Anthropic SDK for LLM calls and astro-prod RAG for
retrieval. Tests inject mocks directly — the critic's value is its
orchestration + schema enforcement, not the specific prompts.

Output is always passed through `loader._validate_rule` before return,
so a caller can assume a Rule returned here is structurally valid
(though the regression harness still decides whether to commit it).
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..rules.loader import RuleLoadError, _validate_rule
from ..schemas.rules import Rule
from ..schemas.trace import ExecutionTrace
from . import regression as _regression


log = logging.getLogger(__name__)


# Parashari corpus via astro-prod.
_ASTRO_PROD_PATH = Path(
    "C:/Users/ravii/.gemini/antigravity/playground/astro-prod"
)
if str(_ASTRO_PROD_PATH) not in sys.path:
    sys.path.insert(0, str(_ASTRO_PROD_PATH))

try:
    import rag_engine as _rag  # type: ignore
except Exception as e:  # pragma: no cover
    _rag = None
    log.warning("rag_engine unavailable: %s", e)


# ── Types ──────────────────────────────────────────────────────────

@dataclass
class ProposedRule:
    rule: Rule                    # validated via loader
    raw_yaml: Dict[str, Any]      # ready to dump to YAML library
    trace_query_id: str           # trace that triggered this proposal
    chunks: List[Dict[str, Any]]  # RAG chunks used as provenance


class CriticError(RuntimeError):
    pass


# ── Defaults ───────────────────────────────────────────────────────

def _default_rag_fn(
    query: str,
    tradition: str,
    houses: Optional[List[int]] = None,
    planets: Optional[List[str]] = None,
    category: Optional[str] = None,
    natal_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Wrap astro-prod's retrieve_for_tradition with graceful fallback
    when the corpus config is not present (offline / CI).
    """
    if _rag is None:
        return []
    return _rag.retrieve_for_tradition(
        query=query,
        tradition=tradition,
        houses=houses or [],
        planets=planets or [],
        category=category,
        natal_context=natal_context or {},
    )


def _default_query_gen(
    trace: ExecutionTrace,
    natal_context: Dict[str, Any],
    target_relationship: str,
    target_life_area: str,
) -> List[str]:
    """Fallback query generator — purely deterministic, no LLM. Useful
    for offline runs and as a test-default when no LLM is wired in.
    """
    fired = [fr.rule_id for fr in trace.rules_fired]
    seed = f"{target_relationship} {target_life_area}"
    q1 = f"{seed} classical Parashari rules"
    if fired:
        q2 = f"{seed} exceptions to {fired[0]}"
    else:
        q2 = f"{seed} why no rule fired"
    if trace.veto_fired:
        q3 = f"{seed} yoga-bhanga counter-yoga for {trace.veto_fired}"
    else:
        q3 = f"{seed} rare combinations Bhavat-Bhavam"
    return [q1, q2, q3]


def _default_rule_synth(
    trace: ExecutionTrace,
    chunks: List[Dict[str, Any]],
    natal_context: Dict[str, Any],
    target_relationship: str,
    target_life_area: str,
) -> Dict[str, Any]:
    """Offline / no-LLM fallback: emit a deterministic placeholder rule
    built entirely from RAG provenance. Useful for wiring tests and
    for air-gapped runs where no Anthropic API key is available.

    Production runs should supply a real `rule_synth` that calls an
    LLM with proper prompting; this fallback is a schema-valid stub
    that the regression gate will reject on merit.
    """
    if not chunks:
        raise CriticError(
            "default rule_synth requires at least one RAG chunk for "
            "provenance.citations — got zero. Provide a richer "
            "query_gen / rag_fn or supply a real rule_synth."
        )
    # Sign: if trace says "event expected but we didn't predict it",
    # lean negative. If trace fired a veto and the event occurred
    # anyway, we need a counter-rule. For v1 we emit a mild negative
    # signal (-0.3) keyed on the same primary planet as the first
    # fired rule, or Sun as Parashari's longevity karaka default.
    primary_planet = "Sun"
    first_fired = trace.rules_fired[0] if trace.rules_fired else None
    if first_fired:
        # The base trace doesn't carry primary_planet; try to pull
        # from natal_context planets table for a sensible default.
        pass
    citations = [
        {"source_id": c["source"], "text_chunk": c["text"]}
        for c in chunks[:3]
    ]
    proposal_id = (
        f"parashari.{target_life_area}.critic_proposed."
        f"{trace.query_id[:12] or 'anon'}"
    )
    return {
        "rule_id": proposal_id,
        "school": "parashari",
        "source": "llm_critic",
        "rule_type": "critic_proposed",
        "applicable_to": {
            "relationships": [target_relationship],
            "life_areas": [target_life_area],
            "effects": ["event_negative"],
        },
        # Re-use an antecedent path that exists in features_schema to
        # keep the proposal loader-valid. Real rule_synth will produce
        # a semantically meaningful antecedent.
        "antecedent": [{
            "feature": "primary_house_data.rotated.lord_house",
            "op": "in",
            "value": [6, 8, 12],
        }],
        "base_cf": -0.3,
        "primary_planet": primary_planet,
        "priority_tier": 3,
        "tags": ["critic_proposed", "placeholder"],
        "provenance": {
            "author": "llm_critic",
            "confidence": 0.3,
            "citations": citations,
        },
    }


# ── Orchestration ──────────────────────────────────────────────────

def propose_rule(
    trace: ExecutionTrace,
    natal_context: Dict[str, Any],
    target_relationship: str = "father",
    target_life_area: str = "longevity",
    query_gen: Optional[Callable] = None,
    rag_fn: Optional[Callable] = None,
    rule_synth: Optional[Callable] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> ProposedRule:
    """Run the critic on one failing trace.

    v1 scope: target_relationship defaults to 'father', target_life_area
    to 'longevity'. Callers may pass other combinations but no other
    ground-truth aspect has been audited for this loop yet.

    Raises `CriticError` on failed synthesis or `RuleLoadError` if the
    synthesized rule doesn't round-trip through the loader validator.
    """
    query_gen = query_gen or _default_query_gen
    rag_fn = rag_fn or _default_rag_fn
    rule_synth = rule_synth or _default_rule_synth

    queries = query_gen(
        trace, natal_context, target_relationship, target_life_area,
    )
    if not queries:
        raise CriticError("query_gen produced no queries")
    if len(queries) > 5:
        log.warning("query_gen produced %d queries; clipping to 5",
                    len(queries))
        queries = queries[:5]

    all_chunks: List[Dict[str, Any]] = []
    seen_texts = set()
    for q in queries:
        chunks = rag_fn(
            query=q, tradition="parashari",
            natal_context=natal_context,
            category=target_life_area,
        ) or []
        for c in chunks:
            txt = c.get("text", "")[:120]  # dedupe by prefix
            if txt in seen_texts:
                continue
            seen_texts.add(txt)
            all_chunks.append(c)

    raw_rule = rule_synth(
        trace, all_chunks, natal_context,
        target_relationship, target_life_area,
    )
    if not isinstance(raw_rule, dict):
        raise CriticError(
            f"rule_synth must return dict, got {type(raw_rule).__name__}"
        )

    # Loader validation round-trips the rule through every invariant
    # check (CF range, primary_planet requirement, yoga-bhanga,
    # schema-valid antecedent clauses, provenance.citations structure).
    schema = schema or _load_features_schema()
    try:
        rule = _validate_rule(raw_rule, schema)
    except RuleLoadError as e:
        raise CriticError(
            f"synthesized rule failed loader validation: {e}"
        ) from e

    # Enforce v1 scope: LLM-critic rules must carry llm_critic author.
    if rule.provenance is None or rule.provenance.author != "llm_critic":
        raise CriticError(
            f"critic proposal {rule.rule_id!r} must set "
            f"provenance.author='llm_critic' (got author="
            f"{rule.provenance.author if rule.provenance else None!r})"
        )

    return ProposedRule(
        rule=rule,
        raw_yaml=raw_rule,
        trace_query_id=trace.query_id,
        chunks=all_chunks,
    )


def _load_features_schema() -> Dict[str, Any]:
    import yaml
    p = Path(__file__).resolve().parents[1] / "rules" / (
        "features_schema.yaml"
    )
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Critic → regression gate cycle ─────────────────────────────────

@dataclass
class CycleResult:
    proposed: Optional[ProposedRule]
    committed: bool
    reason: str
    baseline_metrics: Optional[Any] = None     # regression.Metrics
    with_rule_metrics: Optional[Any] = None    # regression.Metrics


def critic_gate_cycle(
    failing_trace: ExecutionTrace,
    natal_context: Dict[str, Any],
    holdout: List[Dict[str, Any]],
    baseline_predict_fn: Callable[[Dict[str, Any]], Any],
    augmented_predict_fn_factory: Callable[
        [Rule], Callable[[Dict[str, Any]], Any]
    ],
    target_relationship: str = "father",
    target_life_area: str = "longevity",
    truth_field: str = "father_death_date",
    per_aspect_tolerance: float = 0.02,
    critic_kwargs: Optional[Dict[str, Any]] = None,
) -> CycleResult:
    """End-to-end cycle: propose a rule, evaluate on holdout, gate.

    `augmented_predict_fn_factory(proposed_rule)` must return a new
    predict function that incorporates the proposal. Returned object
    carries both the baseline and the with-rule metrics so the critic
    loop can persist them or feed them into the next iteration.
    """
    try:
        proposed = propose_rule(
            failing_trace, natal_context,
            target_relationship=target_relationship,
            target_life_area=target_life_area,
            **(critic_kwargs or {}),
        )
    except (CriticError, RuleLoadError) as e:
        return CycleResult(
            proposed=None, committed=False,
            reason=f"proposal failed: {e}",
        )

    baseline = _regression.evaluate(
        holdout, baseline_predict_fn,
        truth_field=truth_field, aspect=target_life_area,
    )
    augmented_fn = augmented_predict_fn_factory(proposed.rule)
    augmented = _regression.evaluate(
        holdout, augmented_fn,
        truth_field=truth_field, aspect=target_life_area,
    )
    ok, reason = _regression.commit_gate(
        augmented, baseline, target_life_area,
        per_aspect_tolerance=per_aspect_tolerance,
    )
    return CycleResult(
        proposed=proposed,
        committed=ok,
        reason=reason,
        baseline_metrics=baseline,
        with_rule_metrics=augmented,
    )
