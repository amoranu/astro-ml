"""CF inference engine (Module B of NEUROSYMBOLIC_ENGINE_DESIGN.md §3.B).

Consumes pre-fired rules (from the existing `RuleEngine` clause
evaluator) and layers the CF neuro-symbolic pipeline:

  filter CF-native → yoga-bhanga prune → veto short-circuit →
  shadbala μ modulation → MYCIN aggregation → emit execution trace

Modifier evaluation is deferred — the schema already carries them, but
v1 does not re-evaluate modifier conditions (they were not in scope for
the initial CF pipeline). They will be wired in when the LLM critic
starts proposing modifier-bearing rules.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..schemas.rules import FiredRule, Rule
from ..schemas.trace import ExecutionTrace, FiredRuleTrace
from . import cf_math


class CFEngineError(RuntimeError):
    pass


def _is_cf_relevant(rule: Rule) -> bool:
    """A rule participates in CF inference iff it is CF-native or a
    veto. Legacy / graded rules without an explicit base_cf or
    is_veto flag are out-of-scope for v1.
    """
    return rule.base_cf is not None or rule.is_veto


def _planet_mu(
    rule: Rule, mu_by_planet: Dict[str, float],
) -> float:
    """Look up μ for the rule's primary planet. Non-veto CF-native
    rules are loader-guaranteed to declare primary_planet, so the
    lookup should always succeed. Missing planet raises.
    """
    planet = rule.primary_planet
    if planet is None:
        # Only vetoes are allowed here (short-circuit; μ is irrelevant).
        return 1.0
    if planet not in mu_by_planet:
        raise CFEngineError(
            f"rule {rule.rule_id} references primary_planet="
            f"{planet!r} but μ table has keys {sorted(mu_by_planet)}"
        )
    return mu_by_planet[planet]


def _collect_subsumed(active: List[FiredRule]) -> Set[str]:
    """Every rule_id referenced by the subsumes_rules of some active
    rule, intersected with the set of active rule_ids. Yoga-bhanga
    validity (only veto can subsume veto) is enforced at load time.
    """
    active_ids = {fr.rule.rule_id for fr in active}
    subsumed: Set[str] = set()
    for fr in active:
        for target in fr.rule.subsumes_rules:
            if target in active_ids:
                subsumed.add(target)
    return subsumed


def _resolve_veto(
    surviving_vetoes: List[FiredRule],
) -> Tuple[float, str]:
    """Pick the final score when one or more vetoes survive pruning.

    All positive  → +1.0 (protective yoga like Mahamrityunjaya)
    All negative  → -1.0 (absolute denial like durmarana confluence)
    Mixed         → rule-library inconsistency; raise so the author can
                    add an explicit subsumption linking them.
    """
    signs = {
        (1 if fr.rule.effective_base_cf > 0 else -1)
        for fr in surviving_vetoes
    }
    if 1 in signs and -1 in signs:
        pos = [fr.rule.rule_id for fr in surviving_vetoes
               if fr.rule.effective_base_cf > 0]
        neg = [fr.rule.rule_id for fr in surviving_vetoes
               if fr.rule.effective_base_cf < 0]
        raise CFEngineError(
            f"Conflicting vetoes fired without mutual subsumption: "
            f"positive={pos} negative={neg}. Add an explicit "
            f"subsumes_rules link so one yoga-bhangs the other, or "
            f"narrow one veto's antecedent."
        )
    # Single-sign: just pick the first. All same-sign vetoes are ±1.0
    # so any one represents the outcome.
    first = surviving_vetoes[0]
    return first.rule.effective_base_cf, first.rule.rule_id


def infer_cf(
    fired_rules: Iterable[FiredRule],
    mu_by_planet: Dict[str, float],
    target_aspect: str,
    query_id: str = "",
) -> Tuple[float, ExecutionTrace]:
    """Run the Module B CF pipeline on pre-fired rules.

    Returns `(final_score, ExecutionTrace)`. `final_score` is in
    [-1, 1]; ±1.0 exactly only for surviving vetoes.

    Callers are responsible for upstream steps (clause evaluation,
    chart_applicability gating, etc.) via the existing rule engine.
    """
    fired = [fr for fr in fired_rules if _is_cf_relevant(fr.rule)]
    subsumed = _collect_subsumed(fired)
    surviving = [fr for fr in fired if fr.rule.rule_id not in subsumed]

    trace = ExecutionTrace(
        query_id=query_id,
        target_aspect=target_aspect,
        final_score=0.0,
        rules_subsumed=sorted(subsumed),
    )

    surviving_vetoes = [fr for fr in surviving if fr.rule.is_veto]
    if surviving_vetoes:
        score, veto_id = _resolve_veto(surviving_vetoes)
        trace.veto_fired = veto_id
        trace.final_score = score
        # Log the firing veto(s) in rules_fired so the critic sees
        # what triggered the short-circuit.
        for fr in surviving_vetoes:
            trace.rules_fired.append(FiredRuleTrace(
                rule_id=fr.rule.rule_id,
                initial_cf=fr.rule.effective_base_cf,
                strength_multiplier=1.0,  # irrelevant for vetoes
                modifiers_applied=[],
                final_cf=fr.rule.effective_base_cf,
            ))
        return score, trace

    # No vetoes: apply per-rule modifiers, modulate by μ, then
    # aggregate via MYCIN.
    modulated: List[float] = []
    for fr in surviving:
        base = fr.rule.effective_base_cf
        if base == 0.0:
            continue  # no-op rule after derivation; skip
        # Modifier composition: each fired modifier's effect_cf
        # combines with base via the same MYCIN formula. This keeps
        # things bounded and order-invariant. Loader enforces that
        # vetoes carry no modifiers, so we only reach this branch
        # for non-vetoes.
        modifier_explanations: List[str] = []
        adj_base = base
        for idx in fr.fired_modifier_indices:
            if 0 <= idx < len(fr.rule.modifiers):
                mod = fr.rule.modifiers[idx]
                adj_base = cf_math.combine(adj_base, mod.effect_cf)
                modifier_explanations.append(
                    mod.explanation or f"modifier[{idx}]"
                )
        mu = _planet_mu(fr.rule, mu_by_planet)
        final_cf = adj_base * mu
        # base is in the strict-open (-1, 1) interval (loader-enforced
        # for CF-native); μ ∈ [0, 1] can only shrink magnitude. With
        # the post-combine clip in cf_math, adj_base also stays inside
        # the interval.
        if final_cf == 0.0:
            continue
        trace.rules_fired.append(FiredRuleTrace(
            rule_id=fr.rule.rule_id,
            initial_cf=base,
            strength_multiplier=mu,
            modifiers_applied=modifier_explanations,
            final_cf=final_cf,
        ))
        modulated.append(final_cf)

    score = cf_math.aggregate(modulated)
    trace.final_score = score
    return score, trace
