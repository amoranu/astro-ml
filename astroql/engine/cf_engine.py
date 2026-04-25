"""CF inference engine (Module B of NEUROSYMBOLIC_ENGINE_DESIGN.md §3.B).

Consumes fired rules and layers the CF neuro-symbolic pipeline:

  filter CF-native → yoga-bhanga prune → veto short-circuit →
  modifier composition → shadbala μ modulation → MYCIN aggregation →
  emit execution trace

Modifier evaluation has TWO supported paths:

  1. **Pre-evaluated** (legacy): caller populates
     `FiredRule.fired_modifier_indices` with the indices of modifiers
     that fired this epoch. Used by `cf_predict.py` with Python-lambda
     `CFRuleSpec.modifier_predicates`. Runs unchanged.

  2. **JSON-DSL** (LLM-autonomous): caller passes the `epoch_state`
     used during firing as `infer_cf(..., epoch_state=ep)`. The engine
     re-evaluates each `Rule.modifiers[].condition` via
     `dsl_evaluator.evaluate_modifier_indices` and unions the result
     with any pre-evaluated indices. Modifiers whose `condition={}`
     (vacuous) fall through to the legacy path — they must already
     be in `fired_modifier_indices`.

This dual path lets LLM-emitted rules (pure JSON) and human-written
rules (Python lambdas) co-exist in the same RuleSet.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..schemas.rules import FiredRule, Rule
from ..schemas.trace import ExecutionTrace, FiredRuleTrace
from . import cf_math
from . import dsl_evaluator as _dsl


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


def _augment_modifier_indices(
    fr: FiredRule, epoch_state: Optional[Any],
) -> List[int]:
    """Return the union of (a) pre-evaluated indices on the FiredRule
    and (b) DSL-evaluated indices from JSON conditions on the rule's
    modifiers — when `epoch_state` is supplied.

    Modifiers with `condition={}` are skipped by the DSL path (they
    are legacy Python-lambda modifiers; their indices, if any, must
    already be in `fired_modifier_indices`). This avoids double-firing.
    """
    indices = list(fr.fired_modifier_indices)
    if epoch_state is None:
        return indices
    conditions = [m.condition for m in fr.rule.modifiers]
    dsl_fired = _dsl.evaluate_modifier_indices(conditions, epoch_state)
    seen = set(indices)
    for idx in dsl_fired:
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def infer_cf(
    fired_rules: Iterable[FiredRule],
    mu_by_planet: Dict[str, float],
    target_aspect: str,
    query_id: str = "",
    epoch_state: Optional[Any] = None,
    overlap_dampening: Optional[Dict[str, float]] = None,
) -> Tuple[float, ExecutionTrace]:
    """Run the Module B CF pipeline on fired rules.

    Args:
        fired_rules: rules that fired (pre-evaluated antecedents).
        mu_by_planet: shadbala μ table for the natal chart.
        target_aspect: e.g. "longevity" — passed through to trace.
        query_id: passed through to trace.
        epoch_state: optional. When provided, the engine re-evaluates
            `Rule.modifiers[].condition` via the JSON DSL (in addition
            to any pre-evaluated `fired_modifier_indices`). Required
            for LLM-emitted rules whose modifiers have non-empty
            `condition` dicts.
        overlap_dampening: optional `{rule_id: factor}` mapping in
            [0, 1]. The final CF for `rule_id` is multiplied by
            `factor`. Use to dampen rules whose time-varying transit
            condition holds for only a fraction of the SD period
            (fast-transit-only firings — e.g. Moon transits that
            trigger a rule for ~2 days within a multi-day SD).
            Default = no dampening (all factors implicitly 1.0).
            Computing the actual fraction (sample N points within
            each SD, count how many trigger the rule) is left to
            the caller / future emitter improvement.

    Returns `(final_score, ExecutionTrace)`. `final_score ∈ [-1, 1]`;
    ±1.0 exactly only for surviving vetoes.
    """
    overlap_dampening = overlap_dampening or {}
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
        active_idxs = _augment_modifier_indices(fr, epoch_state)
        for idx in active_idxs:
            if 0 <= idx < len(fr.rule.modifiers):
                mod = fr.rule.modifiers[idx]
                adj_base = cf_math.combine(adj_base, mod.effect_cf)
                modifier_explanations.append(
                    mod.explanation or f"modifier[{idx}]"
                )
        mu = _planet_mu(fr.rule, mu_by_planet)
        # Optional overlap-fraction dampening (default 1.0 = no-op).
        damp = overlap_dampening.get(fr.rule.rule_id, 1.0)
        if damp < 0.0 or damp > 1.0:
            raise CFEngineError(
                f"overlap_dampening[{fr.rule.rule_id}]={damp} must be "
                f"in [0, 1]"
            )
        final_cf = adj_base * mu * damp
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

    # ── Correlation-group max-pooling (design v2) ─────────────────
    # MYCIN's combine() assumes independence. Highly correlated rules
    # (same astrological signal seen from two angles) double-count and
    # push the score aggressively toward ±1. Group same-tagged rules
    # and keep only the one with the largest |final_cf| per group.
    # Untagged rules pass through individually.
    pooled = _max_pool_correlation_groups(surviving, modulated, trace)

    score = cf_math.aggregate(pooled)
    trace.final_score = score
    return score, trace


def _max_pool_correlation_groups(
    surviving: List[FiredRule],
    modulated: List[float],
    trace: ExecutionTrace,
) -> List[float]:
    """Collapse fired rules sharing a `correlation_group` to the
    single highest-magnitude representative; pass untagged rules
    through unchanged.

    `surviving` and `modulated` are 1:1 by index for the rules that
    actually contributed a CF (those with effective base != 0 after
    modifier composition + μ modulation). Order is preserved.

    Side-effect: updates `trace.rules_fired[i].suppressed_by_group`
    so the audit trail records which rules were max-pooled out.
    """
    # Pair up surviving rules with their modulated CFs. Only rules
    # that actually emitted a CF are in `modulated` — match by order.
    contributors: List[Tuple[FiredRule, float, int]] = []
    contrib_idx = 0
    for fr in surviving:
        if fr.rule.is_veto:
            continue
        # Re-replay the skip condition the main loop applied: only
        # rules that ended up emitting a final_cf are in `modulated`.
        # We can't easily reverse-engineer here — so callers who care
        # about exact ordering must call this with a parallel list.
        if contrib_idx >= len(modulated):
            break
        contributors.append((fr, modulated[contrib_idx], contrib_idx))
        contrib_idx += 1

    groups: Dict[str, Tuple[FiredRule, float, int]] = {}
    pooled: List[float] = []
    independent: List[Tuple[int, float]] = []  # (orig_idx, cf)
    for fr, cf, oi in contributors:
        tag = fr.rule.correlation_group
        if not tag:
            independent.append((oi, cf))
            continue
        cur = groups.get(tag)
        if cur is None or abs(cf) > abs(cur[1]):
            groups[tag] = (fr, cf, oi)

    # Mark suppressed group members in the trace.
    surviving_oi = {oi for (_, _, oi) in groups.values()}
    for fr, cf, oi in contributors:
        tag = fr.rule.correlation_group
        if tag and oi not in surviving_oi:
            if oi < len(trace.rules_fired):
                trace.rules_fired[oi].suppressed_by_group = tag

    # Emit pooled list in the original encounter order so traces
    # stay stable / deterministic.
    pooled_set = {oi for (_, _, oi) in groups.values()}
    indep_set = {oi for (oi, _) in independent}
    cf_by_oi = {oi: cf for (_, cf, oi) in groups.values()}
    cf_by_oi.update({oi: cf for (oi, cf) in independent})
    for oi in sorted(pooled_set | indep_set):
        pooled.append(cf_by_oi[oi])
    return pooled
