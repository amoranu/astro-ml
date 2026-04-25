"""Within-line scorer (Phase F.2).

A reasoning line collects two pools of evidence on a candidate window:

  support : yoga firings (rule_ids in line.supporting_yogas) +
            virtual graded scoring of line.supporting_facts
  attack  : yoga firings (rule_ids in line.attacking_yogas) +
            virtual graded scoring of line.attacking_facts

Aggregation:
    raw_support = noisy_or([tier_weighted(strength) for s in support])
    raw_attack  = noisy_or([tier_weighted(strength) for a in attack])
    net = raw_support * (1 - alpha * raw_attack)

Exception application (Phase F.3 hook) is then layered via
`apply_exceptions(line, line_ev, fired_exceptions)`:

  yoga-scoped : recompute raw_support after multiplying matching yogas'
                strengths by the exception attenuation
  line-scoped : multiply final net_strength by the exception attenuation

The within-line scorer is per-line, per-candidate. Cross-line fusion is
the RRF module's job (F.4).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..engine.rule_engine import _evaluate_antecedent
from ..schemas.features import FeatureBundle
from ..schemas.results import ExceptionFiring, LineEvidence, YogaFiring
from ..schemas.rules import FiredRule
from .plan import ExceptionRule, ReasoningLine


_TIER_MULTIPLIER = {1: 1.0, 2: 0.7, 3: 0.4}

# Virtual yoga ids for chart_facts evaluations (so exceptions can target
# them by line.line_id+suffix if needed).
_FACT_SUPPORT_SUFFIX = "._facts_support"
_FACT_ATTACK_SUFFIX = "._facts_attack"


def _tier_weighted(strength: float, tier: int) -> float:
    return float(strength) * _TIER_MULTIPLIER.get(tier, 0.7)


def _noisy_or(strengths: List[float]) -> float:
    prod = 1.0
    for s in strengths:
        prod *= max(0.0, 1.0 - float(s))
    return max(0.0, min(1.0, 1.0 - prod))


def _evaluate_facts(
    facts: List[Dict[str, Any]],
    bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]],
    base_strength: float,
    min_factors: int,
) -> Tuple[float, int, int, List[str]]:
    """Treat a list of clauses as a virtual graded rule.

    Each clause may carry a `weight` (default 1.0). Strength is
    base_strength * (sum_matched_weights / sum_total_weights), capped
    to [0, 1]. Returns (strength, n_matched, n_total, matched_keys).
    Returns (0, ...) when fewer than min_factors match.
    """
    if not facts:
        return (0.0, 0, 0, [])
    sum_matched = 0.0
    sum_total = 0.0
    n_matched = 0
    matched_keys: List[str] = []
    for clause in facts:
        weight = float(clause.get("weight", 1.0))
        sum_total += weight
        ok, _ = _evaluate_antecedent([clause], bundle, candidate)
        if ok:
            sum_matched += weight
            n_matched += 1
            matched_keys.append(clause.get("feature", "?"))
    if n_matched < int(min_factors) or sum_total <= 0:
        return (0.0, n_matched, len(facts), matched_keys)
    strength = max(0.0, min(1.0, float(base_strength) * (sum_matched / sum_total)))
    return (strength, n_matched, len(facts), matched_keys)


def _yoga_firing_from(fr: FiredRule, yoga_id: str) -> YogaFiring:
    tier = int(getattr(fr.rule, "priority_tier", 2))
    tw = _tier_weighted(fr.strength, tier)
    return YogaFiring(
        yoga_id=yoga_id,
        strength=float(fr.strength),
        polarity=str(fr.polarity),
        tier=tier,
        tier_weighted_strength=tw,
        n_factors_matched=int(fr.bindings.get("_graded_n_matched", 0)),
        n_factors_total=int(fr.bindings.get("_graded_n_total", 0)),
        matched_keys=list(fr.bindings.get("_graded_matched_keys", [])),
        source=str(fr.rule.source),
    )


def _firings_for_window(
    rule_id: str, window: Tuple[datetime, datetime],
    fired_by_rule_id: Dict[str, List[FiredRule]],
) -> List[FiredRule]:
    """Return firings of rule_id that match this window or are static
    (window=None applies uniformly to every candidate)."""
    out: List[FiredRule] = []
    for fr in fired_by_rule_id.get(rule_id, []):
        if fr.window is None:
            out.append(fr)
        elif fr.window == window:
            out.append(fr)
    return out


def _max_firing(firings: List[FiredRule]) -> Optional[FiredRule]:
    """Pick the strongest firing of a rule (de-duplicate when multiple
    timing instances coincide on the same window — same as legacy
    aggregator's MAX-strength dedup)."""
    if not firings:
        return None
    return max(firings, key=lambda f: float(f.strength))


def evaluate_line(
    line: ReasoningLine,
    candidate: Dict[str, Any],
    candidate_window: Tuple[datetime, datetime],
    fired_by_rule_id: Dict[str, List[FiredRule]],
    bundle: FeatureBundle,
) -> LineEvidence:
    """Score one reasoning line on one candidate window.

    Yoga firings are deduplicated (MAX strength per rule_id per window),
    matching the legacy aggregator's behavior for parity.
    """
    support_yogas: List[YogaFiring] = []
    attack_yogas: List[YogaFiring] = []

    for yid in line.supporting_yogas:
        firings = _firings_for_window(yid, candidate_window, fired_by_rule_id)
        best = _max_firing(firings)
        if best is not None:
            support_yogas.append(_yoga_firing_from(best, yid))
    for yid in line.attacking_yogas:
        firings = _firings_for_window(yid, candidate_window, fired_by_rule_id)
        best = _max_firing(firings)
        if best is not None:
            attack_yogas.append(_yoga_firing_from(best, yid))

    if line.supporting_facts:
        s, nm, nt, mk = _evaluate_facts(
            line.supporting_facts, bundle, candidate,
            line.base_strength, line.min_factors,
        )
        if s > 0:
            support_yogas.append(YogaFiring(
                yoga_id=f"{line.line_id}{_FACT_SUPPORT_SUFFIX}",
                strength=s, polarity="negative", tier=2,
                tier_weighted_strength=_tier_weighted(s, 2),
                n_factors_matched=nm, n_factors_total=nt,
                matched_keys=mk, source="chart_facts",
            ))
    if line.attacking_facts:
        s, nm, nt, mk = _evaluate_facts(
            line.attacking_facts, bundle, candidate,
            line.base_strength, line.min_factors,
        )
        if s > 0:
            attack_yogas.append(YogaFiring(
                yoga_id=f"{line.line_id}{_FACT_ATTACK_SUFFIX}",
                strength=s, polarity="positive", tier=2,
                tier_weighted_strength=_tier_weighted(s, 2),
                n_factors_matched=nm, n_factors_total=nt,
                matched_keys=mk, source="chart_facts",
            ))

    # Cap pools at top-K by tier-weighted strength so noisy-OR doesn't
    # saturate at 1.0 when a line has many supporting yogas. We cap
    # AFTER collecting all firings (so LineEvidence.support still shows
    # everything that fired) but BEFORE noisy-OR, by computing strength
    # from the top-K only.
    cap = int(line.max_yogas_per_window or 0)
    if cap > 0 and len(support_yogas) > cap:
        capped_sup = sorted(
            support_yogas, key=lambda y: -y.tier_weighted_strength,
        )[:cap]
    else:
        capped_sup = support_yogas
    if cap > 0 and len(attack_yogas) > cap:
        capped_atk = sorted(
            attack_yogas, key=lambda y: -y.tier_weighted_strength,
        )[:cap]
    else:
        capped_atk = attack_yogas

    raw_support = _noisy_or([y.tier_weighted_strength for y in capped_sup])
    raw_attack = _noisy_or([y.tier_weighted_strength for y in capped_atk])
    net = raw_support * max(0.0, 1.0 - line.alpha * raw_attack)
    net = max(0.0, min(1.0, net))

    return LineEvidence(
        line_id=line.line_id,
        description=line.description,
        net_strength=net,
        raw_support=raw_support,
        raw_attack=raw_attack,
        support=support_yogas,
        attacks=attack_yogas,
        exceptions_fired=[],
    )


def apply_exceptions(
    line: ReasoningLine,
    line_ev: LineEvidence,
    fired_exceptions: List[Tuple[ExceptionRule, str]],
) -> LineEvidence:
    """Layer exception attenuations onto a LineEvidence (mutates and
    returns the same object).

    Order:
      1. Apply yoga-scoped attenuations → recompute raw_support → net.
      2. Apply line-scoped attenuations → multiply net.

    Yoga-scoped exceptions only apply to a line if at least one of its
    supporting yogas matches the exception's `applies_to_yogas`.
    Line-scoped exceptions only apply if the line is in
    `applies_to_lines` (or applies_to_lines is empty = applies-to-all).
    """
    line_scoped: List[ExceptionRule] = []
    yoga_scoped: List[ExceptionRule] = []
    for exc, reason in fired_exceptions:
        if exc.scope == "line":
            if not exc.applies_to_lines or line.line_id in exc.applies_to_lines:
                line_scoped.append(exc)
                line_ev.exceptions_fired.append(ExceptionFiring(
                    exception_id=exc.exception_id,
                    scope="line",
                    attenuation=exc.attenuation,
                    applies_to_yogas=list(exc.applies_to_yogas),
                    applies_to_lines=list(exc.applies_to_lines),
                    reason=reason,
                    source=exc.source,
                ))
        elif exc.scope == "yoga":
            support_ids = {y.yoga_id for y in line_ev.support}
            if set(exc.applies_to_yogas) & support_ids:
                yoga_scoped.append(exc)
                line_ev.exceptions_fired.append(ExceptionFiring(
                    exception_id=exc.exception_id,
                    scope="yoga",
                    attenuation=exc.attenuation,
                    applies_to_yogas=list(exc.applies_to_yogas),
                    applies_to_lines=list(exc.applies_to_lines),
                    reason=reason,
                    source=exc.source,
                ))

    if yoga_scoped:
        adj_pairs: List[float] = []
        for y in line_ev.support:
            mult = 1.0
            for exc in yoga_scoped:
                if y.yoga_id in exc.applies_to_yogas:
                    mult *= float(exc.attenuation)
            adj_pairs.append(y.tier_weighted_strength * mult)
        # Honor the same top-K cap used in evaluate_line so the
        # exception-adjusted support stays consistent with the original.
        cap = int(line.max_yogas_per_window or 0)
        if cap > 0 and len(adj_pairs) > cap:
            adj_pairs = sorted(adj_pairs, reverse=True)[:cap]
        line_ev.raw_support = _noisy_or(adj_pairs)
        line_ev.net_strength = max(0.0, min(1.0,
            line_ev.raw_support * max(0.0, 1.0 - line.alpha * line_ev.raw_attack)
        ))

    for exc in line_scoped:
        line_ev.net_strength = max(0.0, min(1.0,
            line_ev.net_strength * float(exc.attenuation)
        ))

    return line_ev
