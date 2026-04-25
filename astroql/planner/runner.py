"""Per-tradition planner runner (Phase F.5).

Per-tradition isolated execution:

    for each requested school:
        resolved = resolver.resolve(query)
        chart    = ChartComputer.compute(...)
        bundle   = <School>FeatureExtractor.extract(chart, resolved)
        rules    = StructuredRuleLibrary.load_rules(school, resolved)
        fired    = RuleEngine.apply(bundle, rules)
        plan     = load_plan(school, plan_name)
        excs     = load_exception_library(school)
        chart_static_exceptions = evaluate_chart_static(...)
        for each candidate window:
            for each reasoning_line in plan:
                line_ev = evaluate_line(...)
                window_exceptions = evaluate_window_dynamic(...)
                apply_exceptions(line, line_ev,
                                 chart_static + window_exceptions)
        rrf_scores = fuse_rrf(per_line_scores)
        annotate ranks, build RankedWindow list, return TraditionResult

No cross-tradition state. The MultiTraditionResult bundle is purely a
container — caller compares schools side by side.

The chart computation is also per-school (different vargas / features
needed for KP vs Parashari vs Jaimini), so we don't share chart state
either, keeping the per-tradition output truly independent.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..rules import StructuredRuleLibrary
from ..schemas.enums import QueryType, School
from ..schemas.features import FeatureBundle
from ..schemas.focus import FocusQuery, ResolvedFocus
from ..schemas.results import (
    LineEvidence, MultiTraditionResult, RankedWindow, TraditionResult,
)
from ..schemas.rules import FiredRule
from .exceptions import evaluate_chart_static, evaluate_window_dynamic
from .fusion import annotate_line_ranks, fuse_rrf, fuse_weighted_sum
from .lines import apply_exceptions, evaluate_line
from .plan import (
    ExceptionLibrary, PlanLoadError, QueryPlan,
    load_exception_library, load_plan,
)


_EXTRACTORS = {
    School.PARASHARI: ParashariFeatureExtractor,
    School.JAIMINI: JaiminiFeatureExtractor,
    School.KP: KPFeatureExtractor,
}


_WindowKey = Tuple[datetime, datetime]


# Empirical bounds on father-death timing, expressed as native's age:
#   - Hard floor: 0 years from native birth (event can't be before native
#     exists in any practical sense for this dataset)
#   - Hard ceiling: 90 years from native birth (father can't live past
#     ~120; native unlikely to live past ~100; combined bound ~90 from
#     native birth covers >99% of empirical distribution)
#   - Soft prior centered on 39 years (empirical mean of our 373-chart
#     dataset for father longevity), sigma=25, max ±15% RRF adjustment
_AGE_HARD_MAX_YEARS = 90.0
_AGE_HARD_MIN_YEARS = 0.0
_AGE_PRIOR_MODE = 39.0
_AGE_PRIOR_SIGMA = 25.0
_AGE_PRIOR_MAX_ADJUST = 0.15


def _candidate_window(cand: Dict[str, Any]) -> Optional[_WindowKey]:
    s = cand.get("start")
    e = cand.get("end")
    if not s or not e:
        return None
    try:
        return (datetime.fromisoformat(s), datetime.fromisoformat(e))
    except Exception:
        return None


def _native_age_at(window_mid: datetime, birth_dt: datetime) -> float:
    return (window_mid - birth_dt).total_seconds() / (86400 * 365.25)


def _age_prior_multiplier(age_years: float) -> float:
    """Soft Gaussian prior on native age at father's death.

    Returns a multiplier in [1 - max_adjust, 1 + max_adjust] to apply
    to the RRF score. Density 1.0 (mode) → +max_adjust; density 0
    (extreme tails) → -max_adjust. Doesn't filter — only re-ranks.
    """
    z = (age_years - _AGE_PRIOR_MODE) / _AGE_PRIOR_SIGMA
    density = math.exp(-0.5 * z * z)
    return 1.0 + _AGE_PRIOR_MAX_ADJUST * (2.0 * density - 1.0)


def _index_firings(fired: List[FiredRule]) -> Dict[str, List[FiredRule]]:
    out: Dict[str, List[FiredRule]] = {}
    for fr in fired:
        out.setdefault(fr.rule.rule_id, []).append(fr)
    return out


def _build_argument(
    line_evs: Dict[str, LineEvidence], rrf_score: float, rank: int,
) -> str:
    """One-paragraph human-readable structured argument for a window.

    Lists each line's net + top supporting yoga + any exceptions fired.
    """
    lines: List[str] = [
        f"#{rank} (RRF={rrf_score:.4f}): "
        f"{len(line_evs)} reasoning lines fired."
    ]
    for line_id, ev in sorted(
        line_evs.items(), key=lambda kv: -kv[1].net_strength,
    ):
        if ev.net_strength <= 0:
            continue
        top_supports = sorted(
            ev.support, key=lambda y: -y.tier_weighted_strength,
        )[:3]
        sup_str = ", ".join(
            f"{y.yoga_id}({y.tier_weighted_strength:.2f})"
            for y in top_supports
        ) or "—"
        atk_str = ""
        if ev.attacks:
            top_attacks = sorted(
                ev.attacks, key=lambda y: -y.tier_weighted_strength,
            )[:2]
            atk_str = " vs " + ", ".join(
                f"{y.yoga_id}({y.tier_weighted_strength:.2f})"
                for y in top_attacks
            )
        exc_str = ""
        if ev.exceptions_fired:
            exc_str = " [exceptions: " + ", ".join(
                f"{e.exception_id}×{e.attenuation:.2f}"
                for e in ev.exceptions_fired
            ) + "]"
        lines.append(
            f"  • {line_id}={ev.net_strength:.3f} "
            f"(rank {ev.rank_in_line}): {sup_str}{atk_str}{exc_str}"
        )
    return "\n".join(lines)


def _build_chart(
    query: FocusQuery, resolved: ResolvedFocus, school: School,
):
    """Compute chart with the union of artifacts the school needs."""
    return ChartComputer().compute(
        query.birth, query.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_kp=(school == School.KP),
        need_jaimini=(school == School.JAIMINI),
        need_strength=(resolved.query_type in (
            QueryType.MAGNITUDE, QueryType.YES_NO,
        )),
    )


def run_tradition(
    query: FocusQuery,
    school: School,
    plan_name: str = "longevity",
    max_windows: int = 20,
) -> TraditionResult:
    """Run the planner for a single school. Self-contained; no shared
    state with other schools."""
    resolver = FocusResolver()
    resolved = resolver.resolve(query)

    try:
        plan = load_plan(school, plan_name)
    except PlanLoadError as e:
        return TraditionResult(
            school=school,
            plan_id=f"{school.value}.{plan_name}",
            inconclusive_reason=f"plan load failed: {e}",
        )
    excs: ExceptionLibrary = load_exception_library(school)

    chart = _build_chart(query, resolved, school)
    extractor_cls = _EXTRACTORS[school]
    bundle: FeatureBundle = extractor_cls().extract(chart, resolved)

    lib = StructuredRuleLibrary()
    rules = lib.load_rules(school, resolved)
    engine = RuleEngine()
    fired = engine.apply(bundle, rules)
    fired_by_rule_id = _index_firings(fired)

    chart_static_excs = evaluate_chart_static(excs, bundle, plan.exception_ids)

    # Hard age cutoff: drop candidate windows beyond plausible
    # father-death range. Father can't reasonably die before native is
    # born or 90+ years after (combined human-lifespan bound).
    birth = query.birth
    if birth is not None:
        birth_dt = datetime(birth.date.year, birth.date.month, birth.date.day)
        all_candidates = bundle.dasha_candidates or []
        kept: List[Dict[str, Any]] = []
        for cand in all_candidates:
            w = _candidate_window(cand)
            if w is None:
                continue
            mid = w[0] + (w[1] - w[0]) / 2
            age = _native_age_at(mid, birth_dt)
            if age < _AGE_HARD_MIN_YEARS or age > _AGE_HARD_MAX_YEARS:
                continue
            # v33: per-window native-age, used by BPHS-13 age-specific
            # yogas (chart-static patterns gated to a tight age window).
            cand["native_age_at_midpoint"] = float(age)
            kept.append(cand)
        candidates = kept
    else:
        candidates = bundle.dasha_candidates or []

    per_line_scores: Dict[str, Dict[_WindowKey, LineEvidence]] = {
        ln.line_id: {} for ln in plan.reasoning_lines
    }
    candidate_meta_by_window: Dict[_WindowKey, Dict[str, Any]] = {}

    for cand in candidates:
        w = _candidate_window(cand)
        if w is None:
            continue
        candidate_meta_by_window.setdefault(w, {
            "md": cand.get("md"),
            "ad": cand.get("ad"),
            "pad": cand.get("pad"),
            "level": cand.get("level"),
            "matched_lords": cand.get("matched_lords"),
        })
        window_excs = evaluate_window_dynamic(
            excs, bundle, cand, plan.exception_ids,
        )
        all_excs = chart_static_excs + window_excs

        for line in plan.reasoning_lines:
            line_ev = evaluate_line(
                line, cand, w, fired_by_rule_id, bundle,
            )
            if all_excs:
                apply_exceptions(line, line_ev, all_excs)
            existing = per_line_scores[line.line_id].get(w)
            if existing is None or line_ev.net_strength > existing.net_strength:
                per_line_scores[line.line_id][w] = line_ev

    annotate_line_ranks(per_line_scores)

    if plan.aggregation_method == "weighted_sum":
        fused = fuse_weighted_sum(per_line_scores, plan.line_weights)
    else:
        fused = fuse_rrf(per_line_scores, plan.line_weights, k=plan.rrf_k)

    # Soft age prior on RRF scores (relationship-specific). Boosts windows
    # near the empirical mode of native_age_at_father_death; gently
    # penalizes outliers. Doesn't filter (hard cutoff above did the
    # filtering).
    if birth is not None:
        for w in list(fused.keys()):
            mid = w[0] + (w[1] - w[0]) / 2
            age = _native_age_at(mid, birth_dt)
            fused[w] = fused[w] * _age_prior_multiplier(age)

    ranked: List[Tuple[_WindowKey, float]] = sorted(
        fused.items(), key=lambda r: -r[1],
    )

    windows_out: List[RankedWindow] = []
    for rank_idx, (w, score) in enumerate(ranked[:max_windows], start=1):
        line_evs: Dict[str, LineEvidence] = {}
        for line_id, scores in per_line_scores.items():
            ev = scores.get(w)
            if ev is not None and ev.net_strength > 0:
                line_evs[line_id] = ev
        rw = RankedWindow(
            start=w[0],
            end=w[1],
            rrf_score=score,
            final_rank=rank_idx,
            line_evidence=line_evs,
            structured_argument=_build_argument(line_evs, score, rank_idx),
            candidate_meta=candidate_meta_by_window.get(w, {}),
        )
        windows_out.append(rw)

    explanation = (
        f"{school.value.capitalize()} plan {plan.plan_id}: "
        f"{len(plan.reasoning_lines)} reasoning lines, "
        f"{len(candidates)} candidate windows, "
        f"{len(fired)} rule firings, "
        f"{len(chart_static_excs)} chart-static exceptions fired."
    )

    return TraditionResult(
        school=school,
        plan_id=plan.plan_id,
        windows=windows_out,
        explanation=explanation,
        n_lines_total=len(plan.reasoning_lines),
        n_windows_evaluated=len(candidates),
        aggregation_method=plan.aggregation_method,
        inconclusive_reason=(
            None if windows_out
            else "no window had positive net strength on any reasoning line"
        ),
    )


def run_multi_tradition(
    query: FocusQuery,
    plan_name: str = "longevity",
    max_windows: int = 20,
) -> MultiTraditionResult:
    """Run all requested schools in isolation, return per-tradition
    bundle. No cross-tradition fusion happens."""
    resolved = FocusResolver().resolve(query)
    out = MultiTraditionResult(
        query=query,
        resolved=resolved,
        query_type=resolved.query_type,
    )
    if School.PARASHARI in query.schools:
        out.parashari = run_tradition(
            query, School.PARASHARI, plan_name, max_windows,
        )
    if School.JAIMINI in query.schools:
        out.jaimini = run_tradition(
            query, School.JAIMINI, plan_name, max_windows,
        )
    if School.KP in query.schools:
        out.kp = run_tradition(
            query, School.KP, plan_name, max_windows,
        )
    return out
