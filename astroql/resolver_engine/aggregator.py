"""Aggregator (spec §6.8, §7 aggregation, Phase 2 cross-school extension).

Phase 9 (timing fixes) added:
    - Age prior on TIMING windows for life-area-specific events (CAV-036)
      drawn from empirical distributions (currently father longevity).
    - Specificity bonus by # distinct rules + # distinct schools (CAV-009).
    - Per-rule strength multipliers loaded from calibrated_strengths.yaml
      (CAV-034) when present.

Phase 1 behaviour (single-school):
    1. Greedy temporal overlap clustering of FiredRules within a school.
    2. Per cluster: noisy-OR of negative rule strengths.
    3. Static rules (window=None) applied as a global multiplier.
    4. Filter below min_confidence, rank, take top-N.

Phase 2 cross-school behaviour (spec §6.8):
    1. Cluster windows within each school first.
    2. Across-school clustering by temporal IoU > 0.3 OR containment.
    3. Per cluster: school_conf(S) = noisy-OR over school S's rules.
       aggregate = weighted sum across schools × confluence_bonus(n_schools).
    4. Same filter + rank as Phase 1.

DESCRIPTION / MAGNITUDE / YES_NO still raise NotImplementedError.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schemas.enums import QueryType, School
from ..schemas.focus import FocusQuery, ResolvedFocus
from ..schemas.results import (
    CandidateWindow, DescriptiveAttribute, QueryResult,
)
from ..schemas.rules import FiredRule


class AggregatorError(RuntimeError):
    pass


def _noisy_or(strengths: List[float]) -> float:
    prod = 1.0
    for s in strengths:
        prod *= max(0.0, 1.0 - float(s))
    return max(0.0, min(1.0, 1.0 - prod))


# ── Learned-weights aggregator (Phase C) ─────────────────────────────
# When ASTROQL_USE_LEARNED_WEIGHTS=1 and rules/learned_weights.yaml
# exists, the per-cluster score becomes:
#     sigmoid(bias + sum_j w_j * tier_weighted_strength_signed_j)
# Trained via astroql.benchmark.learn_weights on the rule-firing matrix
# from a labeled train set. Bypasses noisy-OR entirely.

_LEARNED_PATH = (
    Path(__file__).resolve().parents[1] / "rules" / "learned_weights.yaml"
)


def _load_learned_weights() -> Optional[dict]:
    import os
    if os.environ.get("ASTROQL_USE_LEARNED_WEIGHTS", "0") != "1":
        return None
    if not _LEARNED_PATH.exists():
        return None
    try:
        import yaml as _yaml
        with open(_LEARNED_PATH, encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception:
        return None
    weights = data.get("weights") or {}
    meta = data.get("meta") or {}
    return {
        "weights": {str(k): float(v) for k, v in weights.items()},
        "bias": float(meta.get("bias", 0.0)),
    }


def _sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    import math as _m
    return 1.0 / (1.0 + _m.exp(-x))


# Tier multipliers: tier-1 rules carry full weight, tier-2 ~70%, tier-3 ~40%.
# Applied to each rule's strength before noisy-OR aggregation so tier-3
# rules can amplify a window already supported by tier-1 evidence but
# can't single-handedly push a window to high confidence.
_TIER_MULTIPLIER = {1: 1.0, 2: 0.7, 3: 0.4}


def _tier_weighted_strength(fr) -> float:
    tier = getattr(fr.rule, "priority_tier", 2)
    return float(fr.strength) * _TIER_MULTIPLIER.get(tier, 0.7)


def _window_overlap(a: Tuple[_dt.datetime, _dt.datetime],
                    b: Tuple[_dt.datetime, _dt.datetime]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge_windows(
    a: Tuple[_dt.datetime, _dt.datetime],
    b: Tuple[_dt.datetime, _dt.datetime],
) -> Tuple[_dt.datetime, _dt.datetime]:
    return (min(a[0], b[0]), max(a[1], b[1]))


def _cluster_timing_rules(
    timing_fired: List[FiredRule],
) -> List[Tuple[Tuple[_dt.datetime, _dt.datetime], List[FiredRule]]]:
    """Greedy temporal clustering — sort by start, merge overlapping
    windows into clusters. For PoC this is simpler than IoU thresholds
    and produces tight clusters (each candidate window is narrow).
    """
    if not timing_fired:
        return []
    rows = [(fr.window, fr) for fr in timing_fired if fr.window]
    rows.sort(key=lambda r: r[0][0])

    clusters: List[List[Tuple]] = []
    for w, fr in rows:
        placed = False
        for cluster in clusters:
            cluster_window = (
                min(x[0][0] for x in cluster),
                max(x[0][1] for x in cluster),
            )
            if _window_overlap(cluster_window, w):
                cluster.append((w, fr))
                placed = True
                break
        if not placed:
            clusters.append([(w, fr)])

    out: List[Tuple[Tuple[_dt.datetime, _dt.datetime], List[FiredRule]]] = []
    for cluster in clusters:
        window = (min(x[0][0] for x in cluster),
                  max(x[0][1] for x in cluster))
        rules_in_cluster = [x[1] for x in cluster]
        out.append((window, rules_in_cluster))
    return out


# ── Phase 2 cross-school clustering ──────────────────────────────────

def _window_iou(
    a: Tuple[_dt.datetime, _dt.datetime],
    b: Tuple[_dt.datetime, _dt.datetime],
) -> float:
    """Intersection-over-union of two temporal windows."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    if inter_start >= inter_end:
        return 0.0
    inter = (inter_end - inter_start).total_seconds()
    union_start = min(a[0], b[0])
    union_end = max(a[1], b[1])
    union = (union_end - union_start).total_seconds()
    if union <= 0:
        return 0.0
    return inter / union


def _window_contains(
    outer: Tuple[_dt.datetime, _dt.datetime],
    inner: Tuple[_dt.datetime, _dt.datetime],
) -> bool:
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def _cross_school_clusters(
    per_school_clusters: Dict[
        School,
        List[Tuple[Tuple[_dt.datetime, _dt.datetime], List[FiredRule]]],
    ],
    iou_threshold: float = 0.3,
) -> List[Dict[School, List[Tuple]]]:
    """Cross-school cluster merger.

    Input: per-school list of (window, fired_rules) from within-school
    clustering.
    Output: list of {school -> [(window, rules)]} where each dict
    represents one cross-school cluster.
    """
    # Flatten with school tag.
    flat: List[Tuple[School, Tuple, List[FiredRule]]] = []
    for school, clusters in per_school_clusters.items():
        for window, rules in clusters:
            flat.append((school, window, rules))
    flat.sort(key=lambda r: r[1][0])

    groups: List[List[Tuple[School, Tuple, List[FiredRule]]]] = []
    for school, window, rules in flat:
        placed = False
        for group in groups:
            # Cluster window from current group.
            gw = (
                min(x[1][0] for x in group),
                max(x[1][1] for x in group),
            )
            if (_window_iou(gw, window) >= iou_threshold
                    or _window_contains(gw, window)
                    or _window_contains(window, gw)):
                group.append((school, window, rules))
                placed = True
                break
        if not placed:
            groups.append([(school, window, rules)])

    out: List[Dict[School, List[Tuple]]] = []
    for group in groups:
        by_school: Dict[School, List[Tuple]] = {}
        for school, window, rules in group:
            by_school.setdefault(school, []).append((window, rules))
        out.append(by_school)
    return out


def _confluence_bonus(n_schools: int) -> float:
    """Multiplicative bonus for cross-school agreement.

    1 school: 1.00 (baseline — no bonus)
    2 schools: 1.10
    3 schools: 1.20
    """
    return 1.0 + 0.10 * max(0, n_schools - 1)


class Aggregator:
    """Phase 1 single-school aggregator.

    For TIMING queries, returns a QueryResult with a ranked list of
    CandidateWindow. For other query types, raises NotImplementedError
    (implemented in later phases).
    """

    def aggregate(
        self,
        query: FocusQuery,
        resolved: ResolvedFocus,
        fired_by_school: Dict[School, List[FiredRule]],
        min_confidence: Optional[float] = None,
        max_windows: int = 20,
        iou_threshold: float = 0.3,
    ) -> QueryResult:
        qt = resolved.query_type
        if qt == QueryType.DESCRIPTION:
            return self._aggregate_description(
                query, resolved, fired_by_school,
            )
        if qt == QueryType.MAGNITUDE:
            return self._aggregate_magnitude(
                query, resolved, fired_by_school,
            )
        if qt == QueryType.YES_NO:
            return self._aggregate_yes_no(
                query, resolved, fired_by_school,
                threshold_gap=0.2,
            )
        if qt not in (QueryType.TIMING, QueryType.PROBABILITY):
            raise NotImplementedError(
                f"Aggregator: unknown query_type {qt.value}"
            )

        threshold = (
            query.min_confidence
            if min_confidence is None else min_confidence
        )
        school_weights = dict(query.school_weights or {})
        # Normalize: if only one school fired, it gets weight 1.0.
        active_schools = [
            s for s, fired in fired_by_school.items() if fired
        ]
        if len(active_schools) == 0:
            return QueryResult(
                query=query, resolved=resolved, query_type=qt,
                windows=[], inconclusive_reason="no school fired rules",
            )
        total_weight = sum(
            school_weights.get(s, 1.0) for s in active_schools
        )
        if total_weight <= 0:
            school_weights = {s: 1.0 / len(active_schools)
                              for s in active_schools}
        else:
            school_weights = {
                s: school_weights.get(s, 1.0) / total_weight
                for s in active_schools
            }

        # ── Stage 1: per-school clustering + static multiplier. ──
        per_school_clusters: Dict[School, List[Tuple]] = {}
        per_school_static: Dict[School, List[FiredRule]] = {}
        for school, fired in fired_by_school.items():
            if not fired:
                continue
            static = [fr for fr in fired if fr.window is None]
            timing = [fr for fr in fired if fr.window is not None]
            per_school_static[school] = static
            per_school_clusters[school] = _cluster_timing_rules(timing)

        # ── Stage 2: cross-school cluster merge. ──
        cross_clusters = _cross_school_clusters(
            per_school_clusters, iou_threshold=iou_threshold,
        )

        # ── Stage 3: per-cluster aggregation. ──
        candidates: List[CandidateWindow] = []
        for by_school in cross_clusters:
            # Merged window = union of all constituent school windows.
            all_starts = [
                w[0] for rows in by_school.values() for w, _ in rows
            ]
            all_ends = [
                w[1] for rows in by_school.values() for w, _ in rows
            ]
            window = (min(all_starts), max(all_ends))

            per_school_conf: Dict[School, float] = {}
            all_rules: List[FiredRule] = []
            learned = _load_learned_weights()
            for school, rows in by_school.items():
                # Deduplicate by rule_id (keep MAX |strength|), preserving
                # polarity sign. For learned-weights mode we also need
                # positive-polarity rule firings (they can have NEGATIVE
                # learned weights that push the score down).
                best_signed_by_rule: Dict[str, float] = {}
                best_strength_by_rule: Dict[str, float] = {}
                for _, rules in rows:
                    for fr in rules:
                        s_unsigned = _tier_weighted_strength(fr)
                        s_signed = (
                            -s_unsigned if fr.polarity == "positive"
                            else s_unsigned
                        )
                        prev_signed = best_signed_by_rule.get(
                            fr.rule.rule_id, 0.0,
                        )
                        if abs(s_signed) > abs(prev_signed):
                            best_signed_by_rule[fr.rule.rule_id] = s_signed
                        if fr.polarity == "negative":
                            prev = best_strength_by_rule.get(
                                fr.rule.rule_id, 0.0,
                            )
                            if s_unsigned > prev:
                                best_strength_by_rule[
                                    fr.rule.rule_id
                                ] = s_unsigned
                if learned is not None:
                    # Learned-weights path: score = sigmoid(b + w·x)
                    weights = learned["weights"]
                    bias = learned["bias"]
                    z = bias
                    for rid, s_signed in best_signed_by_rule.items():
                        z += weights.get(rid, 0.0) * s_signed
                    # Add static rule contributions (window=None) too.
                    for fr in per_school_static.get(school, []):
                        s_unsigned = _tier_weighted_strength(fr)
                        s_signed = (
                            -s_unsigned if fr.polarity == "positive"
                            else s_unsigned
                        )
                        z += weights.get(fr.rule.rule_id, 0.0) * s_signed
                    school_conf = _sigmoid(z)
                    per_school_conf[school] = school_conf
                    for _, rules in rows:
                        all_rules.extend(rules)
                    all_rules.extend(per_school_static.get(school, []))
                    continue
                # Legacy noisy-OR path (default).
                cluster_conf = _noisy_or(
                    list(best_strength_by_rule.values()),
                )
                # CAV-009: distinct-rule confluence is the key
                # discriminator. A window with 4 distinct rule firings
                # is doctrinally stronger than one with 1 rule even if
                # noisy-OR doesn't fully reflect it. Add a bonus
                # proportional to log(n_rules).
                # v6 inspection found noisy-OR saturates near 1.0 once
                # 5+ rules fire on a window, collapsing the gap between
                # death windows and competitors with similar role load.
                # Bumped weight 0.05 -> 0.10 so confluence count actually
                # discriminates at the saturation regime.
                import math
                n_distinct = len(best_strength_by_rule)
                if n_distinct >= 2:
                    bonus = 0.10 * math.log2(n_distinct)
                    cluster_conf = min(1.0, cluster_conf + bonus)
                static_strengths = [
                    _tier_weighted_strength(fr)
                    for fr in per_school_static.get(school, [])
                    if fr.polarity == "negative"
                ]
                static_mult = _noisy_or(static_strengths)
                # Conjoint static+timing combination: classical doctrine
                # holds that dasha activates a static promise. Reward
                # presence of static evidence but don't cripple windows
                # that fire only on transit/dasha confluence.
                if static_mult > 0.0:
                    school_conf = cluster_conf * (0.7 + 0.3 * static_mult)
                else:
                    school_conf = cluster_conf * 0.7
                per_school_conf[school] = max(0.0, min(1.0, school_conf))

                for _, rules in rows:
                    all_rules.extend(rules)
                all_rules.extend(per_school_static.get(school, []))

            # Weighted combination + confluence bonus.
            weighted_conf = sum(
                school_weights.get(s, 0.0) * per_school_conf[s]
                for s in per_school_conf
            )
            bonus = _confluence_bonus(len(per_school_conf))
            aggregate = max(0.0, min(1.0, weighted_conf * bonus))

            candidates.append(CandidateWindow(
                start=window[0],
                end=window[1],
                confidence_per_school=per_school_conf,
                aggregate_confidence=aggregate,
                contributing_rules=all_rules,
                contradictions=[],   # populated by contradiction-detector
            ))

        # Contradiction annotation + damping (CAV-012).
        from .contradictions import annotate_all
        annotate_all(candidates)
        for c in candidates:
            if c.contradictions:
                c.aggregate_confidence = max(
                    0.0,
                    c.aggregate_confidence * (0.85 ** len(c.contradictions)),
                )

        # Soft age prior (relationship-specific). Caps at ±5% adjustment
        # so outliers (childhood deaths, very late deaths) are never
        # excluded — only marginally re-ranked. Mode = empirical mean
        # for father longevity from our 373-chart dataset.
        if (resolved.query.life_area.value == "longevity"
                and resolved.query.relationship.value == "father"):
            self._apply_soft_age_prior(
                candidates, query.birth.date if query.birth else None,
                mode_age=39.0, sigma=20.0, max_adjust=0.05,
            )

        # Filter + rank + cap.
        candidates = [c for c in candidates
                      if c.aggregate_confidence >= threshold]
        candidates.sort(
            key=lambda c: (
                -len(c.confidence_per_school),   # prefer more schools
                -c.aggregate_confidence,          # then higher confidence
            )
        )
        candidates = candidates[:max_windows]

        return QueryResult(
            query=query,
            resolved=resolved,
            query_type=qt,
            windows=candidates,
            probability=None,
            attributes=None,
            magnitude=None,
            yes_no=None,
            explain=None,
            inconclusive_reason=(
                None if candidates else
                f"no window met min_confidence={threshold:.2f}"
            ),
        )

    def _apply_soft_age_prior(
        self, candidates, birth_date, mode_age: float,
        sigma: float, max_adjust: float,
    ) -> None:
        """Apply a SOFT Gaussian age prior, capped at ±max_adjust.

        Doesn't exclude any window — only marginally re-ranks. Outliers
        (e.g. childhood paternal death at native age <5, very late at
        age >70) still receive their full astrological score, just
        slightly de-prioritized in tied-confidence cases.
        """
        if not birth_date:
            return
        import math
        from datetime import datetime as _dt
        for c in candidates:
            mid = c.start + (c.end - c.start) / 2
            try:
                native_age = (mid - _dt(birth_date.year, birth_date.month,
                                        birth_date.day)).days / 365.25
            except Exception:
                continue
            # Gaussian centered at mode_age. PDF normalized to peak=1.0.
            z = (native_age - mode_age) / sigma
            density = math.exp(-0.5 * z * z)   # 0..1
            # Adjustment: density 1.0 → +max_adjust, density 0 → -max_adjust.
            adj = max_adjust * (2.0 * density - 1.0)
            c.aggregate_confidence = max(
                0.0, min(1.0, c.aggregate_confidence + adj),
            )

    # ── DESCRIPTION flow (spec §10.3) ───────────────────────────────
    def _aggregate_description(
        self, query: FocusQuery, resolved: ResolvedFocus,
        fired_by_school: Dict[School, List[FiredRule]],
        conflict_epsilon: float = 0.15,
    ) -> QueryResult:
        """Group fired rules by consequent attribute; for each attribute
        list values with confidences. When two values for the same attribute
        come from different schools at similar confidence (within
        `conflict_epsilon`), both are tagged as `conflict=True` (CAV-026).
        """
        # Bucket: attribute -> value -> [FiredRule, ...]
        buckets: Dict[str, Dict[str, List[FiredRule]]] = {}
        for fired in fired_by_school.values():
            for fr in fired:
                attrs = fr.rule.consequent.get("attributes") or {}
                for attr_name, val in attrs.items():
                    values = val if isinstance(val, list) else [val]
                    for v in values:
                        (buckets.setdefault(attr_name, {})
                               .setdefault(str(v), []).append(fr))
        attrs_out: List[DescriptiveAttribute] = []
        for attr_name, values in buckets.items():
            # Compute per-value confidence first.
            value_conf = {
                value: (_noisy_or([r.strength for r in rules]), rules)
                for value, rules in values.items()
            }
            # Detect cross-school conflicts: 2+ values for same attribute
            # from different schools within epsilon of top confidence.
            top_conf = max((c for c, _ in value_conf.values()), default=0.0)
            for value, (conf, rules) in value_conf.items():
                schools_for_value = {fr.rule.school for fr in rules}
                # Other values for this attribute from other schools at
                # similar confidence?
                competing_schools = set()
                for other_val, (other_conf, other_rules) in value_conf.items():
                    if other_val == value:
                        continue
                    if abs(conf - other_conf) > conflict_epsilon:
                        continue
                    competing_schools |= {
                        fr.rule.school for fr in other_rules
                    }
                in_conflict = bool(
                    competing_schools - schools_for_value
                )
                attr = DescriptiveAttribute(
                    attribute=attr_name,
                    value=value + (" [CONFLICT]" if in_conflict else ""),
                    confidence=conf,
                    contributing_rules=rules,
                )
                attrs_out.append(attr)
        attrs_out.sort(key=lambda a: -a.confidence)
        return QueryResult(
            query=query, resolved=resolved,
            query_type=QueryType.DESCRIPTION,
            attributes=attrs_out,
            inconclusive_reason=(
                None if attrs_out
                else "no descriptive attributes emitted by rules"
            ),
        )

    # ── MAGNITUDE flow (spec §10.4) ─────────────────────────────────
    def _aggregate_magnitude(
        self, query: FocusQuery, resolved: ResolvedFocus,
        fired_by_school: Dict[School, List[FiredRule]],
    ) -> QueryResult:
        """Weighted signed sum of rule strengths → ordinal bucket.

        CAV-025: bucket cutoffs picked so a single strong (s=0.6) rule
        in either direction lands in 'high'/'low'; two such rules same
        direction land in 'very_high'/'very_low'. Calibration via
        retrodiction is future work.
        """
        pos_total = 0.0
        neg_total = 0.0
        for fired in fired_by_school.values():
            for fr in fired:
                if fr.polarity == "positive":
                    pos_total += fr.strength
                elif fr.polarity == "negative":
                    neg_total += fr.strength
        score = pos_total - neg_total
        # Symmetric thresholds around 0.
        if score >= 1.2:           # ≥ 2 strong positive rules
            ordinal = "very_high"
        elif score >= 0.4:         # ≥ 1 strong positive rule
            ordinal = "high"
        elif score >= -0.4:        # near zero / contested
            ordinal = "moderate"
        elif score >= -1.2:
            ordinal = "low"
        else:
            ordinal = "very_low"
        return QueryResult(
            query=query, resolved=resolved,
            query_type=QueryType.MAGNITUDE,
            magnitude={
                "score": score,
                "positive_sum": pos_total,
                "negative_sum": neg_total,
                "ordinal": ordinal,
            },
            inconclusive_reason=(
                None if (pos_total + neg_total) > 0
                else "no magnitude evidence"
            ),
        )

    # ── YES_NO flow (spec §10.5) ────────────────────────────────────
    def _aggregate_yes_no(
        self, query: FocusQuery, resolved: ResolvedFocus,
        fired_by_school: Dict[School, List[FiredRule]],
        threshold_gap: float = 0.2,
    ) -> QueryResult:
        """pos > neg + gap → yes. neg > pos + gap → no. Otherwise inconclusive."""
        pos = _noisy_or([
            fr.strength for lst in fired_by_school.values()
            for fr in lst if fr.polarity == "positive"
        ])
        neg = _noisy_or([
            fr.strength for lst in fired_by_school.values()
            for fr in lst if fr.polarity == "negative"
        ])
        if pos - neg > threshold_gap:
            answer = "yes"
        elif neg - pos > threshold_gap:
            answer = "no"
        else:
            answer = "inconclusive"
        return QueryResult(
            query=query, resolved=resolved,
            query_type=QueryType.YES_NO,
            yes_no=answer,
            magnitude={"positive_evidence": pos, "negative_evidence": neg},
            inconclusive_reason=(
                None if answer != "inconclusive"
                else f"pos={pos:.2f} neg={neg:.2f} within gap {threshold_gap}"
            ),
        )
