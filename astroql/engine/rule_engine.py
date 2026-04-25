"""Rule application engine (spec §6.7).

Forward-chains structured rules against a FeatureBundle. For TIMING rules,
iterates over `feature_bundle.dasha_candidates` binding each as `candidate`
in the evaluation context. Returns `FiredRule` instances with polarity,
strength, optional window, and evidence excerpt.

Phase 9 additions:
    - rule.priority_tier influences post-aggregation weighting (engine
      itself just stores it on FiredRule via the rule reference).
    - Calibrated per-rule strength multipliers loaded from
      rules/calibrated_strengths.yaml when present (CAV-034).
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..schemas.features import FeatureBundle, Passage
from ..schemas.rules import FiredRule, Rule


_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "rules" / "calibrated_strengths.yaml"
)


def _load_calibration() -> Dict[str, float]:
    """Load per-rule strength multipliers if calibrated_strengths.yaml exists.

    File format (written by benchmark/calibrate.py):
        rules:
          parashari.longevity.maraka_dasha.01:
            hit_rate: 0.0012
            strength_adjustment: -0.0244

    Returns {rule_id: multiplier} where multiplier maps adjustment to
    a positive scale: hit_rate=0.05 (prior) → multiplier 1.0,
    hit_rate=0.0 → multiplier 0.5, hit_rate=0.1 → multiplier 1.5.
    """
    if not _CALIBRATION_PATH.exists():
        return {}
    try:
        with open(_CALIBRATION_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    rules = data.get("rules", {}) or {}
    meta = data.get("meta", {}) or {}
    prior = float(meta.get("prior_hit_rate", 0.05))
    out: Dict[str, float] = {}
    for rid, row in rules.items():
        hit_rate = float(row.get("hit_rate", prior))
        # Linear map: hit_rate / prior → multiplier, clamped [0.3, 2.0].
        if prior <= 0:
            out[rid] = 1.0
        else:
            mult = hit_rate / prior
            out[rid] = max(0.3, min(2.0, mult))
    return out


class RuleEngineError(RuntimeError):
    pass


# ── Feature-path resolution ──────────────────────────────────────────

def _resolve_path(path: str, bundle: FeatureBundle,
                  candidate: Optional[Dict[str, Any]] = None,
                  ) -> Tuple[bool, Any]:
    """Walk a dotted path into the FeatureBundle.

    Returns (found, value). `found=False` means the path isn't present
    (e.g. karaka data for a planet that's missing from the chart). In
    that case, the antecedent clause is considered unsatisfied.
    """
    parts = path.split(".")
    # candidate.* paths
    if parts[0] == "candidate":
        if candidate is None:
            return (False, None)
        node: Any = candidate
        for p in parts[1:]:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                return (False, None)
        return (True, node)

    # Static paths — walk FeatureBundle as nested dict.
    # Map top-level to the corresponding FeatureBundle field.
    field_map = {
        "primary_house_data": bundle.primary_house_data,
        "karaka_data": bundle.karaka_data,
        "varga_features": bundle.varga_features,
        # Jaimini / KP school-specific payloads (None when not loaded).
        "jaimini_features": bundle.jaimini_features or {},
        "kp_features": bundle.kp_features or {},
        # v38: chart-level applicability (Laghu Parashari roles etc).
        "chart_applicability": bundle.chart_applicability or {},
    }
    if parts[0] not in field_map:
        return (False, None)
    node = field_map[parts[0]]
    for p in parts[1:]:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            return (False, None)
    return (True, node)


# ── Operator evaluation ──────────────────────────────────────────────

def _op_eq(a, b) -> bool:
    return a == b


def _op_in(a, b) -> bool:
    if not isinstance(b, (list, tuple, set)):
        return False
    return a in b


def _op_contains_any(a, b) -> bool:
    """True iff collection `a` contains at least one element of `b`."""
    if not isinstance(a, (list, tuple, set)):
        return False
    if not isinstance(b, (list, tuple, set)):
        return a_contains_scalar(a, b)
    return any(x in a for x in b)


def a_contains_scalar(collection, scalar) -> bool:
    return scalar in collection


def _op_contains(a, b) -> bool:
    if not isinstance(a, (list, tuple, set, str)):
        return False
    return b in a


def _apply_op(
    op: str, feature_val: Any, value: Any, bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]] = None,
) -> bool:
    if op == "eq":
        return _op_eq(feature_val, value)
    if op == "neq":
        return not _op_eq(feature_val, value)
    if op == "in":
        return _op_in(feature_val, value)
    if op == "not_in":
        return not _op_in(feature_val, value)
    if op == "gt":
        try:
            return float(feature_val) > float(value)
        except (TypeError, ValueError):
            return False
    if op == "lt":
        try:
            return float(feature_val) < float(value)
        except (TypeError, ValueError):
            return False
    if op == "gte":
        try:
            return float(feature_val) >= float(value)
        except (TypeError, ValueError):
            return False
    if op == "lte":
        try:
            return float(feature_val) <= float(value)
        except (TypeError, ValueError):
            return False
    if op == "contains":
        return _op_contains(feature_val, value)
    if op == "contains_any":
        return _op_contains_any(feature_val, value)
    if op == "expr":
        # Restricted expression evaluator with classical helpers (CAV-006).
        try:
            return bool(_safe_eval(str(value), bundle, candidate, feature_val))
        except Exception:
            return False
    raise RuleEngineError(f"unknown op: {op}")


# ── Safe expression evaluator (CAV-006) ──────────────────────────────

def _classical_helpers(bundle: FeatureBundle):
    """Build a dict of classical helper functions usable inside `expr`.

    Available functions in the eval context:
        lord_of(house_num: int) -> str        — sign-lord of natal house
        marakas_of(target_house: int) -> list — [2L_from_target, 7L_from_target]
        eighth_from(target: int) -> int       — 8th-from-target house num
        sign_of_house(h: int) -> str          — sign in D1 house h
    """
    from ..features.classical import SIGN_LORD, house_from, maraka_houses

    # Pull house_signs from D1 if available via primary_house_data.
    # The bundle doesn't carry the full Chart, so we work off
    # primary_house_data['rotated' / 'direct'].sign and the focus.
    rotated = bundle.primary_house_data.get("rotated", {})
    direct = bundle.primary_house_data.get("direct", {})
    target_rotated = rotated.get("house") or 0

    house_to_sign: Dict[int, str] = {}
    if rotated.get("house") and rotated.get("sign"):
        house_to_sign[rotated["house"]] = rotated["sign"]
    if direct.get("house") and direct.get("sign"):
        house_to_sign[direct["house"]] = direct["sign"]

    def lord_of(h: int) -> str:
        sign = house_to_sign.get(h)
        if not sign:
            raise ValueError(
                f"lord_of({h}) not available — only target houses cached"
            )
        return SIGN_LORD[sign]

    def marakas_of(h: int) -> list:
        return [maraka_houses(h)[i] for i in range(2)]

    def eighth_from(h: int) -> int:
        return house_from(h, 8)

    def sign_of_house(h: int) -> str:
        return house_to_sign.get(h, "")

    return {
        "lord_of": lord_of,
        "marakas_of": marakas_of,
        "eighth_from": eighth_from,
        "sign_of_house": sign_of_house,
    }


def _safe_eval(
    expr: str, bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]],
    feature_val: Any,
):
    """Evaluate an expression in a sandboxed namespace.

    Allowed builtins: `min`, `max`, `len`, `set`, `list`, `tuple`, `int`,
    `bool`, `abs`, `sum`. Plus classical helpers from `_classical_helpers`.
    """
    safe_builtins = {
        "min": min, "max": max, "len": len, "set": set, "list": list,
        "tuple": tuple, "int": int, "bool": bool, "abs": abs, "sum": sum,
        "True": True, "False": False, "None": None,
    }
    helpers = _classical_helpers(bundle)
    namespace = {
        **safe_builtins,
        **helpers,
        "feature": feature_val,
        "candidate": candidate or {},
        "primary": bundle.primary_house_data,
        "karakas": bundle.karaka_data,
        "varga": bundle.varga_features,
    }
    return eval(expr, {"__builtins__": {}}, namespace)


def _resolve_value(
    clause: Dict[str, Any], bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]],
) -> Tuple[bool, Any]:
    """Return the literal value the clause compares against.

    Supports `value` (literal) or `value_expr` (resolved as another
    feature path). For Phase 1 we only support one path-to-path comparison
    (e.g. Sun.house == Saturn.house) — no arithmetic yet.
    """
    if "value" in clause:
        return (True, clause["value"])
    expr = clause.get("value_expr")
    if isinstance(expr, str) and expr.count(".") >= 1 and " " not in expr:
        # Treat as a feature path.
        return _resolve_path(expr, bundle, candidate)
    # More complex expressions not yet supported.
    return (False, None)


# ── Evaluation ───────────────────────────────────────────────────────

def _evaluate_antecedent(
    antecedent: List[Dict[str, Any]],
    bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate all clauses. Returns (matched, bindings)."""
    bindings: Dict[str, Any] = {}
    for clause in antecedent:
        path = clause["feature"]
        op = clause["op"]
        found_feature, feature_val = _resolve_path(path, bundle, candidate)
        if not found_feature:
            return (False, bindings)
        found_val, value = _resolve_value(clause, bundle, candidate)
        if not found_val:
            return (False, bindings)
        if not _apply_op(op, feature_val, value, bundle, candidate):
            return (False, bindings)
        bindings[path] = feature_val
    return (True, bindings)


def _is_timing_rule(rule: Rule) -> bool:
    if rule.rule_type in {"dasha_timing", "transit_timing"}:
        return True
    # A rule that references candidate.* is a timing rule even if it
    # doesn't say so in rule_type — keep tolerant of authoring mistakes.
    all_clauses = (
        list(rule.antecedent)
        + list(rule.required)
        + list(rule.factors)
    )
    for clause in all_clauses:
        if clause.get("feature", "").startswith("candidate."):
            return True
    return False


def _is_graded(rule: Rule) -> bool:
    return bool(rule.factors)


def _evaluate_graded(
    rule: Rule, bundle: FeatureBundle,
    candidate: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, int, int, Dict[str, Any]]:
    """Evaluate a graded rule against a window.

    Returns (fired, strength, n_matched, n_total, bindings).
    Steps:
      1. Required clauses are a hard gate — all must match.
      2. For each factor, evaluate; sum its weight if matched.
      3. If matched count < rule.min_factors, the rule does NOT fire.
      4. Strength = base_strength * (sum_matched / sum_total) for
         scoring='weighted_fraction', or
         base_strength * (n_matched / n_total) for 'linear_count'.
    """
    bindings: Dict[str, Any] = {}

    # Required gate — same semantics as legacy antecedent.
    for clause in rule.required:
        path = clause["feature"]
        op = clause["op"]
        found, fv = _resolve_path(path, bundle, candidate)
        if not found:
            return (False, 0.0, 0, len(rule.factors), bindings)
        found_v, value = _resolve_value(clause, bundle, candidate)
        if not found_v:
            return (False, 0.0, 0, len(rule.factors), bindings)
        if not _apply_op(op, fv, value, bundle, candidate):
            return (False, 0.0, 0, len(rule.factors), bindings)
        bindings[path] = fv

    # Factor evaluation.
    sum_matched = 0.0
    sum_total = 0.0
    n_matched = 0
    n_total = len(rule.factors)
    matched_keys: List[str] = []
    for clause in rule.factors:
        weight = float(clause.get("weight", 1.0))
        sum_total += weight
        path = clause["feature"]
        op = clause["op"]
        found, fv = _resolve_path(path, bundle, candidate)
        if not found:
            continue
        found_v, value = _resolve_value(clause, bundle, candidate)
        if not found_v:
            continue
        if _apply_op(op, fv, value, bundle, candidate):
            sum_matched += weight
            n_matched += 1
            matched_keys.append(path)
            bindings[path] = fv

    if n_matched < int(rule.min_factors):
        return (False, 0.0, n_matched, n_total, bindings)

    if rule.scoring == "linear_count":
        frac = (n_matched / n_total) if n_total else 0.0
    else:  # weighted_fraction (default)
        frac = (sum_matched / sum_total) if sum_total else 0.0
    strength = max(0.0, min(1.0, float(rule.base_strength) * frac))
    bindings["_graded_n_matched"] = n_matched
    bindings["_graded_n_total"] = n_total
    bindings["_graded_matched_keys"] = matched_keys
    return (True, strength, n_matched, n_total, bindings)


def _parse_iso(s: str) -> _dt.datetime:
    # Feature extractor emits ISO-format strings for start/end.
    return _dt.datetime.fromisoformat(s)


def _window_from_candidate(candidate: Dict[str, Any]) -> Optional[tuple]:
    s = candidate.get("start")
    e = candidate.get("end")
    if not s or not e:
        return None
    try:
        return (_parse_iso(s), _parse_iso(e))
    except ValueError:
        return None


class RuleEngine:
    """Forward-chain structured rules against a FeatureBundle.

    For static rules: evaluate antecedent once, fire once.
    For timing rules (reference candidate.*): evaluate per candidate,
        fire once per matching candidate (window recorded).
    """

    def __init__(self):
        self._calibration = _load_calibration()

    def apply(
        self,
        features: FeatureBundle,
        structured_rules: List[Rule],
        rag_passages: Optional[List[Passage]] = None,
    ) -> List[FiredRule]:
        fired: List[FiredRule] = []
        fired.extend(self._apply_structured(features, structured_rules))
        if rag_passages:
            fired.extend(self._apply_rag_passages(features, rag_passages))
        # CAV-034: per-rule calibration multiplier from retrodiction data.
        if self._calibration:
            for fr in fired:
                mult = self._calibration.get(fr.rule.rule_id, 1.0)
                fr.strength = max(0.0, min(1.0, fr.strength * mult))
        # Spec §12.2: dampen strengths when birth time is approximate.
        accuracy = (
            features.focus.query.birth.time_accuracy
            if features.focus.query.birth else "exact"
        )
        damp = {"exact": 1.0, "approximate": 0.9, "unknown": 0.7}.get(
            accuracy, 1.0,
        )
        if damp < 1.0:
            for fr in fired:
                fr.strength = max(0.0, min(1.0, fr.strength * damp))
        return fired

    def _apply_structured(
        self, features: FeatureBundle, rules: List[Rule],
    ) -> List[FiredRule]:
        out: List[FiredRule] = []
        for rule in rules:
            # v38: chart-applicability gate — evaluated ONCE per (rule,
            # chart). If any clause fails, skip this rule entirely (it
            # contributes to NO windows). Clauses evaluate against the
            # chart-static bundle (no candidate binding).
            if rule.applicable_when:
                matched, _ = _evaluate_antecedent(
                    rule.applicable_when, features,
                )
                if not matched:
                    continue
            graded = _is_graded(rule)
            if _is_timing_rule(rule):
                for cand in features.dasha_candidates or []:
                    if graded:
                        fired, strength, _nm, _nt, bindings = (
                            _evaluate_graded(rule, features, candidate=cand)
                        )
                        if not fired:
                            continue
                    else:
                        matched, bindings = _evaluate_antecedent(
                            rule.antecedent, features, candidate=cand,
                        )
                        if not matched:
                            continue
                        strength = float(
                            rule.consequent.get(
                                "strength", rule.confidence,
                            )
                        )
                    window = _window_from_candidate(cand)
                    out.append(FiredRule(
                        rule=rule,
                        bindings={
                            **bindings,
                            "candidate": {
                                k: cand.get(k) for k in
                                ("md", "ad", "pad", "matched_lords",
                                 "level", "reason")
                            },
                        },
                        polarity=rule.consequent.get("polarity", "neutral"),
                        strength=strength,
                        window=window,
                        evidence_excerpt=rule.source,
                    ))
            else:
                if graded:
                    fired, strength, _nm, _nt, bindings = (
                        _evaluate_graded(rule, features)
                    )
                    if not fired:
                        continue
                else:
                    matched, bindings = _evaluate_antecedent(
                        rule.antecedent, features,
                    )
                    if not matched:
                        continue
                    strength = float(
                        rule.consequent.get("strength", rule.confidence)
                    )
                out.append(FiredRule(
                    rule=rule,
                    bindings=bindings,
                    polarity=rule.consequent.get("polarity", "neutral"),
                    strength=strength,
                    window=None,
                    evidence_excerpt=rule.source,
                ))
        return out

    def _apply_rag_passages(
        self, features: FeatureBundle, passages: List[Passage],
    ) -> List[FiredRule]:
        """Synthesize FiredRules from retrieved passages (spec §6.7 step 2).

        Phase 3 simplification: heuristic synthesis without an LLM call.
        Each passage becomes a `rule_type: unstructured` FiredRule with
        strength scaled from its retrieval score. This preserves
        citation-to-evidence traceability (spec §13.4 invariant) even
        when no LLM is configured. Proper LLM-based rule extraction is
        CAV-019 (see CAVEATS.md).
        """
        from ..schemas.rules import Rule
        out: List[FiredRule] = []
        polarity_by_effect = {
            "event_negative": "negative",
            "event_positive": "positive",
        }
        pol = polarity_by_effect.get(
            features.focus.query.effect.value, "neutral",
        )
        for p in passages:
            # Retrieval score: lower distance = more relevant. astro-prod
            # filters below _MAX_DISTANCE=0.36; map [0.0, 0.36] → [0.45, 0.25].
            score = max(0.0, min(1.0, p.score))
            strength = max(0.2, min(0.5, 0.5 - 0.7 * score))
            synth = Rule(
                rule_id=f"rag.{features.school.value}.{p.passage_id}",
                school=features.school,
                source=p.source,
                source_uri=None,
                rule_type="unstructured",
                applicable_to={
                    "relationships": [
                        features.focus.query.relationship.value,
                    ],
                    "life_areas": [features.focus.query.life_area.value],
                    "effects": [features.focus.query.effect.value],
                },
                antecedent=[],
                consequent={"polarity": pol, "strength": strength},
                confidence=strength,
                tags=["rag", "synthesized"],
                notes=p.text[:400],
            )
            out.append(FiredRule(
                rule=synth,
                bindings={"passage_score": p.score},
                polarity=pol,
                strength=strength,
                window=None,
                evidence_excerpt=p.text[:200],
            ))
        return out
