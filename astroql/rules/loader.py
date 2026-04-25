"""Structured rule library loader (spec §6.6, §8.1-8.4).

Loads YAML rule files from `rules/<school>/*.yaml`, validates each against
`features_schema.yaml` (feature-path shape + operator + applicable_to),
and returns typed `Rule` instances filtered by `(school, relationship,
life_area, effect)`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..schemas.enums import Effect, LifeArea, Relationship, School
from ..schemas.focus import ResolvedFocus
from ..schemas.rules import CFModifier, Citation, Provenance, Rule


_CF_EPSILON = 1e-9  # Strict-open interval margin for non-veto base_cf

# 9-planet Parashari set. Kept in sync with engine/shadbala.py.
_PARASHARI_PLANETS = frozenset({
    "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn",
    "Rahu", "Ketu",
})


_RULES_ROOT = Path(__file__).parent


class RuleLoadError(ValueError):
    pass


# Ops the rule engine will support. Kept here because the loader validates
# rules against the allowed op set.
_VALID_OPS = {
    "eq", "neq", "in", "not_in", "gt", "lt", "gte", "lte",
    "contains", "contains_any", "expr",
}


def _canonicalize_feature_path(path: str) -> str:
    """Replace concrete planet names in a feature path with the `<planet>`
    placeholder so schema lookup can match wildcard entries.

    Example: 'karaka_data.Sun.dignity' -> 'karaka_data.<planet>.dignity'
             'varga_features.D9.saturn_house' -> 'varga_features.<varga>.<planet-lc>_house'
    """
    # varga_features.<Dx>.<planet-lc>_<attr>  -> canonical form
    m = re.match(
        r"^varga_features\.(D\d+)\.(target_[a-z_]+)$", path,
    )
    if m:
        return f"varga_features.<varga>.{m.group(2)}"
    m = re.match(
        r"^varga_features\.(D\d+)\.([a-z]+)_(sign|house|dignity)$", path,
    )
    if m:
        return f"varga_features.<varga>.<planet-lc>_{m.group(3)}"
    # karaka_data.<Planet>.<attr>
    m = re.match(r"^karaka_data\.([A-Z][a-z]+)\.([a-z_]+)$", path)
    if m:
        return f"karaka_data.<planet>.{m.group(2)}"
    return path


def _load_schema() -> Dict[str, Any]:
    path = _RULES_ROOT / "features_schema.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_antecedent(
    clause: Dict[str, Any], schema: Dict[str, Any],
) -> None:
    if "feature" not in clause:
        raise RuleLoadError(f"antecedent clause missing 'feature': {clause}")
    if "op" not in clause:
        raise RuleLoadError(f"antecedent clause missing 'op': {clause}")
    if clause["op"] not in _VALID_OPS:
        raise RuleLoadError(
            f"unknown op {clause['op']!r} "
            f"(allowed: {sorted(_VALID_OPS)})"
        )
    if "value" not in clause and "value_expr" not in clause:
        raise RuleLoadError(
            f"antecedent clause needs 'value' or 'value_expr': {clause}"
        )
    # Feature-path schema check.
    path = clause["feature"]
    canon = _canonicalize_feature_path(path)
    known_static = set(schema.get("static", {}).keys())
    known_candidate = set(schema.get("per_candidate", {}).keys())
    known_kp_static = set(schema.get("kp_static", {}).keys())
    known_jaimini_static = set(schema.get("jaimini_static", {}).keys())
    known_chart_app = set(schema.get("chart_applicability", {}).keys())
    # Canonicalize karaka_code paths (AK/AmK/BK/MK/PuK/GK/DK) similarly.
    canon_jaimini = path
    import re as _re
    m = _re.match(
        r"^karaka_data\.(AK|AmK|BK|MK|PuK|GK|DK)\.([a-z_]+)$", path,
    )
    if m:
        canon_jaimini = f"karaka_data.<karaka_code>.{m.group(2)}"
    for known in (known_static, known_candidate, known_kp_static,
                  known_jaimini_static, known_chart_app):
        if (canon in known or path in known or canon_jaimini in known):
            return
    raise RuleLoadError(
        f"unknown feature path {path!r} (canonical: {canon!r}). "
        f"Add it to features_schema.yaml before referencing in rules."
    )


def _validate_rule(
    raw: Dict[str, Any], schema: Dict[str, Any],
) -> Rule:
    # Three valid rule forms (2026-04-24):
    #   legacy:    antecedent + consequent (fixed polarity + strength)
    #   graded:    required + factors + base_strength (weighted fraction)
    #   cf_native: antecedent + base_cf (neuro-symbolic CF schema)
    is_graded = bool(raw.get("factors"))
    is_cf_native = (
        raw.get("base_cf") is not None or bool(raw.get("is_veto"))
    )
    base_required = ("rule_id", "school", "source")
    for f in base_required:
        if f not in raw:
            raise RuleLoadError(
                f"rule missing required field {f!r}: "
                f"{raw.get('rule_id', raw)}"
            )
    if is_graded:
        for f in ("factors", "base_strength"):
            if f not in raw:
                raise RuleLoadError(
                    f"graded rule {raw['rule_id']} missing field {f!r}"
                )
    elif is_cf_native:
        if "antecedent" not in raw:
            raise RuleLoadError(
                f"cf_native rule {raw['rule_id']} missing 'antecedent'"
            )
    else:
        for f in ("antecedent", "consequent"):
            if f not in raw:
                raise RuleLoadError(
                    f"legacy rule {raw['rule_id']} missing field {f!r}"
                )

    try:
        school = School(raw["school"])
    except ValueError as e:
        raise RuleLoadError(
            f"rule {raw['rule_id']} has unknown school {raw['school']!r}"
        ) from e

    antecedent = raw.get("antecedent") or []
    if not isinstance(antecedent, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} antecedent must be a list"
        )
    for clause in antecedent:
        _validate_antecedent(clause, schema)

    # Graded-rule validation: required + factors must each be lists of
    # standard antecedent clauses; weights must be positive floats.
    required_clauses = raw.get("required") or []
    factors = raw.get("factors") or []
    if not isinstance(required_clauses, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} 'required' must be a list"
        )
    if not isinstance(factors, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} 'factors' must be a list"
        )
    for clause in required_clauses:
        _validate_antecedent(clause, schema)
    for clause in factors:
        _validate_antecedent(clause, schema)
        w = clause.get("weight", 1.0)
        try:
            w = float(w)
        except (TypeError, ValueError) as e:
            raise RuleLoadError(
                f"rule {raw['rule_id']}: factor weight must be float, "
                f"got {clause.get('weight')!r}"
            ) from e
        if w <= 0:
            raise RuleLoadError(
                f"rule {raw['rule_id']}: factor weight must be > 0"
            )

    applicable = raw.get("applicable_to", {}) or {}
    for enum_cls, key in (
        (Relationship, "relationships"),
        (LifeArea, "life_areas"),
        (Effect, "effects"),
    ):
        for val in applicable.get(key, []) or []:
            try:
                enum_cls(val)
            except ValueError as e:
                raise RuleLoadError(
                    f"rule {raw['rule_id']} applicable_to.{key} has "
                    f"unknown value {val!r}"
                ) from e

    tier_raw = raw.get("priority_tier", 2)
    try:
        priority_tier = int(tier_raw)
    except (TypeError, ValueError) as e:
        raise RuleLoadError(
            f"rule {raw['rule_id']} priority_tier must be int 1-3"
        ) from e
    if priority_tier not in (1, 2, 3):
        raise RuleLoadError(
            f"rule {raw['rule_id']} priority_tier={priority_tier} "
            f"must be 1 (primary), 2 (secondary), or 3 (circumstantial)"
        )
    scoring = raw.get("scoring", "weighted_fraction")
    if scoring not in ("weighted_fraction", "linear_count"):
        raise RuleLoadError(
            f"rule {raw['rule_id']} scoring must be one of "
            f"weighted_fraction|linear_count, got {scoring!r}"
        )

    # v38: chart_applicability clauses (optional). Validated same as
    # antecedent — feature/op/value shape, features must be in schema.
    applicable_when = raw.get("applicable_when") or []
    if not isinstance(applicable_when, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} applicable_when must be a list"
        )
    for clause in applicable_when:
        _validate_antecedent(clause, schema)

    # ── CF neuro-symbolic fields (v1) ────────────────────────────────
    is_veto = bool(raw.get("is_veto", False))
    base_cf_raw = raw.get("base_cf")
    if base_cf_raw is not None:
        try:
            base_cf: Optional[float] = float(base_cf_raw)
        except (TypeError, ValueError) as e:
            raise RuleLoadError(
                f"rule {raw['rule_id']}: base_cf must be float, "
                f"got {base_cf_raw!r}"
            ) from e
    else:
        base_cf = None
    if is_veto:
        if base_cf is None or base_cf not in (-1.0, 1.0):
            raise RuleLoadError(
                f"rule {raw['rule_id']}: is_veto=true requires "
                f"base_cf == -1.0 or +1.0 explicitly, got {base_cf!r}"
            )
    elif base_cf is not None:
        # Non-veto strict-open invariant: -1 < base_cf < 1
        if not (-1.0 + _CF_EPSILON <= base_cf <= 1.0 - _CF_EPSILON):
            raise RuleLoadError(
                f"rule {raw['rule_id']}: non-veto base_cf={base_cf} "
                f"must be in open interval (-1, 1). Use is_veto=true "
                f"for ±1.0 absolute certainty."
            )
        if base_cf == 0.0:
            raise RuleLoadError(
                f"rule {raw['rule_id']}: base_cf=0.0 is a no-op rule"
            )

    subsumes_raw = raw.get("subsumes_rules") or []
    if not isinstance(subsumes_raw, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} subsumes_rules must be a list"
        )
    for rid in subsumes_raw:
        if not isinstance(rid, str):
            raise RuleLoadError(
                f"rule {raw['rule_id']} subsumes_rules entry must be "
                f"str rule_id, got {rid!r}"
            )
    subsumes_rules = list(subsumes_raw)

    modifiers_raw = raw.get("modifiers") or []
    if not isinstance(modifiers_raw, list):
        raise RuleLoadError(
            f"rule {raw['rule_id']} modifiers must be a list"
        )
    if is_veto and modifiers_raw:
        # Vetoes short-circuit to ±1.0 before aggregation; modifiers
        # would never fire. Reject to prevent silent dead data.
        raise RuleLoadError(
            f"rule {raw['rule_id']}: veto rules cannot carry modifiers "
            f"(modifiers are inert since vetoes short-circuit to ±1.0). "
            f"To make the effect conditional, gate it via antecedent or "
            f"applicable_when instead."
        )
    modifiers: List[CFModifier] = []
    for i, m in enumerate(modifiers_raw):
        if not isinstance(m, dict):
            raise RuleLoadError(
                f"rule {raw['rule_id']} modifier[{i}] must be a dict"
            )
        cond = m.get("condition")
        if not isinstance(cond, dict):
            raise RuleLoadError(
                f"rule {raw['rule_id']} modifier[{i}] needs "
                f"'condition' dict"
            )
        _validate_antecedent(cond, schema)
        if "effect_cf" not in m:
            raise RuleLoadError(
                f"rule {raw['rule_id']} modifier[{i}] missing effect_cf"
            )
        try:
            effect_cf = float(m["effect_cf"])
        except (TypeError, ValueError) as e:
            raise RuleLoadError(
                f"rule {raw['rule_id']} modifier[{i}] effect_cf must "
                f"be float, got {m.get('effect_cf')!r}"
            ) from e
        if not (-1.0 + _CF_EPSILON <= effect_cf <= 1.0 - _CF_EPSILON):
            raise RuleLoadError(
                f"rule {raw['rule_id']} modifier[{i}] effect_cf="
                f"{effect_cf} must be in open interval (-1, 1)"
            )
        modifiers.append(CFModifier(
            condition=cond,
            effect_cf=effect_cf,
            explanation=str(m.get("explanation", "")),
        ))

    primary_planet = raw.get("primary_planet")
    if primary_planet is not None:
        if primary_planet not in _PARASHARI_PLANETS:
            raise RuleLoadError(
                f"rule {raw['rule_id']}: primary_planet="
                f"{primary_planet!r} must be one of "
                f"{sorted(_PARASHARI_PLANETS)}"
            )
    elif is_cf_native and not is_veto:
        raise RuleLoadError(
            f"rule {raw['rule_id']}: CF-native non-veto rule must "
            f"declare primary_planet (the planet whose shadbala μ "
            f"modulates base_cf). Use is_veto=true if the rule is "
            f"absolute regardless of planetary strength."
        )

    provenance: Optional[Provenance] = None
    prov_raw = raw.get("provenance")
    if prov_raw is not None:
        if not isinstance(prov_raw, dict):
            raise RuleLoadError(
                f"rule {raw['rule_id']} provenance must be a dict"
            )
        author = str(prov_raw.get("author", "human"))
        if author not in ("human", "llm_critic"):
            raise RuleLoadError(
                f"rule {raw['rule_id']} provenance.author must be "
                f"'human' or 'llm_critic', got {author!r}"
            )
        try:
            prov_conf = float(prov_raw.get("confidence", 0.6))
        except (TypeError, ValueError) as e:
            raise RuleLoadError(
                f"rule {raw['rule_id']} provenance.confidence must "
                f"be float"
            ) from e
        if not (0.0 <= prov_conf <= 1.0):
            raise RuleLoadError(
                f"rule {raw['rule_id']} provenance.confidence="
                f"{prov_conf} must be in [0, 1]"
            )
        citations_raw = prov_raw.get("citations") or []
        if not isinstance(citations_raw, list):
            raise RuleLoadError(
                f"rule {raw['rule_id']} provenance.citations must be "
                f"a list"
            )
        citations: List[Citation] = []
        for j, c in enumerate(citations_raw):
            if not isinstance(c, dict):
                raise RuleLoadError(
                    f"rule {raw['rule_id']} provenance.citations[{j}] "
                    f"must be a dict"
                )
            sid = c.get("source_id")
            txt = c.get("text_chunk")
            if not isinstance(sid, str) or not sid:
                raise RuleLoadError(
                    f"rule {raw['rule_id']} provenance.citations[{j}]"
                    f" missing non-empty source_id"
                )
            if not isinstance(txt, str) or not txt:
                raise RuleLoadError(
                    f"rule {raw['rule_id']} provenance.citations[{j}]"
                    f" missing non-empty text_chunk"
                )
            citations.append(Citation(source_id=sid, text_chunk=txt))
        provenance = Provenance(
            author=author, confidence=prov_conf, citations=citations,
        )

    return Rule(
        rule_id=raw["rule_id"],
        school=school,
        source=raw["source"],
        source_uri=raw.get("source_uri"),
        rule_type=raw.get("rule_type", "static"),
        applicable_to=applicable,
        antecedent=antecedent,
        consequent=raw.get("consequent") or {},
        confidence=float(raw.get("confidence", 0.6)),
        priority_tier=priority_tier,
        tags=list(raw.get("tags", []) or []),
        notes=raw.get("notes", "") or "",
        required=required_clauses,
        factors=factors,
        min_factors=int(raw.get("min_factors", 1)),
        base_strength=float(raw.get("base_strength", 0.0)),
        scoring=scoring,
        applicable_when=applicable_when,
        base_cf=base_cf,
        is_veto=is_veto,
        subsumes_rules=subsumes_rules,
        modifiers=modifiers,
        provenance=provenance,
        primary_planet=primary_planet,
    )


def validate_yoga_bhanga(rules: List[Rule]) -> None:
    """Yoga-bhanga (§3.B step 3): only a veto can subsume a veto.

    Call after loading all rules across all files, since subsumption
    targets may live in a different file. Raises RuleLoadError on
    violation.
    """
    by_id = {r.rule_id: r for r in rules}
    for r in rules:
        for target_id in r.subsumes_rules:
            target = by_id.get(target_id)
            if target is None:
                continue  # Forward reference to a rule not yet loaded
                # or cross-tradition reference; let the engine report
                # missing targets at evaluation time rather than fail
                # the whole load.
            if target.is_veto and not r.is_veto:
                raise RuleLoadError(
                    f"yoga-bhanga violation: non-veto rule "
                    f"{r.rule_id!r} cannot subsume veto rule "
                    f"{target_id!r}. Only another veto (yoga-bhanga) "
                    f"may neutralize a veto."
                )


class StructuredRuleLibrary:
    """Loads + filters curated rules per spec §6.6, §8."""

    def __init__(self, rules_root: Optional[Path] = None) -> None:
        self._root = rules_root or _RULES_ROOT
        self._schema = _load_schema()
        # Cache: {(school, yaml_path) -> list[Rule]}
        self._cache: Dict[tuple, List[Rule]] = {}

    def _load_file(self, school: School, path: Path) -> List[Rule]:
        key = (school.value, str(path))
        if key in self._cache:
            return self._cache[key]
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or []
        if not isinstance(raw, list):
            raise RuleLoadError(
                f"{path}: expected a YAML list of rules, got {type(raw)}"
            )
        rules = [_validate_rule(r, self._schema) for r in raw]
        # Hard-check the school matches the directory
        for r in rules:
            if r.school != school:
                raise RuleLoadError(
                    f"{r.rule_id}: declares school={r.school} but lives "
                    f"under {school.value}/"
                )
        self._cache[key] = rules
        return rules

    def _load_all_for_school(self, school: School) -> List[Rule]:
        school_dir = self._root / school.value
        if not school_dir.exists():
            return []
        rules: List[Rule] = []
        for yf in sorted(school_dir.glob("*.yaml")):
            rules.extend(self._load_file(school, yf))
        validate_yoga_bhanga(rules)
        return rules

    def load_rules(
        self, school: School, focus: ResolvedFocus,
    ) -> List[Rule]:
        """Return rules whose applicable_to matches the resolved focus."""
        all_rules = self._load_all_for_school(school)
        rel = focus.query.relationship.value
        life = focus.query.life_area.value
        eff = focus.query.effect.value
        out: List[Rule] = []
        for r in all_rules:
            app = r.applicable_to
            if app.get("relationships") and rel not in app["relationships"]:
                continue
            if app.get("life_areas") and life not in app["life_areas"]:
                continue
            if app.get("effects") and eff not in app["effects"]:
                continue
            out.append(r)
        return out

    def all_rules(self, school: School) -> List[Rule]:
        """Escape hatch for introspection (EXPLAIN, tests)."""
        return list(self._load_all_for_school(school))
