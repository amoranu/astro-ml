"""Per-tradition query plan schemas + YAML loader (Phase F.1).

A QueryPlan is a tradition-specific, declarative description of how to
score candidate windows for a (relationship, life_area, effect) goal.
It is composed of:

  - reasoning_lines : independent argument lines, each with supporting
                      yogas (rules with negative polarity), attacking
                      yogas (positive polarity protective rules), and
                      optional graded chart-fact clauses.
  - aggregation     : RRF (reciprocal rank fusion) by default, with
                      per-line weights.
  - exceptions      : list of ExceptionRule ids (loaded from
                      `query_plans/<school>/exceptions.yaml`) that
                      can dampen or amplify line/yoga strengths.

YAML on disk lives at:
    astroql/query_plans/parashari/longevity.yaml
    astroql/query_plans/parashari/exceptions.yaml
    astroql/query_plans/jaimini/longevity.yaml
    astroql/query_plans/jaimini/exceptions.yaml
    astroql/query_plans/kp/longevity.yaml
    astroql/query_plans/kp/exceptions.yaml

Each plan executes in isolation per school (Phase F goal: no cross-
tradition fusion). The planner runner (Phase F.5) loads the plan, runs
each line via the within-line scorer (Phase F.2) + exception evaluator
(Phase F.3), and fuses lines via RRF (Phase F.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..schemas.enums import School


_PLANS_ROOT = Path(__file__).resolve().parents[1] / "query_plans"


class PlanLoadError(ValueError):
    pass


# ── Schemas ──────────────────────────────────────────────────────────

@dataclass
class ReasoningLine:
    """One independent argument inside a tradition's plan.

    A line has two evidence pools:
      - supporting_yogas : rule_ids whose firings count as support
                           (treated as event-supporting regardless of
                           the rule's natural polarity — the line is
                           the orientation, not the rule)
      - attacking_yogas  : rule_ids whose firings count as attacks
                           (benefic / protective evidence; reduces net)

    For chart-fact-based lines (no full graded yoga authored yet), the
    line carries supporting_facts / attacking_facts: lists of antecedent
    clauses each with an optional `weight`. The within-line scorer
    evaluates them as a virtual graded rule.

    `alpha` controls how much attack noisy-OR subtracts from support
    noisy-OR: net = support * (1 - alpha * attack). Default 0.5.
    """
    line_id: str
    description: str = ""
    supporting_yogas: List[str] = field(default_factory=list)
    attacking_yogas: List[str] = field(default_factory=list)
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    attacking_facts: List[Dict[str, Any]] = field(default_factory=list)
    alpha: float = 0.5
    requires_dasha: Optional[str] = None     # e.g. 'chara' for Jaimini lines
    base_strength: float = 1.0               # multiplier for fact-based scoring
    min_factors: int = 1
    # Cap the number of yoga firings (per pool) noisy-OR'd together for
    # this line. 0 = no cap. Set this when a line has many supporting
    # yogas that would saturate noisy-OR at 1.0 across most candidates
    # (e.g. dasha_chain_strength with 17 yogas → top-3 keeps the metric
    # discriminative).
    max_yogas_per_window: int = 0
    notes: str = ""


@dataclass
class ExceptionRule:
    """A classical override (e.g. Neecha Bhanga, Vipreet Raja Yoga,
    Karako Bhava Nashaya).

    scope=line : attenuates the named lines' net strength
    scope=yoga : attenuates only the named yogas inside any line they
                 appear in (recompute support after attenuation)

    `condition` is a list of clauses (AND-combined) using the same
    feature-path / op grammar as rule antecedents. Conditions can be:
      - chart_static   : evaluated once per chart (e.g. natal Saturn
                         in own sign)
      - window_dynamic : evaluated per candidate window (e.g. transit
                         configuration at midpoint)

    `attenuation` is a multiplier:
      - 0.0 — hard override (Vipreet Raja Yoga: bad becomes good → kill
              entire line)
      - 0..1 — softening (e.g. 0.4 = 60% reduction)
      - >1.0 — amplification (e.g. Karako Bhava Nashaya = 1.5)
    """
    exception_id: str
    scope: str                                  # 'line' | 'yoga'
    applies_to_yogas: List[str] = field(default_factory=list)
    applies_to_lines: List[str] = field(default_factory=list)
    condition_type: str = "chart_static"        # 'chart_static' | 'window_dynamic'
    condition: List[Dict[str, Any]] = field(default_factory=list)
    attenuation: float = 1.0
    source: str = ""
    notes: str = ""


@dataclass
class QueryPlan:
    """A tradition's complete plan for a (relationship, life_area, effect)
    goal."""
    plan_id: str
    school: School
    relationship: str
    life_area: str
    effect: str
    goal: str = ""
    reasoning_lines: List[ReasoningLine] = field(default_factory=list)
    aggregation_method: str = "rrf"
    rrf_k: int = 60
    line_weights: Dict[str, float] = field(default_factory=dict)
    exception_ids: List[str] = field(default_factory=list)
    notes: str = ""

    def line_by_id(self, line_id: str) -> Optional[ReasoningLine]:
        for ln in self.reasoning_lines:
            if ln.line_id == line_id:
                return ln
        return None


@dataclass
class ExceptionLibrary:
    by_id: Dict[str, ExceptionRule] = field(default_factory=dict)

    def filter(self, ids: List[str]) -> List[ExceptionRule]:
        return [self.by_id[i] for i in ids if i in self.by_id]


# ── YAML loader ──────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_clause(clause: Dict[str, Any], where: str) -> None:
    if not isinstance(clause, dict):
        raise PlanLoadError(f"{where}: clause must be a mapping, got {clause!r}")
    if "feature" not in clause or "op" not in clause:
        raise PlanLoadError(
            f"{where}: clause needs 'feature' and 'op' (got {clause})"
        )
    if "value" not in clause and "value_expr" not in clause:
        raise PlanLoadError(
            f"{where}: clause needs 'value' or 'value_expr' (got {clause})"
        )


def _parse_reasoning_line(raw: Dict[str, Any]) -> ReasoningLine:
    if "id" not in raw:
        raise PlanLoadError(f"reasoning_line missing 'id': {raw}")
    line_id = raw["id"]
    yogas = raw.get("yogas") or {}
    facts = raw.get("chart_facts") or {}
    supporting_facts = list(facts.get("supporting") or [])
    attacking_facts = list(facts.get("attacking") or [])
    for c in supporting_facts + attacking_facts:
        _validate_clause(c, f"line {line_id} chart_facts")
    return ReasoningLine(
        line_id=line_id,
        description=str(raw.get("description", "")),
        supporting_yogas=list(yogas.get("supporting") or []),
        attacking_yogas=list(yogas.get("attacking") or []),
        supporting_facts=supporting_facts,
        attacking_facts=attacking_facts,
        alpha=float(raw.get("alpha", 0.5)),
        requires_dasha=raw.get("requires_dasha"),
        base_strength=float(raw.get("base_strength", 1.0)),
        min_factors=int(raw.get("min_factors", 1)),
        max_yogas_per_window=int(raw.get("max_yogas_per_window", 0)),
        notes=str(raw.get("notes", "")),
    )


def _parse_exception_rule(raw: Dict[str, Any]) -> ExceptionRule:
    if "id" not in raw:
        raise PlanLoadError(f"exception missing 'id': {raw}")
    eid = raw["id"]
    scope = raw.get("scope", "line")
    if scope not in ("line", "yoga"):
        raise PlanLoadError(
            f"exception {eid}: scope must be 'line' or 'yoga', got {scope!r}"
        )
    cond_type = raw.get("condition_type", "chart_static")
    if cond_type not in ("chart_static", "window_dynamic"):
        raise PlanLoadError(
            f"exception {eid}: condition_type must be 'chart_static' or "
            f"'window_dynamic', got {cond_type!r}"
        )
    condition = raw.get("condition") or []
    if not isinstance(condition, list):
        raise PlanLoadError(
            f"exception {eid}: 'condition' must be a list of clauses"
        )
    for c in condition:
        _validate_clause(c, f"exception {eid}")
    return ExceptionRule(
        exception_id=eid,
        scope=scope,
        applies_to_yogas=list(raw.get("applies_to_yogas") or []),
        applies_to_lines=list(raw.get("applies_to_lines") or []),
        condition_type=cond_type,
        condition=condition,
        attenuation=float(raw.get("attenuation", 1.0)),
        source=str(raw.get("source", "")),
        notes=str(raw.get("notes", "")),
    )


def load_plan(school: School, plan_name: str = "longevity",
              root: Optional[Path] = None) -> QueryPlan:
    """Load `query_plans/<school>/<plan_name>.yaml`."""
    base = root or _PLANS_ROOT
    path = base / school.value / f"{plan_name}.yaml"
    if not path.exists():
        raise PlanLoadError(f"no plan at {path}")
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise PlanLoadError(f"{path}: top-level must be a mapping")
    try:
        plan_school = School(raw.get("school", school.value))
    except ValueError as e:
        raise PlanLoadError(f"{path}: unknown school {raw.get('school')!r}") from e
    if plan_school != school:
        raise PlanLoadError(
            f"{path}: declared school={plan_school.value} but lives under "
            f"{school.value}/"
        )
    lines_raw = raw.get("reasoning_lines") or []
    if not isinstance(lines_raw, list):
        raise PlanLoadError(f"{path}: reasoning_lines must be a list")
    lines = [_parse_reasoning_line(ln) for ln in lines_raw]
    if not lines:
        raise PlanLoadError(f"{path}: at least one reasoning_line required")

    agg = raw.get("aggregation") or {}
    method = agg.get("method", "rrf")
    if method not in ("rrf", "weighted_sum"):
        raise PlanLoadError(
            f"{path}: aggregation.method must be 'rrf' or 'weighted_sum', "
            f"got {method!r}"
        )
    rrf_k = int(agg.get("k", 60))
    weights = {str(k): float(v) for k, v in (agg.get("weights") or {}).items()}

    exception_ids = list(raw.get("exceptions") or [])
    return QueryPlan(
        plan_id=raw.get("plan_id", f"{school.value}.{plan_name}"),
        school=school,
        relationship=str(raw.get("relationship", "")),
        life_area=str(raw.get("life_area", "")),
        effect=str(raw.get("effect", "event_negative")),
        goal=str(raw.get("goal", "")),
        reasoning_lines=lines,
        aggregation_method=method,
        rrf_k=rrf_k,
        line_weights=weights,
        exception_ids=exception_ids,
        notes=str(raw.get("notes", "")),
    )


def load_exception_library(school: School,
                           root: Optional[Path] = None) -> ExceptionLibrary:
    """Load `query_plans/<school>/exceptions.yaml`. Missing file = empty lib."""
    base = root or _PLANS_ROOT
    path = base / school.value / "exceptions.yaml"
    if not path.exists():
        return ExceptionLibrary(by_id={})
    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise PlanLoadError(f"{path}: top-level must be a mapping")
    items = raw.get("exceptions") or []
    if not isinstance(items, list):
        raise PlanLoadError(f"{path}: 'exceptions' must be a list")
    by_id: Dict[str, ExceptionRule] = {}
    for r in items:
        e = _parse_exception_rule(r)
        if e.exception_id in by_id:
            raise PlanLoadError(
                f"{path}: duplicate exception id {e.exception_id!r}"
            )
        by_id[e.exception_id] = e
    return ExceptionLibrary(by_id=by_id)
