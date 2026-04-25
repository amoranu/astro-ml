"""Minimal JSON API (spec §14 Phase 8).

No HTTP server — exposes a single `run_query_json(query_dict) -> dict`
function that can be wrapped by Flask/FastAPI or called directly from
notebooks / integration tests. Keeps AstroQL framework-agnostic.

Also enforces spec §12.6 severity gating: longevity+event_negative
queries must include `severity_ack=True` in the request, else a soft
refusal dict is returned.
"""
from __future__ import annotations

from datetime import date as _date
from typing import Any, Dict

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..explainer import Explainer
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..resolver_engine import Aggregator
from ..rules import StructuredRuleLibrary
from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    Relationship, School,
)


_SENSITIVE = {
    (LifeArea.LONGEVITY, Effect.EVENT_NEGATIVE),
    (LifeArea.HEALTH, Effect.EVENT_NEGATIVE),
    (LifeArea.HEALTH, Effect.MAGNITUDE),     # CAV-043
    (LifeArea.LITIGATION, Effect.EVENT_NEGATIVE),
}


def _build_query(req: Dict[str, Any]) -> FocusQuery:
    rel = Relationship(req["relationship"])
    life = LifeArea(req["life_area"])
    eff = Effect(req["effect"])
    mod = Modifier(req.get("modifier", "null"))
    schools = [
        School(s) for s in req.get("schools", ["parashari"])
    ]
    b = req["birth"]
    birth = BirthDetails(
        date=_date.fromisoformat(b["date"]),
        time=b.get("time"),
        tz=b.get("tz", "UTC"),
        lat=float(b["lat"]),
        lon=float(b["lon"]),
        time_accuracy=b.get("time_accuracy", "exact"),
    )
    return FocusQuery(
        relationship=rel, life_area=life, effect=eff, modifier=mod,
        birth=birth, config=ChartConfig(),
        schools=schools, gender=b.get("gender"),
        min_confidence=req.get("min_confidence", 0.55),
    )


def run_query_json(req: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point — dict in, dict out. Enforces severity gating."""
    try:
        query = _build_query(req)
    except (KeyError, ValueError) as e:
        return {
            "error": "invalid_request",
            "detail": str(e),
        }

    # Spec §12.6: sensitive content gating.
    if ((query.life_area, query.effect) in _SENSITIVE
            and not req.get("severity_ack", False)):
        return {
            "error": "severity_gate",
            "detail": (
                "Queries about longevity or serious health outcomes "
                "require severity_ack=True in the request. Results are "
                "probabilistic, never deterministic, and should not be "
                "treated as medical or life advice."
            ),
            "required_field": "severity_ack",
        }
    # Auto-raise min_confidence by 0.1 for sensitive queries.
    if (query.life_area, query.effect) in _SENSITIVE:
        query.min_confidence = min(1.0, query.min_confidence + 0.1)

    resolver = FocusResolver()
    resolved = resolver.resolve(query)
    chart = ChartComputer().compute(
        query.birth, query.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_kp=(School.KP in query.schools),
        need_jaimini=(School.JAIMINI in query.schools),
    )
    lib = StructuredRuleLibrary()
    eng = RuleEngine()
    fired = {}
    if School.PARASHARI in query.schools:
        fired[School.PARASHARI] = eng.apply(
            ParashariFeatureExtractor().extract(chart, resolved),
            lib.load_rules(School.PARASHARI, resolved),
        )
    if School.KP in query.schools:
        fired[School.KP] = eng.apply(
            KPFeatureExtractor().extract(chart, resolved),
            lib.load_rules(School.KP, resolved),
        )
    if School.JAIMINI in query.schools:
        fired[School.JAIMINI] = eng.apply(
            JaiminiFeatureExtractor().extract(chart, resolved),
            lib.load_rules(School.JAIMINI, resolved),
        )

    result = Aggregator().aggregate(query, resolved, fired)
    explainer = Explainer()
    out = explainer.explain(result)
    out["narration"] = explainer.narrate(result)
    return out
