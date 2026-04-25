"""Unified pipeline runner with tracing (spec §9).

Collects stage timings, rule-fire counts, RAG retrieval counts into an
ExecutionTrace and threads that through to the Explainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..rag import RAGPipeline
from ..resolver import FocusResolver
from ..resolver_engine import Aggregator
from ..rules import StructuredRuleLibrary
from ..schemas.enums import School
from ..schemas.focus import FocusQuery
from ..schemas.results import QueryResult
from .trace import ExecutionTrace


@dataclass
class PipelineResult:
    result: QueryResult
    trace: ExecutionTrace


def run_pipeline(
    query: FocusQuery,
    use_rag: bool = False,
) -> PipelineResult:
    trace = ExecutionTrace()

    with trace.stage("resolve"):
        resolved = FocusResolver().resolve(query)

    with trace.stage("compute_chart",
                      vargas=list(resolved.vargas_required),
                      dashas=list(resolved.dashas_required)):
        # Compute strength only when query type needs it.
        from ..schemas.enums import QueryType
        need_strength = resolved.query_type in (
            QueryType.MAGNITUDE, QueryType.YES_NO,
        )
        chart = ChartComputer().compute(
            query.birth, query.config,
            vargas=resolved.vargas_required,
            dashas=resolved.dashas_required,
            need_kp=(School.KP in query.schools),
            need_jaimini=(School.JAIMINI in query.schools),
            need_strength=need_strength,
        )

    lib = StructuredRuleLibrary()
    engine = RuleEngine()
    rag = RAGPipeline() if use_rag else None

    fired = {}

    def _run_school(school: School, extractor_cls):
        with trace.stage(f"extract:{school.value}") as s:
            fb = extractor_cls().extract(chart, resolved)
            s["feature_bundle_size"] = len(
                (fb.dasha_candidates or [])
            )
        with trace.stage(f"load_rules:{school.value}") as s:
            rules = lib.load_rules(school, resolved)
            s["n_rules"] = len(rules)
        passages = None
        if rag is not None and rag.is_available():
            with trace.stage(f"rag:{school.value}") as s:
                passages = rag.retrieve(fb, top_k=10)
                s["n_passages"] = len(passages)
        with trace.stage(f"apply:{school.value}") as s:
            fr = engine.apply(fb, rules, passages)
            s["n_fired"] = len(fr)
        fired[school] = fr

    if School.PARASHARI in query.schools:
        _run_school(School.PARASHARI, ParashariFeatureExtractor)
    if School.KP in query.schools:
        _run_school(School.KP, KPFeatureExtractor)
    if School.JAIMINI in query.schools:
        _run_school(School.JAIMINI, JaiminiFeatureExtractor)

    with trace.stage("aggregate") as s:
        result = Aggregator().aggregate(query, resolved, fired)
        s["n_windows"] = len(result.windows or [])
        s["n_attributes"] = len(result.attributes or [])

    return PipelineResult(result=result, trace=trace)
