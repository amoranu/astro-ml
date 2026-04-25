"""Sensitivity analysis (spec §11.7).

Given a FocusQuery with approximate birth time, perturb the time by
±N minutes and report which conclusions are stable vs time-sensitive.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..resolver_engine import Aggregator
from ..rules import StructuredRuleLibrary
from ..schemas.focus import FocusQuery
from ..schemas.enums import School
from ..schemas.results import QueryResult


@dataclass
class SensitivityReport:
    base_result: QueryResult
    perturbation_results: List[Dict[str, Any]]
    stability: Dict[str, float]


def _shift_query_time(query: FocusQuery, minutes: int) -> FocusQuery:
    q = copy.deepcopy(query)
    if not q.birth or not q.birth.time:
        return q
    h, m, s = [int(x) for x in q.birth.time.split(":")]
    base = datetime(2000, 1, 1, h, m, s) + timedelta(minutes=minutes)
    q.birth.time = base.strftime("%H:%M:%S")
    return q


def _run_full(query: FocusQuery):
    resolved = FocusResolver().resolve(query)
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
    return Aggregator().aggregate(query, resolved, fired)


def sensitivity_scan(
    query: FocusQuery, offsets_min: List[int] = (-60, -30, 0, 30, 60),
    top_n: int = 10,
) -> SensitivityReport:
    """Run the pipeline at each time offset; compare top-N windows.

    Stability metric: for each top-1 window in the base run, fraction of
    perturbations whose top-N includes a window overlapping the base top-1.
    """
    base = _run_full(query)
    perturbation_results: List[Dict[str, Any]] = []
    for off in offsets_min:
        if off == 0:
            result = base
        else:
            result = _run_full(_shift_query_time(query, off))
        perturbation_results.append({
            "offset_minutes": off,
            "n_windows": len(result.windows or []),
            "top_aggregate": (
                result.windows[0].aggregate_confidence
                if result.windows else 0.0
            ),
            "top_window_start": (
                result.windows[0].start.isoformat()
                if result.windows else None
            ),
            "top_window_end": (
                result.windows[0].end.isoformat()
                if result.windows else None
            ),
        })

    # Stability: for base top-1, fraction of perturbations whose top-N
    # contains an overlapping window.
    stability: Dict[str, float] = {}
    if base.windows:
        base_top = base.windows[0]
        overlap_count = 0
        for off in offsets_min:
            if off == 0:
                overlap_count += 1
                continue
            r = _run_full(_shift_query_time(query, off))
            for w in (r.windows or [])[:top_n]:
                if w.start < base_top.end and base_top.start < w.end:
                    overlap_count += 1
                    break
        stability["top1_stable_fraction"] = overlap_count / len(offsets_min)
    return SensitivityReport(
        base_result=base,
        perturbation_results=perturbation_results,
        stability=stability,
    )
