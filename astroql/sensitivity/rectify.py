"""Rectification (spec §11.6, CAV-039 multi-life-area scoring).

Given a base query (with birth) and a list of (life_area, event_date)
pairs, scores candidate birth times by inverse rank of the engine's
predicted window covering each known event. Each event uses its own
life_area / effect — not just the base query's.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from ..schemas.enums import Effect, LifeArea, Modifier, QueryType, Relationship
from ..schemas.focus import FocusQuery
from .sensitivity import _run_full


@dataclass
class RectificationResult:
    best_time: str
    best_score: float
    candidates: List[Tuple[str, float]]


def _build_query_at_time(base: FocusQuery, hhmmss: str) -> FocusQuery:
    q = copy.deepcopy(base)
    q.birth.time = hhmmss
    return q


def rectify(
    base_query: FocusQuery,
    known_events: List[Tuple[str, date]],
    window_min: int = 60,
    step_min: int = 10,
) -> RectificationResult:
    """Score candidate birth times by engine agreement with known events.

    `known_events` is a list of (life_area_str, event_date) pairs. For
    each candidate time, run TIMING queries for each event's life_area,
    and score by the rank of the candidate window containing the event
    (lower rank = higher score).
    """
    if not base_query.birth or not base_query.birth.time:
        raise ValueError("base_query must have a birth time")
    h, m, s = [int(x) for x in base_query.birth.time.split(":")]
    base_dt = datetime(2000, 1, 1, h, m, s)

    offsets = list(range(-window_min, window_min + 1, step_min))
    candidates: List[Tuple[str, float]] = []

    for off in offsets:
        t = (base_dt + timedelta(minutes=off)).strftime("%H:%M:%S")
        q = _build_query_at_time(base_query, t)
        score = 0.0
        for life_area_str, event_date in known_events:
            # CAV-039: build a per-event FocusQuery from the event's
            # own life_area, with effect/modifier defaults appropriate
            # for that life area.
            event_query = copy.deepcopy(q)
            try:
                event_query.life_area = LifeArea(life_area_str)
            except ValueError:
                continue
            # Heuristic effect/modifier defaults per life_area.
            if event_query.life_area == LifeArea.LONGEVITY:
                event_query.effect = Effect.EVENT_NEGATIVE
                event_query.modifier = Modifier.TIMING
                # Father's death = relationship.father, mother's = mother.
                # Caller should set base_query.relationship; we don't override.
            elif event_query.life_area == LifeArea.MARRIAGE:
                event_query.effect = Effect.EVENT_POSITIVE
                event_query.modifier = Modifier.TIMING
            elif event_query.life_area == LifeArea.CHILDREN:
                event_query.effect = Effect.EVENT_POSITIVE
                event_query.modifier = Modifier.TIMING
            elif event_query.life_area == LifeArea.CAREER:
                event_query.effect = Effect.EVENT_POSITIVE
                event_query.modifier = Modifier.TIMING

            result = _run_full(event_query)
            event_dt = datetime(
                event_date.year, event_date.month, event_date.day,
            )
            rank = None
            for i, w in enumerate(result.windows or [], 1):
                if w.start <= event_dt <= w.end:
                    rank = i
                    break
            if rank is None:
                score -= 1.0   # miss penalty
            else:
                score += 1.0 / rank
        candidates.append((t, score))

    candidates.sort(key=lambda x: -x[1])
    return RectificationResult(
        best_time=candidates[0][0],
        best_score=candidates[0][1],
        candidates=candidates,
    )
