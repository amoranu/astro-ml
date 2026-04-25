"""AstroQL CLI (spec §10).

Usage:
    python -u -m astroql.cli \\
        --birth-date 1965-03-15 --birth-time 14:20:00 \\
        --birth-tz Asia/Kolkata --lat 28.6139 --lon 77.2090 \\
        --gender M --relationship father --life-area longevity \\
        --effect event_negative --modifier timing --min-confidence 0.6 \\
        --schools parashari [--explain] [--top-n 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date as _date
from typing import List

from .chart import ChartComputer
from .engine import RuleEngine
from .explainer import Explainer
from .features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from .parser import parse_dsl
from .resolver import FocusResolver
from .resolver_engine import Aggregator
from .rules import StructuredRuleLibrary
from .schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea,
    Modifier, Relationship, School,
)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="astroql", description=__doc__)
    p.add_argument("--birth-date", required=False,
                   help="ISO birth date, e.g. 1965-03-15")
    p.add_argument("--birth-time", default="12:00:00",
                   help="HH:MM:SS")
    p.add_argument("--birth-tz", default="UTC",
                   help="IANA timezone, e.g. Asia/Kolkata")
    p.add_argument("--lat", type=float, required=False)
    p.add_argument("--lon", type=float, required=False)
    p.add_argument("--gender", default=None, choices=[None, "M", "F"])
    p.add_argument("--relationship", required=False,
                   choices=[r.value for r in Relationship])
    p.add_argument("--life-area", required=False,
                   choices=[l.value for l in LifeArea])
    p.add_argument("--effect", required=False,
                   choices=[e.value for e in Effect])
    p.add_argument("--modifier", default="null",
                   choices=[m.value for m in Modifier])
    p.add_argument("--schools", default="parashari",
                   help="comma-separated subset of parashari,jaimini,kp")
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--explain", action="store_true",
                   help="emit structured EXPLAIN JSON in addition to narration")
    p.add_argument("--dasha-window-years", type=float, default=100.0,
                   help="Vimshottari horizon to compute")
    p.add_argument("--sensitivity", action="store_true",
                   help="run ±60min sensitivity scan after main query")
    p.add_argument("--dsl", default=None,
                   help="AstroQL DSL file; when given, argparse flags"
                        " for relationship/life-area/etc are ignored")
    return p.parse_args(argv)


def _build_query(ns: argparse.Namespace) -> FocusQuery:
    schools = [School(s.strip()) for s in ns.schools.split(",") if s.strip()]
    birth = BirthDetails(
        date=_date.fromisoformat(ns.birth_date),
        time=ns.birth_time,
        tz=ns.birth_tz,
        lat=ns.lat,
        lon=ns.lon,
    )
    life_area = LifeArea(ns.life_area)
    relationship = Relationship(ns.relationship)

    # Let the resolver decide vargas/dashas, but pre-populate a sensible
    # default ChartConfig so ChartComputer can fall back if a caller omits.
    config = ChartConfig(
        vargas=["D1"], dasha_systems=[]   # overridden via resolved values
    )
    return FocusQuery(
        relationship=relationship,
        life_area=life_area,
        effect=Effect(ns.effect),
        modifier=Modifier(ns.modifier),
        birth=birth,
        config=config,
        schools=schools,
        min_confidence=ns.min_confidence,
        gender=ns.gender,
        explain=ns.explain,
    )


def run(argv: List[str]) -> int:
    ns = _parse_args(argv)
    if ns.dsl:
        with open(ns.dsl, encoding="utf-8") as f:
            query = parse_dsl(f.read())
    else:
        # Validate argparse flags required when not using DSL.
        for required in ("birth_date", "lat", "lon", "relationship",
                         "life_area", "effect"):
            if getattr(ns, required) is None:
                raise SystemExit(
                    f"--{required.replace('_','-')} required unless --dsl used"
                )
        query = _build_query(ns)

    from .planner import run_pipeline
    pipeline_out = run_pipeline(query, use_rag=False)
    result = pipeline_out.result
    # Re-cap windows by user's --top-n.
    if result.windows and len(result.windows) > ns.top_n:
        result.windows = result.windows[:ns.top_n]

    explainer = Explainer()
    print(explainer.narrate(result))
    if ns.explain:
        print("\n---EXPLAIN---")
        print(json.dumps(
            explainer.explain(result, trace=pipeline_out.trace),
            indent=2, default=str,
        ))
    if ns.sensitivity:
        print("\n---SENSITIVITY---")
        from .sensitivity import sensitivity_scan
        report = sensitivity_scan(
            query, offsets_min=[-60, -30, 0, 30, 60], top_n=5,
        )
        for p in report.perturbation_results:
            print(
                f"  offset={p['offset_minutes']:+4d}m  "
                f"top_agg={p['top_aggregate']:.3f}  "
                f"top_start={p['top_window_start']}"
            )
        print(f"  stability: {report.stability}")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
