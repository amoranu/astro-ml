"""Show the top-ranked windows in a chart so we can see what is
outscoring the death window.

Usage:
    python -u -m astroql.discovery.debug_top_window \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --name "Anais Nin" --top 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from ..chart import ChartComputer
from ..engine import RuleEngine
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    rec = next((r for r in data if r.get("name") == args.name), None)
    if rec is None:
        print(f"NOT FOUND: {args.name}")
        return 1

    bd = date.fromisoformat(rec["birth_date"])
    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(rec["lat"]), lon=float(rec["lon"]),
    )
    death_dt = datetime.fromisoformat(rec["father_death_date"])
    q = FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI, School.KP, School.JAIMINI],
        gender=rec.get("gender") or "M",
        min_confidence=0.6,
    )

    resolver = FocusResolver()
    resolved = resolver.resolve(q)
    chart = ChartComputer().compute(
        birth, q.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_kp=True, need_jaimini=True,
    )

    lib = StructuredRuleLibrary()
    engine = RuleEngine()
    fired_by_school = {}
    fired_by_school[School.PARASHARI] = engine.apply(
        ParashariFeatureExtractor().extract(chart, resolved),
        lib.load_rules(School.PARASHARI, resolved),
    )
    fired_by_school[School.KP] = engine.apply(
        KPFeatureExtractor().extract(chart, resolved),
        lib.load_rules(School.KP, resolved),
    )
    fired_by_school[School.JAIMINI] = engine.apply(
        JaiminiFeatureExtractor().extract(chart, resolved),
        lib.load_rules(School.JAIMINI, resolved),
    )
    result = Aggregator().aggregate(
        q, resolved, fired_by_school,
        max_windows=10**9, min_confidence=0.0,
    )

    death_rank = None
    for i, w in enumerate(result.windows, 1):
        if w.start <= death_dt <= w.end:
            death_rank = i
            break

    print(f"### {rec['name']}  death={rec['father_death_date']}")
    print(f"DEATH RANK: {death_rank} of {len(result.windows)}")
    death_window = (
        result.windows[death_rank - 1] if death_rank else None
    )
    if death_window:
        print(
            f"DEATH WINDOW: {death_window.start.date()}..{death_window.end.date()} "
            f"agg={death_window.aggregate_confidence:.4f} "
            f"n_rules={len(death_window.contributing_rules)}"
        )
        print("  rules:")
        for fr in sorted(
            death_window.contributing_rules,
            key=lambda fr: -fr.strength,
        )[:15]:
            print(
                f"    {fr.strength:.3f} {fr.polarity:8} "
                f"tier={fr.rule.priority_tier} {fr.rule.rule_id}"
            )

    print(f"\n=== TOP {args.top} WINDOWS (outscoring the death) ===")
    for i, w in enumerate(result.windows[:args.top], 1):
        is_death = death_rank == i
        marker = " <- DEATH" if is_death else ""
        print(
            f"\n#{i}{marker}: {w.start.date()}..{w.end.date()} "
            f"agg={w.aggregate_confidence:.4f} n_rules={len(w.contributing_rules)}"
        )
        for fr in sorted(
            w.contributing_rules, key=lambda fr: -fr.strength,
        )[:10]:
            print(
                f"    {fr.strength:.3f} {fr.polarity:8} "
                f"tier={fr.rule.priority_tier} {fr.rule.rule_id}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
