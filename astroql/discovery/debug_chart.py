"""Dump engine's view of a chart: primary_house_data + which rules fire
on a specific window.

Usage:
    python -u -m astroql.discovery.debug_chart \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --name "Anais Nin" \\
        --on-date 1949-10-24
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import ParashariFeatureExtractor
from ..resolver import FocusResolver
from ..rules import StructuredRuleLibrary
from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    Relationship, School,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--on-date", required=True,
                   help="ISO date; show rules that fire on the window covering it")
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
        schools=[School.PARASHARI],
        gender=rec.get("gender") or "M",
        min_confidence=0.6,
    )

    resolver = FocusResolver()
    resolved = resolver.resolve(q)
    chart = ChartComputer().compute(
        birth, q.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_kp=False, need_jaimini=False,
    )

    print(f"### {rec['name']}  birth={bd}  death={rec.get('father_death_date')}")
    print(f"target_house_rotated={resolved.target_house_rotated}")
    print(f"target_house_direct={resolved.target_house_direct}")

    bundle = ParashariFeatureExtractor().extract(chart, resolved)

    print("\n=== primary_house_data.rotated ===")
    for k, v in (bundle.primary_house_data.get("rotated") or {}).items():
        print(f"  {k}: {v}")
    print("\n=== primary_house_data.direct ===")
    for k, v in (bundle.primary_house_data.get("direct") or {}).items():
        print(f"  {k}: {v}")

    # Find the candidate window covering the target date.
    target = datetime.fromisoformat(args.on_date)
    matching_cands = []
    for cand in (bundle.dasha_candidates or []):
        s = datetime.fromisoformat(cand["start"])
        e = datetime.fromisoformat(cand["end"])
        if s <= target <= e:
            matching_cands.append(cand)

    print(f"\n=== Candidate windows covering {args.on_date} ({len(matching_cands)}) ===")
    for cand in matching_cands:
        print(f"\n--- {cand['start']}..{cand['end']} ---")
        for k in sorted(cand.keys()):
            v = cand[k]
            print(f"  {k}: {v}")

    # Apply rules and report which fire on this window.
    lib = StructuredRuleLibrary()
    engine = RuleEngine()
    rules = lib.load_rules(School.PARASHARI, resolved)
    fired = engine.apply(bundle, rules)

    # Filter to fired rules whose window covers the target date.
    print(f"\n=== Rules fired on the matching candidate (with strength) ===")
    matching_fires = []
    for fr in fired:
        if fr.window is None:
            continue
        if fr.window[0] <= target <= fr.window[1]:
            matching_fires.append(fr)
    matching_fires.sort(key=lambda fr: -fr.strength)
    for fr in matching_fires:
        print(
            f"  {fr.strength:0.3f} {fr.polarity:8} tier={fr.rule.priority_tier} "
            f"{fr.rule.rule_id}"
        )

    # Also dump static (window=None) negative rules.
    print(f"\n=== Static rules (window=None) ===")
    for fr in sorted(
        [fr for fr in fired if fr.window is None],
        key=lambda fr: -fr.strength,
    ):
        print(
            f"  {fr.strength:0.3f} {fr.polarity:8} tier={fr.rule.priority_tier} "
            f"{fr.rule.rule_id}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
