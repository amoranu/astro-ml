"""Phase 6 calibration — learn per-rule strengths from retrodiction data.

Simple point-mass estimator:
    For each rule_id, hit_rate = fraction of FIRED instances whose window
    contains the true event date. A rule that fires 500 times and contains
    the truth 10 of those times has hit_rate = 2%.

Calibrated strength = baseline_strength + learning_rate * (hit_rate - prior).
Writes results to rules/calibrated_strengths.yaml for a follow-up pipeline
to apply as a strength multiplier.

This is NOT a proper MLE / Bayesian update — it's a transparent baseline
that beats random ranking. Proper calibration is CAV-033.

Usage:
    python -u -m astroql.benchmark.calibrate \\
        --dataset ml/father_passing_date_clean_clean2.json \\
        --max 50 --out astroql/rules/calibrated_strengths.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import yaml

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..rules import StructuredRuleLibrary
from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    Relationship, School,
)


def _collect_rule_fire_hits(
    rec: dict,
) -> Dict[str, Dict[str, int]]:
    """Return {rule_id: {"fires": n, "hits": h}} for one chart."""
    try:
        bd = date.fromisoformat(rec["birth_date"])
    except Exception:
        return {}

    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    lat, lon = rec.get("lat"), rec.get("lon")
    if lat is None or lon is None:
        return {}

    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(lat), lon=float(lon),
    )
    try:
        true_dt = datetime.fromisoformat(rec["father_death_date"])
    except Exception:
        return {}

    q = FocusQuery(
        relationship=Relationship.FATHER, life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE, modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI, School.KP, School.JAIMINI],
        gender=rec.get("gender") or "M",
        min_confidence=0.0,
    )
    r = FocusResolver().resolve(q)
    try:
        c = ChartComputer().compute(
            birth, q.config,
            vargas=r.vargas_required,
            dashas=r.dashas_required,
            need_kp=True, need_jaimini=True,
        )
    except Exception:
        return {}

    lib = StructuredRuleLibrary()
    eng = RuleEngine()

    all_fired = []
    all_fired.extend(eng.apply(
        ParashariFeatureExtractor().extract(c, r),
        lib.load_rules(School.PARASHARI, r),
    ))
    all_fired.extend(eng.apply(
        KPFeatureExtractor().extract(c, r),
        lib.load_rules(School.KP, r),
    ))
    all_fired.extend(eng.apply(
        JaiminiFeatureExtractor().extract(c, r),
        lib.load_rules(School.JAIMINI, r),
    ))

    per_rule: Dict[str, Dict[str, int]] = {}
    for fr in all_fired:
        rid = fr.rule.rule_id
        entry = per_rule.setdefault(rid, {"fires": 0, "hits": 0})
        entry["fires"] += 1
        if fr.window is not None and fr.window[0] <= true_dt <= fr.window[1]:
            entry["hits"] += 1
    return per_rule


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--max", type=int, default=50)
    p.add_argument("--out", default="astroql/rules/calibrated_strengths.yaml")
    p.add_argument("--learning-rate", type=float, default=0.5)
    p.add_argument("--prior", type=float, default=0.05,
                   help="Prior hit rate (random baseline).")
    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    if args.max > 0:
        data = data[:args.max]

    totals: Dict[str, Dict[str, int]] = {}
    for i, rec in enumerate(data, 1):
        per = _collect_rule_fire_hits(rec)
        for rid, counts in per.items():
            t = totals.setdefault(rid, {"fires": 0, "hits": 0})
            t["fires"] += counts["fires"]
            t["hits"] += counts["hits"]
        print(f"  [{i}/{len(data)}] {rec.get('name','?')}  "
              f"rules_fired={len(per)}")

    # Build calibrated strengths.
    out_rules: Dict[str, Dict] = {}
    for rid, c in sorted(totals.items()):
        fires = c["fires"]
        hits = c["hits"]
        if fires == 0:
            continue
        hit_rate = hits / fires
        # adjustment: positive if rule out-performs prior, negative if under.
        adj = args.learning_rate * (hit_rate - args.prior)
        out_rules[rid] = {
            "fires": fires,
            "hits": hits,
            "hit_rate": round(hit_rate, 4),
            "strength_adjustment": round(adj, 4),
        }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({
            "meta": {
                "dataset": str(args.dataset),
                "n_charts": len(data),
                "prior_hit_rate": args.prior,
                "learning_rate": args.learning_rate,
            },
            "rules": out_rules,
        }, f, sort_keys=False)

    # Print ranked.
    print("\n===== CALIBRATED RANKINGS =====")
    ranked = sorted(
        out_rules.items(),
        key=lambda kv: -kv[1]["hit_rate"],
    )
    for rid, row in ranked[:20]:
        print(f"  {rid:60s}  fires={row['fires']:5d}  hits={row['hits']:4d}  "
              f"hit_rate={row['hit_rate']:.4f}  adj={row['strength_adjustment']:+.4f}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
