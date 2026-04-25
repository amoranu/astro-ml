"""Retrodiction benchmark (spec §13.3).

For each chart with a known father-death date, runs the full AstroQL
pipeline (TIMING), then checks where the true event date ranks among
the engine's candidate windows.

Metrics:
    hit@k      — true event falls within top-k candidate windows
    avg_rank   — average rank of the true-containing window
    abstain    — count of charts where no window covered the true date

Usage:
    python -u -m astroql.benchmark.retrodiction \\
        --dataset ml/father_passing_date_clean_clean2.json \\
        --max 50
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

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


@dataclass
class BenchResult:
    chart_name: str
    true_date: Optional[date]
    rank_of_hit: Optional[int]        # 1-based; None if no window covers it
    top_k_abstain: bool
    n_windows: int
    # ±N-month tolerance: rank of best window whose midpoint is within
    # N months of the true date. Looser than rank_of_hit and matches the
    # ±6mo benchmark cited by classical texts (BPHS Ch.46, Phaladeepika 18).
    rank_within_6mo: Optional[int] = None
    months_off_at_rank1: Optional[float] = None   # |best_window_mid - true| in months


def _run_chart(rec: dict, top_n: int = 20) -> BenchResult:
    try:
        bd = date.fromisoformat(rec["birth_date"])
    except Exception:
        return BenchResult(rec.get("name", "?"), None, None, True, 0)

    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    lat = rec.get("lat")
    lon = rec.get("lon")
    if lat is None or lon is None:
        return BenchResult(rec.get("name", "?"), None, None, True, 0)

    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(lat), lon=float(lon),
    )
    try:
        true_death = date.fromisoformat(rec["father_death_date"])
    except Exception:
        true_death = None

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
    computer = ChartComputer()
    chart = computer.compute(
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

    # Run with min_confidence=0 and max_windows huge to measure the true
    # event's rank among ALL candidate windows.
    result = Aggregator().aggregate(
        q, resolved, fired_by_school,
        max_windows=10**9, min_confidence=0.0,
    )
    # Cap after ranking for the returned-to-user view.
    if top_n < len(result.windows):
        user_view_truncated = True
    else:
        user_view_truncated = False

    if true_death is None:
        return BenchResult(
            rec.get("name", "?"), None, None, True, len(result.windows),
        )

    true_dt = datetime(true_death.year, true_death.month, true_death.day)
    rank = None
    for i, w in enumerate(result.windows, 1):
        if w.start <= true_dt <= w.end:
            rank = i
            break
    # ±6mo tolerance metric: rank of the best window whose midpoint is
    # within 6 months of true death. This matches the classical benchmark
    # of ±6 month timing precision.
    SIX_MO = 30.4 * 6  # ~182.5 days
    rank_within_6mo = None
    for i, w in enumerate(result.windows, 1):
        mid = w.start + (w.end - w.start) / 2
        diff_days = abs((mid - true_dt).total_seconds() / 86400)
        if diff_days <= SIX_MO:
            rank_within_6mo = i
            break
    # Months_off at rank-1: how far is the engine's #1 window from truth.
    months_off = None
    if result.windows:
        top_w = result.windows[0]
        top_mid = top_w.start + (top_w.end - top_w.start) / 2
        months_off = abs(
            (top_mid - true_dt).total_seconds() / (86400 * 30.4)
        )
    return BenchResult(
        chart_name=rec.get("name", "?"),
        true_date=true_death,
        rank_of_hit=rank,
        top_k_abstain=rank is None,
        n_windows=len(result.windows),
        rank_within_6mo=rank_within_6mo,
        months_off_at_rank1=months_off,
    )


def _summarize(rows: List[BenchResult], k_list=(1, 3, 5, 10, 20)) -> dict:
    eval_rows = [r for r in rows if r.true_date is not None]
    out = {
        "n_total": len(rows),
        "n_evaluable": len(eval_rows),
        "n_abstained": sum(1 for r in eval_rows if r.top_k_abstain),
    }
    for k in k_list:
        hit = sum(
            1 for r in eval_rows
            if r.rank_of_hit is not None and r.rank_of_hit <= k
        )
        out[f"hit@{k}"] = hit / max(1, len(eval_rows))
    # ±6 month tolerance hit rates — the classical-text benchmark.
    for k in k_list:
        hit = sum(
            1 for r in eval_rows
            if r.rank_within_6mo is not None and r.rank_within_6mo <= k
        )
        out[f"hit@{k}_pm6mo"] = hit / max(1, len(eval_rows))
    ranks = [r.rank_of_hit for r in eval_rows if r.rank_of_hit is not None]
    if ranks:
        out["mean_rank_when_hit"] = sum(ranks) / len(ranks)
        out["median_rank_when_hit"] = sorted(ranks)[len(ranks) // 2]
    months_off = [
        r.months_off_at_rank1 for r in eval_rows
        if r.months_off_at_rank1 is not None
    ]
    if months_off:
        out["mean_months_off_at_rank1"] = sum(months_off) / len(months_off)
        out["median_months_off_at_rank1"] = (
            sorted(months_off)[len(months_off) // 2]
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="path to cleaned father-death json")
    p.add_argument("--max", type=int, default=30,
                   help="if --start/--end not given, take data[:max]")
    p.add_argument("--start", type=int, default=None,
                   help="data[start:end] slice (test-split)")
    p.add_argument("--end", type=int, default=None,
                   help="data[start:end] slice (test-split)")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--out", default=None,
                   help="write per-chart results as JSON")
    args = p.parse_args()

    path = Path(args.dataset)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end if args.end is not None else len(data)
        data = data[s:e]
    elif args.max > 0:
        data = data[:args.max]

    rows: List[BenchResult] = []
    for i, rec in enumerate(data, 1):
        try:
            r = _run_chart(rec, top_n=args.top_n)
        except Exception as e:
            r = BenchResult(
                rec.get("name", "?"), None, None, True, 0,
            )
            print(f"  [{i}/{len(data)}] {rec.get('name', '?')}: ERROR {e}")
        else:
            rank_str = f"rank={r.rank_of_hit}" if r.rank_of_hit else "MISS"
            print(f"  [{i}/{len(data)}] {r.chart_name}: {rank_str}"
                  f"  windows={r.n_windows}")
        rows.append(r)

    summary = _summarize(rows)
    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "summary": summary,
                "rows": [
                    {"name": r.chart_name,
                     "true_date": (r.true_date.isoformat()
                                    if r.true_date else None),
                     "rank": r.rank_of_hit,
                     "windows": r.n_windows}
                    for r in rows
                ],
            }, f, indent=2)
        print(f"\nwrote per-chart results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
