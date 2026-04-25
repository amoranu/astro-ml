"""Per-tradition retrodiction benchmark (Phase F.12).

Runs the new per-tradition planner (`run_multi_tradition`) on the
labeled father-death dataset and reports Hit@K + ±6mo Hit@K + median
months-off, INDEPENDENTLY per school. The classical text benchmark of
±6 month precision is the headline metric.

This benchmark intentionally does NOT combine traditions — it shows
each school's independent performance so the user can compare
side-by-side and pick a tradition (or use them as triangulation).

Usage:
    python -u -m astroql.benchmark.retrodiction_per_tradition \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --start 200 --end 400 \\
        --schools parashari,kp \\
        --top-n 20

Compare against legacy combined Aggregator path:
    python -u -m astroql.benchmark.retrodiction \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --start 200 --end 400
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..planner import run_tradition
from ..schemas.birth import BirthDetails
from ..schemas.chart import ChartConfig
from ..schemas.enums import (
    Effect, LifeArea, Modifier, Relationship, School,
)
from ..schemas.focus import FocusQuery


_SIX_MO_DAYS = 30.4 * 6   # ~182.5


@dataclass
class PerSchoolResult:
    school: School
    rank_of_hit: Optional[int] = None
    rank_within_6mo: Optional[int] = None
    months_off_at_rank1: Optional[float] = None
    n_windows: int = 0
    inconclusive_reason: Optional[str] = None


@dataclass
class ChartRow:
    chart_name: str
    true_date: Optional[date]
    by_school: Dict[School, PerSchoolResult] = field(default_factory=dict)


def _build_query(rec: dict, schools: List[School]) -> Optional[FocusQuery]:
    try:
        bd = date.fromisoformat(rec["birth_date"])
    except Exception:
        return None
    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    if rec.get("lat") is None or rec.get("lon") is None:
        return None
    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(rec["lat"]), lon=float(rec["lon"]),
    )
    return FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D9", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=schools,
        gender=rec.get("gender") or "M",
        min_confidence=0.0,
    )


def _evaluate_one(
    rec: dict, schools: List[School], top_n: int = 20,
) -> ChartRow:
    name = rec.get("name", "?")
    try:
        true_death = date.fromisoformat(rec["father_death_date"])
    except Exception:
        true_death = None
    row = ChartRow(chart_name=name, true_date=true_death)

    q = _build_query(rec, schools)
    if q is None:
        for s in schools:
            row.by_school[s] = PerSchoolResult(
                school=s, inconclusive_reason="bad birth data",
            )
        return row

    true_dt = (
        datetime(true_death.year, true_death.month, true_death.day)
        if true_death else None
    )

    for school in schools:
        try:
            tr = run_tradition(q, school, max_windows=top_n * 5)
        except Exception as e:
            row.by_school[school] = PerSchoolResult(
                school=school, inconclusive_reason=f"runtime error: {e}",
            )
            continue

        ps = PerSchoolResult(
            school=school, n_windows=len(tr.windows),
            inconclusive_reason=tr.inconclusive_reason,
        )
        if true_dt is not None and tr.windows:
            for i, w in enumerate(tr.windows, 1):
                if w.start <= true_dt <= w.end:
                    ps.rank_of_hit = i
                    break
            for i, w in enumerate(tr.windows, 1):
                mid = w.start + (w.end - w.start) / 2
                diff = abs((mid - true_dt).total_seconds() / 86400)
                if diff <= _SIX_MO_DAYS:
                    ps.rank_within_6mo = i
                    break
            top_w = tr.windows[0]
            top_mid = top_w.start + (top_w.end - top_w.start) / 2
            ps.months_off_at_rank1 = abs(
                (top_mid - true_dt).total_seconds() / (86400 * 30.4)
            )
        row.by_school[school] = ps
    return row


def _summarize_school(
    rows: List[ChartRow], school: School,
    k_list=(1, 3, 5, 10, 20),
) -> dict:
    eval_rows = [
        r for r in rows
        if r.true_date is not None and school in r.by_school
    ]
    out = {
        "n_total": len(rows),
        "n_evaluable": len(eval_rows),
        "n_runtime_errors": sum(
            1 for r in eval_rows
            if r.by_school[school].inconclusive_reason
            and "runtime error" in (r.by_school[school].inconclusive_reason or "")
        ),
        "n_no_windows": sum(
            1 for r in eval_rows
            if r.by_school[school].n_windows == 0
        ),
    }
    for k in k_list:
        n_hit = sum(
            1 for r in eval_rows
            if r.by_school[school].rank_of_hit is not None
            and r.by_school[school].rank_of_hit <= k
        )
        out[f"hit@{k}"] = n_hit / max(1, len(eval_rows))
    for k in k_list:
        n_hit = sum(
            1 for r in eval_rows
            if r.by_school[school].rank_within_6mo is not None
            and r.by_school[school].rank_within_6mo <= k
        )
        out[f"hit@{k}_pm6mo"] = n_hit / max(1, len(eval_rows))
    months_off = [
        r.by_school[school].months_off_at_rank1 for r in eval_rows
        if r.by_school[school].months_off_at_rank1 is not None
    ]
    if months_off:
        out["mean_months_off_at_rank1"] = sum(months_off) / len(months_off)
        out["median_months_off_at_rank1"] = sorted(months_off)[
            len(months_off) // 2
        ]
    ranks = [
        r.by_school[school].rank_of_hit for r in eval_rows
        if r.by_school[school].rank_of_hit is not None
    ]
    if ranks:
        out["mean_rank_when_hit"] = sum(ranks) / len(ranks)
        out["median_rank_when_hit"] = sorted(ranks)[len(ranks) // 2]
    return out


def _parse_schools(s: str) -> List[School]:
    out: List[School] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        try:
            out.append(School(tok))
        except ValueError:
            raise ValueError(f"unknown school {tok!r}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--max", type=int, default=30)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--schools", type=str, default="parashari,jaimini,kp",
                   help="comma-separated subset of {parashari,jaimini,kp}")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--out", default=None,
                   help="write per-chart, per-school results as JSON")
    p.add_argument("--workers", type=int, default=1,
                   help="thread pool size (per-chart parallelism). Each "
                        "chart's evaluation is independent; swisseph "
                        "calls release the GIL so threading helps even "
                        "for CPU-bound work.")
    args = p.parse_args()

    schools = _parse_schools(args.schools)
    path = Path(args.dataset)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end if args.end is not None else len(data)
        data = data[s:e]
    elif args.max > 0:
        data = data[:args.max]

    def _format_row(i: int, row: ChartRow) -> str:
        cells: List[str] = []
        for s in schools:
            ps = row.by_school.get(s)
            if ps is None:
                cells.append(f"{s.value}=?")
                continue
            r = ps.rank_of_hit
            r6 = ps.rank_within_6mo
            cells.append(
                f"{s.value}=rank{r if r else '-'} "
                f"r6mo{r6 if r6 else '-'} (n={ps.n_windows})"
            )
        return (f"  [{i}/{len(data)}] {row.chart_name[:30]:30s}  "
                + "  ".join(cells))

    rows: List[ChartRow] = []
    if args.workers <= 1:
        for i, rec in enumerate(data, 1):
            row = _evaluate_one(rec, schools, top_n=args.top_n)
            print(_format_row(i, row), flush=True)
            rows.append(row)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Preserve input ordering in output. Each future returns
        # (input_index, ChartRow); we collect into a slot map.
        slot: Dict[int, ChartRow] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_idx = {
                ex.submit(_evaluate_one, rec, schools, args.top_n): idx
                for idx, rec in enumerate(data, 1)
            }
            done = 0
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                done += 1
                try:
                    row = fut.result()
                except Exception as e:
                    print(f"  [{idx}/{len(data)}] FAILED: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                slot[idx] = row
                print(_format_row(idx, row), flush=True)
                if done % 5 == 0:
                    print(f"  ...progress {done}/{len(data)}", flush=True)
        rows = [slot[k] for k in sorted(slot)]

    print("\n=" * 1, "PER-SCHOOL SUMMARY", "=" * 60)
    summary_by_school: Dict[str, dict] = {}
    for school in schools:
        summary = _summarize_school(rows, school)
        summary_by_school[school.value] = summary
        print(f"\n--- {school.value.upper()} ---")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "summary": summary_by_school,
                "rows": [
                    {
                        "name": r.chart_name,
                        "true_date": (r.true_date.isoformat()
                                      if r.true_date else None),
                        "by_school": {
                            s.value: {
                                "rank_of_hit": ps.rank_of_hit,
                                "rank_within_6mo": ps.rank_within_6mo,
                                "months_off_at_rank1": ps.months_off_at_rank1,
                                "n_windows": ps.n_windows,
                                "inconclusive_reason": ps.inconclusive_reason,
                            }
                            for s, ps in r.by_school.items()
                        },
                    }
                    for r in rows
                ],
            }, f, indent=2)
        print(f"\nwrote per-chart results to {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
