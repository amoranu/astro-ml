"""Stratified subject selection for inverted-mining.

Targets 100 charts balanced across (gender x age-at-father-loss).

Strata: 5 age buckets x 2 genders = 10 cells. Target 10/cell. F/60+ has
only 2 records in the source dataset, so the 8-record shortfall is
redistributed across the other F cells (+2 each).

Usage:
  python -m astroql.inverted_mining.select_subjects \
      --src ml/father_passing_date_clean.json \
      --out astroql/inverted_mining/data/subjects.json \
      --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

AGE_BUCKETS = [
    ("<15",    0,   15),
    ("15-30", 15,   30),
    ("30-45", 30,   45),
    ("45-60", 45,   60),
    ("60+",   60,  120),
]


def _bucket(age: float) -> str:
    for label, lo, hi in AGE_BUCKETS:
        if lo <= age < hi:
            return label
    return "60+"


def _age_years(record: dict) -> float | None:
    try:
        b = datetime.strptime(record["birth_date"], "%Y-%m-%d")
        d = datetime.strptime(record["father_death_date"], "%Y-%m-%d")
        return (d - b).days / 365.25
    except Exception:
        return None


def _gender(record: dict) -> str | None:
    g = record.get("gender")
    if g in ("M", "F"):
        return g
    return None


def select(src_path: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    # bucket records: (gender, age-bucket) -> [records...]
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in data:
        g = _gender(r)
        a = _age_years(r)
        if g is None or a is None or a < 0 or a > 90:
            continue
        # require birth_time too (snapshot needs it)
        if not r.get("birth_time"):
            continue
        buckets[(g, _bucket(a))].append(r)

    # quotas: balanced (gender, age) target ~ n / 10 per cell, with
    # F/60+ shortfall redistributed across other F cells.
    base = n // 10  # 10 with n=100
    quota: dict[tuple[str, str], int] = {}
    for g in ("M", "F"):
        for label, _, _ in AGE_BUCKETS:
            quota[(g, label)] = base

    # redistribute F/60+ shortfall
    f_60plus_avail = len(buckets.get(("F", "60+"), []))
    if f_60plus_avail < base:
        deficit = base - f_60plus_avail
        quota[("F", "60+")] = f_60plus_avail
        # spread across other F cells, +1 each round-robin
        f_other = [("F", lbl) for lbl, _, _ in AGE_BUCKETS if lbl != "60+"]
        i = 0
        while deficit > 0 and any(quota[c] < len(buckets.get(c, [])) for c in f_other):
            cell = f_other[i % len(f_other)]
            if quota[cell] < len(buckets.get(cell, [])):
                quota[cell] += 1
                deficit -= 1
            i += 1

    # sample
    chosen: list[dict] = []
    rows: list[dict] = []
    for cell, q in quota.items():
        avail = buckets.get(cell, [])
        take = min(q, len(avail))
        picks = rng.sample(avail, take) if take < len(avail) else list(avail)
        for r in picks:
            chosen.append(r)
            rows.append({
                "subject_id": r["id"],
                "name": r["name"],
                "gender": r["gender"],
                "age_at_loss": round(_age_years(r), 2),
                "age_bucket": _bucket(_age_years(r)),
                "birth_date": r["birth_date"],
                "birth_time": r["birth_time"],
                "lat": r["lat"],
                "lon": r["lon"],
                "tz": r["tz"],
                "father_death_date": r["father_death_date"],
                "time_rating": r.get("time_rating", ""),
                "adb_slug": r.get("adb_slug", ""),
            })

    # sanity log
    out_buckets = defaultdict(int)
    for row in rows:
        out_buckets[(row["gender"], row["age_bucket"])] += 1
    print(f"selected n={len(rows)} from src n={len(data)}")
    print("strata:")
    for cell in sorted(out_buckets.keys()):
        print(f"  {cell}: {out_buckets[cell]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="ml/father_passing_date_clean.json")
    ap.add_argument("--out", default="astroql/inverted_mining/data/subjects.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = select(args.src, args.n, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
