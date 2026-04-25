"""Evaluation harness for father-longevity CF rules.

Loads the verified subject set, runs `predict_extreme_epoch` over a
3-year window per chart, and reports both dasha-level matching
(Hit@MD/AD/PD/SD) and distance metrics (|days_off|, PD-distance
within matched AD).

Distance metrics are robust to ±day-level data noise that exact
PD/SD matching over-penalizes. Use them as the primary success
signal at small sample sizes.

Run:
  python -X utf8 -m astroql.applications.father_longevity.eval [version]

where [version] is v12 | v13 | v14 | v15 (default: current = v15).
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from astroql.engine import epoch_emitter as _ee
from astroql.engine.cf_predict import (
    PredictionResult,
    predict_extreme_epoch,
)
from astroql.schemas.birth import BirthDetails

from .rules import ALL_VERSIONS, RULES

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = Path(__file__).resolve().parent / "data" / "verified.json"
WINDOW_YEARS = 3.0
SPLIT_SEED = 42


def _to_birth(rec: Dict[str, Any]) -> BirthDetails:
    return BirthDetails(
        date=date.fromisoformat(rec["birth_date"][:10]),
        time=rec["birth_time"],
        tz=rec["tz"],
        lat=float(rec["lat"]),
        lon=float(rec["lon"]),
    )


def _window_for(
    truth: date, birth_tz: str, rng: random.Random,
) -> tuple[datetime, datetime, float, datetime]:
    """Return (start, end, offset_years, truth_dt).

    truth_dt is constructed in birth.tz so it aligns with the frame
    used by epoch_emitter._parse_sd_date. Earlier UTC framing
    caused a ±1-day shift at the truth-epoch matching step.
    """
    offset_years = rng.uniform(0.0, WINDOW_YEARS)
    days_before = int(offset_years * 365.25)
    days_after = int((WINDOW_YEARS - offset_years) * 365.25)
    truth_dt = datetime(truth.year, truth.month, truth.day,
                        tzinfo=ZoneInfo(birth_tz))
    return (
        truth_dt - timedelta(days=days_before),
        truth_dt + timedelta(days=days_after),
        offset_years,
        truth_dt,
    )


def _dasha_match_level(pred_dashas, truth_dashas) -> str:
    if (pred_dashas.sookshma == truth_dashas.sookshma
        and pred_dashas.pratyantar == truth_dashas.pratyantar
        and pred_dashas.antar == truth_dashas.antar
        and pred_dashas.maha == truth_dashas.maha):
        return "SD"
    if (pred_dashas.pratyantar == truth_dashas.pratyantar
        and pred_dashas.antar == truth_dashas.antar
        and pred_dashas.maha == truth_dashas.maha):
        return "PD"
    if (pred_dashas.antar == truth_dashas.antar
        and pred_dashas.maha == truth_dashas.maha):
        return "AD"
    if pred_dashas.maha == truth_dashas.maha:
        return "MD"
    return "NONE"


def _abbrev_dashas(d) -> str:
    return (f"{d.maha[:3]}-{d.antar[:3]}-"
            f"{d.pratyantar[:3]}-{d.sookshma[:3]}")


def _pd_index(epochs: List, target_ep) -> int:
    """Sequential index of a PD-bucket within its AD."""
    if target_ep is None:
        return -1
    same_md_ad = [
        ep for ep in epochs
        if ep.dashas.maha == target_ep.dashas.maha
        and ep.dashas.antar == target_ep.dashas.antar
    ]
    seen_pds: List[tuple] = []
    for ep in sorted(same_md_ad, key=lambda e: e.start_time):
        key = (ep.dashas.maha, ep.dashas.antar, ep.dashas.pratyantar)
        if key not in seen_pds:
            seen_pds.append(key)
    target_key = (
        target_ep.dashas.maha,
        target_ep.dashas.antar,
        target_ep.dashas.pratyantar,
    )
    return seen_pds.index(target_key) if target_key in seen_pds else -1


def evaluate(rules, records, *, seed: int = SPLIT_SEED,
             verbose: bool = True) -> Dict[str, Any]:
    """Run prediction over `records` and return aggregate stats."""
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    if verbose:
        print(
            f"\n{'#':>2} {'id':<5} {'subject':<22} "
            f"{'truth':<11} {'pred':<11} {'days_off':>8} "
            f"{'truth_dashas':<22} {'pred_dashas':<22} {'match':<5}"
        )
        print("-" * 130)

    for i, rec in enumerate(records):
        truth = date.fromisoformat(rec["father_death_date"][:10])
        birth = _to_birth(rec)
        start, end, _, truth_dt = _window_for(truth, birth.tz, rng)
        rid = str(rec.get("id", "?"))[:5]
        subject = (rec.get("name") or "?")[:22]

        try:
            pred = predict_extreme_epoch(
                birth, start, end, rules,
                polarity="negative",
                max_window_years=WINDOW_YEARS + 0.5,
            )
            all_epochs = _ee.emit_epochs(
                birth, start, end,
                max_window_years=WINDOW_YEARS + 0.5,
            )
        except Exception as e:
            if verbose:
                print(f"{i:>2} {rid:<5} {subject:<22} ERROR: "
                      f"{type(e).__name__}: {e}")
            continue

        truth_ep = next(
            (ep for ep in all_epochs
             if ep.start_time <= truth_dt <= ep.end_time),
            None,
        )
        pred_ep = (
            next((ep for ep in all_epochs
                  if ep.epoch_id == pred.extreme_epoch.epoch_id),
                 None)
            if pred else None
        )

        if pred is None:
            match = "NONE"
            pred_str = "NO_PREDICTION"
            days_off = None
            truth_d = (_abbrev_dashas(truth_ep.dashas)
                       if truth_ep else "-")
            pred_d = "-"
        else:
            pred_str = pred.predicted_date.isoformat()
            days_off = (pred.predicted_date - truth).days
            truth_d = (_abbrev_dashas(truth_ep.dashas)
                       if truth_ep else "-")
            pred_d = (_abbrev_dashas(pred_ep.dashas)
                      if pred_ep else "-")
            match = (
                _dasha_match_level(pred_ep.dashas, truth_ep.dashas)
                if (pred_ep and truth_ep) else "NONE"
            )

        if verbose:
            days_off_str = (f"{days_off:+5d}" if days_off is not None
                            else "  -  ")
            print(
                f"{i:>2} {rid:<5} {subject:<22} "
                f"{truth.isoformat():<11} {pred_str:<11} "
                f"{days_off_str:>8} "
                f"{truth_d:<22} {pred_d:<22} {match:<5}"
            )

        pd_dist = None
        if (pred_ep is not None and truth_ep is not None
            and pred_ep.dashas.maha == truth_ep.dashas.maha
            and pred_ep.dashas.antar == truth_ep.dashas.antar):
            pred_idx = _pd_index(all_epochs, pred_ep)
            truth_idx = _pd_index(all_epochs, truth_ep)
            if pred_idx >= 0 and truth_idx >= 0:
                pd_dist = abs(pred_idx - truth_idx)

        results.append({
            "i": i, "id": rid, "subject": subject,
            "truth": truth.isoformat(),
            "pred_date": pred.predicted_date.isoformat() if pred else None,
            "days_off": days_off,
            "pd_dist_within_ad": pd_dist,
            "match_level": match,
        })

    n = len(results)
    levels = [r["match_level"] for r in results]
    md = sum(1 for m in levels if m in ("MD", "AD", "PD", "SD"))
    ad = sum(1 for m in levels if m in ("AD", "PD", "SD"))
    pd = sum(1 for m in levels if m in ("PD", "SD"))
    sd = sum(1 for m in levels if m == "SD")

    days_offs = [abs(r["days_off"]) for r in results
                 if r["days_off"] is not None]
    pd_dists = [r["pd_dist_within_ad"] for r in results
                if r["pd_dist_within_ad"] is not None]

    if verbose:
        print(f"\nDasha-level hits (deeper = stricter):")
        print(f"  Hit@MD = {md}/{n} = {md/max(n,1):.0%}")
        print(f"  Hit@AD = {ad}/{n} = {ad/max(n,1):.0%}")
        print(f"  Hit@PD = {pd}/{n} = {pd/max(n,1):.0%}")
        print(f"  Hit@SD = {sd}/{n} = {sd/max(n,1):.0%}")
        if days_offs:
            print(f"\nDistance metrics (lower = better):")
            print(f"  |days_off|: median={median(days_offs):.0f}d  "
                  f"mean={mean(days_offs):.0f}d  max={max(days_offs)}d")
            print(f"   ≤30d: {sum(1 for d in days_offs if d<=30)}/{n}  "
                  f"≤90d: {sum(1 for d in days_offs if d<=90)}/{n}  "
                  f"≤180d: {sum(1 for d in days_offs if d<=180)}/{n}")
        if pd_dists:
            print(f"  PD-distance within matched AD: "
                  f"median={median(pd_dists):.0f}  "
                  f"mean={mean(pd_dists):.1f}  "
                  f"(n={len(pd_dists)} of AD-matched cases)")

    return {
        "n": n, "results": results,
        "hits": {"MD": md, "AD": ad, "PD": pd, "SD": sd},
        "days_offs": days_offs, "pd_dists": pd_dists,
    }


def main() -> None:
    version = (sys.argv[1] if len(sys.argv) > 1 else "current").lower()
    if version == "current":
        rules = RULES
        version_label = "current (v15)"
    elif version in ALL_VERSIONS:
        rules = ALL_VERSIONS[version]
        version_label = version
    else:
        print(f"Unknown version: {version}. Valid: current, "
              f"{list(ALL_VERSIONS.keys())}")
        sys.exit(1)

    print(f"Ruleset: {version_label} ({len(rules)} rules)")
    print(f"Loading verified GT from {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  {len(records)} verified records.")
    evaluate(rules, records, verbose=True)


if __name__ == "__main__":
    main()
