"""Death-state snapshot generator (Phase B.2).

For each subject in subjects.json, computes:
  - Natal chart (D1 + D9)
  - Per-tradition feature bundle (Parashari, Jaimini, KP)
  - The deepest dasha-candidate window containing father_death_date
    (this carries the per-window candidate.* features: matched_lords,
    chain_strength, transit overlays, etc.)

Output: astroql/inverted_mining/data/snapshots/{subject_id}.json

Each snapshot is the input to RAG mining (Phase C). It is intentionally
unbiased by current rule firings — only raw natal features + the
candidate-window features that the existing extractors already compute.

Usage:
  python -m astroql.inverted_mining.snapshot \
      --subjects astroql/inverted_mining/data/subjects.json \
      --out astroql/inverted_mining/data/snapshots
  # single subject:
  python -m astroql.inverted_mining.snapshot --subject-id 1
"""
from __future__ import annotations

import argparse
import json
import logging
import traceback
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chart import ChartComputer
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..schemas.birth import BirthDetails, ChartConfig
from ..schemas.enums import Effect, LifeArea, Relationship, School
from ..schemas.focus import FocusQuery
from ..schemas.features import FeatureBundle

log = logging.getLogger(__name__)

_EXTRACTORS = {
    School.PARASHARI: ParashariFeatureExtractor,
    School.JAIMINI: JaiminiFeatureExtractor,
    School.KP: KPFeatureExtractor,
}


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / datetimes / sets to JSON types."""
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _build_birth(subject: Dict[str, Any]) -> BirthDetails:
    bd = subject["birth_date"]
    bt = subject["birth_time"]
    y, m, d = (int(x) for x in bd.split("-"))
    return BirthDetails(
        date=date(y, m, d),
        time=bt,
        tz=subject["tz"],
        lat=float(subject["lat"]),
        lon=float(subject["lon"]),
        time_accuracy=subject.get("time_rating") or "exact",
    )


def _focus_query(subject: Dict[str, Any], school: School) -> FocusQuery:
    return FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        birth=_build_birth(subject),
        config=ChartConfig(),
        schools=[school],
        gender=subject.get("gender"),
    )


def _candidate_at(
    candidates: List[Dict[str, Any]], target: datetime,
) -> Optional[Dict[str, Any]]:
    """Pick the deepest-leaf candidate whose [start, end] contains target.

    Levels (in order of specificity): SD < PAD < AD < MD. We pick the
    smallest-window candidate that contains the date.
    """
    LEVEL_RANK = {"SD": 0, "PAD": 1, "AD": 2, "MD": 3}
    best: Optional[Dict[str, Any]] = None
    best_rank = 99
    best_span = None
    for c in candidates or []:
        s = c.get("start"); e = c.get("end")
        if not s or not e:
            continue
        try:
            sdt = datetime.fromisoformat(s)
            edt = datetime.fromisoformat(e)
        except Exception:
            continue
        if not (sdt <= target <= edt):
            continue
        lvl = c.get("level", "MD")
        rank = LEVEL_RANK.get(lvl, 99)
        span = (edt - sdt).total_seconds()
        # prefer deeper level; tie-break on shorter span
        if (rank < best_rank) or (rank == best_rank and (
            best_span is None or span < best_span
        )):
            best = c
            best_rank = rank
            best_span = span
    return best


def _candidates_at_each_level(
    candidates: List[Dict[str, Any]], target: datetime,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return one candidate per level (MD, AD, PAD, SD) containing target."""
    out: Dict[str, Optional[Dict[str, Any]]] = {
        "MD": None, "AD": None, "PAD": None, "SD": None,
    }
    for c in candidates or []:
        s = c.get("start"); e = c.get("end")
        if not s or not e:
            continue
        try:
            sdt = datetime.fromisoformat(s)
            edt = datetime.fromisoformat(e)
        except Exception:
            continue
        if not (sdt <= target <= edt):
            continue
        lvl = c.get("level")
        if lvl in out:
            cur = out[lvl]
            if cur is None:
                out[lvl] = c
            else:
                # tighter span wins (shouldn't happen normally)
                cur_s = (datetime.fromisoformat(cur["end"])
                         - datetime.fromisoformat(cur["start"])).total_seconds()
                if (edt - sdt).total_seconds() < cur_s:
                    out[lvl] = c
    return out


def _bundle_chart_static(bundle: FeatureBundle) -> Dict[str, Any]:
    """Pull chart-static fields (independent of dasha window) from bundle."""
    out: Dict[str, Any] = {
        "primary_house_data": bundle.primary_house_data,
        "karaka_data": bundle.karaka_data,
    }
    if bundle.varga_features:
        out["varga_features"] = bundle.varga_features
    if bundle.jaimini_features:
        out["jaimini_features"] = bundle.jaimini_features
    if bundle.kp_features:
        out["kp_features"] = bundle.kp_features
    return out


def _resolved_summary(bundle: FeatureBundle) -> Dict[str, Any]:
    """The resolved-focus values the school used (target houses, karakas)."""
    rf = bundle.focus
    return {
        "target_house_rotated": rf.target_house_rotated,
        "target_house_direct": rf.target_house_direct,
        "relevant_houses": rf.relevant_houses,
        "relation_karakas": rf.relation_karakas,
        "domain_karakas": rf.domain_karakas,
        "jaimini_karakas": rf.jaimini_karakas,
    }


_SIGN_LORD = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury",
    "Cancer": "Moon", "Leo": "Sun", "Virgo": "Mercury",
    "Libra": "Venus", "Scorpio": "Mars", "Sagittarius": "Jupiter",
    "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter",
}


def _natal_chart_metadata(chart) -> Dict[str, Any]:
    """v2: chart-level metadata for richer RAG natal_context.

    Returns:
      - lagna_sign, lagna_lord (computed from D1 house 1)
      - all_planets: {planet: {rashi, house, nakshatra, dignity, ...}}
        — full D1 positions, NOT filtered by karaka relevance, so
        dignity_features can compute lagna-lord state for any lagna
      - navamsa.planets: {planet: {rashi}} for vargottama detection
    """
    out: Dict[str, Any] = {}
    d1 = chart.vargas.get("D1")
    if d1:
        lagna_sign = d1.house_signs.get(1, "")
        out["lagna_sign"] = lagna_sign
        out["lagna_lord"] = _SIGN_LORD.get(lagna_sign, "")
        out["all_planets"] = {
            p: {
                "rashi": pp.sign,
                "house": pp.house,
                "nakshatra": getattr(pp, "nakshatra", ""),
                "dignity": getattr(pp, "dignity", ""),
                "retrograde": bool(getattr(pp, "retrograde", False)),
                "combust": bool(getattr(pp, "combust", False)),
            }
            for p, pp in d1.planet_positions.items()
            if p != "Lagna"
        }
    d9 = chart.vargas.get("D9")
    if d9:
        out["navamsa"] = {
            "planets": {
                p: {"rashi": pp.sign}
                for p, pp in d9.planet_positions.items()
                if p != "Lagna"
            }
        }
    return out


def snapshot_subject(subject: Dict[str, Any]) -> Dict[str, Any]:
    """Build the death-state snapshot for one subject (all 3 traditions)."""
    death_str = subject["father_death_date"]
    y, m, d = (int(x) for x in death_str.split("-"))
    death_dt = datetime(y, m, d, 12, 0)  # noon, tz-naive (matches candidate keys)

    chart = ChartComputer().compute(
        birth=_build_birth(subject),
        config=ChartConfig(),
        vargas=["D1", "D9"],
        dashas=["vimshottari"],
        need_jaimini=True,
        need_kp=True,
    )

    natal_meta = _natal_chart_metadata(chart)

    bundles: Dict[str, Dict[str, Any]] = {}
    resolver = FocusResolver()
    for school in (School.PARASHARI, School.JAIMINI, School.KP):
        try:
            fq = _focus_query(subject, school)
            resolved = resolver.resolve(fq)
            bundle: FeatureBundle = _EXTRACTORS[school]().extract(chart, resolved)

            cands = bundle.dasha_candidates or []
            leaf = _candidate_at(cands, death_dt)
            per_level = _candidates_at_each_level(cands, death_dt)

            bundles[school.value] = {
                "resolved": _resolved_summary(bundle),
                "chart_static": _bundle_chart_static(bundle),
                "death_window": {
                    "leaf": leaf,
                    "by_level": per_level,
                },
                "n_candidates_total": len(cands),
            }
        except Exception as e:
            log.warning(
                "[%s/%s] extractor failed: %s",
                subject.get("subject_id"), school.value, e,
            )
            bundles[school.value] = {"error": f"{type(e).__name__}: {e}"}

    return {
        "subject_id": subject["subject_id"],
        "name": subject["name"],
        "gender": subject["gender"],
        "age_at_loss": subject["age_at_loss"],
        "age_bucket": subject["age_bucket"],
        "birth": {
            "date": subject["birth_date"], "time": subject["birth_time"],
            "tz": subject["tz"], "lat": subject["lat"], "lon": subject["lon"],
            "time_rating": subject.get("time_rating"),
        },
        "natal_meta": natal_meta,
        "father_death_date": death_str,
        "traditions": bundles,
    }


def run(subjects_path: str, out_dir: str, only_id: Optional[int] = None,
        force: bool = False) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(subjects_path, encoding="utf-8") as f:
        subjects = json.load(f)
    if only_id is not None:
        subjects = [s for s in subjects if s["subject_id"] == only_id]
        if not subjects:
            print(f"no subject with id={only_id}")
            return

    n_done = n_skip = n_fail = 0
    for i, subject in enumerate(subjects, start=1):
        sid = subject["subject_id"]
        out_file = out_path / f"{sid}.json"
        if out_file.exists() and not force:
            n_skip += 1
            continue
        try:
            snap = snapshot_subject(subject)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(snap), f, indent=2, ensure_ascii=False)
            n_done += 1
            print(f"[{i}/{len(subjects)}] id={sid} {subject['name']} OK")
        except Exception as e:
            n_fail += 1
            print(f"[{i}/{len(subjects)}] id={sid} {subject['name']} FAIL: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"done. ok={n_done} skipped={n_skip} failed={n_fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects",
                    default="astroql/inverted_mining/data/subjects.json")
    ap.add_argument("--out",
                    default="astroql/inverted_mining/data/snapshots")
    ap.add_argument("--subject-id", type=int, default=None,
                    help="only run for this subject_id (debug)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing snapshots")
    args = ap.parse_args()
    run(args.subjects, args.out, only_id=args.subject_id, force=args.force)


if __name__ == "__main__":
    main()
