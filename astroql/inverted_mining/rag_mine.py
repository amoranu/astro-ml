"""RAG mining (Phase C).

For each (snapshot × tradition) pair, build a query payload that
describes the death state, plus 3 dynamic per-feature queries, and call
astro-prod's `retrieve_for_tradition` (8-pass + dynamic).

Saves retrieved passages to:
  astroql/inverted_mining/data/rag_results/{subject_id}_{tradition}.json

Each result file contains the full payload echo (so it's reproducible)
and the passages with scores + sources.

Usage:
  python -m astroql.inverted_mining.rag_mine \
      --snapshots astroql/inverted_mining/data/snapshots \
      --out astroql/inverted_mining/data/rag_results
  # single subject + tradition for debug:
  python -m astroql.inverted_mining.rag_mine --subject-id 1 --tradition parashari
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_ASTRO_PROD_PATH = Path(
    "C:/Users/ravii/.gemini/antigravity/playground/astro-prod"
)
if str(_ASTRO_PROD_PATH) not in sys.path:
    sys.path.insert(0, str(_ASTRO_PROD_PATH))

try:
    import rag_engine as _rag  # type: ignore
except Exception as e:
    _rag = None
    log.warning("rag_engine unavailable: %s", e)


_GENDER_WORD = {"M": "male native", "F": "female native"}


def _bhavat_bhavam(direct_house: int) -> int:
    """Return 'house from house': 9th house from 9th house = 5th."""
    return ((direct_house - 1) + (direct_house - 1)) % 12 + 1


def build_payload(snapshot: Dict[str, Any], tradition: str) -> Optional[Dict[str, Any]]:
    """Build the retrieve_for_tradition kwargs from a snapshot bundle."""
    bundle = snapshot.get("traditions", {}).get(tradition)
    if not bundle or "error" in bundle:
        return None

    chart_static = bundle.get("chart_static", {})
    karaka_data = chart_static.get("karaka_data", {}) or {}
    primary_house = chart_static.get("primary_house_data", {}) or {}
    direct = primary_house.get("direct", {}) or {}
    rotated = primary_house.get("rotated", {}) or {}
    leaf = (bundle.get("death_window", {}) or {}).get("leaf")

    gender = snapshot.get("gender")
    gender_word = _GENDER_WORD.get(gender, "native")
    age = snapshot.get("age_at_loss") or 0
    age_int = int(age)

    # ── query string ─────────────────────────────────────────
    parts: List[str] = [
        f"Father death timing for {gender_word} aged {age_int} years.",
        "Pitri karaka Sun, ayushkaraka Saturn, longevity for father, "
        "9th house pitru bhava, marana maraka dasha activation, "
        "transit overlay sade sati ashtama shani.",
    ]
    if leaf:
        chain_bits = []
        if leaf.get("md"):
            chain_bits.append(f"Mahadasha {leaf['md']}")
        if leaf.get("ad"):
            chain_bits.append(f"Antardasha {leaf['ad']}")
        if leaf.get("pad"):
            chain_bits.append(f"Pratyantar dasha {leaf['pad']}")
        if chain_bits:
            parts.append(", ".join(chain_bits) + ".")
        cs = leaf.get("chain_strength", "")
        roles = leaf.get("lord_roles_present", []) or []
        if cs or roles:
            parts.append(
                f"Chain strength {cs}. Active roles: {', '.join(roles) or 'none'}."
            )
    query = " ".join(parts)

    # ── houses ──────────────────────────────────────────────
    # Tradition-specific target + bhavat-bhavam (house-from-house) +
    # marakas (2H, 7H) + 8H (death) + 12H (loss).
    h_dir = int(direct.get("house") or 9)
    h_rot = int(rotated.get("house") or 4)
    houses = sorted(set([
        h_dir, h_rot,
        _bhavat_bhavam(h_dir),  # 9-from-9 = 5
        2, 7, 8, 12,
    ]))

    # ── planets ─────────────────────────────────────────────
    planets: List[str] = []
    if leaf:
        for p in leaf.get("matched_lords", []) or []:
            if p:
                planets.append(p)
    for p in ("Sun", "Saturn", "Jupiter"):  # always include karakas
        planets.append(p)
    planets = list(dict.fromkeys(planets))  # dedupe, keep order

    # ── natal_context (v2: full enrichment) ─────────────────
    # Pass ALL planets with rashi+house+nakshatra+dignity (was just
    # rashi+house). The natal_suffix in retrieve_for_tradition only
    # uses the first 4 karaka planets, but having richer per-planet
    # data means downstream consumers (e.g., dignity_features below)
    # work properly.
    nc_planets: Dict[str, Dict[str, Any]] = {}
    for pname, pdata in karaka_data.items():
        if not isinstance(pdata, dict):
            continue
        nc_planets[pname] = {
            "rashi": pdata.get("sign", ""),
            "house": pdata.get("house", ""),
            "nakshatra": pdata.get("nakshatra", ""),
            "nakshatra_lord": pdata.get("nakshatra_lord", ""),
            "dignity": pdata.get("dignity", ""),
            "retrograde": pdata.get("retrograde", False),
            "combust": pdata.get("combust", False),
        }
    natal_context: Dict[str, Any] = {"planets": nc_planets}

    # Lagna: prefer the proper value from natal_meta (v2 snapshot),
    # fall back to inferring from any planet in 1H (v1 behavior).
    natal_meta = snapshot.get("natal_meta") or {}
    lagna_sign = natal_meta.get("lagna_sign", "")
    if not lagna_sign:
        for pname, pdata in karaka_data.items():
            if isinstance(pdata, dict) and pdata.get("house") == 1:
                lagna_sign = pdata.get("sign", "")
                break
    if lagna_sign:
        natal_context["lagna"] = {"sign": lagna_sign, "rashi": lagna_sign}

    # Full MD/AD/PAD chain at the death window (v2: was MD+AD only).
    if leaf:
        natal_context["dasha"] = {
            "md": leaf.get("md"),
            "ad": leaf.get("ad"),
            "pad": leaf.get("pad"),
            "current_date_dasha": leaf.get("md"),
            "current_antardasha": leaf.get("ad"),
        }

    # Rotated-target context (v2): so passes that mention rotated lord
    # / 9-from-9 see the bhavat-bhavam state. Embedded as a synthetic
    # "rotated_context" key — rag_engine doesn't read this directly,
    # but the dignity_features helper and dynamic_queries can.
    if rotated:
        natal_context["rotated_target"] = {
            "house_in_rotated": rotated.get("house"),
            "sign": rotated.get("sign"),
            "lord": rotated.get("lord"),
            "lord_house": rotated.get("lord_house"),
            "lord_dignity": rotated.get("lord_dignity"),
        }

    # ── dignity_features (unlocks pass 7 CHART_EXCEPTIONS) ──
    # Built via astro-prod's compute_dignity_features. Needs
    # lagna + ALL 9 planets (not just karakas) + navamsa for accurate
    # yogakaraka/vargottama/lagna-lord-state detection.
    if lagna_sign:
        navamsa = (natal_meta.get("navamsa") or {}).get("planets") or {}
        all_planets = natal_meta.get("all_planets") or {}
        dig_planets = dict(all_planets) if all_planets else dict(nc_planets)
        # Ensure every planet entry is dict-shaped {rashi, house, ...}
        dig_natal = {
            "lagna": {"rashi": lagna_sign},
            "planets": dig_planets,
            "dasha": natal_context.get("dasha", {}),
        }
        try:
            from focus_areas.common.dignity_features import (
                compute_dignity_features,
            )
            dig = compute_dignity_features(dig_natal, navamsa)
            natal_context["dignity_features"] = dig
        except Exception as e:
            log.debug("dignity_features compute failed: %s", e)
        # Also overlay all_planets into natal_context.planets so the
        # natal_suffix template (top 4 of 'planets') has more material
        # to choose from when 'planets' arg is sparse.
        if all_planets:
            for p, pdata in all_planets.items():
                if p not in nc_planets:
                    nc_planets[p] = pdata

    # ── sub_topics (gender-aware, multi-lingual, v2 expanded) ────
    sub_topics = [
        f"father longevity death timing {gender_word}",
        "pitri marana maraka ayur ayushkaraka 9th house",
        "father death dasha transit gochar saturn maraka",
        # v2: bhavat-bhavam + arudha-aware topics
        "rotated lagna 9th from 9th bhavat bhavam pitri",
        "yogakaraka vargottama lagna lord state pitri marana",
    ]

    return {
        "query": query,
        "tradition": tradition,
        "houses": houses,
        "planets": planets,
        "category": "longevity",
        "natal_context": natal_context,
        "sub_topics": sub_topics,
    }


def build_dynamic_queries(snapshot: Dict[str, Any], tradition: str) -> List[str]:
    """3 tight queries (cap = 3 inside retrieve_dynamic) targeting the
    most distinctive features of THIS death state."""
    bundle = snapshot.get("traditions", {}).get(tradition)
    if not bundle or "error" in bundle:
        return []
    leaf = (bundle.get("death_window", {}) or {}).get("leaf") or {}
    chart_static = bundle.get("chart_static", {})
    karaka_data = chart_static.get("karaka_data", {}) or {}
    primary_house = chart_static.get("primary_house_data", {}) or {}
    direct = primary_house.get("direct", {}) or {}

    queries: List[str] = []

    # Q1: dasha activation pattern
    md = leaf.get("md", "")
    ad = leaf.get("ad", "")
    pad = leaf.get("pad", "") or ""
    roles = leaf.get("lord_roles_present", []) or []
    if md or ad:
        bits = []
        if md: bits.append(f"{md} mahadasha")
        if ad: bits.append(f"{ad} antardasha")
        if pad: bits.append(f"{pad} pratyantar")
        roles_str = ", ".join(roles) if roles else "no role"
        queries.append(
            " ".join(bits)
            + f" father death pitri marana, chain roles: {roles_str}. "
            "Classical dasha activation for parental death timing."
        )

    # Q2: transit overlay (only fired flags)
    transit_facts = []
    if leaf.get("saturn_over_natal_sun"):
        transit_facts.append("Saturn transit over natal Sun pitri karaka")
    if leaf.get("ashtama_shani"):
        transit_facts.append("Ashtama Shani Saturn 8th house from natal Moon")
    if leaf.get("kantaka_shani"):
        transit_facts.append("Kantaka Shani Saturn 4th from natal Moon")
    if leaf.get("sade_sati"):
        transit_facts.append("Sade Sati Saturn 12th 1st 2nd from Moon")
    if leaf.get("mars_transit_target"):
        transit_facts.append("Mars transit on father house 9th pitru bhava")
    if leaf.get("saturn_transit_target"):
        transit_facts.append("Saturn transit on father house 9th")
    if leaf.get("jupiter_transit_target"):
        transit_facts.append("Jupiter transit on father house 9th benefic protection")
    if leaf.get("mars_saturn_conjunction_transit"):
        transit_facts.append("Mars-Saturn conjunction Mrityu Yoga BPHS")
    if leaf.get("chain_lord_in_8h_from_target"):
        transit_facts.append("Chain dasha lord placed 8th from father karaka")
    if leaf.get("chain_lord_in_mrityu_bhaga"):
        transit_facts.append("Chain dasha lord at Mrityu Bhaga critical degree BPHS Ch.40")
    if transit_facts:
        queries.append(
            "Father death transit overlay: " + "; ".join(transit_facts)
            + ". Gochara classical death timing."
        )

    # Q3 (v2): natal placement + LAGNA-LORD STATE + ROTATED context.
    # Replaces v1's bare Sun/Saturn/9L summary with bhavat-bhavam +
    # dignity-aware framing, so dynamic-pass retrieval surfaces
    # chart-specific exception/cancellation rules.
    sun_h = karaka_data.get("Sun", {}).get("house", "")
    sun_s = karaka_data.get("Sun", {}).get("sign", "")
    sat_h = karaka_data.get("Saturn", {}).get("house", "")
    sat_s = karaka_data.get("Saturn", {}).get("sign", "")
    nineL = direct.get("lord", "")
    nineL_h = direct.get("lord_house", "")
    nineL_d = direct.get("lord_dignity", "")
    natal_meta = snapshot.get("natal_meta") or {}
    lagna_sign = natal_meta.get("lagna_sign", "")
    lagna_lord = natal_meta.get("lagna_lord", "")
    lagna_lord_data = karaka_data.get(lagna_lord, {}) if lagna_lord else {}
    ll_h = lagna_lord_data.get("house", "")
    ll_d = lagna_lord_data.get("dignity", "")
    rotated = (chart_static.get("primary_house_data", {}) or {}).get("rotated", {})
    rotL = rotated.get("lord", "")
    rotL_h = rotated.get("lord_house", "")
    rotL_d = rotated.get("lord_dignity", "")
    queries.append(
        f"Natal father longevity for {lagna_sign} lagna ({lagna_lord} lagna lord "
        f"in house {ll_h} dignity {ll_d}). Sun pitri karaka in {sun_s} "
        f"house {sun_h}; Saturn ayushkaraka in {sat_s} house {sat_h}. "
        f"9th lord direct {nineL} in house {nineL_h} dignity {nineL_d}. "
        f"Rotated 9-lagna lord {rotL} in house {rotL_h} dignity {rotL_d} "
        f"(bhavat bhavam pitru chart). Yogakaraka, vargottama, "
        f"neechabhanga exception classical death timing for father."
    )

    return queries[:3]


def mine_one(
    snapshot: Dict[str, Any], tradition: str,
) -> Optional[Dict[str, Any]]:
    """Run 8-pass + dynamic RAG for one (snapshot, tradition)."""
    if _rag is None or not _rag.is_available():
        log.warning("rag_engine unavailable")
        return None

    payload = build_payload(snapshot, tradition)
    if payload is None:
        return None
    dyn_queries = build_dynamic_queries(snapshot, tradition)

    try:
        dyn_results = (
            _rag.retrieve_dynamic(tradition, dyn_queries) if dyn_queries else []
        )
    except Exception as e:
        log.warning("retrieve_dynamic failed: %s", e)
        dyn_results = []

    try:
        passages = _rag.retrieve_for_tradition(
            query=payload["query"],
            tradition=payload["tradition"],
            houses=payload["houses"],
            planets=payload["planets"],
            category=payload["category"],
            natal_context=payload["natal_context"],
            sub_topics=payload["sub_topics"],
            dynamic_results=dyn_results,
        )
    except Exception as e:
        log.warning("retrieve_for_tradition failed: %s", e)
        passages = []

    return {
        "subject_id": snapshot["subject_id"],
        "name": snapshot["name"],
        "gender": snapshot["gender"],
        "age_at_loss": snapshot["age_at_loss"],
        "tradition": tradition,
        "payload_summary": {
            "query": payload["query"],
            "houses": payload["houses"],
            "planets": payload["planets"],
            "sub_topics": payload["sub_topics"],
            "dynamic_queries": dyn_queries,
        },
        "n_passages": len(passages),
        "n_dynamic_passages": len(dyn_results),
        "passages": passages,
    }


def _process_one(
    sf: Path, tr: str, out_path: Path, force: bool,
) -> Tuple[str, str, int, int]:
    """Worker: load snapshot, mine for tradition, write file. Returns
    (status, label, n_passages, n_dynamic). status: ok/skip/fail/no-payload."""
    label = f"{sf.stem}/{tr}"
    try:
        with open(sf, encoding="utf-8") as f:
            snap = json.load(f)
    except Exception as e:
        return ("fail", f"{label} load:{e}", 0, 0)

    out_file = out_path / f"{snap['subject_id']}_{tr}.json"
    if out_file.exists() and not force:
        return ("skip", label, 0, 0)

    try:
        res = mine_one(snap, tr)
    except Exception as e:
        return ("fail", f"{label} mine:{e}", 0, 0)
    if res is None:
        return ("no-payload", label, 0, 0)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    except Exception as e:
        return ("fail", f"{label} write:{e}", 0, 0)
    return ("ok", f"id={snap['subject_id']} {tr}",
            res["n_passages"], res["n_dynamic_passages"])


def run(
    snapshots_dir: str, out_dir: str,
    only_subject_id: Optional[int] = None,
    only_tradition: Optional[str] = None,
    force: bool = False,
    workers: int = 1,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import sys as _sys

    snap_path = Path(snapshots_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    snap_files = sorted(snap_path.glob("*.json"))
    if not snap_files:
        print(f"no snapshots in {snap_path}")
        return

    traditions = (
        [only_tradition] if only_tradition else ["parashari", "jaimini", "kp"]
    )
    # Build full task list (filter by subject_id if requested).
    tasks: List[Tuple[Path, str]] = []
    for sf in snap_files:
        if only_subject_id is not None:
            try:
                with open(sf, encoding="utf-8") as f:
                    sid = json.load(f).get("subject_id")
            except Exception:
                continue
            if sid != only_subject_id:
                continue
        for tr in traditions:
            tasks.append((sf, tr))

    n_done = n_skip = n_fail = n_nopay = 0
    t0 = time.time()
    total = len(tasks)
    print(f"starting RAG mining: {total} tasks, workers={workers}", flush=True)

    if workers <= 1:
        for i, (sf, tr) in enumerate(tasks, start=1):
            status, label, npas, ndyn = _process_one(sf, tr, out_path, force)
            if status == "ok":
                n_done += 1
            elif status == "skip":
                n_skip += 1
            elif status == "no-payload":
                n_nopay += 1
            else:
                n_fail += 1
            elapsed = time.time() - t0
            eta = (total - i) / max(i / max(elapsed, 0.01), 0.01)
            print(
                f"[{i}/{total}] {status:>6} {label}: "
                f"npas={npas} ndyn={ndyn} | ETA {eta/60:.1f}m",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_task = {
                ex.submit(_process_one, sf, tr, out_path, force): (sf, tr)
                for (sf, tr) in tasks
            }
            i = 0
            for fut in as_completed(future_to_task):
                i += 1
                try:
                    status, label, npas, ndyn = fut.result()
                except Exception as e:
                    status, label, npas, ndyn = ("fail", f"future:{e}", 0, 0)
                if status == "ok":
                    n_done += 1
                elif status == "skip":
                    n_skip += 1
                elif status == "no-payload":
                    n_nopay += 1
                else:
                    n_fail += 1
                elapsed = time.time() - t0
                eta = (total - i) / max(i / max(elapsed, 0.01), 0.01)
                print(
                    f"[{i}/{total}] {status:>6} {label}: "
                    f"npas={npas} ndyn={ndyn} | ETA {eta/60:.1f}m",
                    flush=True,
                )

    print(f"done. ok={n_done} skipped={n_skip} no-payload={n_nopay} "
          f"failed={n_fail} elapsed={(time.time()-t0)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots",
                    default="astroql/inverted_mining/data/snapshots")
    ap.add_argument("--out",
                    default="astroql/inverted_mining/data/rag_results")
    ap.add_argument("--subject-id", type=int, default=None)
    ap.add_argument("--tradition", choices=["parashari", "jaimini", "kp"],
                    default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="thread pool size for parallel RAG calls (I/O bound)")
    args = ap.parse_args()
    run(args.snapshots, args.out,
        only_subject_id=args.subject_id,
        only_tradition=args.tradition,
        force=args.force,
        workers=args.workers)


if __name__ == "__main__":
    main()
