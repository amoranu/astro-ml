"""KP feature extractor (spec §6.4, KP branch).

Primary KP signals for a TIMING/EVENT query:
    - Cuspal sub-lord (CSL) of target house — does it signify the
      relevant house cluster (target + 8L + badhaka + marakas)?
    - Significators for each relevant house — union of planets whose
      5-level chain touches the house.
    - DBAS overlap: a dasha-antardasha-pratyantardasha-sookshma period
      whose lords are all significators of the relevant-house cluster
      fires an event-timing candidate.

Extensibility:
    - Reuse of _dasha_candidates pattern from Parashari but role-tagged
      by CSL/significator membership rather than classical lords.
    - Feature paths are exposed through a KP namespace
      (`kp.primary_house_data.*`, `kp.csl_significations.*`, etc.) to
      avoid collision with Parashari feature keys.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas.chart import Chart, DashaNode
from ..schemas.enums import QueryType, School
from ..schemas.features import FeatureBundle
from ..schemas.focus import ResolvedFocus
from .classical import (
    badhaka_house,
    house_from,
    maraka_houses,
)


# ── helpers ──────────────────────────────────────────────────────────

def _csl_of_house(chart: Chart, house_num: int) -> Optional[str]:
    if not chart.kp_cuspal_sublords:
        return None
    return chart.kp_cuspal_sublords.get(house_num)


def _sigs_of_planet(chart: Chart, planet: str) -> List[int]:
    if not chart.kp_significators:
        return []
    return list(chart.kp_significators.get(planet, []))


def _strong_sigs_of_planet(
    chart: Chart, planet: str,
) -> List[int]:
    """Tightened KP significators (CAV-011) — keep only houses signified
    via levels 1-2 (occupation by star-lord OR by planet itself).
    Excludes ownership-only (level 3-4) and sub-lord-star (level 5)
    "null lord" connections per classical KP doctrine.

    Falls back to all-levels if no level 1-2 hits exist.
    """
    if not chart.kp_significators or not chart.kp_planet_houses:
        return _sigs_of_planet(chart, planet)
    if not chart.kp_planet_details:
        return _sigs_of_planet(chart, planet)
    star_lord = chart.kp_planet_details.get(planet, {}).get("star_lord")
    strong: List[int] = []
    # Level 1: house occupied by star_lord
    if star_lord and star_lord in chart.kp_planet_houses:
        strong.append(chart.kp_planet_houses[star_lord])
    # Level 2: house occupied by planet itself
    if planet in chart.kp_planet_houses:
        strong.append(chart.kp_planet_houses[planet])
    # Dedup, preserve order
    seen = set()
    out = []
    for h in strong:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out or _sigs_of_planet(chart, planet)


def _planets_signifying_houses(
    chart: Chart, houses: List[int], strong_only: bool = True,
) -> List[str]:
    """Planets whose significator chain touches ANY of the target houses.

    `strong_only=True` (CAV-011) restricts to level 1-2 connections
    (occupation), excluding ownership-only planets that would otherwise
    over-saturate the cluster.
    """
    if not chart.kp_significators:
        return []
    target = set(houses)
    out = set()
    for p in chart.kp_significators:
        sigs = (
            _strong_sigs_of_planet(chart, p) if strong_only
            else _sigs_of_planet(chart, p)
        )
        if set(sigs) & target:
            out.add(p)
    return sorted(out)


def _walk_dasha(root: Optional[DashaNode]) -> List[DashaNode]:
    out: List[DashaNode] = []

    def rec(n: DashaNode):
        for c in n.children:
            out.append(c)
            rec(c)
    if root is None:
        return out
    rec(root)
    return out


def _kp_dasha_candidates(
    chart: Chart, relevant_houses: List[int],
) -> List[Dict[str, Any]]:
    """KP-style: any MD/AD/PAD whose lord-chain consists of STRONG
    significators of the relevant-house set (CAV-011 — only level 1-2
    occupation links count).
    """
    out: List[Dict[str, Any]] = []
    if chart.vimshottari is None or not chart.kp_significators:
        return out
    target_houses = set(relevant_houses)

    def signifies(planet: str) -> bool:
        return bool(
            set(_strong_sigs_of_planet(chart, planet)) & target_houses
        )

    for md in chart.vimshottari.children:
        md_sig = signifies(md.lord)
        for ad in md.children:
            ad_sig = signifies(ad.lord)
            if not (md_sig or ad_sig):
                continue
            pad_hit = False
            for pad in ad.children:
                pad_sig = signifies(pad.lord)
                # Strong KP candidate: ALL three lords signify the cluster.
                if md_sig and ad_sig and pad_sig:
                    out.append({
                        "level": "PAD",
                        "md": md.lord, "ad": ad.lord, "pad": pad.lord,
                        "start": pad.start.isoformat(),
                        "end": pad.end.isoformat(),
                        "all_signify": True,
                        "signifying_lords": sorted(
                            {md.lord, ad.lord, pad.lord}
                        ),
                        "reason": "DBA chain fully signifies target cluster",
                    })
                    pad_hit = True
                elif md_sig and ad_sig:
                    # MD+AD signify; PAD doesn't — weaker candidate.
                    # Record at AD level once (after the PAD loop).
                    pass
            if not pad_hit and md_sig and ad_sig:
                out.append({
                    "level": "AD",
                    "md": md.lord, "ad": ad.lord, "pad": None,
                    "start": ad.start.isoformat(),
                    "end": ad.end.isoformat(),
                    "all_signify": False,
                    "signifying_lords": sorted({md.lord, ad.lord}),
                    "reason": "MD+AD signify target cluster; PAD misses",
                })
    return out


class KPFeatureExtractor:
    """KP school feature extractor for TIMING queries."""

    def extract(
        self, chart: Chart, resolved: ResolvedFocus,
    ) -> FeatureBundle:
        if chart.kp_cuspal_sublords is None:
            raise ValueError(
                "KP chart not computed. Pass need_kp=True to ChartComputer."
            )

        target_rot = resolved.target_house_rotated
        target_dir = resolved.target_house_direct

        # KP relevant-house cluster is effect-specific. For event_negative
        # (death/illness), the cluster is houses that INDICATE harm to the
        # relationship's significations, NOT the relationship's own house:
        #   - target_rotated (= 8th-from-direct for longevity queries)
        #   - marakas-from-direct (2nd + 7th from the relationship's house)
        #   - badhaka-from-direct
        #   - 12th-from-direct (loss)
        # target_direct itself (e.g. 9H for father) is sustaining, not
        # death-indicating; excluded from the negative cluster.
        effect_val = resolved.query.effect.value
        if effect_val == "event_negative":
            direct_sign = chart.vargas["D1"].house_signs[target_dir]
            badhaka_from_direct = house_from(
                target_dir, badhaka_house(direct_sign),
            )
            relevant_houses = sorted({
                target_rot,                            # 8th-from-direct
                *maraka_houses(target_dir),            # 2nd + 7th from direct
                badhaka_from_direct,
                house_from(target_dir, 12),            # loss-from-direct
            })
        else:
            # For positive/nature/magnitude queries, the cluster centers
            # on the direct/rotated target houses themselves.
            relevant_houses = sorted({target_rot, target_dir})

        # ── Primary house data (KP flavour) ─────────────────────────
        rotated_csl = _csl_of_house(chart, target_rot)
        direct_csl = _csl_of_house(chart, target_dir)
        rotated_csl_sigs = _sigs_of_planet(chart, rotated_csl or "")
        direct_csl_sigs = _sigs_of_planet(chart, direct_csl or "")

        signifiers_of_cluster = _planets_signifying_houses(
            chart, relevant_houses,
        )
        cluster_set = set(relevant_houses)

        primary_house_data: Dict[str, Any] = {
            "rotated": {
                "house": target_rot,
                "csl": rotated_csl,
                "csl_signifies": rotated_csl_sigs,
                "csl_signifies_negative_cluster": bool(
                    set(rotated_csl_sigs) & cluster_set
                ),
            },
            "direct": {
                "house": target_dir,
                "csl": direct_csl,
                "csl_signifies": direct_csl_sigs,
                "csl_signifies_negative_cluster": bool(
                    set(direct_csl_sigs) & cluster_set
                ),
            },
            "relevant_houses": relevant_houses,
            "signifiers_of_cluster": signifiers_of_cluster,
        }

        # ── Karaka data (KP angle): sub-lords of relation/domain karakas ──
        karaka_data: Dict[str, Dict[str, Any]] = {}
        for k in set(resolved.relation_karakas + resolved.domain_karakas):
            if chart.kp_planet_details is None:
                continue
            kd = chart.kp_planet_details.get(k)
            if kd is None:
                continue
            karaka_data[k] = {
                "planet": k,
                "star_lord": kd.get("star_lord"),
                "sub_lord": kd.get("sub_lord"),
                "sub_lord_star_lord": kd.get("sub_lord_star_lord"),
                "placidus_house": (
                    chart.kp_planet_houses.get(k)
                    if chart.kp_planet_houses else None
                ),
                "significators": _sigs_of_planet(chart, k),
            }

        # ── DBAS-style dasha candidates ────────────────────────────
        dasha_candidates = None
        if resolved.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            dasha_candidates = _kp_dasha_candidates(chart, relevant_houses)

        return FeatureBundle(
            school=School.KP,
            focus=resolved,
            primary_house_data=primary_house_data,
            karaka_data=karaka_data,
            varga_features={},
            dasha_candidates=dasha_candidates,
            transit_events=None,
            kp_features={
                "relevant_houses": relevant_houses,
                "signifiers_of_cluster": signifiers_of_cluster,
            },
        )
