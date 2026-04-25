"""Jaimini feature extractor (spec §6.4, Jaimini branch).

Extracts:
    - Chara karakas (AK, AmK, BK, MK, PuK, GK, DK) — normalized short codes
    - Relation-specific karaka (via RELATION_JAIMINI_KARAKA)
    - Arudha pada of target house
    - Sign occupied by relevant karaka + 8th-from-karaka (longevity trigger)
    - Argala planets (spec §6.4) — planets in 2nd/4th/5th/8th/11th from
      target sign provide intervention; opposing planets block.

Phase 3 limitations (see CAV-020):
    - No Chara dasha — Jaimini timing rules currently borrow the
      Vimshottari dasha tree from Parashari analysis (noted as heuristic).
    - Argala detection simplified — just lists planets in argala houses.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..schemas.chart import Chart
from ..schemas.enums import QueryType, School
from ..schemas.features import FeatureBundle
from ..schemas.focus import ResolvedFocus
from ..resolver.tables import RELATION_JAIMINI_KARAKA
from .classical import (
    house_from,
    maraka_houses,
    occupants_of_house,
)


def _arudha_sign_num(chart: Chart, house_num: int) -> int:
    """Sign num (1-12) of the Arudha pada of `house_num`."""
    return chart.arudhas.get(house_num, 0) if chart.arudhas else 0


def _sign_num(chart: Chart, varga: str, house_num: int) -> int:
    """Sign num of a house in a given varga."""
    rashis = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    ]
    sign = chart.vargas[varga].house_signs.get(house_num, "")
    if not sign:
        return 0
    return rashis.index(sign) + 1


_NATURAL_BENEFICS = {"Jupiter", "Venus", "Moon", "Mercury"}
_NATURAL_MALEFICS = {"Saturn", "Mars", "Sun", "Rahu", "Ketu"}


def _split_polarity(planets: List[str]) -> Dict[str, List[str]]:
    return {
        "benefic": [p for p in planets if p in _NATURAL_BENEFICS],
        "malefic": [p for p in planets if p in _NATURAL_MALEFICS],
    }


def _argala_planets(
    chart: Chart, target_house: int,
) -> Dict[str, Any]:
    """Argala intervention on target house, with benefic/malefic split
    and virodhargala cancellation (CAV-023).

    Subhargala houses from target: 2nd, 4th, 5th, 8th, 11th.
    Virodhargala (cancellation): 12th, 10th, 9th, 6th, 3rd respectively.
    Each (subhargala, virodhargala) pair: argala is *cancelled* if the
    virodhargala has equal-or-stronger intervention.
    """
    d1_positions = chart.vargas["D1"].planet_positions
    pairs = [(2, 12), (4, 10), (5, 9), (8, 6), (11, 3)]
    out: Dict[str, Any] = {
        "argala_houses": [],
        "virodhargala_houses": [],
        "argala_benefic_planets": [],
        "argala_malefic_planets": [],
        "net_argala_polarity": "neutral",
    }
    pos_score = 0
    neg_score = 0
    for arg_off, vir_off in pairs:
        arg_h = house_from(target_house, arg_off)
        vir_h = house_from(target_house, vir_off)
        arg_occs = occupants_of_house(d1_positions, arg_h)
        vir_occs = occupants_of_house(d1_positions, vir_h)
        # Cancellation: virodhargala with malefics cancels malefic argala;
        # benefics cancel benefic argala. Simplified — count per-side
        # nets by polarity-weighted occupant count.
        arg_split = _split_polarity(arg_occs)
        vir_split = _split_polarity(vir_occs)
        net_benefic = max(
            0, len(arg_split["benefic"]) - len(vir_split["benefic"]),
        )
        net_malefic = max(
            0, len(arg_split["malefic"]) - len(vir_split["malefic"]),
        )
        if net_benefic > 0:
            out["argala_houses"].append(arg_h)
            out["argala_benefic_planets"].extend(arg_split["benefic"])
            pos_score += net_benefic
        if net_malefic > 0:
            out["argala_houses"].append(arg_h)
            out["argala_malefic_planets"].extend(arg_split["malefic"])
            neg_score += net_malefic
    if pos_score > neg_score:
        out["net_argala_polarity"] = "benefic"
    elif neg_score > pos_score:
        out["net_argala_polarity"] = "malefic"
    return out


class JaiminiFeatureExtractor:
    """Jaimini school feature extractor."""

    def extract(
        self, chart: Chart, resolved: ResolvedFocus,
    ) -> FeatureBundle:
        if not chart.chara_karakas:
            raise ValueError(
                "Jaimini karakas not computed. "
                "Pass need_jaimini=True to ChartComputer."
            )

        rel_val = resolved.query.relationship.value
        jaimini_karaka_codes = RELATION_JAIMINI_KARAKA.get(rel_val, [])

        # Primary karaka planet for this relationship.
        # PiK code from 8-karaka scheme maps to Pitru (father); we don't
        # yet compute that. Fall back to Putrakaraka only for children.
        # For father with code "PiK", no Jaimini karaka available in
        # 7-karaka scheme — skip and rely on arudha + 8th-from-X analysis.
        relation_karaka_planet = None
        for code in jaimini_karaka_codes:
            planet = chart.chara_karakas.get(code)
            if planet:
                relation_karaka_planet = planet
                break

        target_rot = resolved.target_house_rotated
        target_dir = resolved.target_house_direct

        d1_positions = chart.vargas["D1"].planet_positions

        primary_house_data: Dict[str, Any] = {
            "rotated": {
                "house": target_rot,
                "sign_num": _sign_num(chart, "D1", target_rot),
                "occupants": occupants_of_house(d1_positions, target_rot),
            },
            "direct": {
                "house": target_dir,
                "sign_num": _sign_num(chart, "D1", target_dir),
                "occupants": occupants_of_house(d1_positions, target_dir),
            },
            "argala": _argala_planets(chart, target_rot),
            "arudha_target_rotated_sign": _arudha_sign_num(chart, target_rot),
            "arudha_target_direct_sign": _arudha_sign_num(chart, target_dir),
        }

        # Karaka data (Chara-karaka view).
        karaka_data: Dict[str, Dict[str, Any]] = {}
        for code, planet in chart.chara_karakas.items():
            pos = d1_positions.get(planet)
            if pos is None:
                continue
            karaka_data[code] = {
                "planet": planet,
                "sign": pos.sign,
                "house": pos.house,
                "dignity": pos.dignity,
                # 8th-from-karaka sign — classical longevity trigger:
                "eighth_from_sign_num": ((pos.house - 1 + 7) % 12) + 1,
            }

        # Karakamsha (CAV-024): AK's sign in D9 (Navamsa).
        karakamsha_d9_sign = None
        karakamsha_d9_house = None
        ak = chart.chara_karakas.get("AK")
        if ak and "D9" in chart.vargas:
            d9_pos = chart.vargas["D9"].planet_positions.get(ak)
            if d9_pos is not None:
                karakamsha_d9_sign = d9_pos.sign
                karakamsha_d9_house = d9_pos.house

        # Jaimini-specific payload.
        jaimini_features: Dict[str, Any] = {
            "relation_karaka_code": (
                jaimini_karaka_codes[0]
                if jaimini_karaka_codes else None
            ),
            "relation_karaka_planet": relation_karaka_planet,
            "atmakaraka": chart.chara_karakas.get("AK"),
            "karakamsha_d9_sign": karakamsha_d9_sign,
            "karakamsha_d9_house": karakamsha_d9_house,
            # Backward-compat alias kept for existing rules:
            "karakamsha_house": karaka_data.get("AK", {}).get("house"),
        }

        # Timing: Phase 3 heuristic — reuse Vimshottari tree for Jaimini
        # rules. True Chara dasha is CAV-020.
        dasha_candidates = None
        if resolved.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            # Placeholder — leave empty so only static Jaimini rules fire
            # until Chara dasha is implemented.
            dasha_candidates = []

        return FeatureBundle(
            school=School.JAIMINI,
            focus=resolved,
            primary_house_data=primary_house_data,
            karaka_data=karaka_data,
            varga_features={},
            dasha_candidates=dasha_candidates,
            transit_events=None,
            jaimini_features=jaimini_features,
        )
