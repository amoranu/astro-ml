"""Parashari feature extractor (spec §6.4, Parashari branch).

Takes a Chart + ResolvedFocus, returns a FeatureBundle with:
    primary_house_data — rotated + direct target houses
    karaka_data       — positions/dignity for relation + domain karakas
    varga_features    — key facts per varga_required
    dasha_candidates  — sub-dasha windows where relevant lord fires (TIMING)
    transit_events    — stubbed for Phase 1 (wire in Phase 2+)

Extensibility:
    Jaimini/KP branches live in sibling modules `jaimini.py` / `kp.py`.
    The feature schema (keys produced) is declared in
    `rules/features_schema.yaml` so rule authors see what they can reference.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from datetime import datetime, timedelta

from ..schemas.chart import Chart, DashaNode
from ..schemas.enums import QueryType, School
from ..schemas.features import FeatureBundle
from ..schemas.focus import ResolvedFocus
from .classical import (
    SIGN_LORD,
    badhaka_house,
    compute_father_natal_context,
    compute_functional_roles,
    detect_neecha_bhanga,
    detect_vipreet_raja_yoga,
    house_from,
    in_mrityu_bhaga,
    maraka_houses,
    occupants_of_house,
    planets_aspecting_house,
)


def _compute_transit_table(
    chart: Chart, planets: list, n_months: int = 1200,
) -> dict:
    """Precompute monthly transit positions for slow movers.

    Returns {planet: [(month_dt, sign_num, longitude_deg)]} sampled
    monthly from birth. Saturn moves ~1 sign / 2.5 years, Jupiter
    ~1 sign / year, so monthly sampling is sufficient for sign and
    nakshatra precision (each nakshatra is 13.333° = ~5.5 months for
    Saturn, ~1.3 months for Jupiter — Jupiter just at the edge, but
    the misclassification rate is acceptable for an aspect-class signal).
    """
    import sys
    from pathlib import Path
    _APATH = Path("C:/Users/ravii/.gemini/antigravity/playground/astro-prod")
    if str(_APATH) not in sys.path:
        sys.path.insert(0, str(_APATH))
    from astro_engine import AstroEngine
    eng = AstroEngine()

    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return {}

    birth = chart.birth
    try:
        zone = ZoneInfo(birth.tz)
    except Exception:
        return {}
    bd = datetime(birth.date.year, birth.date.month, birth.date.day,
                  12, 0, 0, tzinfo=zone)

    out = {p: [] for p in planets}
    cursor = bd
    for _ in range(n_months):
        try:
            positions = eng.calculate_planetary_positions(
                cursor, birth.lat, birth.lon,
            )
            for p in planets:
                pdata = positions.get(p)
                if pdata:
                    out[p].append((
                        cursor,
                        int(pdata["sign_num"]),
                        float(pdata.get("longitude") or 0.0),
                    ))
        except Exception:
            pass
        cursor = cursor + timedelta(days=30)
    return out


def _nakshatra_index(longitude_deg: float) -> int:
    """1-based nakshatra (1=Ashwini ... 27=Revati). Each is 13°20'."""
    return int(longitude_deg // (360.0 / 27.0)) + 1


def _closest_transit_row(
    transit_table: dict, planet: str, when: datetime,
):
    """Return the nearest sample row for (planet, when), or None."""
    series = transit_table.get(planet, [])
    if not series:
        return None
    when_naive = when.replace(tzinfo=None) if when.tzinfo else when
    return min(
        series,
        key=lambda r: abs(
            (r[0].replace(tzinfo=None) - when_naive).total_seconds()
        ),
    )


def _transit_active_at(
    transit_table: dict, planet: str, when: datetime,
    target_sign_nums: set,
) -> bool:
    """True iff `planet` is in any of `target_sign_nums` at `when`
    (nearest-month lookup)."""
    closest = _closest_transit_row(transit_table, planet, when)
    if closest is None:
        return False
    return closest[1] in target_sign_nums


def _transit_in_nakshatras(
    transit_table: dict, planet: str, when: datetime,
    target_nakshatras: set,
) -> bool:
    """True iff `planet`'s longitude at `when` falls in any of the
    target nakshatras (1-based indices)."""
    closest = _closest_transit_row(transit_table, planet, when)
    if closest is None or len(closest) < 3:
        return False
    return _nakshatra_index(closest[2]) in target_nakshatras


def _house_sign(chart: Chart, varga: str, house_num: int) -> str:
    return chart.vargas[varga].house_signs[house_num]


def _lord_of_house(chart: Chart, varga: str, house_num: int) -> str:
    return SIGN_LORD[_house_sign(chart, varga, house_num)]


def _planet_pos(chart: Chart, varga: str, planet: str):
    return chart.vargas[varga].planet_positions.get(planet)


def _resolve_lord_tokens(
    chart: Chart, target_house: int, tokens: List[str],
) -> List[str]:
    """Expand domain-karaka tokens like '8L' → 'Saturn' relative to target."""
    out: List[str] = []
    for t in tokens:
        if len(t) >= 2 and t[-1] == "L" and t[:-1].isdigit():
            offset = int(t[:-1])
            from_house = house_from(target_house, offset)
            out.append(_lord_of_house(chart, "D1", from_house))
        else:
            out.append(t)
    return out


def _primary_house(chart: Chart, house_num: int) -> Dict[str, Any]:
    """Summarise a target house for rule consumption."""
    sign = _house_sign(chart, "D1", house_num)
    lord = SIGN_LORD[sign]
    d1_positions = chart.vargas["D1"].planet_positions
    lord_pos = d1_positions.get(lord)
    occupants = occupants_of_house(d1_positions, house_num)
    aspects = planets_aspecting_house(d1_positions, house_num)
    data: Dict[str, Any] = {
        "house": house_num,
        "sign": sign,
        "lord": lord,
        "occupants": occupants,
        "aspects_from": aspects,
        "badhaka_house": badhaka_house(sign),
        "marakas_houses": maraka_houses(house_num),
    }
    if lord_pos is not None:
        data.update({
            "lord_sign": lord_pos.sign,
            "lord_house": lord_pos.house,
            "lord_dignity": lord_pos.dignity,
            "lord_retrograde": lord_pos.retrograde,
            "lord_combust": lord_pos.combust,
        })
    return data


def _karaka_facts(chart: Chart, planet: str) -> Dict[str, Any]:
    pos = chart.vargas["D1"].planet_positions.get(planet)
    if pos is None:
        return {"planet": planet, "missing": True}
    return {
        "planet": planet,
        "sign": pos.sign,
        "house": pos.house,
        "nakshatra": pos.nakshatra,
        "nakshatra_lord": pos.nakshatra_lord,
        "dignity": pos.dignity,
        "retrograde": pos.retrograde,
        "combust": pos.combust,
    }


def _varga_snapshot(chart: Chart, varga: str,
                    target_house: int, karakas: List[str]) -> Dict[str, Any]:
    if varga not in chart.vargas:
        return {"missing": True}
    v = chart.vargas[varga]
    target_sign = v.house_signs.get(target_house)
    occupants = occupants_of_house(v.planet_positions, target_house)
    snap: Dict[str, Any] = {
        "target_sign": target_sign,
        "target_occupants": occupants,
    }
    for k in karakas:
        pp = v.planet_positions.get(k)
        if pp is None:
            continue
        snap[f"{k.lower()}_sign"] = pp.sign
        snap[f"{k.lower()}_house"] = pp.house
        snap[f"{k.lower()}_dignity"] = pp.dignity
    return snap


def _walk_dasha(node: Optional[DashaNode]) -> List[DashaNode]:
    """Flatten the dasha tree in depth-first order (excluding root sentinel)."""
    out: List[DashaNode] = []

    def rec(n: DashaNode):
        for c in n.children:
            out.append(c)
            rec(c)
    if node is None:
        return out
    rec(node)
    return out


def _lord_role_map(
    target_house_rotated: int, target_house_direct: int,
    chart: Chart, rel_karakas: List[str], dom_karakas: List[str],
    rotated_lord: str, direct_lord: str, badhaka_lord: str,
    eighth_from_target: str, marakas_lords: List[str],
) -> Dict[str, List[str]]:
    """{planet -> [roles]} mapping the classical function each planet plays
    for this specific query.
    """
    roles: Dict[str, List[str]] = {}

    def tag(planet: str, role: str) -> None:
        if planet in ("", None, "Lagna"):
            return
        roles.setdefault(planet, []).append(role)

    tag(rotated_lord, "rotated_lord")
    tag(direct_lord, "direct_lord")
    tag(badhaka_lord, "badhaka_lord")
    tag(eighth_from_target, "eighth_from_target")
    for m in marakas_lords:
        tag(m, "maraka")
    for k in rel_karakas:
        tag(k, "relation_karaka")
    for k in dom_karakas:
        tag(k, "domain_karaka")
    return roles


_MALEFICS = {"Sun", "Mars", "Saturn", "Rahu", "Ketu"}
_BENEFICS = {"Moon", "Mercury", "Jupiter", "Venus"}
_DUSTHANA = {6, 8, 12}


def _dasha_candidates(
    chart: Chart, lord_roles: Dict[str, List[str]],
    target_house_rotated: int = 0,
    transit_table: Dict[str, list] = None,
    natal_sun_sign_num: int = 0,
    natal_moon_sign_num: int = 0,
) -> List[Dict[str, Any]]:
    """All MD/AD/PAD windows whose lord path touches any relevant lord.

    Each candidate carries `lord_roles`: {lord -> [roles]} for the subset
    of MD/AD/PAD lords that are relevant. Timing rules can filter on role.

    Also emits extended fields for the manually-derived rules:
      chain_lord_in_8h / _12h / _dusthana / _8h_from_target
      n_malefics_in_chain, n_benefics_in_chain
      n_debilitated_in_chain, n_combust_in_chain
      md_pad_same_planet, md_ad_same_planet
      window_duration_days, window_short, window_long
    """
    relevant = set(lord_roles.keys())
    out: List[Dict[str, Any]] = []
    if chart.vimshottari is None:
        return out

    d1 = chart.vargas.get("D1")
    d1_pos = d1.planet_positions if d1 else {}
    eighth_from_target_house = (
        ((target_house_rotated - 1 + 7) % 12) + 1
        if target_house_rotated else 0
    )

    def _planet_house(p):
        pp = d1_pos.get(p)
        return pp.house if pp else 0

    def _planet_dignity(p):
        pp = d1_pos.get(p)
        return pp.dignity if pp else ""

    def _planet_combust(p):
        pp = d1_pos.get(p)
        return bool(pp.combust) if pp else False

    def _planet_nak_lord(p):
        pp = d1_pos.get(p)
        return pp.nakshatra_lord if pp else ""

    def _planet_in_mrityu_bhaga(p):
        pp = d1_pos.get(p)
        if pp is None:
            return False
        return in_mrityu_bhaga(p, pp.longitude, tolerance=1.0)

    # Pre-compute chart-wide Jupiter dignity (used by per-candidate
    # jupiter_well_placed_in_chain field). Jupiter as greatest benefic
    # only carries protective weight when its natal dignity is good.
    jup_pos = d1_pos.get("Jupiter")
    jupiter_well_placed = (
        jup_pos is not None
        and jup_pos.dignity in {"exalted", "own", "friend"}
    )

    def _window(kind: str, md_lord, ad_lord, pad_lord, start, end):
        chain_lords = [md_lord, ad_lord]
        if pad_lord is not None:
            chain_lords.append(pad_lord)
        n_chain = len(chain_lords)
        hit = [l for l in chain_lords if l in relevant]
        if not hit:
            return None
        # Role chain strength (CAV-007 / CAV-009):
        #   "full"    — ALL chain levels in relevant set
        #   "partial" — n-1 of n levels in relevant set
        #   "weak"    — 1 of n levels (or 1/2 for AD-only windows)
        n_hit = len(hit)
        if n_hit == n_chain:
            chain_strength = "full"
        elif n_hit >= max(1, n_chain - 1):
            chain_strength = "partial"
        else:
            chain_strength = "weak"
        roles_present = sorted({
            role for l in hit for role in lord_roles.get(l, [])
        })

        # Extended factor extraction (manual-rule support).
        chain_houses = [
            _planet_house(p) for p in chain_lords
        ]
        chain_houses = [h for h in chain_houses if h]
        chain_in_8h = any(h == 8 for h in chain_houses)
        chain_in_12h = any(h == 12 for h in chain_houses)
        chain_in_dust = any(h in _DUSTHANA for h in chain_houses)
        chain_in_8h_target = (
            eighth_from_target_house != 0
            and any(h == eighth_from_target_house for h in chain_houses)
        )
        n_mal = sum(1 for p in chain_lords if p in _MALEFICS)
        n_ben = sum(1 for p in chain_lords if p in _BENEFICS)
        n_deb = sum(
            1 for p in chain_lords
            if _planet_dignity(p) == "debilitated"
        )
        n_comb = sum(1 for p in chain_lords if _planet_combust(p))
        md_pad_same = pad_lord is not None and md_lord == pad_lord
        md_ad_same = md_lord == ad_lord
        duration_days = max(0, (end - start).days)
        win_short = duration_days < 60
        win_long = duration_days > 365

        # Per-planet chain-membership flags (graded yogas reference these).
        chain_set = set(chain_lords)
        saturn_in_chain = "Saturn" in chain_set
        sun_in_chain = "Sun" in chain_set
        jupiter_in_chain = "Jupiter" in chain_set
        mars_in_chain = "Mars" in chain_set
        rahu_or_ketu_in_chain = bool({"Rahu", "Ketu"} & chain_set)

        # Nakshatra-lord shadow chain: when chain lords don't carry roles
        # but their nakshatra dispositors do, classical doctrine treats
        # the period as still activating the role through dispositorship.
        nak_lord_roles = []
        for p in chain_lords:
            nl = _planet_nak_lord(p)
            if nl and nl in lord_roles:
                nak_lord_roles.extend(lord_roles[nl])
        n_nak_lord_roles = len(set(nak_lord_roles))
        any_nak_lord_relevant = n_nak_lord_roles > 0

        # Per-level role lists — most discriminating signal because the
        # MD lord (longest sub-period) carries the most weight, then AD,
        # then PAD. Expose separately so rules can say "MD plays maraka"
        # vs "PAD plays maraka" (very different astrological force).
        md_roles = sorted(set(lord_roles.get(md_lord, [])))
        ad_roles = sorted(set(lord_roles.get(ad_lord, [])))
        pad_roles = (
            sorted(set(lord_roles.get(pad_lord, [])))
            if pad_lord else []
        )
        # Role-overlay counts: how many distinct roles each chain-level
        # lord plays. A planet wearing multiple hats (e.g. Saturn = 8L +
        # domain_karaka + maraka) concentrates malefic significance into
        # a single sub-period — much stronger than 3 different planets
        # each playing one role across the chain.
        n_md_roles = len(md_roles)
        n_ad_roles = len(ad_roles)
        n_pad_roles = len(pad_roles)
        n_max_level_roles = max(n_md_roles, n_ad_roles, n_pad_roles)

        # Mrityu Bhaga (BPHS Ch.40): chain lord at its critical degree.
        chain_in_mrityu_bhaga = any(
            _planet_in_mrityu_bhaga(p) for p in chain_lords
        )
        n_chain_in_mrityu_bhaga = sum(
            1 for p in chain_lords if _planet_in_mrityu_bhaga(p)
        )
        # Jupiter benefic only counts as protective when natally well-
        # placed (exalted/own/friend). A debilitated Jupiter in chain is
        # itself afflicted and not protective.
        jupiter_well_placed_in_chain = (
            jupiter_well_placed and jupiter_in_chain
        )
        # Classical test: Jupiter "relieves" a sub-period only when the
        # chain is NOT carrying death karma. When the chain's overall
        # lord_roles_present includes any death-cluster role (maraka /
        # eighth_from_target / badhaka_lord), the chain is executing
        # the affliction — Jupiter in such a chain is part of the
        # executing group, not an external softener (Phaladeepika Ch.26,
        # Saravali 30, v19 regression analysis on Victor Hugo + J. S.
        # Holliday). Only when the chain as a whole has no death
        # cluster role is a well-placed Jupiter truly protective —
        # this correctly suppresses competitor noise windows that had
        # Jupiter-in-chain in a benign chain (Fritz Huitfeldt v17→v18
        # evidence).
        _DEATH_CLUSTER_ROLES = {
            "maraka", "eighth_from_target", "badhaka_lord",
        }
        chain_has_death_role = bool(
            set(roles_present) & _DEATH_CLUSTER_ROLES
        )
        jupiter_safe_protective = (
            jupiter_well_placed_in_chain
            and not chain_has_death_role
        )

        # Mars transit overlays (v23 — Phaladeepika Ch.18 + BPHS Ch.46):
        # Mars over natal Sun = acute affliction to pitri karaka.
        # Mars in 8H-from-natal-Moon = sudden death trigger.
        # Mars over natal target sign = malefic activation of relation.
        mars_over_sun = False
        mars_8h_from_moon = False
        mars_over_target = False
        # Saturn over natal Sun at window midpoint (father longevity).
        sat_over_sun = False
        # Sade Sati: Saturn transits the 12th, 1st, or 2nd from natal Moon
        # (the 7.5-year affliction; Phaladeepika Ch.26, Saravali 31.40).
        sade_sati = False
        # Ashtama Shani: Saturn in 8H from natal Moon — sub-period of
        # Sade Sati most associated with parental loss (BPHS Ch.46).
        ashtama_shani = False
        # Kantaka Shani: Saturn in 4H from natal Moon (afflicts the
        # 4H significations — mother / domestic peace / father's house).
        kantaka_shani = False
        mid_dt = start + (end - start) / 2
        if transit_table and natal_sun_sign_num:
            sat_over_sun = _transit_active_at(
                transit_table, "Saturn", mid_dt, {natal_sun_sign_num},
            )
            mars_over_sun = _transit_active_at(
                transit_table, "Mars", mid_dt, {natal_sun_sign_num},
            )
        # Mars-Saturn conjunction (Mrityu Yoga, BPHS Ch.46): both
        # malefics in same sign at midpoint = classical sudden-death
        # yoga. We use same-sign as a 30°-wide proxy for the strict
        # 8°-orb classical definition (transit table only stores sign,
        # not exact longitude).
        mars_saturn_conjunction = False
        if transit_table:
            mars_series = transit_table.get("Mars", [])
            sat_series = transit_table.get("Saturn", [])
            if mars_series and sat_series:
                mid_naive = (
                    mid_dt.replace(tzinfo=None) if mid_dt.tzinfo else mid_dt
                )
                m_closest = min(
                    mars_series,
                    key=lambda r: abs(
                        (r[0].replace(tzinfo=None) - mid_naive).total_seconds()
                    ),
                )
                s_closest = min(
                    sat_series,
                    key=lambda r: abs(
                        (r[0].replace(tzinfo=None) - mid_naive).total_seconds()
                    ),
                )
                mars_saturn_conjunction = (m_closest[1] == s_closest[1])
        if transit_table and natal_moon_sign_num:
            # 8H from natal Moon = ashtama-from-Moon (death-cluster).
            mars_8h_from_moon_sign = (
                ((natal_moon_sign_num - 1 + 7) % 12) + 1
            )
            mars_8h_from_moon = _transit_active_at(
                transit_table, "Mars", mid_dt, {mars_8h_from_moon_sign},
            )
        # Saturn transit through critical nakshatras for father — BPHS
        # via RAG cluster C7 (freq=45): "If Saturn transits through
        # Uttarabhadrapada, Pushyami or Anuradha nakshatras, the death
        # of father takes place if an inauspicious dasha is in operation."
        # Nakshatras (1-based): Pushya=8, Anuradha=17, Uttarabhadra=26.
        saturn_in_father_critical_nakshatra = False
        if transit_table:
            saturn_in_father_critical_nakshatra = _transit_in_nakshatras(
                transit_table, "Saturn", mid_dt, {8, 17, 26},
            )

        # Saturn transit through 9H from natal Moon — Patel
        # Ashtakavarga sloka 11 (RAG cluster C8): "When Saturn transits
        # the 9th house from the Lagna or the Moon, the demise of one's
        # parents may be predicted, if associated/aspected by malefic."
        # The malefic-association requirement is enforced in the rule
        # antecedent (chain has malefic).
        saturn_transit_9h_from_moon = False
        if transit_table and natal_moon_sign_num:
            # 12th, 1st, 2nd from natal Moon (1-based sign indices).
            sade_signs = {
                ((natal_moon_sign_num - 2) % 12) + 1,   # 12th
                natal_moon_sign_num,                     # 1st (Janma Shani)
                (natal_moon_sign_num % 12) + 1,          # 2nd
            }
            sade_sati = _transit_active_at(
                transit_table, "Saturn", mid_dt, sade_signs,
            )
            ashtama_sign = ((natal_moon_sign_num - 1 + 7) % 12) + 1
            ashtama_shani = _transit_active_at(
                transit_table, "Saturn", mid_dt, {ashtama_sign},
            )
            kantaka_sign = ((natal_moon_sign_num - 1 + 3) % 12) + 1
            kantaka_shani = _transit_active_at(
                transit_table, "Saturn", mid_dt, {kantaka_sign},
            )
            ninth_from_moon_sign = ((natal_moon_sign_num - 1 + 8) % 12) + 1
            saturn_transit_9h_from_moon = _transit_active_at(
                transit_table, "Saturn", mid_dt, {ninth_from_moon_sign},
            )

        # Saturn co-active with chain-maraka lord (Chatterjee Medical
        # Astrology, RAG cluster C15): "When malefic Saturn is associated
        # with any of the planets causing death, he overrides all others
        # and himself causes death." Proxy: Saturn is in chain AND chain
        # contains a death-cluster role (8L/maraka/badhaka).
        chain_has_death_role = bool(
            {"eighth_from_target", "maraka", "badhaka"} & set(roles_present)
        )
        malefic_saturn_with_maraka_chain = (
            saturn_in_chain and chain_has_death_role
        )

        return {
            "level": kind,
            "md": md_lord, "ad": ad_lord, "pad": pad_lord,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "matched_lords": sorted(set(hit)),
            "lord_roles_present": roles_present,
            "n_distinct_roles": len(roles_present),
            "chain_strength": chain_strength,
            "chain_lord_in_8h": chain_in_8h,
            "chain_lord_in_12h": chain_in_12h,
            "chain_lord_in_dusthana": chain_in_dust,
            "chain_lord_in_8h_from_target": chain_in_8h_target,
            "n_malefics_in_chain": n_mal,
            "n_benefics_in_chain": n_ben,
            "n_debilitated_in_chain": n_deb,
            "n_combust_in_chain": n_comb,
            "md_pad_same_planet": md_pad_same,
            "md_ad_same_planet": md_ad_same,
            "window_duration_days": duration_days,
            "window_short": win_short,
            "window_long": win_long,
            "saturn_in_chain": saturn_in_chain,
            "sun_in_chain": sun_in_chain,
            "jupiter_in_chain": jupiter_in_chain,
            "mars_in_chain": mars_in_chain,
            "rahu_or_ketu_in_chain": rahu_or_ketu_in_chain,
            "any_nak_lord_relevant": any_nak_lord_relevant,
            "n_nak_lord_roles": n_nak_lord_roles,
            "saturn_over_natal_sun": sat_over_sun,
            "sade_sati": sade_sati,
            "ashtama_shani": ashtama_shani,
            "kantaka_shani": kantaka_shani,
            "md_roles": md_roles,
            "ad_roles": ad_roles,
            "pad_roles": pad_roles,
            "n_md_roles": n_md_roles,
            "n_ad_roles": n_ad_roles,
            "n_pad_roles": n_pad_roles,
            "n_max_level_roles": n_max_level_roles,
            "chain_lord_in_mrityu_bhaga": chain_in_mrityu_bhaga,
            "n_chain_in_mrityu_bhaga": n_chain_in_mrityu_bhaga,
            "jupiter_well_placed_in_chain": jupiter_well_placed_in_chain,
            "jupiter_safe_protective": jupiter_safe_protective,
            # v23 Mars transit overlays (Phaladeepika Ch.18, BPHS Ch.46).
            "mars_over_natal_sun": mars_over_sun,
            "mars_8h_from_moon": mars_8h_from_moon,
            # v28 Mrityu Yoga (BPHS Ch.46) — Mars-Saturn same-sign at midpoint.
            "mars_saturn_conjunction_transit": mars_saturn_conjunction,
            # v32 inverted-mining additions:
            #  - Patel Ashtakavarga sloka 11 (saturn over 9H from Moon)
            #  - Chatterjee Medical (Saturn dominates death when in chain
            #    with maraka-role lord)
            "saturn_transit_9h_from_moon": saturn_transit_9h_from_moon,
            "malefic_saturn_with_maraka_chain": malefic_saturn_with_maraka_chain,
            # v32 — BPHS Santhanam, Saturn in critical father-death
            # nakshatras (Pushya / Anuradha / Uttarabhadra).
            "saturn_in_father_critical_nakshatra":
                saturn_in_father_critical_nakshatra,
            "reason": (
                f"chain={chain_strength} ({n_hit}/{n_chain})"
                f" roles({len(roles_present)}): {','.join(roles_present)}"
            ),
        }

    for md in chart.vimshottari.children:
        md_lord = md.lord
        for ad in md.children:
            ad_lord = ad.lord
            pad_hit = False
            for pad in ad.children:
                w = _window("PAD", md_lord, ad_lord, pad.lord,
                            pad.start, pad.end)
                if w:
                    out.append(w)
                    pad_hit = True
            if not pad_hit:
                w = _window("AD", md_lord, ad_lord, None,
                            ad.start, ad.end)
                if w:
                    out.append(w)
    return out


class ParashariFeatureExtractor:
    """Parashari school feature extractor."""

    def extract(
        self, chart: Chart, resolved: ResolvedFocus,
    ) -> FeatureBundle:
        # Houses to analyse: rotated (analytical) + direct (relationship's own).
        houses_to_analyse: List[int] = []
        for h in (resolved.target_house_rotated, resolved.target_house_direct):
            if h not in houses_to_analyse:
                houses_to_analyse.append(h)

        # Primary-house blocks keyed by role.
        primary_house_data: Dict[str, Any] = {
            "rotated": _primary_house(chart, resolved.target_house_rotated),
            "direct": _primary_house(chart, resolved.target_house_direct),
        }
        # Phase F.9: classical-override flags (chart-static facts).
        d1_for_overrides = chart.vargas.get("D1")
        if d1_for_overrides is not None:
            vry = detect_vipreet_raja_yoga(
                d1_for_overrides.planet_positions,
                d1_for_overrides.house_signs,
            )
            nb = detect_neecha_bhanga(
                d1_for_overrides.planet_positions,
                d1_for_overrides.house_signs,
            )
            primary_house_data["vipreet_raja_yoga_present"] = bool(vry)
            primary_house_data["vipreet_raja_yoga_count"] = len(vry)
            primary_house_data["neecha_bhanga_present"] = bool(nb)
            primary_house_data["neecha_bhanga_planets"] = [
                str(item["planet"]) for item in nb
            ]
        else:
            primary_house_data["vipreet_raja_yoga_present"] = False
            primary_house_data["vipreet_raja_yoga_count"] = 0
            primary_house_data["neecha_bhanga_present"] = False
            primary_house_data["neecha_bhanga_planets"] = []

        # v32 inverted-mining: malefics in 4H from natal Sun (Patel
        # Ashtakavarga sloka 10) — "A person born having Rahu, Saturn or
        # Mars in the 4th house from the Sun will be the cause of his
        # father's early death, provided that house is not aspected by
        # Jupiter or Venus." Chart-static fact, applied across all
        # candidate windows in static_promise line.
        d1_pp = (chart.vargas["D1"].planet_positions
                 if chart.vargas.get("D1") else {})
        from .classical import (
            occupants_of_house, planets_aspecting_house, house_from,
        )
        sun_pp = d1_pp.get("Sun")
        sun_house = sun_pp.house if sun_pp else 0
        if sun_house:
            fourth_from_sun = house_from(sun_house, 4)
            occ = occupants_of_house(d1_pp, fourth_from_sun)
            asp = planets_aspecting_house(d1_pp, fourth_from_sun)
            mal_in = [p for p in occ if p in {"Rahu", "Saturn", "Mars"}]
            ben_asp = [p for p in asp if p in {"Jupiter", "Venus"}]
            primary_house_data["fourth_from_sun_house"] = fourth_from_sun
            primary_house_data["malefics_in_4h_from_sun"] = mal_in
            primary_house_data["n_malefics_in_4h_from_sun"] = len(mal_in)
            primary_house_data["benefics_aspect_4h_from_sun"] = ben_asp
            primary_house_data["malefics_4h_from_sun_unprotected"] = (
                len(mal_in) > 0 and len(ben_asp) == 0
            )
        else:
            primary_house_data["fourth_from_sun_house"] = 0
            primary_house_data["malefics_in_4h_from_sun"] = []
            primary_house_data["n_malefics_in_4h_from_sun"] = 0
            primary_house_data["benefics_aspect_4h_from_sun"] = []
            primary_house_data["malefics_4h_from_sun_unprotected"] = False

        # v33 inverted-mining (v2): BPHS Ch.13 verses 13-25 — age-specific
        # father-death combinations. Each chart-static pattern below is
        # paired with a tight age-window gate in the rule YAML.
        # All conditions reference the NATAL frame (not rotated); BPHS
        # speaks of "8th lord", "12th lord" etc. relative to the lagna.
        if d1_pp:
            sun_h = d1_pp["Sun"].house if "Sun" in d1_pp else 0
            mars_h = d1_pp["Mars"].house if "Mars" in d1_pp else 0
            saturn_h = d1_pp["Saturn"].house if "Saturn" in d1_pp else 0
            rahu_h = d1_pp["Rahu"].house if "Rahu" in d1_pp else 0
            try:
                eighth_lord = _lord_of_house(chart, "D1", 8)
                ninth_lord = _lord_of_house(chart, "D1", 9)
                twelfth_lord = _lord_of_house(chart, "D1", 12)
                lagna_lord = _lord_of_house(chart, "D1", 1)
            except Exception:
                eighth_lord = ninth_lord = twelfth_lord = lagna_lord = ""
            eighth_lord_h = (d1_pp[eighth_lord].house
                             if eighth_lord in d1_pp else 0)
            ninth_lord_h = (d1_pp[ninth_lord].house
                            if ninth_lord in d1_pp else 0)
            twelfth_lord_h = (d1_pp[twelfth_lord].house
                              if twelfth_lord in d1_pp else 0)
            lagna_lord_h = (d1_pp[lagna_lord].house
                            if lagna_lord in d1_pp else 0)

            # Pattern 1 — within 1 year:
            #   "Sun in 8th + 8L in 9th"
            primary_house_data["bphs_13_pattern_within_1y"] = bool(
                sun_h == 8 and eighth_lord_h == 9
            )

            # Pattern 2 — year 2 OR year 12:
            #   "Lagna lord in 8H + 8L conjunct (same house as) Sun"
            primary_house_data["bphs_13_pattern_year_2_or_12"] = bool(
                lagna_lord_h == 8
                and eighth_lord_h
                and eighth_lord_h == sun_h
            )

            # Pattern 3 — year 16 OR year 18:
            #   "Rahu in 4H from lagna + Sun in 5H from lagna"
            primary_house_data["bphs_13_pattern_year_16_or_18"] = bool(
                rahu_h == 4 and sun_h == 5
            )

            # Pattern 4 — year 44 (mutual exchange / parivartana):
            #   "9L in 12H + 12L in 9H"
            primary_house_data["bphs_13_pattern_year_44_parivartana"] = bool(
                ninth_lord_h == 12 and twelfth_lord_h == 9
            )

            # Pattern 5 — year 50:
            #   "Sun is the 9L AND Sun conjunct Mars AND Sun conjunct Saturn"
            primary_house_data["bphs_13_pattern_year_50_sun_mars_saturn"] = (
                bool(
                    ninth_lord == "Sun"
                    and sun_h
                    and mars_h == sun_h
                    and saturn_h == sun_h
                )
            )

            # Pattern 6 — year 26 OR year 30:
            #   "9L debilitated AND dispositor of 9L in 9H"
            ninth_lord_dignity = (
                d1_pp[ninth_lord].dignity if ninth_lord in d1_pp else ""
            )
            ninth_lord_disp = ""
            if ninth_lord in d1_pp:
                try:
                    ninth_lord_disp = SIGN_LORD[d1_pp[ninth_lord].sign]
                except Exception:
                    ninth_lord_disp = ""
            ninth_lord_disp_h = (
                d1_pp[ninth_lord_disp].house
                if ninth_lord_disp in d1_pp else 0
            )
            primary_house_data["bphs_13_pattern_year_26_or_30"] = bool(
                ninth_lord_dignity == "debilitated"
                and ninth_lord_disp_h == 9
            )
        else:
            for k in (
                "bphs_13_pattern_within_1y",
                "bphs_13_pattern_year_2_or_12",
                "bphs_13_pattern_year_16_or_18",
                "bphs_13_pattern_year_44_parivartana",
                "bphs_13_pattern_year_50_sun_mars_saturn",
                "bphs_13_pattern_year_26_or_30",
            ):
                primary_house_data[k] = False

        # Convenience: include target_house_lord for rule antecedents.
        rotated_lord = primary_house_data["rotated"]["lord"]
        direct_lord = primary_house_data["direct"]["lord"]

        # Relation karakas: resolve "Lagna_Lord" sentinel if present.
        rel_karakas: List[str] = []
        for k in resolved.relation_karakas:
            if k == "Lagna_Lord":
                lagna = chart.vargas["D1"].planet_positions.get("Lagna")
                if lagna is not None:
                    rel_karakas.append(SIGN_LORD[lagna.sign])
            else:
                rel_karakas.append(k)

        # Domain karakas: expand {N}L tokens relative to rotated target.
        dom_karakas = _resolve_lord_tokens(
            chart, resolved.target_house_rotated, resolved.domain_karakas,
        )

        karaka_data: Dict[str, Dict[str, Any]] = {}
        # v36: include all 9 planets unconditionally so graded rules
        # can reference natal placement of Rahu/Ketu/Mars/Jupiter etc.
        # as modulators (e.g., "Rahu in 8H natally" amplifies Rahu/Ketu
        # in chain). Was: only karakas of the focus area.
        all_planets = rel_karakas + dom_karakas + [
            rotated_lord, direct_lord,
            "Sun", "Moon", "Mars", "Mercury", "Jupiter",
            "Venus", "Saturn", "Rahu", "Ketu",
        ]
        for k in all_planets:
            if k in ("Lagna", ""):
                continue
            if k in karaka_data:
                continue
            try:
                karaka_data[k] = _karaka_facts(chart, k)
            except Exception:
                # tolerate planets missing from D1 (shouldn't happen
                # but keep extraction robust)
                continue

        # Varga snapshots.
        varga_features: Dict[str, Dict[str, Any]] = {}
        for v in resolved.vargas_required:
            varga_features[v] = _varga_snapshot(
                chart, v, resolved.target_house_rotated,
                list(set(rel_karakas + dom_karakas +
                         [rotated_lord, direct_lord])),
            )

        # Transit table: precompute Saturn + Jupiter + Mars monthly
        # positions over 100 years from birth (~1200 months) for transit
        # overlay. Mars added v23 (Phaladeepika Ch.18 — Mars transits
        # mark acute death triggers; BPHS Ch.46 — Mars-Moon contact
        # afflicts longevity).
        transit_table = {}
        target_sign_nums = set()
        if resolved.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            transit_table = _compute_transit_table(
                chart, planets=["Saturn", "Jupiter", "Mars"], n_months=1200,
            )
            # Target signs: rotated + direct + 8th-from-direct (death-house
            # for the relationship's longevity arc).
            from .classical import sign_index
            d1 = chart.vargas.get("D1")
            if d1:
                for h in (resolved.target_house_rotated,
                          resolved.target_house_direct):
                    s = d1.house_signs.get(h)
                    if s:
                        target_sign_nums.add(sign_index(s))

        # Dasha candidates (TIMING / PROBABILITY only).
        dasha_candidates: Optional[List[Dict[str, Any]]] = None
        if resolved.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            marakas_lords = [
                _lord_of_house(chart, "D1", h)
                for h in maraka_houses(resolved.target_house_rotated)
            ]
            # Badhaka is the X-th house from the target (X depends on sign
            # nature: movable->11, fixed->9, dual->7), not the X-th natal
            # house. See classical.badhaka_house for the offset.
            badhaka_offset = badhaka_house(
                _house_sign(chart, "D1", resolved.target_house_rotated),
            )
            badhaka_lord = _lord_of_house(
                chart, "D1",
                house_from(resolved.target_house_rotated, badhaka_offset),
            )
            eighth_from_target = _lord_of_house(
                chart, "D1",
                house_from(resolved.target_house_rotated, 8),
            )
            lord_roles = _lord_role_map(
                resolved.target_house_rotated,
                resolved.target_house_direct,
                chart,
                rel_karakas, dom_karakas,
                rotated_lord, direct_lord, badhaka_lord,
                eighth_from_target, marakas_lords,
            )
            # Natal Sun sign number — for the saturn_over_natal_sun
            # transit overlay (Saturn over father karaka).
            sun_pp = chart.vargas["D1"].planet_positions.get("Sun")
            natal_sun_sign_num = (
                sign_index(sun_pp.sign) if sun_pp and sun_pp.sign else 0
            )
            # Natal Moon sign — for Sade Sati / Ashtama / Kantaka Shani.
            moon_pp = chart.vargas["D1"].planet_positions.get("Moon")
            natal_moon_sign_num = (
                sign_index(moon_pp.sign) if moon_pp and moon_pp.sign else 0
            )
            dasha_candidates = _dasha_candidates(
                chart, lord_roles,
                target_house_rotated=resolved.target_house_rotated,
                transit_table=transit_table,
                natal_sun_sign_num=natal_sun_sign_num,
                natal_moon_sign_num=natal_moon_sign_num,
            )
            # CAV-008: tag each candidate with Saturn/Jupiter transit
            # status over the natal target signs at window midpoint.
            if transit_table and target_sign_nums:
                for cand in dasha_candidates:
                    try:
                        s = datetime.fromisoformat(cand["start"])
                        e = datetime.fromisoformat(cand["end"])
                        mid = s + (e - s) / 2
                        cand["saturn_transit_target"] = _transit_active_at(
                            transit_table, "Saturn", mid, target_sign_nums,
                        )
                        cand["jupiter_transit_target"] = _transit_active_at(
                            transit_table, "Jupiter", mid, target_sign_nums,
                        )
                        # v23: Mars transit over natal target sign —
                        # Phaladeepika Ch.18 acute death trigger.
                        cand["mars_transit_target"] = _transit_active_at(
                            transit_table, "Mars", mid, target_sign_nums,
                        )
                    except Exception:
                        cand["saturn_transit_target"] = False
                        cand["jupiter_transit_target"] = False
                        cand["mars_transit_target"] = False

        # v38: chart applicability (Laghu Parashari functional roles).
        # Evaluated once per chart; rules' applicable_when clauses
        # reference these lists (and derived flags) to decide whether
        # to apply to this chart at all (before per-window matching).
        d1 = chart.vargas.get("D1")
        lagna_sign_d1 = d1.house_signs.get(1, "") if d1 else ""
        roles = compute_functional_roles(lagna_sign_d1)
        fm = roles["functional_malefics"]
        yk = roles["yogakarakas"]
        fb = roles["functional_benefics"]
        chart_applicability = {
            "lagna_sign": lagna_sign_d1,
            "lagna_lord": roles["lagna_lord"],
            "yogakarakas": sorted(yk),
            "functional_malefics": sorted(fm),
            "functional_benefics": sorted(fb),
            # derived per-planet boolean flags (easier to reference
            # in rule.applicable_when than list-contains checks)
            "saturn_is_fm":  "Saturn"  in fm,
            "saturn_is_yk":  "Saturn"  in yk,
            "saturn_is_fb":  "Saturn"  in fb,
            "mars_is_fm":    "Mars"    in fm,
            "mars_is_yk":    "Mars"    in yk,
            "mars_is_fb":    "Mars"    in fb,
            "jupiter_is_fm": "Jupiter" in fm,
            "jupiter_is_fb": "Jupiter" in fb,
            "jupiter_is_yk": "Jupiter" in yk,
            "sun_is_fm":     "Sun"     in fm,
            "sun_is_fb":     "Sun"     in fb,
            "mercury_is_fm": "Mercury" in fm,
            "mercury_is_fb": "Mercury" in fb,
            "venus_is_fm":   "Venus"   in fm,
            "venus_is_fb":   "Venus"   in fb,
            "venus_is_yk":   "Venus"   in yk,
            "moon_is_fm":    "Moon"    in fm,
            "moon_is_fb":    "Moon"    in fb,
            "moon_is_yk":    "Moon"    in yk,
        }
        # v38.7: father natal-context primitives (Pitri karaka + 9th bhava).
        if d1:
            father_ctx = compute_father_natal_context(
                d1.planet_positions, d1.house_signs,
            )
            chart_applicability.update(father_ctx)

        return FeatureBundle(
            school=School.PARASHARI,
            focus=resolved,
            primary_house_data=primary_house_data,
            karaka_data=karaka_data,
            varga_features=varga_features,
            dasha_candidates=dasha_candidates,
            transit_events=None,   # Phase 2+: wire in Saturn/Jupiter transits
            chart_applicability=chart_applicability,
        )
