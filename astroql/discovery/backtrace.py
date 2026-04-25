"""Backtrace from a known event date to the active dasha context.

For each chart + event-date pair, this module:
    1. Computes the natal chart (positions, vargas, vimshottari).
    2. Resolves the FATHER/LONGEVITY focus to obtain target houses,
       relation karakas, domain karakas.
    3. Builds the lord-role map (rotated_lord, direct_lord, badhaka_lord,
       eighth_from_target, maraka set, relation/domain karakas, plus
       a few extras: nakshatra_lord chains, ayushkaraka Saturn).
    4. Walks the dasha tree to find the PAD window containing the
       event date (PAD ⊂ AD ⊂ MD, so all three lords are identified).
    5. Captures per-window features for the death window + every other
       window in the lifetime, so the aggregator can compare them.

The "richer" feature set (per user direction) includes:
    - core role tags at each chain level (md/ad/pad)
    - chain_strength (full/partial/weak)
    - n_distinct_roles
    - saturn_transit_target / jupiter_transit_target
    - nakshatra_lord of MD/AD/PAD lords
    - md_in_kendra_from_pad / md_in_dusthana_from_pad — chain-internal
      angularity
    - ayushkaraka_saturn_in_chain — Saturn (longevity karaka) appears
      anywhere in the chain
    - sun_in_chain — relation karaka (Sun, for father) appears anywhere
    - jupiter_in_chain — protective; suggests positive override
    - rahu_or_ketu_in_chain — node in chain (commonly noted in death timing)
    - md_lord_dignity, ad_lord_dignity — birth-chart dignity of the MD/AD
      lords (links static chart to timing)

Anything new can be added in `_window_features()` without touching the
aggregator — the aggregator iterates over whatever keys the dict carries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from ..chart import ChartComputer
from ..features.classical import (
    SIGN_LORD, badhaka_house, house_from, maraka_houses, sign_index,
)
from ..features.parashari import (
    _compute_transit_table, _transit_active_at,
)
from ..resolver import FocusResolver
from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    QueryType, Relationship, School,
)
from ..schemas.chart import Chart, DashaNode


# ── Per-window feature record ───────────────────────────────────────

@dataclass
class WindowFeatures:
    """Features for one (MD, AD, PAD) window relative to the focus."""
    level: str                          # "PAD" | "AD" (when no PAD child)
    md: str
    ad: str
    pad: Optional[str]
    start: datetime
    end: datetime

    # Role tags by chain level (each is a sorted list of role-name strings).
    md_roles: List[str] = field(default_factory=list)
    ad_roles: List[str] = field(default_factory=list)
    pad_roles: List[str] = field(default_factory=list)

    # Chain-level summary (also used by the existing rule library).
    chain_strength: str = "weak"        # "full" | "partial" | "weak"
    n_distinct_roles: int = 0
    roles_present: List[str] = field(default_factory=list)

    # Transit overlay at window midpoint.
    saturn_transit_target: bool = False
    jupiter_transit_target: bool = False

    # Nakshatra-lord chain (links Vimshottari sub-divisions to deeper
    # natal-chart structure — often cited in classical death timing).
    md_nak_lord: str = ""
    ad_nak_lord: str = ""
    pad_nak_lord: str = ""
    md_nak_lord_role: List[str] = field(default_factory=list)
    ad_nak_lord_role: List[str] = field(default_factory=list)
    pad_nak_lord_role: List[str] = field(default_factory=list)

    # Karaka-in-chain shortcuts (boolean — most useful for lift bucketing).
    saturn_in_chain: bool = False       # ayushkaraka
    sun_in_chain: bool = False          # father karaka (relation)
    jupiter_in_chain: bool = False      # protective
    rahu_or_ketu_in_chain: bool = False
    mars_in_chain: bool = False         # malefic / 8th-from-Sun karaka

    # Chain dignity (birth-chart dignity of each chain lord).
    md_dignity: str = ""
    ad_dignity: str = ""
    pad_dignity: str = ""
    n_debilitated_in_chain: int = 0
    n_combust_in_chain: int = 0

    # Chain-internal angularity (PAD lord viewed from MD lord's house).
    pad_house_from_md_house: int = 0     # 1..12; 0 if undefined
    pad_in_dusthana_from_md: bool = False  # 6/8/12 from MD

    # ── Extended factors (compound-discovery amplifiers) ──────────
    # Where each chain lord sits in the natal D1 chart.
    md_house: int = 0
    ad_house: int = 0
    pad_house: int = 0
    # Any chain lord in 8H/12H from natal lagna (classical death houses).
    chain_lord_in_8h: bool = False
    chain_lord_in_12h: bool = False
    chain_lord_in_dusthana: bool = False     # 6/8/12 from lagna
    # Any chain lord in 8H from the rotated target house (deep affliction).
    chain_lord_in_8h_from_target: bool = False
    # Chain composition counts.
    n_malefics_in_chain: int = 0     # Sun + Mars + Saturn + Rahu + Ketu
    n_benefics_in_chain: int = 0     # Moon + Mercury + Jupiter + Venus
    md_pad_same_planet: bool = False    # MD lord == PAD lord
    md_ad_same_planet: bool = False     # MD lord == AD lord
    # Window-duration band (PAD windows are short; MD-only are long).
    window_duration_days: int = 0
    window_short: bool = False           # < 60 days (sudden-event tier)
    window_long: bool = False            # > 365 days (year-scale)
    # Age band at window midpoint.
    age_at_window_mid_years: float = 0.0
    age_band: str = ""                   # "child" / "young" / "mid" / "old"
    # Saturn transit over natal Sun (father karaka transit) at midpoint.
    saturn_over_sun_at_mid: bool = False
    # Multi-malefic transit overlay.
    saturn_or_jupiter_transit: bool = False     # at least one
    saturn_and_jupiter_transit: bool = False    # both at once

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "md": self.md, "ad": self.ad, "pad": self.pad,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "md_roles": list(self.md_roles),
            "ad_roles": list(self.ad_roles),
            "pad_roles": list(self.pad_roles),
            "chain_strength": self.chain_strength,
            "n_distinct_roles": self.n_distinct_roles,
            "roles_present": list(self.roles_present),
            "saturn_transit_target": self.saturn_transit_target,
            "jupiter_transit_target": self.jupiter_transit_target,
            "md_nak_lord": self.md_nak_lord,
            "ad_nak_lord": self.ad_nak_lord,
            "pad_nak_lord": self.pad_nak_lord,
            "md_nak_lord_role": list(self.md_nak_lord_role),
            "ad_nak_lord_role": list(self.ad_nak_lord_role),
            "pad_nak_lord_role": list(self.pad_nak_lord_role),
            "saturn_in_chain": self.saturn_in_chain,
            "sun_in_chain": self.sun_in_chain,
            "jupiter_in_chain": self.jupiter_in_chain,
            "rahu_or_ketu_in_chain": self.rahu_or_ketu_in_chain,
            "mars_in_chain": self.mars_in_chain,
            "md_dignity": self.md_dignity,
            "ad_dignity": self.ad_dignity,
            "pad_dignity": self.pad_dignity,
            "n_debilitated_in_chain": self.n_debilitated_in_chain,
            "n_combust_in_chain": self.n_combust_in_chain,
            "pad_house_from_md_house": self.pad_house_from_md_house,
            "pad_in_dusthana_from_md": self.pad_in_dusthana_from_md,
            "md_house": self.md_house,
            "ad_house": self.ad_house,
            "pad_house": self.pad_house,
            "chain_lord_in_8h": self.chain_lord_in_8h,
            "chain_lord_in_12h": self.chain_lord_in_12h,
            "chain_lord_in_dusthana": self.chain_lord_in_dusthana,
            "chain_lord_in_8h_from_target":
                self.chain_lord_in_8h_from_target,
            "n_malefics_in_chain": self.n_malefics_in_chain,
            "n_benefics_in_chain": self.n_benefics_in_chain,
            "md_pad_same_planet": self.md_pad_same_planet,
            "md_ad_same_planet": self.md_ad_same_planet,
            "window_duration_days": self.window_duration_days,
            "window_short": self.window_short,
            "window_long": self.window_long,
            "age_at_window_mid_years": self.age_at_window_mid_years,
            "age_band": self.age_band,
            "saturn_over_sun_at_mid": self.saturn_over_sun_at_mid,
            "saturn_or_jupiter_transit": self.saturn_or_jupiter_transit,
            "saturn_and_jupiter_transit":
                self.saturn_and_jupiter_transit,
        }


@dataclass
class DeathBacktrace:
    """Result of backtracing one chart's known event date."""
    name: str
    death_date: date
    age_at_death_years: float
    death_window: Optional[WindowFeatures]   # the window containing death
    all_windows: List[WindowFeatures]        # every PAD/AD window in life


# ── Helpers ─────────────────────────────────────────────────────────

_DUSTHANA = {6, 8, 12}
_MALEFICS = {"Sun", "Mars", "Saturn", "Rahu", "Ketu"}
_BENEFICS = {"Moon", "Mercury", "Jupiter", "Venus"}


def _planet_natal_sign_num(chart: Chart, planet: str) -> int:
    """1..12 sign index of a planet at birth in D1, or 0 if missing."""
    pp = chart.vargas["D1"].planet_positions.get(planet)
    if pp is None or not pp.sign:
        return 0
    try:
        return sign_index(pp.sign)
    except KeyError:
        return 0


def _age_band(age_years: float) -> str:
    if age_years < 18:
        return "child"
    if age_years < 35:
        return "young"
    if age_years < 60:
        return "mid"
    return "old"


def _flat_pad_windows(root: Optional[DashaNode]) -> List[tuple]:
    """Return list of (md_node, ad_node, pad_node|None) for every leaf
    window. If an AD has no PAD children, emit (md, ad, None) so the
    AD itself is the leaf."""
    if root is None:
        return []
    out = []
    for md in root.children:
        for ad in md.children:
            if ad.children:
                for pad in ad.children:
                    out.append((md, ad, pad))
            else:
                out.append((md, ad, None))
    return out


def _build_lord_role_map(
    chart: Chart, resolved,
) -> Dict[str, List[str]]:
    """Replicates the role-tagging from features.parashari._lord_role_map
    so backtrace produces the same role labels rules consume.
    """
    target = resolved.target_house_rotated
    direct = resolved.target_house_direct
    d1 = chart.vargas["D1"]

    def _lord_of_house(h: int) -> str:
        return SIGN_LORD[d1.house_signs[h]]

    rotated_lord = _lord_of_house(target)
    direct_lord = _lord_of_house(direct)
    badhaka_off = badhaka_house(d1.house_signs[target])
    badhaka_lord = _lord_of_house(house_from(target, badhaka_off))
    eighth_from = _lord_of_house(house_from(target, 8))
    marakas = [_lord_of_house(h) for h in maraka_houses(target)]

    # Resolve relation/domain karakas (mirroring the feature extractor).
    rel_karakas: List[str] = []
    for k in resolved.relation_karakas:
        if k == "Lagna_Lord":
            lagna = d1.planet_positions.get("Lagna")
            if lagna is not None:
                rel_karakas.append(SIGN_LORD[lagna.sign])
        else:
            rel_karakas.append(k)
    dom_karakas: List[str] = []
    for t in resolved.domain_karakas:
        if len(t) >= 2 and t[-1] == "L" and t[:-1].isdigit():
            offset = int(t[:-1])
            dom_karakas.append(_lord_of_house(house_from(target, offset)))
        else:
            dom_karakas.append(t)

    roles: Dict[str, List[str]] = {}

    def tag(p: str, r: str) -> None:
        if p in ("", None, "Lagna"):
            return
        roles.setdefault(p, []).append(r)

    tag(rotated_lord, "rotated_lord")
    tag(direct_lord, "direct_lord")
    tag(badhaka_lord, "badhaka_lord")
    tag(eighth_from, "eighth_from_target")
    for m in marakas:
        tag(m, "maraka")
    for k in rel_karakas:
        tag(k, "relation_karaka")
    for k in dom_karakas:
        tag(k, "domain_karaka")
    return roles


def _planet_house(chart: Chart, planet: str) -> int:
    pp = chart.vargas["D1"].planet_positions.get(planet)
    return pp.house if pp else 0


def _planet_dignity(chart: Chart, planet: str) -> str:
    pp = chart.vargas["D1"].planet_positions.get(planet)
    return pp.dignity if pp else ""


def _planet_combust(chart: Chart, planet: str) -> bool:
    pp = chart.vargas["D1"].planet_positions.get(planet)
    return bool(pp.combust) if pp else False


def _planet_nak_lord(chart: Chart, planet: str) -> str:
    pp = chart.vargas["D1"].planet_positions.get(planet)
    return pp.nakshatra_lord if pp else ""


def _window_features(
    md: DashaNode, ad: DashaNode, pad: Optional[DashaNode],
    chart: Chart, lord_roles: Dict[str, List[str]],
    transit_table: Dict[str, list], target_sign_nums: set,
    *,
    target_house_rotated: int,
    natal_sun_sign_num: int,
    birth_dt: datetime,
) -> WindowFeatures:
    md_lord = md.lord
    ad_lord = ad.lord
    pad_lord = pad.lord if pad is not None else None

    chain = [md_lord, ad_lord]
    if pad_lord:
        chain.append(pad_lord)
    n_chain = len(chain)
    hits = [l for l in chain if l in lord_roles]
    n_hit = len(hits)
    if n_hit == n_chain:
        chain_strength = "full"
    elif n_hit >= max(1, n_chain - 1):
        chain_strength = "partial"
    else:
        chain_strength = "weak"

    md_roles = sorted(lord_roles.get(md_lord, []))
    ad_roles = sorted(lord_roles.get(ad_lord, []))
    pad_roles = sorted(lord_roles.get(pad_lord, [])) if pad_lord else []

    roles_present = sorted({
        r for level_roles in (md_roles, ad_roles, pad_roles)
        for r in level_roles
    })

    start = pad.start if pad is not None else ad.start
    end = pad.end if pad is not None else ad.end

    saturn_trans = False
    jupiter_trans = False
    if transit_table and target_sign_nums:
        mid = start + (end - start) / 2
        saturn_trans = _transit_active_at(
            transit_table, "Saturn", mid, target_sign_nums,
        )
        jupiter_trans = _transit_active_at(
            transit_table, "Jupiter", mid, target_sign_nums,
        )

    chain_set = set(chain)
    md_nak = _planet_nak_lord(chart, md_lord)
    ad_nak = _planet_nak_lord(chart, ad_lord)
    pad_nak = _planet_nak_lord(chart, pad_lord) if pad_lord else ""

    md_h = _planet_house(chart, md_lord)
    pad_h = _planet_house(chart, pad_lord) if pad_lord else 0
    if md_h and pad_h:
        # 1-based offset from MD lord's house to PAD lord's house.
        offset = ((pad_h - md_h) % 12) + 1
    else:
        offset = 0

    n_deb = sum(
        1 for p in chain if _planet_dignity(chart, p) == "debilitated"
    )
    n_comb = sum(1 for p in chain if _planet_combust(chart, p))

    # ── Extended factor extraction ───────────────────────────────
    md_h = _planet_house(chart, md_lord)
    ad_h = _planet_house(chart, ad_lord)
    pad_h_n = _planet_house(chart, pad_lord) if pad_lord else 0
    chain_houses = [h for h in (md_h, ad_h, pad_h_n) if h]

    chain_in_8h = any(h == 8 for h in chain_houses)
    chain_in_12h = any(h == 12 for h in chain_houses)
    chain_in_dust = any(h in _DUSTHANA for h in chain_houses)

    # 8th from rotated target (relation's death-house).
    eighth_from_target_house = ((target_house_rotated - 1 + 7) % 12) + 1
    chain_in_8h_target = any(
        h == eighth_from_target_house for h in chain_houses
    )

    n_mal = sum(1 for p in chain if p in _MALEFICS)
    n_ben = sum(1 for p in chain if p in _BENEFICS)

    md_pad_same = bool(pad_lord) and md_lord == pad_lord
    md_ad_same = md_lord == ad_lord

    duration_days = max(0, (end - start).days)
    short = duration_days < 60
    long_ = duration_days > 365

    mid = start + (end - start) / 2
    age_mid_years = max(0.0, (mid - birth_dt).total_seconds()
                        / (365.25 * 86400))
    band = _age_band(age_mid_years)

    sat_over_sun = False
    if transit_table and natal_sun_sign_num:
        sat_over_sun = _transit_active_at(
            transit_table, "Saturn", mid, {natal_sun_sign_num},
        )

    sat_or_jup = saturn_trans or jupiter_trans
    sat_and_jup = saturn_trans and jupiter_trans

    return WindowFeatures(
        level="PAD" if pad is not None else "AD",
        md=md_lord, ad=ad_lord, pad=pad_lord,
        start=start, end=end,
        md_roles=md_roles, ad_roles=ad_roles, pad_roles=pad_roles,
        chain_strength=chain_strength,
        n_distinct_roles=len(roles_present),
        roles_present=roles_present,
        saturn_transit_target=saturn_trans,
        jupiter_transit_target=jupiter_trans,
        md_nak_lord=md_nak, ad_nak_lord=ad_nak, pad_nak_lord=pad_nak,
        md_nak_lord_role=sorted(lord_roles.get(md_nak, [])),
        ad_nak_lord_role=sorted(lord_roles.get(ad_nak, [])),
        pad_nak_lord_role=sorted(lord_roles.get(pad_nak, [])),
        saturn_in_chain="Saturn" in chain_set,
        sun_in_chain="Sun" in chain_set,
        jupiter_in_chain="Jupiter" in chain_set,
        rahu_or_ketu_in_chain=bool({"Rahu", "Ketu"} & chain_set),
        mars_in_chain="Mars" in chain_set,
        md_dignity=_planet_dignity(chart, md_lord),
        ad_dignity=_planet_dignity(chart, ad_lord),
        pad_dignity=_planet_dignity(chart, pad_lord) if pad_lord else "",
        n_debilitated_in_chain=n_deb,
        n_combust_in_chain=n_comb,
        pad_house_from_md_house=offset,
        pad_in_dusthana_from_md=offset in _DUSTHANA,
        md_house=md_h,
        ad_house=ad_h,
        pad_house=pad_h_n,
        chain_lord_in_8h=chain_in_8h,
        chain_lord_in_12h=chain_in_12h,
        chain_lord_in_dusthana=chain_in_dust,
        chain_lord_in_8h_from_target=chain_in_8h_target,
        n_malefics_in_chain=n_mal,
        n_benefics_in_chain=n_ben,
        md_pad_same_planet=md_pad_same,
        md_ad_same_planet=md_ad_same,
        window_duration_days=duration_days,
        window_short=short,
        window_long=long_,
        age_at_window_mid_years=age_mid_years,
        age_band=band,
        saturn_over_sun_at_mid=sat_over_sun,
        saturn_or_jupiter_transit=sat_or_jup,
        saturn_and_jupiter_transit=sat_and_jup,
    )


def _build_focus(rec: dict) -> Optional[FocusQuery]:
    try:
        bd = date.fromisoformat(rec["birth_date"])
    except Exception:
        return None
    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    lat = rec.get("lat")
    lon = rec.get("lon")
    if lat is None or lon is None:
        return None
    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(lat), lon=float(lon),
    )
    return FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI],
        gender=rec.get("gender") or "M",
        min_confidence=0.6,
    )


def backtrace_chart(rec: dict) -> Optional[DeathBacktrace]:
    """Backtrace one chart record. Returns None if the record is unusable
    (missing fields, ephemeris failure, no death_date)."""
    q = _build_focus(rec)
    if q is None:
        return None

    death_str = rec.get("father_death_date")
    if not death_str:
        return None
    try:
        death_date = date.fromisoformat(death_str)
    except Exception:
        return None

    try:
        resolver = FocusResolver()
        resolved = resolver.resolve(q)
        chart = ChartComputer().compute(
            q.birth, q.config,
            vargas=resolved.vargas_required,
            dashas=resolved.dashas_required,
            need_kp=False, need_jaimini=False,
        )
    except Exception:
        return None

    if chart.vimshottari is None:
        return None

    lord_roles = _build_lord_role_map(chart, resolved)

    # Transit table — same setup as features/parashari.
    target_sign_nums: set = set()
    d1 = chart.vargas.get("D1")
    if d1:
        for h in (resolved.target_house_rotated,
                  resolved.target_house_direct):
            s = d1.house_signs.get(h)
            if s:
                target_sign_nums.add(sign_index(s))
    try:
        transit_table = _compute_transit_table(
            chart, planets=["Saturn", "Jupiter"], n_months=1200,
        )
    except Exception:
        transit_table = {}

    leaf_windows = _flat_pad_windows(chart.vimshottari)
    death_dt = datetime(
        death_date.year, death_date.month, death_date.day,
    )
    natal_sun_sign_num = _planet_natal_sign_num(chart, "Sun")
    birth_dt_naive = datetime(
        q.birth.date.year, q.birth.date.month, q.birth.date.day,
    )

    all_features: List[WindowFeatures] = []
    death_window: Optional[WindowFeatures] = None
    for md, ad, pad in leaf_windows:
        wf = _window_features(
            md, ad, pad, chart, lord_roles,
            transit_table, target_sign_nums,
            target_house_rotated=resolved.target_house_rotated,
            natal_sun_sign_num=natal_sun_sign_num,
            birth_dt=birth_dt_naive,
        )
        all_features.append(wf)
        if wf.start <= death_dt <= wf.end:
            death_window = wf

    age_years = (death_date - q.birth.date).days / 365.25

    return DeathBacktrace(
        name=rec.get("name", "?"),
        death_date=death_date,
        age_at_death_years=age_years,
        death_window=death_window,
        all_windows=all_features,
    )
