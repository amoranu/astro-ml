"""ChartComputer (spec §6.3).

Thin wrapper over `astro-prod/astro_engine.py::AstroEngine`. Responsibilities:
    - build tz-aware datetime from BirthDetails (historical IANA tz via zoneinfo)
    - call AstroEngine for positions, lagna, vargas, dashas, karakas
    - map astro-prod's dicts to our Chart dataclasses

Extensibility:
    - Each school-specific computation (Jaimini karakas, KP cusps, etc.)
      is gated by a flag derived from ResolvedFocus/ChartConfig, so a caller
      wanting Parashari-only pays nothing for Jaimini/KP.
    - Varga set is data-driven from ResolvedFocus.vargas_required.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore

from ..schemas.birth import BirthDetails, ChartConfig
from ..schemas.chart import Chart, DashaNode, PlanetPosition, Varga

# ── astro-prod engine import (see memory/astro_engine_source.md) ─────
_ASTRO_PROD_PATH = Path(
    "C:/Users/ravii/.gemini/antigravity/playground/astro-prod"
)
if str(_ASTRO_PROD_PATH) not in sys.path:
    sys.path.insert(0, str(_ASTRO_PROD_PATH))

from astro_engine import AstroEngine  # noqa: E402


class ChartComputerError(RuntimeError):
    pass


# Nakshatra → Vimshottari dasha lord (for nakshatra_lord field).
_NAK_TO_LORD = {
    "Ashwini": "Ketu", "Bharani": "Venus", "Krittika": "Sun",
    "Rohini": "Moon", "Mrigashira": "Mars", "Ardra": "Rahu",
    "Punarvasu": "Jupiter", "Pushya": "Saturn", "Ashlesha": "Mercury",
    "Magha": "Ketu", "Purva Phalguni": "Venus", "Uttara Phalguni": "Sun",
    "Hasta": "Moon", "Chitra": "Mars", "Swati": "Rahu",
    "Vishakha": "Jupiter", "Anuradha": "Saturn", "Jyeshtha": "Mercury",
    "Mula": "Ketu", "Purva Ashadha": "Venus", "Uttara Ashadha": "Sun",
    "Shravana": "Moon", "Dhanishta": "Mars", "Shatabhisha": "Rahu",
    "Purva Bhadrapada": "Jupiter", "Uttara Bhadrapada": "Saturn",
    "Revati": "Mercury",
}


def _to_aware_datetime(birth: BirthDetails) -> _dt.datetime:
    """Build a tz-aware datetime from BirthDetails using IANA tzdata."""
    if ZoneInfo is None:
        raise ChartComputerError(
            "zoneinfo unavailable; Python 3.9+ required"
        )
    time_str = birth.time or "12:00:00"
    parts = time_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    ss = int(parts[2]) if len(parts) > 2 else 0
    try:
        zone = ZoneInfo(birth.tz)
    except Exception as e:
        raise ChartComputerError(f"unknown tz {birth.tz!r}: {e}") from e
    return _dt.datetime(
        birth.date.year, birth.date.month, birth.date.day,
        hh, mm, ss, tzinfo=zone,
    )


def _whole_sign_house(planet_sign_num: int, lagna_sign_num: int) -> int:
    return ((planet_sign_num - lagna_sign_num) % 12) + 1


def _position_from_engine_data(
    planet: str, data: Dict[str, Any], house: int, dignity: str,
) -> PlanetPosition:
    nak = data.get("nakshatra", "")
    return PlanetPosition(
        planet=planet,
        longitude=float(data.get("longitude", 0.0)),
        sign=data.get("rashi", ""),
        house=house,
        nakshatra=nak,
        nakshatra_lord=_NAK_TO_LORD.get(nak, ""),
        nakshatra_pada=int(data.get("pada", 0) or 0),
        retrograde=bool(data.get("is_retrograde", False)),
        combust=bool(data.get("is_combust", False)),
        dignity=(dignity or "neutral").lower(),
        speed=float(data.get("daily_speed") or 0.0),
    )


def _build_varga_d1(
    positions: Dict[str, Dict[str, Any]],
    lagna_sign_num: int,
    engine: AstroEngine,
) -> Varga:
    house_signs: Dict[int, str] = {}
    rashis = engine.rashis
    for i in range(12):
        sign_idx = (lagna_sign_num - 1 + i) % 12
        house_signs[i + 1] = rashis[sign_idx]
    planet_positions: Dict[str, PlanetPosition] = {}
    for planet, data in positions.items():
        sign_num = int(data["sign_num"])
        house = _whole_sign_house(sign_num, lagna_sign_num)
        dignity = engine.get_planet_dignity(planet, sign_num)
        planet_positions[planet] = _position_from_engine_data(
            planet, data, house, dignity,
        )
    return Varga(
        name="D1",
        planet_positions=planet_positions,
        house_cusps=[],  # whole-sign: cusps = sign boundaries, not per-degree
        house_signs=house_signs,
    )


def _build_divisional_with_lagna(
    name: str,
    division: int,
    positions: Dict[str, Dict[str, Any]],
    lagna_longitude: float,
    engine: AstroEngine,
) -> Varga:
    """Build a Varga using astro-prod's get_divisional_chart for sign
    placement of planets, but compute varga-lagna properly from the
    lagna's longitude (CAV-004 fix)."""
    div = engine.get_divisional_chart(positions, division)
    # Compute varga lagna sign by applying the same divisional formula
    # to the lagna longitude. astro-prod's get_divisional_chart can
    # accept a synthesized "Lagna" entry too.
    fake_positions = {"Lagna": {"longitude": lagna_longitude}}
    try:
        lagna_div = engine.get_divisional_chart(fake_positions, division)
        varga_lagna_sign_num = int(lagna_div["Lagna"]["sign_num"])
    except Exception:
        varga_lagna_sign_num = int(lagna_longitude // 30) + 1

    rashis = engine.rashis
    house_signs: Dict[int, str] = {}
    for i in range(12):
        house_signs[i + 1] = rashis[(varga_lagna_sign_num - 1 + i) % 12]

    planet_positions: Dict[str, PlanetPosition] = {}
    for planet, d in div.items():
        sign_num = int(d["sign_num"])
        house = _whole_sign_house(sign_num, varga_lagna_sign_num)
        dignity = engine.get_planet_dignity(planet, sign_num)
        base = positions.get(planet, {})
        planet_positions[planet] = PlanetPosition(
            planet=planet,
            longitude=float(base.get("longitude", 0.0)),
            sign=d.get("rashi", ""),
            house=house,
            nakshatra="",
            nakshatra_lord="",
            nakshatra_pada=0,
            retrograde=bool(base.get("is_retrograde", False)),
            combust=bool(base.get("is_combust", False)),
            dignity=(dignity or "neutral").lower(),
            speed=float(base.get("daily_speed") or 0.0),
        )
    return Varga(
        name=name, planet_positions=planet_positions,
        house_cusps=[], house_signs=house_signs,
    )


def _build_divisional(
    name: str,
    division: int,
    positions: Dict[str, Dict[str, Any]],
    lagna_sign_num: int,
    engine: AstroEngine,
) -> Varga:
    """Build a divisional Varga using astro-prod's get_divisional_chart."""
    div = engine.get_divisional_chart(positions, division)
    # Determine lagna's divisional sign similarly by running the formula on
    # the lagna longitude. We synthesize a "Lagna" entry in positions for this.
    house_signs: Dict[int, str] = {}
    # Fallback: align house 1 with D1 lagna's whole-sign (good enough for
    # analysis-level features; a strict varga-lagna calc can be wired later).
    rashis = engine.rashis
    for i in range(12):
        house_signs[i + 1] = rashis[(lagna_sign_num - 1 + i) % 12]

    planet_positions: Dict[str, PlanetPosition] = {}
    for planet, d in div.items():
        sign_num = int(d["sign_num"])
        house = _whole_sign_house(sign_num, lagna_sign_num)
        dignity = engine.get_planet_dignity(planet, sign_num)
        # Varga entries from astro-prod only carry sign; reuse D1 longitude
        # for reference, zero out nakshatra info (strictly D-chart sign matters).
        base = positions.get(planet, {})
        planet_positions[planet] = PlanetPosition(
            planet=planet,
            longitude=float(base.get("longitude", 0.0)),
            sign=d.get("rashi", ""),
            house=house,
            nakshatra="",
            nakshatra_lord="",
            nakshatra_pada=0,
            retrograde=bool(base.get("is_retrograde", False)),
            combust=bool(base.get("is_combust", False)),
            dignity=(dignity or "neutral").lower(),
            speed=float(base.get("daily_speed") or 0.0),
        )
    return Varga(
        name=name,
        planet_positions=planet_positions,
        house_cusps=[],
        house_signs=house_signs,
    )


def _sequence_to_dasha_tree(seq: List[Dict[str, Any]]) -> Optional[DashaNode]:
    """Convert astro-prod's flat MD/AD/PAD/SD list into our hierarchical tree.

    Returns a sentinel root whose children are MDs. None if empty.
    """
    if not seq:
        return None

    def _parse(s: str) -> _dt.datetime:
        return _dt.datetime.strptime(s, "%Y-%m-%d")

    root = DashaNode(
        lord="ROOT",
        start=_parse(seq[0]["start"]),
        end=_parse(seq[-1]["end"]),
        level=0,
        children=[],
    )
    stack: List[DashaNode] = [root]
    levels = {"MD": 1, "AD": 2, "PAD": 3, "SD": 4}
    for row in seq:
        lvl = levels.get(row.get("type"), 0)
        if lvl == 0:
            continue
        node = DashaNode(
            lord=row["lord"] if lvl == 1 else (
                row.get("sub_lord") if lvl == 2 else
                row.get("prat_lord") if lvl == 3 else
                row.get("sookshma_lord")
            ),
            start=_parse(row["start"]),
            end=_parse(row["end"]),
            level=lvl,
            children=[],
        )
        # Pop stack until we find parent (level == lvl-1)
        while stack and stack[-1].level >= lvl:
            stack.pop()
        if not stack:
            stack.append(root)
        stack[-1].children.append(node)
        stack.append(node)
    return root


# Divisions implemented by astro-prod's get_divisional_chart.
_VARGA_DIVISIONS = {
    "D1": 1, "D2": 2, "D3": 3, "D4": 4, "D6": 6, "D7": 7, "D9": 9,
    "D10": 10, "D12": 12, "D24": 24, "D30": 30, "D60": 60,
}


def _local_division(longitude: float, division: int) -> int:
    """Compute divisional sign_num (1-12) for the given division.

    Implements canonical formulas for D8, D11, D16, D20, D27, D40, D45
    (the divisions astro-prod doesn't natively support). Each formula
    follows BPHS / standard Parashari sources.
    """
    sign_num = int(longitude // 30) + 1
    deg_in_sign = longitude % 30
    part_span = 30.0 / division
    part = int(deg_in_sign / part_span)
    movable = {1, 4, 7, 10}
    fixed = {2, 5, 8, 11}
    odd = sign_num % 2 == 1

    if division == 8:
        # Spec: movable from sign, fixed from 9th, dual from 5th.
        if sign_num in movable:
            start = sign_num
        elif sign_num in fixed:
            start = ((sign_num + 8) % 12) + 1
        else:
            start = ((sign_num + 4) % 12) + 1
    elif division == 11:
        # D11 (Rudramsa) — odd from itself, even from 12th.
        start = sign_num if odd else ((sign_num + 10) % 12) + 1
    elif division == 16:
        # D16 (Shodashamsa) — movable from Aries, fixed from Leo, dual from Sag.
        if sign_num in movable:
            start = 1
        elif sign_num in fixed:
            start = 5
        else:
            start = 9
    elif division == 20:
        # D20 (Vimshamsa) — movable from Aries, fixed from Sag, dual from Leo.
        if sign_num in movable:
            start = 1
        elif sign_num in fixed:
            start = 9
        else:
            start = 5
    elif division == 27:
        # D27 (Nakshatramsa) — fire from Aries, earth from Cancer,
        # air from Libra, water from Capricorn.
        elem = (sign_num - 1) % 4
        start = [1, 4, 7, 10][elem]
    elif division == 40:
        # D40 (Khavedamsa) — odd from Aries, even from Libra.
        start = 1 if odd else 7
    elif division == 45:
        # D45 (Akshavedamsa) — movable from Aries, fixed from Leo,
        # dual from Sagittarius.
        if sign_num in movable:
            start = 1
        elif sign_num in fixed:
            start = 5
        else:
            start = 9
    else:
        return sign_num

    return ((start - 1 + part) % 12) + 1


def _local_varga_positions(
    planet_positions: dict, division: int,
) -> dict:
    """Build {planet: {sign_num, rashi}} for any local division."""
    rashis = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    ]
    out = {}
    for p, data in planet_positions.items():
        sign_num = _local_division(float(data["longitude"]), division)
        out[p] = {"sign_num": sign_num, "rashi": rashis[sign_num - 1]}
    return out


# Divisions we compute ourselves because astro-prod doesn't.
_LOCAL_VARGA_DIVISIONS = {
    "D8": 8, "D11": 11, "D16": 16, "D20": 20,
    "D27": 27, "D40": 40, "D45": 45,
}


# Normalize astro-prod's verbose Jaimini-karaka keys to canonical short codes.
# astro-prod labels the 5th slot "Putrakaraka (PiK)" — we relabel as PuK to
# avoid collision with the 8-karaka scheme's PiK = Pitrukaraka (father).
_JAIMINI_KEY_NORMALIZATION = {
    "Atmakaraka (AK)": "AK",
    "Amatyakaraka (AmK)": "AmK",
    "Bhratrukaraka (BK)": "BK",
    "Matrukaraka (MK)": "MK",
    "Putrakaraka (PiK)": "PuK",   # corrected from astro-prod's PiK label
    "Gnatikaraka (GK)": "GK",
    "Darakaraka (DK)": "DK",
}


def _normalize_chara_karakas(raw: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for long_key, planet in raw.items():
        short = _JAIMINI_KEY_NORMALIZATION.get(long_key, long_key)
        out[short] = planet
    return out


def _eight_karaka_scheme(
    positions: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Compute 8-karaka Jaimini scheme (CAV-002, CAV-021).

    Standard 8-karaka uses Sun, Moon, Mars, Mercury, Jupiter, Venus,
    Saturn AND Rahu (mean node, longitude reversed). Sorted by
    degrees-within-sign descending. Slots:
        AK   Atmakaraka       (1st = highest degrees)
        AmK  Amatyakaraka     (2nd)
        BK   Bhratrukaraka    (3rd)
        MK   Matrukaraka      (4th)
        PiK  Pitrukaraka      (5th = FATHER karaka — only in 8-karaka)
        PuK  Putrakaraka      (6th = children)
        GK   Gnatikaraka      (7th)
        DK   Darakaraka       (8th = lowest)
    For Rahu, degrees-within-sign use 30 - actual (since it goes
    backward in zodiac).
    """
    candidates = []
    for name in ("Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus",
                 "Saturn"):
        pdata = positions.get(name)
        if pdata is None:
            continue
        candidates.append((name, float(pdata["degree"])))
    rahu = positions.get("Rahu")
    if rahu is not None:
        # Reversed — Rahu's "effective" degree for karaka sort is 30-deg.
        candidates.append(("Rahu", 30.0 - float(rahu["degree"])))
    candidates.sort(key=lambda x: x[1], reverse=True)
    codes = ["AK", "AmK", "BK", "MK", "PiK", "PuK", "GK", "DK"]
    return {codes[i]: candidates[i][0] for i in range(min(8, len(candidates)))}


class ChartComputer:
    """Compute a `Chart` from `BirthDetails` + `ChartConfig`.

    Only the vargas and dashas caller requests are computed — no wasted work.
    Extension to new schools: add a branch inside `compute()` keyed on `config`
    or a flag, populate the corresponding fields on `Chart`.
    """

    def __init__(self) -> None:
        self._engine = AstroEngine()

    def _kp_planet_houses(
        self, positions: Dict[str, Dict[str, Any]],
        cusps_dict: Dict[int, Dict[str, Any]],
    ) -> Dict[str, int]:
        """Placidus house per planet.

        Matches astro-prod's interval logic in calculate_kp_significators
        but exposed as a standalone dict so the KP feature extractor can
        reference it without re-running the significator calc.
        """
        cusp_lons = [
            (h, cusps_dict[h]["longitude"]) for h in range(1, 13)
        ]
        out: Dict[str, int] = {}
        for p_name, p_data in positions.items():
            p_lon = float(p_data["longitude"])
            occupied = None
            for i in range(12):
                h_curr, lon_curr = cusp_lons[i]
                h_next, lon_next = cusp_lons[(i + 1) % 12]
                if lon_curr < lon_next:
                    if lon_curr <= p_lon < lon_next:
                        occupied = h_curr
                        break
                else:
                    if p_lon >= lon_curr or p_lon < lon_next:
                        occupied = h_curr
                        break
            out[p_name] = occupied if occupied is not None else 1
        return out

    def _build_local_varga(
        self, name: str, div_positions: Dict[str, Dict[str, Any]],
        base_positions: Dict[str, Dict[str, Any]], lagna_sign_num: int,
    ) -> Varga:
        """Build a Varga from a locally-computed sign_num/rashi dict.

        Used for divisions astro-prod doesn't implement (D8, D11, etc.).
        """
        rashis = self._engine.rashis
        house_signs: Dict[int, str] = {}
        for i in range(12):
            house_signs[i + 1] = rashis[(lagna_sign_num - 1 + i) % 12]
        planet_positions: Dict[str, PlanetPosition] = {}
        for planet, d in div_positions.items():
            sign_num = int(d["sign_num"])
            house = _whole_sign_house(sign_num, lagna_sign_num)
            dignity = self._engine.get_planet_dignity(planet, sign_num)
            base = base_positions.get(planet, {})
            planet_positions[planet] = PlanetPosition(
                planet=planet,
                longitude=float(base.get("longitude", 0.0)),
                sign=d.get("rashi", ""),
                house=house,
                nakshatra="",
                nakshatra_lord="",
                nakshatra_pada=0,
                retrograde=bool(base.get("is_retrograde", False)),
                combust=bool(base.get("is_combust", False)),
                dignity=(dignity or "neutral").lower(),
                speed=float(base.get("daily_speed") or 0.0),
            )
        return Varga(
            name=name, planet_positions=planet_positions,
            house_cusps=[], house_signs=house_signs,
        )

    def compute(
        self,
        birth: BirthDetails,
        config: Optional[ChartConfig] = None,
        vargas: Optional[List[str]] = None,
        dashas: Optional[List[str]] = None,
        dasha_window_years: float = 120.0,
        need_jaimini: bool = False,
        need_kp: bool = False,
        need_strength: bool = False,
    ) -> Chart:
        cfg = config or ChartConfig()
        varga_list = vargas or cfg.vargas or ["D1"]
        dasha_list = dashas or cfg.dasha_systems or []

        dt = _to_aware_datetime(birth)

        # Planetary positions (sidereal, Lahiri ayanamsa by default in
        # astro-prod — module sets SIDM_LAHIRI at import).
        positions = self._engine.calculate_planetary_positions(
            dt, birth.lat, birth.lon,
        )
        lagna = self._engine.calculate_lagna(dt, birth.lat, birth.lon)
        lagna_sign_num = int(lagna["sign_num"])

        # D1 is always built; additional vargas as requested.
        vargas_out: Dict[str, Varga] = {
            "D1": _build_varga_d1(positions, lagna_sign_num, self._engine)
        }
        # Lagna's longitude — used to compute per-varga lagna-sign.
        lagna_lon = float(lagna["longitude"])
        for vname in varga_list:
            if vname == "D1":
                continue
            if vname in _LOCAL_VARGA_DIVISIONS:
                division = _LOCAL_VARGA_DIVISIONS[vname]
                div_positions = _local_varga_positions(positions, division)
                # Compute the varga-lagna sign properly from lagna's
                # longitude using the same division formula.
                varga_lagna_sign_num = _local_division(lagna_lon, division)
                vargas_out[vname] = self._build_local_varga(
                    vname, div_positions, positions, varga_lagna_sign_num,
                )
                continue
            division = _VARGA_DIVISIONS.get(vname)
            if division is None:
                continue
            try:
                vargas_out[vname] = _build_divisional_with_lagna(
                    vname, division, positions, lagna_lon, self._engine,
                )
            except Exception:
                continue

        # Dasha tree (Vimshottari).
        vimshottari = None
        if "vimshottari" in dasha_list:
            moon_lon = float(positions["Moon"]["longitude"])
            end_dt = dt + _dt.timedelta(days=365.25 * dasha_window_years)
            try:
                seq = self._engine.calculate_dasha_sequence(
                    moon_lon=moon_lon,
                    birth_date=dt,
                    start_date=dt,
                    end_date=end_dt,
                )
                vimshottari = _sequence_to_dasha_tree(seq)
            except Exception as e:
                raise ChartComputerError(
                    f"vimshottari dasha failed: {e}"
                ) from e

        # Jaimini karakas + arudhas (optional). Normalize keys.
        chara_karakas: Dict[str, str] = {}
        arudhas: Dict[int, int] = {}
        if need_jaimini:
            scheme = (cfg.karaka_scheme or "7").strip()
            if scheme == "8":
                try:
                    chara_karakas = _eight_karaka_scheme(positions)
                except Exception:
                    chara_karakas = {}
            else:
                try:
                    raw = self._engine.calculate_jaimini_karakas(positions)
                    chara_karakas = _normalize_chara_karakas(raw)
                except Exception:
                    chara_karakas = {}
            try:
                # calculate_all_arudhas returns {"A1": {longitude,rashi,sign_num}, ...}
                raw_arudhas = self._engine.calculate_all_arudhas(
                    positions,
                    {"longitude": float(lagna["longitude"]),
                     "sign_num": lagna_sign_num,
                     "rashi": lagna["rashi"]},
                )
                # Normalize to {house_num: arudha_sign_num}.
                arudhas = {
                    int(k[1:]): int(v["sign_num"])
                    for k, v in raw_arudhas.items()
                    if isinstance(v, dict) and "sign_num" in v
                }
            except Exception:
                arudhas = {}

        # KP computation path (Phase 2).
        kp_cusps = None
        kp_cuspal_sublords = None
        kp_significators = None
        kp_cusp_details = None
        kp_planet_details = None
        kp_planet_houses = None
        if need_kp:
            # CAV-001: switch to KP ayanamsa for KP-path computations.
            # astro-prod sets SIDM_LAHIRI globally at import. We toggle
            # to SIDM_KRISHNAMURTI for the duration of the KP block, then
            # restore Lahiri so subsequent Parashari/Jaimini calls remain
            # consistent.
            import swisseph as _swe
            _prev_sidmode = None
            try:
                _swe.set_sid_mode(_swe.SIDM_KRISHNAMURTI)
                _prev_sidmode = _swe.SIDM_KRISHNAMURTI
                # Re-fetch positions under KP ayanamsa for the KP path so
                # cusps + sub-lords align to KP's ayanamsa convention.
                kp_positions = self._engine.calculate_planetary_positions(
                    dt, birth.lat, birth.lon,
                )
            except Exception:
                kp_positions = positions
            try:
                cusps_dict = self._engine.calculate_placidus_cusps(
                    dt, birth.lat, birth.lon,
                )
            except Exception as e:
                # CAV-035: Placidus fails at high latitudes or when swisseph
                # rejects the geometry. Fall back to equal-house cusps from
                # the lagna so KP analysis can still proceed (with reduced
                # precision — sub-lords on equal cusps are approximate).
                try:
                    cusps_dict = self._engine._calculate_equal_house_cusps(
                        float(lagna["longitude"]),
                    )
                except Exception:
                    raise ChartComputerError(
                        f"KP cusps failed (Placidus and equal): {e}"
                    ) from e
            try:
                kp_details = self._engine.calculate_kp_details(kp_positions)
                sigs = self._engine.calculate_kp_significators(
                    kp_positions, cusps_dict, kp_details,
                )
                kp_cusps = [cusps_dict[h]["longitude"] for h in range(1, 13)]
                kp_cuspal_sublords = {
                    h: cusps_dict[h]["sub_lord"] for h in range(1, 13)
                }
                kp_significators = sigs
                kp_cusp_details = cusps_dict
                kp_planet_details = kp_details
                kp_planet_houses = self._kp_planet_houses(
                    kp_positions, cusps_dict,
                )
            except Exception as e:
                raise ChartComputerError(
                    f"KP details/significators failed: {e}"
                ) from e
            finally:
                # Always restore Lahiri so non-KP code paths stay consistent.
                if _prev_sidmode is not None:
                    _swe.set_sid_mode(_swe.SIDM_LAHIRI)

        # Shadbala / Ashtakavarga (CAV-018) — computed on demand only.
        shadbala = None
        ashtakavarga = None
        if need_strength:
            lagna_data = {
                "longitude": float(lagna["longitude"]),
                "sign_num": lagna_sign_num,
                "rashi": lagna["rashi"],
            }
            try:
                shadbala = self._engine.calculate_shadbala(
                    positions, lagna_data,
                )
            except Exception:
                shadbala = None
            try:
                av = self._engine.calculate_ashtakavarga(
                    positions, lagna_data,
                )
                # astro-prod returns nested dict; collapse to {house: SAV}.
                if isinstance(av, dict):
                    sav = av.get("SAV") or av.get("sarvashtakavarga") or {}
                    if isinstance(sav, dict):
                        ashtakavarga = {int(k): int(v) for k, v in sav.items()
                                        if str(k).isdigit()}
            except Exception:
                ashtakavarga = None

        chart = Chart(
            birth=birth,
            config=cfg,
            vargas=vargas_out,
            vimshottari=vimshottari,
            chara=None,   # Chara dasha wiring deferred to Phase 3
            yogini=None,
            chara_karakas=chara_karakas,
            arudhas=arudhas,
            kp_cusps=kp_cusps,
            kp_cuspal_sublords=kp_cuspal_sublords,
            kp_significators=kp_significators,
            kp_cusp_details=kp_cusp_details,
            kp_planet_details=kp_planet_details,
            kp_planet_houses=kp_planet_houses,
            shadbala=shadbala,
            ashtakavarga=ashtakavarga,
        )

        # Stash lagna as a synthetic PlanetPosition on D1 so features can
        # reference it uniformly. Lagna has no nakshatra in the classical
        # sense for dignity purposes, so fields beyond sign/house are minimal.
        lagna_pos = PlanetPosition(
            planet="Lagna",
            longitude=float(lagna["longitude"]),
            sign=lagna["rashi"],
            house=1,
            nakshatra=lagna.get("nakshatra", ""),
            nakshatra_lord=_NAK_TO_LORD.get(lagna.get("nakshatra", ""), ""),
            nakshatra_pada=int(lagna.get("pada", 0) or 0),
        )
        chart.vargas["D1"].planet_positions["Lagna"] = lagna_pos

        return chart
