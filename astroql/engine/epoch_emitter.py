"""Epoch state emitter (Module A of NEUROSYMBOLIC_ENGINE_DESIGN.md).

Given a birth chart and an **explicit** [query_start, query_end] window,
emit a list of `EpochState` snapshots at sookshma dasha granularity
(MD → AD → PD → SD).

Hard contract (design §3.A, locked 2026-04-24):
  * `query_start` and `query_end` are required — there is no lifetime /
    open-ended default, anywhere in the stack.
  * `query_end - query_start` must be positive and ≤ `max_window_years`
    (default 10 years). Callers opt in to longer windows explicitly.

Natal-chart derived data (signs, natal houses, shadbala μ) is computed
once and reused for every epoch. Only transit positions + aspects +
dasha lords change per epoch.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from ..schemas.birth import BirthDetails
from ..schemas.epoch_state import (
    DashaStack,
    EpochState,
    PlanetEpochState,
)
from . import aspects as _aspects
from . import ashtakavarga as _ashtakavarga
from . import shadbala as _sb


# astro-prod engine import (mirrors astroql/chart/computer.py).
_ASTRO_PROD_PATH = Path(
    "C:/Users/ravii/.gemini/antigravity/playground/astro-prod"
)
if str(_ASTRO_PROD_PATH) not in sys.path:
    sys.path.insert(0, str(_ASTRO_PROD_PATH))

from astro_engine import AstroEngine  # noqa: E402


DEFAULT_MAX_WINDOW_YEARS = 10.0


class EpochEmissionError(RuntimeError):
    pass


def _birth_to_datetime(birth: BirthDetails) -> _dt.datetime:
    if ZoneInfo is None:
        raise EpochEmissionError("zoneinfo unavailable; Python 3.9+ required")
    time_str = birth.time or "12:00:00"
    parts = time_str.split(":")
    hh = int(parts[0])
    mm = int(parts[1]) if len(parts) > 1 else 0
    ss = int(parts[2]) if len(parts) > 2 else 0
    try:
        zone = ZoneInfo(birth.tz)
    except Exception as e:
        raise EpochEmissionError(f"unknown tz {birth.tz!r}: {e}") from e
    return _dt.datetime(
        birth.date.year, birth.date.month, birth.date.day,
        hh, mm, ss, tzinfo=zone,
    )


def _whole_sign_house(planet_sign_num: int, lagna_sign_num: int) -> int:
    return ((planet_sign_num - lagna_sign_num) % 12) + 1


_SIGN_ORDER = (
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
)

_HOUSE_NAMES = (
    "lagna_lord", "second_lord", "third_lord", "fourth_lord",
    "fifth_lord", "sixth_lord", "seventh_lord", "eighth_lord",
    "ninth_lord", "tenth_lord", "eleventh_lord", "twelfth_lord",
)


def _nth_sign_from(sign: str, n: int) -> str:
    """Return the nth sign forward from `sign` (1-indexed: n=1 → sign)."""
    try:
        i = _SIGN_ORDER.index(sign)
    except ValueError:
        return ""
    return _SIGN_ORDER[(i + n - 1) % 12]


def _compute_derived_lords(
    natal_lagna_sign: str,
    sun_natal_sign: str,
) -> Dict[str, str]:
    """Pre-compute every lord identity that the JSON DSL might query.

    Returns a flat dict the rule author can reference via dot-paths:
      "derived_lords.ninth_lord", "derived_lords.father_8L", etc.

    All values are planet names (e.g. "Saturn"). Empty string when
    a lord cannot be derived (typically chart not fully populated).
    """
    out: Dict[str, str] = {}
    if not natal_lagna_sign:
        return out
    # Native-lagna-relative house lords (1L..12L).
    for house_idx, key in enumerate(_HOUSE_NAMES, start=1):
        sign = _nth_sign_from(natal_lagna_sign, house_idx)
        out[key] = _sb.sign_lord(sign) if sign else ""
    # Derived father-lagna lords (BPHS Ch. 8). Father's lagna =
    # native's 9H sign; F2L/F7L/F8L/F12L are 2nd/7th/8th/12th from
    # that derived lagna.
    father_lagna = _nth_sign_from(natal_lagna_sign, 9)
    out["father_lagna_sign"] = father_lagna
    if father_lagna:
        out["father_2L"] = _sb.sign_lord(_nth_sign_from(father_lagna, 2))
        out["father_7L"] = _sb.sign_lord(_nth_sign_from(father_lagna, 7))
        out["father_8L"] = _sb.sign_lord(_nth_sign_from(father_lagna, 8))
        out["father_12L"] = _sb.sign_lord(_nth_sign_from(father_lagna, 12))
    # Sun-karaka marakas for father (2nd/7th from natal Sun).
    if sun_natal_sign:
        out["sun_2nd_maraka"] = _sb.sign_lord(
            _nth_sign_from(sun_natal_sign, 2))
        out["sun_7th_maraka"] = _sb.sign_lord(
            _nth_sign_from(sun_natal_sign, 7))
    return out


def _ensure_aware(dt: _dt.datetime, fallback_tz: str) -> _dt.datetime:
    """Promote naive datetimes to tz-aware using fallback_tz."""
    if dt.tzinfo is not None:
        return dt
    if ZoneInfo is None:
        raise EpochEmissionError("zoneinfo unavailable")
    return dt.replace(tzinfo=ZoneInfo(fallback_tz))


def _parse_sd_date(s: str, tz_name: str) -> _dt.datetime:
    """Parse astro-prod's 'YYYY-MM-DD' SD boundary as tz-aware at
    start-of-day in the birth tz (matches dasha calculation semantics).
    """
    if ZoneInfo is None:
        raise EpochEmissionError("zoneinfo unavailable")
    d = _dt.date.fromisoformat(s)
    return _dt.datetime(
        d.year, d.month, d.day, 0, 0, 0,
        tzinfo=ZoneInfo(tz_name),
    )


def emit_epochs(
    birth: BirthDetails,
    query_start: _dt.datetime,
    query_end: _dt.datetime,
    max_window_years: float = DEFAULT_MAX_WINDOW_YEARS,
    include_aspects: bool = True,
) -> List[EpochState]:
    """Emit sookshma-granularity `EpochState`s covering [query_start,
    query_end]. Both boundaries are required.

    Raises `EpochEmissionError` for any input-contract violation
    (missing / inverted / oversized window).
    """
    # ── Input-contract validation (design §3.A) ─────────────────────
    if query_start is None or query_end is None:
        raise EpochEmissionError(
            "query_start and query_end are required — no lifetime / "
            "open-ended queries are accepted."
        )
    query_start = _ensure_aware(query_start, birth.tz)
    query_end = _ensure_aware(query_end, birth.tz)
    if query_end <= query_start:
        raise EpochEmissionError(
            f"query_end ({query_end.isoformat()}) must be strictly "
            f"after query_start ({query_start.isoformat()})"
        )
    duration_days = (query_end - query_start).total_seconds() / 86400.0
    max_days = max_window_years * 365.25
    if duration_days > max_days:
        raise EpochEmissionError(
            f"query window is {duration_days:.1f} days > "
            f"{max_days:.1f} (max_window_years={max_window_years}). "
            f"Pass a larger max_window_years explicitly if intentional."
        )

    engine = AstroEngine()
    birth_dt = _birth_to_datetime(birth)

    # ── Natal chart — computed once ─────────────────────────────────
    natal_positions_raw = engine.calculate_planetary_positions(
        birth_dt, birth.lat, birth.lon,
    )
    # astro-prod may include outer planets (Uranus/Neptune/Pluto).
    # Filter to the 9-planet classical Parashari set.
    natal_positions: Dict[str, Any] = {
        p: d for p, d in natal_positions_raw.items()
        if p in _sb.PARASHARI_PLANETS
    }
    natal_lagna = engine.calculate_lagna(birth_dt, birth.lat, birth.lon)
    natal_lagna_sign_num = int(natal_lagna["sign_num"])

    natal_sign_by_planet: Dict[str, str] = {
        p: d["rashi"] for p, d in natal_positions.items()
    }
    natal_sign_num_by_planet: Dict[str, int] = {
        p: int(d["sign_num"]) for p, d in natal_positions.items()
    }
    natal_house_by_planet: Dict[str, int] = {
        p: _whole_sign_house(int(d["sign_num"]), natal_lagna_sign_num)
        for p, d in natal_positions.items()
    }

    natal_shadbala = engine.calculate_shadbala(
        natal_positions,
        {
            "longitude": float(natal_lagna["longitude"]),
            "sign_num": natal_lagna_sign_num,
            "rashi": natal_lagna["rashi"],
        },
    )
    natal_virupas: Dict[str, float] = {
        p: float(v.get("score", 0.0))
        for p, v in natal_shadbala.items()
        if p in _sb.PARASHARI_PLANETS
    }
    mu_by_planet: Dict[str, float] = _sb.normalize_all(
        natal_virupas, natal_sign_by_planet,
    )

    # ── Pre-computed lord identities (chart-static) ──────────────
    # Exposed on every EpochState so DSL paths (e.g. "derived_lords.
    # ninth_lord") resolve without invoking helper functions. This
    # is what unblocks LLM-emitted rules that reference house lords
    # via JSON conditions (see dsl_evaluator.evaluate).
    derived_lords = _compute_derived_lords(
        natal_lagna_sign=natal_lagna["rashi"],
        sun_natal_sign=natal_sign_by_planet.get("Sun", ""),
    )

    # ── Bhinnashtakavarga grid (chart-static, BPHS Ch. 66) ───────
    # Computed once from natal positions; reused across all SD
    # epochs. Used to GATE transit rules — see
    # `engine.ashtakavarga` for the gating-pattern docstring.
    try:
        bav_table = _ashtakavarga.bav_grid(
            sign_by_planet=natal_sign_by_planet,
            lagna_sign=natal_lagna["rashi"],
        )
    except Exception:
        # If natal data is incomplete, leave the grid empty. DSL
        # rules that reference ashtakavarga.* will fail to fire
        # (treated as a missing path); the rest of the engine
        # continues to work. This matches existing tolerance for
        # incomplete shadbala / nodes.
        bav_table = {}

    # ── Sookshma-depth dasha sequence, clipped to query window ──────
    moon_lon = float(natal_positions["Moon"]["longitude"])
    raw_seq = engine.calculate_dasha_sequence(
        moon_lon=moon_lon,
        birth_date=birth_dt,
        start_date=query_start,
        end_date=query_end,
        include_sookshma=True,
    )
    sd_records = [r for r in raw_seq if r.get("type") == "SD"]

    epochs: List[EpochState] = []
    for idx, rec in enumerate(sd_records):
        start = _parse_sd_date(rec["start"], birth.tz)
        end = _parse_sd_date(rec["end"], birth.tz)
        # Clip to query window.
        if start < query_start:
            start = query_start
        if end > query_end:
            end = query_end
        if end <= start:
            continue

        mid = start + (end - start) / 2
        tr_positions_raw = engine.calculate_planetary_positions(
            mid, birth.lat, birth.lon,
        )
        tr_positions: Dict[str, Any] = {
            p: d for p, d in tr_positions_raw.items()
            if p in _sb.PARASHARI_PLANETS
        }
        tr_sign_by_planet: Dict[str, str] = {
            p: d["rashi"] for p, d in tr_positions.items()
        }
        tr_sign_num_by_planet: Dict[str, int] = {
            p: int(d["sign_num"]) for p, d in tr_positions.items()
        }
        tr_retro_by_planet: Dict[str, bool] = {
            p: bool(d.get("is_retrograde", False))
            for p, d in tr_positions.items()
        }

        planets_out: Dict[str, PlanetEpochState] = {}
        for planet in tr_positions.keys():
            tr_sign = tr_sign_by_planet[planet]
            tr_sign_num = tr_sign_num_by_planet[planet]
            tr_house = _whole_sign_house(
                tr_sign_num, natal_lagna_sign_num,
            )
            rec_aspects: List[str] = []
            on_natal: List[str] = []
            if include_aspects:
                rec_aspects = _aspects.aspects_receiving(
                    planet, tr_sign_num, tr_sign_num_by_planet,
                )
                # Transit-on-natal: which transiting planets aspect
                # *this* planet's natal sign? Computed against the
                # current transit-position table for the *aspectors*,
                # but the target sign is this planet's natal sign.
                natal_sign_num = natal_sign_num_by_planet.get(planet)
                if natal_sign_num is not None:
                    on_natal = _aspects.aspects_receiving(
                        planet, natal_sign_num, tr_sign_num_by_planet,
                    )
            planets_out[planet] = PlanetEpochState(
                transit_sign=tr_sign,
                transit_house=tr_house,
                natal_house=natal_house_by_planet.get(planet, 0),
                shadbala_coefficient=mu_by_planet.get(planet, 0.0),
                shadbala_virupas=natal_virupas.get(planet),
                is_retrograde=tr_retro_by_planet[planet],
                aspects_receiving=rec_aspects,
                aspects_on_natal=on_natal,
                natal_sign=natal_sign_by_planet.get(planet, ""),
            )

        dashas = DashaStack(
            maha=rec.get("lord") or "",
            antar=rec.get("sub_lord") or "",
            pratyantar=rec.get("prat_lord") or "",
            sookshma=rec.get("sookshma_lord") or "",
        )
        epoch_id = (
            f"{dashas.maha}-{dashas.antar}-{dashas.pratyantar}-"
            f"{dashas.sookshma}@{start.date().isoformat()}#{idx}"
        )
        epochs.append(EpochState(
            epoch_id=epoch_id,
            start_time=start,
            end_time=end,
            dashas=dashas,
            planets=planets_out,
            natal_lagna_sign=natal_lagna["rashi"],
            derived_lords=derived_lords,
            ashtakavarga=bav_table,
        ))

    return epochs
