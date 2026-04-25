"""AstroQL DSL parser (spec §15.7).

Minimal implementation supporting the spec's EBNF:

    QUERY father.longevity.event_negative
    FROM BIRTH { date: "1965-03-15", time: "14:20:00", tz: "Asia/Kolkata",
                 lat: 28.6, lon: 77.2, gender: "M" }
    USING { vargas: [D1,D8,D12,D30], dashas: [vimshottari] }
    SCHOOLS [parashari, kp]
    WINDOW 1970-01-01 TO 2010-01-01
    WITH { min_confidence: 0.6 }
    EXPLAIN

Returns a FocusQuery. Not a general-purpose parser — narrow regex-based
tokenizer for the common-case syntax above.
"""
from __future__ import annotations

import ast
import re
from datetime import date as _date
from typing import Any, Dict, Optional

from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    Relationship, School,
)


class DSLParseError(ValueError):
    pass


_KW = re.compile(
    r"\b(QUERY|FROM|BIRTH|USING|SCHOOLS|WINDOW|WITH|EXPLAIN|ANALYZE|TO)\b",
    re.IGNORECASE,
)


def _extract_block(src: str, label: str) -> Optional[str]:
    """Extract '{ ... }' body following a keyword label."""
    pat = re.compile(
        rf"{label}\s*\{{([^}}]*)\}}", re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(src)
    return m.group(1) if m else None


def _parse_kv_block(body: str) -> Dict[str, Any]:
    """Parse a `key: value, key: [a,b,c]` block.

    Values may be quoted strings, numbers, bare identifiers, or lists of
    identifiers/quoted strings.
    """
    out: Dict[str, Any] = {}
    # Split on commas at top level (not inside brackets).
    depth = 0
    buf = []
    entries = []
    for ch in body:
        if ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            entries.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        entries.append("".join(buf).strip())
    for entry in entries:
        if not entry:
            continue
        if ":" not in entry:
            raise DSLParseError(f"malformed kv entry: {entry}")
        k, v = entry.split(":", 1)
        k = k.strip()
        v = v.strip()
        out[k] = _parse_value(v)
    return out


def _parse_value(v: str) -> Any:
    v = v.strip()
    if not v:
        return None
    if v[0] == "[" and v[-1] == "]":
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(x.strip()) for x in inner.split(",")]
    # Try Python literal (handles quoted strings, numbers).
    try:
        return ast.literal_eval(v)
    except Exception:
        # Bare identifier (e.g. D9, vimshottari, parashari).
        return v


def parse_dsl(src: str) -> FocusQuery:
    """Parse an AstroQL DSL block into a FocusQuery."""
    # Require QUERY at start (after whitespace).
    m = re.match(r"\s*QUERY\s+([a-z_]+)\.([a-z_]+)(?:\.([a-z_]+))?",
                 src, re.IGNORECASE)
    if not m:
        raise DSLParseError(
            "expected 'QUERY relationship.life_area[.effect]' at start"
        )
    rel_str, life_str, eff_str = m.group(1), m.group(2), m.group(3)
    try:
        rel = Relationship(rel_str)
    except ValueError as e:
        raise DSLParseError(f"unknown relationship: {rel_str}") from e
    try:
        life = LifeArea(life_str)
    except ValueError as e:
        raise DSLParseError(f"unknown life_area: {life_str}") from e
    eff = None
    if eff_str:
        try:
            eff = Effect(eff_str)
        except ValueError as e:
            raise DSLParseError(f"unknown effect: {eff_str}") from e
    # Default effect is EVENT_NEGATIVE for longevity, else NATURE.
    if eff is None:
        eff = (
            Effect.EVENT_NEGATIVE
            if life == LifeArea.LONGEVITY else Effect.NATURE
        )
    # Modifier default: TIMING for events, DESCRIPTION for nature.
    mod = (
        Modifier.TIMING if eff in (Effect.EVENT_NEGATIVE, Effect.EVENT_POSITIVE)
        else Modifier.NULL
    )

    # FROM BIRTH { ... }
    birth_body = _extract_block(src, r"FROM\s+BIRTH")
    if birth_body is None:
        raise DSLParseError("missing FROM BIRTH { ... } block")
    birth_kv = _parse_kv_block(birth_body)
    try:
        birth = BirthDetails(
            date=_date.fromisoformat(birth_kv["date"]),
            time=birth_kv.get("time"),
            tz=birth_kv.get("tz", "UTC"),
            lat=float(birth_kv["lat"]),
            lon=float(birth_kv["lon"]),
            time_accuracy=birth_kv.get("time_accuracy", "exact"),
        )
    except (KeyError, ValueError) as e:
        raise DSLParseError(f"invalid BIRTH block: {e}") from e
    gender = birth_kv.get("gender")

    # USING { ... }
    cfg = ChartConfig()
    using_body = _extract_block(src, "USING")
    if using_body:
        kv = _parse_kv_block(using_body)
        if "vargas" in kv:
            cfg.vargas = list(kv["vargas"])
        if "dashas" in kv:
            cfg.dasha_systems = list(kv["dashas"])
        if "ayanamsa" in kv:
            cfg.ayanamsa = kv["ayanamsa"]
        if "house_system" in kv:
            cfg.house_system = kv["house_system"]

    # SCHOOLS [a,b,c]
    schools_m = re.search(
        r"SCHOOLS\s*\[([^\]]+)\]", src, re.IGNORECASE,
    )
    if schools_m:
        schools = [
            School(s.strip())
            for s in schools_m.group(1).split(",")
            if s.strip()
        ]
    else:
        schools = [School.PARASHARI]

    # WINDOW date TO date
    window = None
    w_m = re.search(
        r"WINDOW\s+(\d{4}-\d{2}-\d{2})\s+TO\s+(\d{4}-\d{2}-\d{2})",
        src, re.IGNORECASE,
    )
    if w_m:
        window = (
            _date.fromisoformat(w_m.group(1)),
            _date.fromisoformat(w_m.group(2)),
        )

    # WITH { ... }
    opts = {}
    opts_body = _extract_block(src, "WITH")
    if opts_body:
        opts = _parse_kv_block(opts_body)

    # EXPLAIN flag
    explain = bool(re.search(r"\bEXPLAIN\b", src, re.IGNORECASE))

    return FocusQuery(
        relationship=rel,
        life_area=life,
        effect=eff,
        modifier=mod,
        window=window,
        birth=birth,
        config=cfg,
        schools=schools,
        min_confidence=float(opts.get("min_confidence", 0.55)),
        require_confluence=int(opts.get("require_confluence", 1)),
        explain=explain,
        gender=gender,
    )
