"""FocusResolver — FocusQuery → ResolvedFocus (spec §6.2).

Pure derivation: house rotation, karaka lookup, varga selection,
query-type inference. No external calls.
"""
from __future__ import annotations

from ..schemas.enums import Effect, LifeArea, Modifier, QueryType, School
from ..schemas.focus import FocusQuery, ResolvedFocus
from .tables import (
    DOMAIN_HOUSE,
    DOMAIN_KARAKA,
    DOMAIN_VARGA,
    RELATION_HOUSE,
    RELATION_JAIMINI_KARAKA,
    RELATION_KARAKA,
    RELATION_VARGA,
)


class ResolverError(ValueError):
    pass


def _rotate(rel_house: int, dom_house: int) -> int:
    """bhavat bhavam: house count from the relationship's house (1-indexed)."""
    return ((rel_house - 1) + (dom_house - 1)) % 12 + 1


def _query_type(effect: Effect, modifier: Modifier) -> QueryType:
    """Spec §2.6 mapping."""
    if effect in (Effect.EVENT_POSITIVE, Effect.EVENT_NEGATIVE):
        if modifier == Modifier.TIMING:
            return QueryType.TIMING
        if modifier == Modifier.PROBABILITY:
            return QueryType.PROBABILITY
        return QueryType.PROBABILITY
    if effect == Effect.NATURE:
        return QueryType.DESCRIPTION
    if effect == Effect.MAGNITUDE:
        return QueryType.MAGNITUDE
    if effect == Effect.EXISTENCE:
        return QueryType.YES_NO
    raise ResolverError(f"cannot derive query_type from {effect=} {modifier=}")


def _relation_karakas(relationship_val: str, gender: str | None) -> list[str]:
    """Spec §2.3. For spouse, filter by gender (male→Venus, female→Jupiter).

    If `gender` is missing for a spouse query, both karakas are returned
    AND a warning is logged. Caller can inspect the resolved focus to
    detect this case (len > 1 for spouse).
    """
    raw = RELATION_KARAKA.get(relationship_val, []).copy()
    if relationship_val == "spouse":
        if gender:
            g = gender.upper()
            if g == "M":
                return [p for p in raw if p == "Venus"] or raw
            if g == "F":
                return [p for p in raw if p == "Jupiter"] or raw
        else:
            import warnings
            warnings.warn(
                "Spouse query without gender — using both Venus + Jupiter "
                "as karakas. Pass gender='M' or 'F' on FocusQuery for "
                "single-karaka analysis.",
                stacklevel=3,
            )
    return raw


def _dashas(query_type: QueryType, schools: list[School]) -> list[str]:
    """Spec §6.2. Vimshottari for timing/probability; Chara if Jaimini in schools."""
    if query_type == QueryType.DESCRIPTION:
        return []
    if query_type not in (QueryType.TIMING, QueryType.PROBABILITY, QueryType.YES_NO):
        return []
    dashas = ["vimshottari"]
    if School.JAIMINI in schools:
        dashas.append("chara")
    return dashas


def _vargas(life_area: LifeArea, relationship_val: str) -> list[str]:
    """Union of domain + relationship-specific vargas, D1 always included."""
    vargas = list(DOMAIN_VARGA.get(life_area.value, ["D1"]))
    for v in RELATION_VARGA.get(relationship_val, []):
        if v not in vargas:
            vargas.append(v)
    if "D1" not in vargas:
        vargas.insert(0, "D1")
    return vargas


class FocusResolver:
    """Resolves a FocusQuery into a ResolvedFocus (houses, karakas, vargas,
    dashas, query_type).

    Extensible: adding a new Relationship or LifeArea only requires a row
    in the lookup tables; the resolver logic is generic over all enum values.
    """

    def resolve(self, query: FocusQuery) -> ResolvedFocus:
        rel_val = query.relationship.value
        life_val = query.life_area.value

        if rel_val not in RELATION_HOUSE:
            raise ResolverError(f"unknown relationship: {rel_val}")
        if life_val not in DOMAIN_HOUSE:
            raise ResolverError(f"unknown life_area: {life_val}")

        rel_house = RELATION_HOUSE[rel_val]
        dom_house = DOMAIN_HOUSE[life_val]

        direct_house = rel_house
        # Spec §6.2: for NATURE, no rotation — target == relationship's own house.
        if query.life_area == LifeArea.NATURE:
            rotated_house = rel_house
        else:
            rotated_house = _rotate(rel_house, dom_house)

        # Collect the houses the engine should examine. Dedup preserving order.
        relevant: list[int] = []
        for h in (rotated_house, direct_house):
            if h not in relevant:
                relevant.append(h)

        rel_karakas = _relation_karakas(rel_val, query.gender)
        dom_karakas = list(DOMAIN_KARAKA.get(life_val, []))
        jaimini = list(RELATION_JAIMINI_KARAKA.get(rel_val, []))

        qt = _query_type(query.effect, query.modifier)
        vargas = _vargas(query.life_area, rel_val)
        dashas = _dashas(qt, query.schools)
        need_transits = qt in (QueryType.TIMING, QueryType.PROBABILITY)

        return ResolvedFocus(
            query=query,
            target_house_rotated=rotated_house,
            target_house_direct=direct_house,
            relevant_houses=relevant,
            relation_karakas=rel_karakas,
            domain_karakas=dom_karakas,
            jaimini_karakas=jaimini,
            vargas_required=vargas,
            dashas_required=dashas,
            need_transits=need_transits,
            query_type=qt,
        )
