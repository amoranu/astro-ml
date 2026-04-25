"""Phase 0 smoke test: resolve a father-death query + compute its chart.

Exercises FocusResolver + ChartComputer end-to-end. No rules yet.
Also verifies a second focus area (spouse nature) to check extensibility.

Run:
    python -u -m astroql.tests.integration.test_phase0_smoke
"""
from __future__ import annotations

import sys
from datetime import date

from astroql.chart import ChartComputer
from astroql.resolver import FocusResolver
from astroql.schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea,
    Modifier, QueryType, Relationship, School,
)


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def run_father_death() -> None:
    _hr("Phase 0 smoke: father-death timing query")

    birth = BirthDetails(
        date=date(1965, 3, 15), time="14:20:00",
        tz="Asia/Kolkata", lat=28.6139, lon=77.2090,
        time_accuracy="exact",
    )

    query = FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI],   # Phase 1 scope
        gender="M",
    )

    resolver = FocusResolver()
    resolved = resolver.resolve(query)
    print("[resolve]")
    print(f"  target_rotated  = {resolved.target_house_rotated}")
    print(f"  target_direct   = {resolved.target_house_direct}")
    print(f"  relevant_houses = {resolved.relevant_houses}")
    print(f"  relation_karaka = {resolved.relation_karakas}")
    print(f"  domain_karakas  = {resolved.domain_karakas}")
    print(f"  vargas_required = {resolved.vargas_required}")
    print(f"  dashas_required = {resolved.dashas_required}")
    print(f"  query_type      = {resolved.query_type.value}")
    assert resolved.target_house_rotated == 4
    assert resolved.target_house_direct == 9
    assert resolved.query_type == QueryType.TIMING

    computer = ChartComputer()
    chart = computer.compute(
        birth, query.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_jaimini=False, need_kp=False,
    )
    print("\n[chart]")

    d1 = chart.vargas["D1"]
    lagna = d1.planet_positions["Lagna"]
    print(f"  Lagna: {lagna.sign} ({lagna.longitude:.2f} deg)")

    # Father-specific facts: 9H (direct) + 4H (rotated) features
    from astroql.resolver.tables import RELATION_HOUSE
    print("  9H (father direct) sign = " + d1.house_signs[9])
    print("  4H (father's 8H rotated) sign = " + d1.house_signs[4])

    # List planets in these houses
    occ_9 = [p for p, pp in d1.planet_positions.items()
             if p != "Lagna" and pp.house == 9]
    occ_4 = [p for p, pp in d1.planet_positions.items()
             if p != "Lagna" and pp.house == 4]
    print(f"  9H occupants: {occ_9 or '(none)'}")
    print(f"  4H occupants: {occ_4 or '(none)'}")

    # Sun (father's karaka)
    sun = d1.planet_positions["Sun"]
    print(f"  Sun: house={sun.house}  sign={sun.sign}  dignity={sun.dignity}")

    # Saturn (longevity karaka)
    sat = d1.planet_positions["Saturn"]
    print(f"  Saturn: house={sat.house}  sign={sat.sign}  dignity={sat.dignity}  retrograde={sat.retrograde}")

    # Dasha sanity
    assert chart.vimshottari is not None
    print(f"\n  Vimshottari MDs computed: {len(chart.vimshottari.children)}")
    for md in chart.vimshottari.children[:4]:
        print(f"    {md.lord:8s} {md.start.date()} -> {md.end.date()}")

    # Varga sanity
    print(f"\n  Vargas computed: {list(chart.vargas.keys())}")
    assert {"D1", "D8", "D12", "D30"} <= set(chart.vargas.keys())

    print("\n[OK] father-death Phase 0 smoke passed")


def run_spouse_nature() -> None:
    """Sanity check: same resolver handles a DESCRIPTION query with no dashas."""
    _hr("Phase 0 smoke: spouse-nature description query (extensibility check)")

    birth = BirthDetails(
        date=date(1990, 11, 22), time="08:45:00",
        tz="America/New_York", lat=40.7128, lon=-74.0060,
    )

    query = FocusQuery(
        relationship=Relationship.SPOUSE,
        life_area=LifeArea.NATURE,
        effect=Effect.NATURE,
        modifier=Modifier.DESCRIPTION,
        birth=birth,
        config=ChartConfig(vargas=["D1", "D9"], dasha_systems=[]),
        schools=[School.PARASHARI],
        gender="M",
    )

    resolver = FocusResolver()
    resolved = resolver.resolve(query)
    print(f"  target=7 (no rotation), karakas={resolved.relation_karakas}, "
          f"vargas={resolved.vargas_required}, dashas={resolved.dashas_required}")
    assert resolved.target_house_rotated == 7
    assert resolved.dashas_required == []
    assert resolved.query_type == QueryType.DESCRIPTION

    chart = ChartComputer().compute(
        birth, query.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
    )
    print(f"  D1 7H sign: {chart.vargas['D1'].house_signs[7]}")
    print(f"  D9 7H sign: {chart.vargas['D9'].house_signs[7]}")
    venus = chart.vargas["D1"].planet_positions["Venus"]
    print(f"  Venus (karaka): house={venus.house}  sign={venus.sign}  dignity={venus.dignity}")
    assert chart.vimshottari is None, "no dasha for DESCRIPTION"
    print("\n[OK] spouse-nature Phase 0 smoke passed")


def main() -> int:
    run_father_death()
    run_spouse_nature()
    print("\n[ALL] Phase 0 smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
