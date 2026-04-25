"""Phase 2 e2e: father-death TIMING query with Parashari + KP.

Exercises both-schools path and confirms:
    - Both extractors run without error
    - KP chart fields populated
    - Cross-school clusters formed
    - At least some windows show 2-school confluence
    - Explainer renders per-school confidences + contradictions if any

Run:
    python -u -m astroql.tests.integration.test_phase2_e2e
"""
from __future__ import annotations

import sys
from datetime import date

from astroql.chart import ChartComputer
from astroql.engine import RuleEngine
from astroql.explainer import Explainer
from astroql.features import KPFeatureExtractor, ParashariFeatureExtractor
from astroql.resolver import FocusResolver
from astroql.resolver_engine import Aggregator
from astroql.rules import StructuredRuleLibrary
from astroql.schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea,
    Modifier, Relationship, School,
)


def main() -> int:
    birth = BirthDetails(
        date=date(1965, 3, 15), time="14:20:00",
        tz="Asia/Kolkata", lat=28.6139, lon=77.2090,
    )
    q = FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI, School.KP],
        gender="M",
        min_confidence=0.80,
        school_weights={School.PARASHARI: 0.6, School.KP: 0.4},
    )

    resolved = FocusResolver().resolve(q)
    chart = ChartComputer().compute(
        birth, q.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
        need_kp=True,
    )

    # KP chart populated
    assert chart.kp_cuspal_sublords, "KP cuspal sublords missing"
    assert chart.kp_significators, "KP significators missing"
    assert chart.kp_planet_details, "KP planet details missing"
    assert chart.kp_cusp_details, "KP cusp details missing"
    print(f"[chart] KP cusps computed: {len(chart.kp_cusps)} cusps, "
          f"CSL for target direct 9H = {chart.kp_cuspal_sublords[9]}, "
          f"CSL for target rotated 4H = {chart.kp_cuspal_sublords[4]}")

    lib = StructuredRuleLibrary()
    engine = RuleEngine()

    fb_par = ParashariFeatureExtractor().extract(chart, resolved)
    rules_par = lib.load_rules(School.PARASHARI, resolved)
    fired_par = engine.apply(fb_par, rules_par)

    fb_kp = KPFeatureExtractor().extract(chart, resolved)
    rules_kp = lib.load_rules(School.KP, resolved)
    fired_kp = engine.apply(fb_kp, rules_kp)

    print(f"[rules]  parashari: {len(rules_par)} loaded, {len(fired_par)} fired")
    print(f"[rules]  kp:        {len(rules_kp)} loaded, {len(fired_kp)} fired")
    assert fired_par, "Parashari fired nothing"
    assert fired_kp, "KP fired nothing"

    result = Aggregator().aggregate(
        q, resolved,
        {School.PARASHARI: fired_par, School.KP: fired_kp},
        max_windows=8,
    )

    # Structural invariants
    assert result.windows, "expected at least one window"
    schools_per_w = [len(w.confidence_per_school) for w in result.windows]
    print(f"[agg] top-8 windows — schools per window: {schools_per_w}")
    # Assert at least one window is 2-school confluence
    assert max(schools_per_w) >= 2, \
        "no cross-school confluence window in top-8"

    # Confidences sorted.
    confs = [w.aggregate_confidence for w in result.windows]
    # Primary sort key is school count desc, then confidence desc — assert
    # within each school-count group confidences are non-increasing.
    grouped_confs = {}
    for w in result.windows:
        k = len(w.confidence_per_school)
        grouped_confs.setdefault(k, []).append(w.aggregate_confidence)
    for k, cs in grouped_confs.items():
        assert cs == sorted(cs, reverse=True), \
            f"confidences not sorted within school-count={k} group"

    print("\n" + Explainer().narrate(result))
    print("\n[OK] Phase 2 e2e passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
