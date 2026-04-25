"""Phase 1 end-to-end test: father-death TIMING query, full pipeline.

Exercises parser (implicit — manual FocusQuery) -> resolver -> chart
-> features -> rules -> engine -> aggregator -> explainer.

Also runs the same pipeline against a mother-longevity query to show
the same rule library + extractor serve multiple relationships.

Run:
    python -u -m astroql.tests.integration.test_phase1_e2e
"""
from __future__ import annotations

import sys
from datetime import date

from astroql.chart import ChartComputer
from astroql.engine import RuleEngine
from astroql.explainer import Explainer
from astroql.features import ParashariFeatureExtractor
from astroql.resolver import FocusResolver
from astroql.resolver_engine import Aggregator
from astroql.rules import StructuredRuleLibrary
from astroql.schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea,
    Modifier, Relationship, School,
)


def run_one(query: FocusQuery, title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)

    resolver = FocusResolver()
    resolved = resolver.resolve(query)

    computer = ChartComputer()
    chart = computer.compute(
        query.birth, query.config,
        vargas=resolved.vargas_required,
        dashas=resolved.dashas_required,
    )
    fb = ParashariFeatureExtractor().extract(chart, resolved)
    lib = StructuredRuleLibrary()
    rules = lib.load_rules(School.PARASHARI, resolved)
    fired = RuleEngine().apply(fb, rules)

    print(f"resolved: target={resolved.target_house_rotated}/"
          f"{resolved.target_house_direct}, "
          f"karakas={resolved.relation_karakas + resolved.domain_karakas}, "
          f"vargas={resolved.vargas_required}")
    print(f"rules applicable: {len(rules)}   fired: {len(fired)}")

    result = Aggregator().aggregate(
        query, resolved, {School.PARASHARI: fired},
        max_windows=5,
    )
    print(Explainer().narrate(result))
    print()
    assert result.windows, "expected at least one window"
    # Windows must be sorted by aggregate confidence (desc).
    confs = [w.aggregate_confidence for w in result.windows]
    assert confs == sorted(confs, reverse=True)


def main() -> int:
    birth = BirthDetails(
        date=date(1965, 3, 15), time="14:20:00",
        tz="Asia/Kolkata", lat=28.6139, lon=77.2090,
    )
    cfg = ChartConfig(
        vargas=["D1", "D8", "D12", "D30"],
        dasha_systems=["vimshottari"],
    )

    # ── Father longevity ─────────────────────────────────────────
    father_q = FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth, config=cfg,
        schools=[School.PARASHARI], gender="M",
        min_confidence=0.70,
    )
    run_one(father_q, "Phase 1 e2e: father longevity (TIMING)")

    # ── Mother longevity (same rule library, different target) ──
    mother_q = FocusQuery(
        relationship=Relationship.MOTHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth, config=cfg,
        schools=[School.PARASHARI], gender="M",
        min_confidence=0.70,
    )
    run_one(mother_q, "Phase 1 e2e: mother longevity (same rules, different rotation)")

    print("[ALL] Phase 1 e2e passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
