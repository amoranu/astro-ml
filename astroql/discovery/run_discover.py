"""CLI: discover rules from a train slice of a death-date dataset.

Usage:
    python -u -m astroql.discovery.run_discover \
        --dataset ml/father_passing_date_v2_clean_clean2.json \
        --train-end 200 --test-start 200 --test-end 250 \
        --out-report ml/discovery_report.json \
        --out-yaml astroql/rules/parashari/longevity_discovered.yaml

Splits the dataset by index (deterministic). Train slice runs the
backtrace + aggregation pipeline. Test slice is left untouched here —
benchmark on it separately via astroql.benchmark.retrodiction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import yaml

from .aggregate import (
    aggregate_lifts, default_pattern_set, render_rule_candidates,
    discover_compounds, render_compound_rules,
)
from .backtrace import DeathBacktrace, backtrace_chart


def _load_data(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-end", type=int, default=200,
                   help="train slice = data[:train_end]")
    p.add_argument("--test-start", type=int, default=None,
                   help="test slice start (default = train_end)")
    p.add_argument("--test-end", type=int, default=None,
                   help="test slice end (default = train_end + 50)")
    p.add_argument("--out-report", default=None,
                   help="JSON: full lift table for inspection")
    p.add_argument("--out-yaml", default=None,
                   help="YAML: generalist + specialist discovered rules")
    p.add_argument("--top-print", type=int, default=30,
                   help="print top-N lifts to stdout")
    args = p.parse_args()

    data = _load_data(Path(args.dataset))
    train = data[:args.train_end]
    print(
        f"Loaded {len(data)} records; train=0..{args.train_end} "
        f"({len(train)} charts)"
    )

    backtraces: List[DeathBacktrace] = []
    n_unusable = 0
    for i, rec in enumerate(train, 1):
        bt = backtrace_chart(rec)
        if bt is None:
            n_unusable += 1
            print(f"  [{i}/{len(train)}] {rec.get('name', '?')}: skip")
            continue
        if bt.death_window is None:
            print(
                f"  [{i}/{len(train)}] {bt.name}: NO_WINDOW_FOUND "
                f"(death={bt.death_date}, windows={len(bt.all_windows)})"
            )
            n_unusable += 1
            continue
        dw = bt.death_window
        print(
            f"  [{i}/{len(train)}] {bt.name}: "
            f"age={bt.age_at_death_years:.1f} "
            f"chain={dw.chain_strength} roles={dw.n_distinct_roles} "
            f"md={dw.md} ad={dw.ad} pad={dw.pad} "
            f"sat_tr={dw.saturn_transit_target} "
            f"jup_tr={dw.jupiter_transit_target}"
        )
        backtraces.append(bt)

    n_valid = sum(1 for b in backtraces if b.death_window is not None)
    print(f"\nDiscovery: {n_valid} usable charts "
          f"({n_unusable} skipped)")

    lifts = aggregate_lifts(backtraces, default_pattern_set())
    # Sort by support desc, then cross_precision.
    lifts.sort(key=lambda L: (L.support, L.cross_precision), reverse=True)

    print("\n===== TOP PATTERNS BY SUPPORT × PRECISION =====")
    print(
        f"{'pattern':<55} {'support':>8} {'fired':>6} {'prec':>6} "
        f"{'spec':>6}"
    )
    for lift in lifts[:args.top_print]:
        print(
            f"{lift.pattern_key:<55} "
            f"{lift.support:>8} {lift.n_charts_fired:>6} "
            f"{lift.cross_precision:>6.2f} {lift.mean_specificity:>6.3f}"
        )

    # Specialist sort: precision × specificity, with support floor.
    specialists = [
        L for L in lifts
        if L.support >= 3 and L.cross_precision >= 0.5
        and L.mean_specificity >= 0.05
    ]
    specialists.sort(
        key=lambda L: L.cross_precision * L.mean_specificity, reverse=True,
    )
    print("\n===== TOP SPECIALIST PATTERNS (rare-but-precise) =====")
    print(
        f"{'pattern':<55} {'support':>8} {'prec':>6} {'spec':>6} "
        f"{'p*s':>7}"
    )
    for lift in specialists[:args.top_print]:
        print(
            f"{lift.pattern_key:<55} "
            f"{lift.support:>8} {lift.cross_precision:>6.2f} "
            f"{lift.mean_specificity:>6.3f} "
            f"{lift.cross_precision * lift.mean_specificity:>7.3f}"
        )

    if args.out_report:
        out = {
            "n_charts_train": len(train),
            "n_usable": n_valid,
            "n_skipped": n_unusable,
            "lifts": [
                {
                    "pattern_key": L.pattern_key,
                    "description": L.description,
                    "support": L.support,
                    "n_charts_fired": L.n_charts_fired,
                    "n_charts_total": L.n_charts_total,
                    "cross_precision": L.cross_precision,
                    "mean_specificity": L.mean_specificity,
                    "coverage": L.coverage,
                }
                for L in lifts
            ],
        }
        Path(args.out_report).write_text(json.dumps(out, indent=2))
        print(f"\nWrote lift report -> {args.out_report}")

    # ── Compound discovery (read more around each broad rule) ──────
    print("\n===== COMPOUND DISCOVERY =====")
    print("(broad pattern AND amplifier feature, ranked by precision uplift)")
    compounds = discover_compounds(backtraces, min_compound_support=5)
    compounds.sort(
        key=lambda c: (c.precision_uplift, c.compound_support), reverse=True,
    )
    print(
        f"{'broad':<35} {'amp':<35} {'sup':>4} {'fired':>5} "
        f"{'prec':>6} {'broad_p':>7} {'uplift':>7} {'spec':>6}"
    )
    for c in compounds[:args.top_print]:
        print(
            f"{c.broad_key:<35} {c.amplifier_key:<35} "
            f"{c.compound_support:>4} {c.compound_n_fired:>5} "
            f"{c.compound_precision:>6.2f} {c.broad_precision:>7.2f} "
            f"{c.precision_uplift:>+7.2f} {c.compound_specificity:>6.3f}"
        )

    if args.out_yaml:
        generals, specs = render_rule_candidates(lifts)
        compound_rules = render_compound_rules(compounds, min_uplift=0.15)
        for r in compound_rules:
            r["tags"] = list(r.get("tags", [])) + ["bucket_compound"]
        # Tag bucket in rule_id so a human reviewer can filter.
        for r in generals:
            r["tags"] = list(r.get("tags", [])) + ["bucket_general"]
        for r in specs:
            r["tags"] = list(r.get("tags", [])) + ["bucket_specialist"]
        all_rules = generals + specs + compound_rules
        Path(args.out_yaml).write_text(
            "# AUTO-GENERATED by astroql.discovery.run_discover.\n"
            "# Review before promoting into longevity.yaml.\n"
            "# Buckets:\n"
            "#   bucket_general    -- high support, broad antecedent\n"
            "#   bucket_specialist -- rare but high-precision\n"
            "#   bucket_compound   -- broad pattern AND narrowing amplifier\n"
            "# Compound rules add specificity to the broad rules without\n"
            "# weakening them; engine fires both, with compound's higher\n"
            "# strength dominating the death window.\n"
            + yaml.safe_dump(all_rules, sort_keys=False)
        )
        print(
            f"Wrote {len(generals)} generalist + {len(specs)} specialist "
            f"+ {len(compound_rules)} compound rule candidates "
            f"-> {args.out_yaml}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
