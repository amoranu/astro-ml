"""Deep-inspect a chart around the father-death date.

Print the death window plus every window within +/-N months of it, side by side,
so we can see what astrologically distinguishes the death window from its
TEMPORAL neighbors (not from arbitrary windows elsewhere in the lifetime).

This is more useful than chart-wide comparator windows because:
- the actual death is in a narrow time band
- competitor windows from different decades are usually in a different MD,
  so the comparison is a chain-vs-chain comparison rather than a
  fine-grained "what made this exact moment the one"

Usage:
    python -u -m astroql.discovery.inspect_around_death \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --names "Brendan Behan" "Michael Chaplin" \\
        --months 18
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .backtrace import backtrace_chart


def _fmt(w) -> str:
    return (
        f"L={w.level} {w.start.date()}..{w.end.date()} "
        f"({w.window_duration_days:>4}d) "
        f"{w.md:<8}/{w.ad:<8}/{(w.pad or '-'):<8} "
        f"chain={w.chain_strength:<7} n={w.n_distinct_roles} "
        f"roles={','.join(w.roles_present) or '-':<60}"
    )


def _flag_str(w) -> str:
    flags = []
    if w.window_short:
        flags.append("SHORT")
    if w.md_pad_same_planet:
        flags.append("md=pad")
    if w.md_ad_same_planet:
        flags.append("md=ad")
    if w.saturn_transit_target:
        flags.append("sat-tr")
    if w.jupiter_transit_target:
        flags.append("jup-tr")
    if w.saturn_over_sun_at_mid:
        flags.append("sat-over-sun")
    if w.chain_lord_in_8h:
        flags.append("chain-8h")
    if w.chain_lord_in_12h:
        flags.append("chain-12h")
    if w.chain_lord_in_dusthana:
        flags.append("chain-dust")
    if w.chain_lord_in_8h_from_target:
        flags.append("chain-8h-tgt")
    if w.n_debilitated_in_chain:
        flags.append(f"deb={w.n_debilitated_in_chain}")
    if w.n_combust_in_chain:
        flags.append(f"comb={w.n_combust_in_chain}")
    nak_roles = set()
    for r in (w.md_nak_lord_role, w.ad_nak_lord_role, w.pad_nak_lord_role):
        nak_roles.update(r)
    if nak_roles:
        flags.append(f"nak-roles={len(nak_roles)}")
    return " ".join(flags) or "-"


def _diff(a, b, label):
    """Return diff line for one feature, only if different."""
    av = getattr(a, label, None)
    bv = getattr(b, label, None)
    if av != bv:
        return f"  {label:30}: death={av!s:<25} other={bv!s}"
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--months", type=int, default=18,
                   help="window radius around death date")
    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    by_name = {r.get("name", ""): r for r in data}

    for name in args.names:
        rec = by_name.get(name)
        if rec is None:
            print(f"\n### {name}: NOT FOUND")
            continue
        print(f"\n{'#' * 80}\n{name}\n{'#' * 80}")
        bt = backtrace_chart(rec)
        if bt is None or bt.death_window is None:
            print("  (no usable backtrace)")
            continue

        dw = bt.death_window
        print(f"father_death: {bt.death_date}  age={bt.age_at_death_years:.2f}")
        print(f"DEATH WINDOW: {_fmt(dw)}")
        print(f"  flags: {_flag_str(dw)}")

        # Temporal neighbors: windows within +/- months around death.
        radius = timedelta(days=30 * args.months)
        d_dt = datetime(
            bt.death_date.year, bt.death_date.month, bt.death_date.day,
        )
        neighbors = [
            w for w in bt.all_windows
            if w is not dw
            and (w.start - radius) <= d_dt <= (w.end + radius)
        ]
        neighbors.sort(key=lambda w: w.start)
        print(
            f"\nTEMPORAL NEIGHBORS within +/-{args.months} months "
            f"({len(neighbors)} windows):"
        )
        for w in neighbors:
            print(f"  {_fmt(w)}")
            print(f"    flags: {_flag_str(w)}")

        # Pairwise diffs between death window and each neighbor.
        print(f"\nDISCRIMINATOR ANALYSIS (death-window unique features vs each neighbor):")
        for w in neighbors:
            diffs = []
            for label in [
                "md", "ad", "pad", "level", "chain_strength",
                "n_distinct_roles", "window_short", "saturn_transit_target",
                "jupiter_transit_target", "saturn_over_sun_at_mid",
                "chain_lord_in_8h", "chain_lord_in_12h",
                "chain_lord_in_dusthana", "chain_lord_in_8h_from_target",
                "md_pad_same_planet", "md_ad_same_planet",
                "n_debilitated_in_chain", "n_combust_in_chain",
                "n_malefics_in_chain", "n_benefics_in_chain",
            ]:
                d = _diff(dw, w, label)
                if d:
                    diffs.append(d)
            if not diffs:
                print(f"  vs {w.start.date()}..{w.end.date()}: IDENTICAL on tracked fields")
                continue
            print(f"  vs {w.start.date()}..{w.end.date()} ({w.md}/{w.ad}/{w.pad}):")
            for d in diffs:
                print(d)

        # Roles diff (compare role sets).
        print(f"\nROLE-SET DIFFS (death vs each neighbor):")
        d_roles = set(dw.roles_present)
        for w in neighbors:
            n_roles = set(w.roles_present)
            only_d = d_roles - n_roles
            only_n = n_roles - d_roles
            if not only_d and not only_n:
                print(f"  vs {w.start.date()}..{w.end.date()}: same role set")
                continue
            print(f"  vs {w.start.date()}..{w.end.date()}:")
            if only_d:
                print(f"    death has only: {sorted(only_d)}")
            if only_n:
                print(f"    neighbor has only: {sorted(only_n)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
