"""Manually inspect specific charts: print full death-window feature
dict so we can read off what's astrologically distinctive and write
rules for it.

Usage:
    python -u -m astroql.discovery.inspect_charts \
        --dataset ml/father_passing_date_v2_clean_clean2.json \
        --names "John Howard" "Dennis Hopper" "Deborah Kerr"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backtrace import backtrace_chart


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--names", nargs="+", required=True,
        help="exact chart names from the dataset",
    )
    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    by_name = {r.get("name", ""): r for r in data}

    for name in args.names:
        rec = by_name.get(name)
        if rec is None:
            print(f"\n### {name}: NOT FOUND in dataset")
            continue
        print(f"\n{'=' * 72}\n### {name}\n{'=' * 72}")
        print(f"birth: {rec.get('birth_date')} {rec.get('birth_time')} "
              f"({rec.get('lat')}, {rec.get('lon')}) tz={rec.get('tz')}")
        print(f"father_death: {rec.get('father_death_date')}")

        bt = backtrace_chart(rec)
        if bt is None or bt.death_window is None:
            print("  -> no usable backtrace (chart compute failed or "
                  "death window not found)")
            continue

        dw = bt.death_window
        print(f"\nage at death: {bt.age_at_death_years:.2f} yr")
        print(f"death window: level={dw.level} "
              f"start={dw.start.date()} end={dw.end.date()} "
              f"({dw.window_duration_days}d, "
              f"short={dw.window_short}, long={dw.window_long}, "
              f"age_band={dw.age_band})")
        print(f"chain: MD={dw.md} ({dw.md_dignity}, h{dw.md_house})  "
              f"AD={dw.ad} ({dw.ad_dignity}, h{dw.ad_house})  "
              f"PAD={dw.pad} ({dw.pad_dignity}, h{dw.pad_house})")
        print(f"chain_strength={dw.chain_strength}  "
              f"n_distinct_roles={dw.n_distinct_roles}  "
              f"roles_present={dw.roles_present}")
        print(f"per-level roles: md={dw.md_roles}  ad={dw.ad_roles}  "
              f"pad={dw.pad_roles}")
        print(f"nakshatra-lord chain: md_nl={dw.md_nak_lord} "
              f"({dw.md_nak_lord_role})  "
              f"ad_nl={dw.ad_nak_lord} ({dw.ad_nak_lord_role})  "
              f"pad_nl={dw.pad_nak_lord} ({dw.pad_nak_lord_role})")
        print(f"karaka in chain: saturn={dw.saturn_in_chain} "
              f"sun={dw.sun_in_chain} jupiter={dw.jupiter_in_chain} "
              f"mars={dw.mars_in_chain} nodes={dw.rahu_or_ketu_in_chain}")
        print(f"natal placements: chain_lord_in_8h={dw.chain_lord_in_8h} "
              f"in_12h={dw.chain_lord_in_12h} "
              f"in_dusthana={dw.chain_lord_in_dusthana} "
              f"in_8h_from_target={dw.chain_lord_in_8h_from_target}")
        print(f"chain composition: "
              f"n_malefics={dw.n_malefics_in_chain} "
              f"n_benefics={dw.n_benefics_in_chain}  "
              f"md=pad? {dw.md_pad_same_planet}  md=ad? {dw.md_ad_same_planet}")
        print(f"chain dignity: deb_count={dw.n_debilitated_in_chain} "
              f"comb_count={dw.n_combust_in_chain}")
        print(f"chain geometry: pad_house_from_md={dw.pad_house_from_md_house}"
              f" pad_in_dust_from_md={dw.pad_in_dusthana_from_md}")
        print(f"transits at midpoint: "
              f"saturn_target={dw.saturn_transit_target}  "
              f"jupiter_target={dw.jupiter_transit_target}  "
              f"saturn_over_natal_sun={dw.saturn_over_sun_at_mid}  "
              f"sat_or_jup={dw.saturn_or_jupiter_transit}  "
              f"sat_and_jup={dw.saturn_and_jupiter_transit}")

        # Find competitor windows that *outscore* the death window: list any
        # window with strictly more roles, or full chain when death is partial,
        # etc. Lightweight competitor analysis since we don't run the engine.
        print(f"\ncomparator windows (same chart) sorted by 'looks more "
              f"like death':")
        scored = []
        for w in bt.all_windows:
            if w is dw:
                continue
            score = (
                w.n_distinct_roles * 2
                + (5 if w.chain_strength == "full" else
                   2 if w.chain_strength == "partial" else 0)
                + (3 if w.saturn_transit_target else 0)
                + w.n_malefics_in_chain
                + (2 if w.chain_lord_in_8h_from_target else 0)
            )
            scored.append((score, w))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, w in scored[:5]:
            print(
                f"  score={s:>3} {w.start.date()}..{w.end.date()} "
                f"{w.md}/{w.ad}/{w.pad}  "
                f"chain={w.chain_strength}  n_roles={w.n_distinct_roles}  "
                f"sat_tr={w.saturn_transit_target}  "
                f"8h_tgt={w.chain_lord_in_8h_from_target}"
            )

        # Heuristic: where would death window rank by this same heuristic?
        my_score = (
            dw.n_distinct_roles * 2
            + (5 if dw.chain_strength == "full" else
               2 if dw.chain_strength == "partial" else 0)
            + (3 if dw.saturn_transit_target else 0)
            + dw.n_malefics_in_chain
            + (2 if dw.chain_lord_in_8h_from_target else 0)
        )
        n_higher = sum(1 for s, _ in scored if s > my_score)
        print(f"  death_window heuristic_score={my_score} "
              f"({n_higher} other windows score higher)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
