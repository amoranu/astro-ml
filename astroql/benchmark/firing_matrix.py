"""Build the rule-firing matrix across train charts.

For each chart:
  - Run the engine on FATHER/LONGEVITY focus
  - For every candidate window, record which rules fire and at what
    strength (after tier weighting)
  - Mark which window contains the true death date

Rows = candidate windows (across all charts, with chart_id + window index)
Cols = rule_ids
Values = tier-weighted strength when the rule fires on that window, else 0
Label = 1 if the window contains true death, 0 otherwise

This matrix is the input to:
  - Per-chart ablation (Phase B): drop one rule's column, recompute scores
  - Learned-weights aggregator (Phase C): logistic regression on (X, y)

Saves a compact .npz with X (sparse), y, chart_ids, window_starts, rule_ids.

Usage:
    python -u -m astroql.benchmark.firing_matrix \\
        --dataset ml/father_passing_date_v2_clean_clean2.json \\
        --start 0 --end 200 \\
        --out ml/firing_matrix_train.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..chart import ChartComputer
from ..engine import RuleEngine
from ..features import (
    JaiminiFeatureExtractor, KPFeatureExtractor, ParashariFeatureExtractor,
)
from ..resolver import FocusResolver
from ..rules import StructuredRuleLibrary
from ..schemas import (
    BirthDetails, ChartConfig, Effect, FocusQuery, LifeArea, Modifier,
    Relationship, School,
)


def _build_focus(rec: dict) -> Optional[FocusQuery]:
    try:
        bd = date.fromisoformat(rec["birth_date"])
    except Exception:
        return None
    bt = rec.get("birth_time") or "12:00"
    if len(bt) == 5:
        bt = bt + ":00"
    lat = rec.get("lat")
    lon = rec.get("lon")
    if lat is None or lon is None:
        return None
    birth = BirthDetails(
        date=bd, time=bt, tz=rec.get("tz") or "UTC",
        lat=float(lat), lon=float(lon),
    )
    return FocusQuery(
        relationship=Relationship.FATHER,
        life_area=LifeArea.LONGEVITY,
        effect=Effect.EVENT_NEGATIVE,
        modifier=Modifier.TIMING,
        birth=birth,
        config=ChartConfig(
            vargas=["D1", "D8", "D12", "D30"],
            dasha_systems=["vimshottari"],
        ),
        schools=[School.PARASHARI],
        gender=rec.get("gender") or "M",
        min_confidence=0.0,
    )


def _tier_mult(tier: int) -> float:
    return {1: 1.0, 2: 0.7, 3: 0.4}.get(tier, 0.7)


def collect_firing_matrix(
    records: List[dict],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Tuple[int, int]],
           List[str], List[float]]:
    """Returns (X, y, rule_ids, (chart_idx, win_idx) keys, chart_names,
    polarities)."""
    lib = StructuredRuleLibrary()
    engine = RuleEngine()

    # First pass: discover the rule_id universe (only Parashari for now).
    sample_rules = lib.all_rules(School.PARASHARI)
    rule_ids = sorted({r.rule_id for r in sample_rules})
    rule_index = {rid: i for i, rid in enumerate(rule_ids)}
    rule_polarity = {}
    rule_tier = {}
    for r in sample_rules:
        rule_polarity[r.rule_id] = r.consequent.get("polarity", "neutral")
        rule_tier[r.rule_id] = getattr(r, "priority_tier", 2)

    rows_X: List[np.ndarray] = []
    rows_y: List[int] = []
    keys: List[Tuple[int, int]] = []
    chart_names: List[str] = []

    for chart_idx, rec in enumerate(records):
        q = _build_focus(rec)
        if q is None:
            continue
        try:
            true_death = date.fromisoformat(rec["father_death_date"])
        except Exception:
            continue
        true_dt = datetime(
            true_death.year, true_death.month, true_death.day,
        )
        try:
            resolver = FocusResolver()
            resolved = resolver.resolve(q)
            chart = ChartComputer().compute(
                q.birth, q.config,
                vargas=resolved.vargas_required,
                dashas=resolved.dashas_required,
                need_kp=False, need_jaimini=False,
            )
            bundle = ParashariFeatureExtractor().extract(chart, resolved)
            rules = lib.load_rules(School.PARASHARI, resolved)
            fired = engine.apply(bundle, rules)
        except Exception as e:
            print(f"  [{chart_idx}] {rec.get('name')}: error {e}")
            continue

        # Collect timing rules per window (group by window key).
        # Static rules (window=None) duplicate across all windows in chart.
        timing_by_window: Dict[Tuple[datetime, datetime],
                               Dict[str, float]] = {}
        static: Dict[str, float] = {}
        for fr in fired:
            if fr.rule.rule_id not in rule_index:
                continue
            mult = _tier_mult(rule_tier.get(fr.rule.rule_id, 2))
            s = mult * float(fr.strength)
            # Negative polarity = positive death-evidence; positive
            # polarity = subtractive (we keep both as signed).
            if fr.polarity == "positive":
                s = -s
            if fr.window is None:
                # Static rule
                prev = static.get(fr.rule.rule_id, 0.0)
                if abs(s) > abs(prev):
                    static[fr.rule.rule_id] = s
            else:
                key = fr.window
                tw = timing_by_window.setdefault(key, {})
                prev = tw.get(fr.rule.rule_id, 0.0)
                if abs(s) > abs(prev):
                    tw[fr.rule.rule_id] = s

        # Each candidate window from the bundle becomes a row.
        for win_idx, cand in enumerate(bundle.dasha_candidates or []):
            try:
                w_start = datetime.fromisoformat(cand["start"])
                w_end = datetime.fromisoformat(cand["end"])
            except Exception:
                continue
            row = np.zeros(len(rule_ids), dtype=np.float32)
            # Static rules add to every row uniformly.
            for rid, s in static.items():
                row[rule_index[rid]] = s
            # Timing rules: look up by window key.
            tw = timing_by_window.get((w_start, w_end), {})
            for rid, s in tw.items():
                row[rule_index[rid]] = s
            label = 1 if (w_start <= true_dt <= w_end) else 0
            rows_X.append(row)
            rows_y.append(label)
            keys.append((chart_idx, win_idx))
            chart_names.append(rec.get("name", "?"))
        print(
            f"  [{chart_idx}/{len(records)}] {rec.get('name')}: "
            f"{len(bundle.dasha_candidates or [])} windows, "
            f"{sum(1 for k, w in zip(keys, rows_y) if w == 1 and k[0] == chart_idx)} death"
        )

    X = np.vstack(rows_X) if rows_X else np.zeros((0, len(rule_ids)))
    y = np.array(rows_y, dtype=np.int8)
    polarities = [rule_polarity.get(rid, "neutral") for rid in rule_ids]
    return X, y, rule_ids, keys, chart_names, polarities


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=200)
    p.add_argument("--out", required=True,
                   help="path to save .npz with X, y, rule_ids, keys")
    args = p.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)
    records = data[args.start:args.end]
    print(f"Loaded {len(records)} charts")

    X, y, rule_ids, keys, chart_names, polarities = collect_firing_matrix(
        records,
    )
    print(f"\nFiring matrix: X={X.shape} y={y.shape} "
          f"n_rules={len(rule_ids)}")
    print(f"  death windows: {int(y.sum())}")
    print(f"  total rows:    {len(y)}")
    np.savez_compressed(
        args.out,
        X=X, y=y,
        rule_ids=np.array(rule_ids),
        keys=np.array(keys),
        chart_names=np.array(chart_names),
        polarities=np.array(polarities),
    )
    print(f"Wrote firing matrix -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
