"""30-day calendar window accuracy from Parashari time unit predictions.

Model predicts at dasha level (PD or SD). Accuracy is measured by mapping
the model's top-ranked period back to calendar time and checking if it
falls in the correct 30-day slot.

24 months = ~730 days = ~24 non-overlapping 30-day slots.
Random baseline = 1/24 ~ 4.2%.

For each chart:
1. Rank all dasha periods in the 24-month window using the ML model
2. Take the model's top-1 (or top-K) prediction
3. Map that period's midpoint to a 30-day calendar slot
4. Check if the death date falls in the same 30-day slot

Tests at both PD (depth=3) and SD (depth=4) granularity.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_30day_accuracy
"""

import json
import os
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'cal30'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

WINDOW_MONTHS = 24
SLOT_DAYS = 30


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


def _n(s):
    r = s.max() - s.min()
    return (s - s.min()) / r if r > 1e-10 else np.zeros_like(s)


def train_seed_avg(cols, df_tr, df_va, gcol, params, n_seeds=5):
    models = []
    for i in range(n_seeds):
        p = {**params, 'seed': 42 + i * 17}
        td = lgb.Dataset(df_tr[cols].values, label=df_tr['label'].values,
                         group=df_tr.groupby(gcol, sort=False).size().values)
        vd = lgb.Dataset(df_va[cols].values, label=df_va['label'].values,
                         group=df_va.groupby(gcol, sort=False).size().values,
                         reference=td)
        m = lgb.train(p, td, num_boost_round=1500, valid_sets=[vd],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        models.append(m)
    return models


def predict_avg(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def jd_to_slot(jd, window_start_jd):
    """Map a JD to a 30-day slot index (0-based) from window start."""
    return int((jd - window_start_jd) / SLOT_DAYS)


# ── Dataset builder that preserves JD info ──────────────────────────────

def build_dataset(records, name, start_index, target_depth,
                   n_augment=1):
    """Build features at a given dasha depth, preserving calendar JD info."""
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .astro_engine.multiref_dasha import compute_multi_reference
    from .features.dasha_window import construct_dasha_window, date_to_jd
    from .features.multiref_features import _maraka_set
    from .features.extended_maraka import (
        precompute_maraka_sets, classify_candidate_extended)
    from .features.d12_features import compute_d12_periods, d12_maraka_lords
    from .features.gochar_features import precompute_gochar_context
    from .features.hierarchy_features import precompute_hierarchy_context
    from .features.lever2_features import precompute_jup_bav
    from .features.yogini_features import precompute_yogini
    from .features.five_new_features import add_relative_features

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_d{target_depth}_aug{n_augment}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    from .run_dasha_depth import extract_features_for_candidate

    dn = {1: 'MD', 2: 'AD', 3: 'PD', 4: 'SD'}[target_depth]
    print(f"  Building {name} depth={target_depth} ({dn}) "
          f"({len(records)} charts x {n_augment} aug)...")

    compute_depth = max(target_depth, 3)
    all_rows = []
    n_ok = 0
    errors = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        base_idx = start_index + i
        try:
            chart, asc = compute_chart(
                rec['birth_date'], rec['birth_time'],
                rec['lat'], rec['lon'])
            birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])
            moon_long = chart['Moon']['longitude']
            death_jd = date_to_jd(rec['father_death_date'])

            all_depth = compute_full_dasha(
                moon_long, birth_jd, max_depth=compute_depth,
                collect_all_depths=True)

            depth1 = compute_full_dasha(moon_long, birth_jd, max_depth=1)
            depth2 = [p for p in all_depth if p['depth'] == 2]
            depth3 = [p for p in all_depth if p['depth'] == 3]

            if target_depth == 1:
                target_periods = depth1
            elif target_depth == 4:
                all_d4 = compute_full_dasha(
                    moon_long, birth_jd, max_depth=4,
                    collect_all_depths=True)
                target_periods = [p for p in all_d4 if p['depth'] == 4]
            else:
                target_periods = [p for p in all_depth
                                   if p['depth'] == target_depth]

            multi_ref = compute_multi_reference(chart, asc, birth_jd, 3)
            d12_periods = compute_d12_periods(chart, birth_jd, 3)
            d12_p3 = [p for p in d12_periods if p['depth'] == 3]
            d1_mk = _maraka_set(asc)
            pre = precompute_maraka_sets(chart, asc)
            gc_ctx = precompute_gochar_context(asc, chart)
            hi_ctx = precompute_hierarchy_context(asc)
            father_mk = hi_ctx['father_marakas']
            jup_bav = precompute_jup_bav(chart, asc)
            yogini_p3 = precompute_yogini(moon_long, birth_jd, father_mk)
            d12_mk = d12_maraka_lords(asc)

            for aug in range(n_augment):
                seed = base_idx * 100 + aug
                idx = base_idx * 100 + aug

                candidates, correct_idx, ws_jd, we_jd = construct_dasha_window(
                    rec['father_death_date'], target_periods,
                    window_months=WINDOW_MONTHS, seed=seed)
                if correct_idx is None:
                    continue

                # Pre-classify for sequence features
                for ci, cand in enumerate(candidates):
                    tier, dscore = classify_candidate_extended(
                        cand, chart, asc, pre)
                    cand['tier'] = tier
                    cand['danger_score'] = dscore

                # Death slot
                death_slot = jd_to_slot(death_jd, ws_jd)

                for ci, cand in enumerate(candidates):
                    f = extract_features_for_candidate(
                        cand, chart, asc, pre, multi_ref,
                        depth1, depth2, depth3,
                        d12_p3, d1_mk, gc_ctx, hi_ctx,
                        jup_bav, yogini_p3, d12_mk,
                        father_mk, candidates, ci)

                    # Calendar info for accuracy mapping
                    mid_jd = (cand['start_jd'] + cand['end_jd']) / 2
                    cand_slot = jd_to_slot(mid_jd, ws_jd)

                    row = {
                        'group_id': idx,
                        'cand_idx': ci,
                        'label': 1 if ci == correct_idx else 0,
                        'mid_jd': mid_jd,
                        'start_jd': cand['start_jd'],
                        'end_jd': cand['end_jd'],
                        'window_start_jd': ws_jd,
                        'death_jd': death_jd,
                        'cand_slot': cand_slot,
                        'death_slot': death_slot,
                        'slot_match': 1 if cand_slot == death_slot else 0,
                        **f,
                    }
                    all_rows.append(row)

            n_ok += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    [WARN] {base_idx}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(records)} ({(i+1)/elapsed:.0f}/s)")

    df = pd.DataFrame(all_rows)
    if len(df) > 0 and 'group_id' in df.columns:
        df = add_relative_features(df, group_col='group_id')
    df.to_parquet(cache_f)
    elapsed = time.time() - t0
    n_groups = df['group_id'].nunique() if len(df) > 0 and 'group_id' in df.columns else 0
    print(f"  {n_ok} OK, {errors} errors, {elapsed:.1f}s")
    print(f"  Rows: {len(df)}, Groups: {n_groups}")

    return df


def cal30_eval(df, scores, topk_values=(1, 2, 3, 5)):
    """Evaluate 30-day calendar accuracy from model scores.

    For each group:
    - Rank candidates by model score
    - Take top-K candidates
    - Map each candidate's midpoint to a 30-day slot
    - Check if ANY of the top-K slots match the death slot
    """
    df = df.copy()
    df['score'] = scores
    results = {k: 0 for k in topk_values}
    n = 0

    for _, grp in df.groupby('group_id', sort=False):
        death_slot = grp['death_slot'].iloc[0]
        ranked = grp.sort_values('score', ascending=False)
        n += 1

        for k in topk_values:
            top_slots = set(ranked.head(k)['cand_slot'].values)
            if death_slot in top_slots:
                results[k] += 1

    return {k: results[k] / n for k in topk_values}, n


def main():
    print("=" * 75)
    print("30-DAY CALENDAR ACCURACY FROM PARASHARI TIME UNITS")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    n_slots = math.ceil(WINDOW_MONTHS * 30.44 / SLOT_DAYS)
    print(f"  Window: {WINDOW_MONTHS} months = {n_slots} slots of {SLOT_DAYS} days")
    print(f"  Random baseline: 1/{n_slots} = {1/n_slots:.1%}")

    META_COLS = ['group_id', 'cand_idx', 'label', 'mid_jd', 'start_jd',
                 'end_jd', 'window_start_jd', 'death_jd', 'cand_slot',
                 'death_slot', 'slot_match', 'tier', 'danger_score',
                 'duration_days']

    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05, 'subsample': 0.8,
        'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 25,
        'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    for depth in [3, 4]:
        dn = {3: 'PD', 4: 'SD'}[depth]
        aug = 3 if depth == 3 else 1

        print(f"\n{'='*75}")
        print(f"DEPTH {depth}: {dn} -> 30-day calendar slots")
        print(f"{'='*75}")

        df_tr = build_dataset(train_recs, 'train', 0, depth, n_augment=aug)
        df_va = build_dataset(val_recs, 'val', len(train_recs) * 100,
                               depth, n_augment=1)

        if len(df_va) == 0 or 'group_id' not in df_va.columns:
            print("  SKIP: no data")
            continue

        # Filter to groups with >= 2 candidates
        gs = df_va.groupby('group_id').size()
        valid = gs[gs >= 2].index
        df_tr_f = df_tr[df_tr['group_id'].isin(
            df_tr.groupby('group_id').filter(
                lambda x: len(x) >= 2)['group_id'].unique())]
        df_va_f = df_va[df_va['group_id'].isin(valid)]

        n_va = df_va_f['group_id'].nunique()
        gs_f = df_va_f.groupby('group_id').size()

        feat_cols = [c for c in df_va_f.columns
                     if c not in META_COLS
                     and isinstance(df_va_f[c].iloc[0],
                                     (int, float, np.integer, np.floating))]
        feat_cols = [c for c in feat_cols
                     if c in df_tr_f.columns and c in df_va_f.columns]

        no_dur_cols = [c for c in feat_cols
                       if 'duration' not in c and 'dur_' not in c
                       and c not in ('seq_duration_days', 'seq_dur_vs_mean',
                                     'seq_dur_log', 'seq_danger_intensity',
                                     'duration_days')]

        print(f"  Val groups: {n_va}")
        print(f"  Candidates/group: mean={gs_f.mean():.1f}")
        print(f"  Features: {len(feat_cols)} (no-dur: {len(no_dur_cols)})")

        # Unique 30-day slots per group
        slots_per_group = df_va_f.groupby('group_id')['cand_slot'].nunique()
        print(f"  Unique slots/group: mean={slots_per_group.mean():.1f}, "
              f"median={slots_per_group.median():.0f}")

        # ── Random baseline (calendar slots) ────────────────────────
        # Random = pick a period uniformly, its slot might match
        # More precisely: for each group, fraction of candidates in death slot
        rnd_probs = []
        for gid, grp in df_va_f.groupby('group_id', sort=False):
            n_match = grp['slot_match'].sum()
            rnd_probs.append(n_match / len(grp))
        rnd_30d = np.mean(rnd_probs)

        # Simpler: 1/n_unique_slots per group
        rnd_slot_based = (1.0 / slots_per_group.values).mean()

        # ── Duration baseline ───────────────────────────────────────
        dur_scores = df_va_f['duration_days'].values if 'duration_days' in df_va_f.columns else np.zeros(len(df_va_f))
        dur_30d, _ = cal30_eval(df_va_f, dur_scores)

        # ── Model (full) ────────────────────────────────────────────
        print(f"  Training full model...")
        models = train_seed_avg(feat_cols, df_tr_f, df_va_f,
                                 'group_id', params)
        model_scores = predict_avg(models, df_va_f[feat_cols].values)
        model_30d, _ = cal30_eval(df_va_f, model_scores)

        # ── Model (no duration) ─────────────────────────────────────
        print(f"  Training no-duration model...")
        nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f,
                                    'group_id', params)
        nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
        nd_30d, _ = cal30_eval(df_va_f, nd_scores)

        # ── Also: "any top-K period overlaps death's 30-day slot" ───
        # More generous: does ANY part of the top-1 period overlap?
        overlap_hits = 0
        for gid, grp in df_va_f.groupby('group_id', sort=False):
            grp2 = grp.copy()
            grp2['score'] = model_scores[grp.index]
            top1 = grp2.sort_values('score', ascending=False).iloc[0]
            death_jd = top1['death_jd']
            # 30-day window: death_jd - 15 to death_jd + 15
            if (top1['start_jd'] <= death_jd + 15 and
                    top1['end_jd'] >= death_jd - 15):
                overlap_hits += 1
        overlap_pct = overlap_hits / n_va

        # ── Results ─────────────────────────────────────────────────
        print(f"\n  {dn} -> 30-Day Calendar Accuracy:")
        print(f"    {'Method':>25s} {'Slot-1':>7s} {'Slot-2':>7s} "
              f"{'Slot-3':>7s} {'Slot-5':>7s}")
        print(f"    {'-'*55}")
        print(f"    {'Random (period-based)':>25s} {rnd_30d:6.1%}")
        print(f"    {'Random (slot-based)':>25s} {rnd_slot_based:6.1%}")
        print(f"    {'Duration only':>25s} "
              f"{dur_30d[1]:6.1%}  {dur_30d[2]:6.1%}  "
              f"{dur_30d[3]:6.1%}  {dur_30d[5]:6.1%}")
        print(f"    {'Model (no duration)':>25s} "
              f"{nd_30d[1]:6.1%}  {nd_30d[2]:6.1%}  "
              f"{nd_30d[3]:6.1%}  {nd_30d[5]:6.1%}")
        print(f"    {'Full model':>25s} "
              f"{model_30d[1]:6.1%}  {model_30d[2]:6.1%}  "
              f"{model_30d[3]:6.1%}  {model_30d[5]:6.1%}")
        print(f"\n    Top-1 overlaps +/-15d:  {overlap_pct:.1%}")
        print(f"    Lift over random:       "
              f"{model_30d[1]/max(rnd_30d, 0.001):.1f}x (full), "
              f"{nd_30d[1]/max(rnd_30d, 0.001):.1f}x (astro-only)")

    # ── Grand summary ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY: 30-Day Calendar Accuracy from Parashari Time Units")
    print("=" * 75)
    print(f"  Window: {WINDOW_MONTHS} months = {n_slots} x {SLOT_DAYS}-day slots")
    print(f"  Pure random (slot): ~{1/n_slots:.1%}")
    print(f"  Model ranks dasha periods, top-1's midpoint mapped to 30-day slot")
    print(f"  'Slot-K' = correct slot appears in the K unique slots")
    print(f"             from the model's top-K ranked periods")
    print("=" * 75)


if __name__ == '__main__':
    main()
