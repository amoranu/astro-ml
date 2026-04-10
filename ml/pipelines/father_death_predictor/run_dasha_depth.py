"""Dasha-native multi-depth prediction: MD -> AD -> PD -> SD.

Predicts at EVERY Parashari time unit:
  Depth 1: Mahadasha      (~7-20 years each, ~2-4 in 24mo window)
  Depth 2: Antardasha     (~6mo-2yr each, ~5-15 in 24mo window)
  Depth 3: Pratyantardasha (~3-90 days each, ~13-80 in 24mo window)
  Depth 4: Sookshma       (~1-10 days each, ~100-500 in 24mo window)

At each depth: train LambdaRank, measure top-1/3/5 accuracy, compare to
random and duration baselines.

Also: hierarchical cascade — pick best MD, then best AD within, then best PD, etc.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_dasha_depth
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'dasha_depth'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

DEPTH_NAMES = {1: 'MD', 2: 'AD', 3: 'PD', 4: 'SD'}


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


def eval_topk(df, scores, k_values=(1, 3, 5), gcol='group_id'):
    df = df.copy()
    df['score'] = scores
    results = {k: 0 for k in k_values}
    n = 0
    for _, grp in df.groupby(gcol, sort=False):
        ranked = grp.sort_values('score', ascending=False)
        cr = ranked['label'].values.tolist()
        if 1 not in cr:
            continue
        rank = cr.index(1) + 1
        n += 1
        for k in k_values:
            if rank <= k:
                results[k] += 1
    return {f'top_{k}': results[k] / n for k in k_values}, n


# ── Feature extraction (works at any depth) ────────────────────────────

def extract_features_for_candidate(cand, chart, asc, pre, multi_ref,
                                    depth1, depth2, depth3,
                                    d12_p3, d1_mk, gc_ctx, hi_ctx,
                                    jup_bav, yogini_p3, d12_mk,
                                    father_mk, candidates, ci):
    """Extract all features for a candidate at any depth."""
    from .features.extended_maraka import (
        classify_candidate_extended, extract_tier_features)
    from .features.multiref_features import (
        extract_multiref_features, extract_lord_discrimination)
    from .features.d12_features import extract_d12_features
    from .features.five_new_features import (
        extract_sookshma_features, extract_sandhi_features,
        extract_identity_features, extract_lord_transit_features)
    from .features.gochar_features import extract_gochar_features
    from .features.hierarchy_features import extract_hierarchy_features
    from .features.navamsha_features import extract_navamsha_features
    from .features.lever2_features import (
        extract_multipoint_features, extract_jup_bav_features,
        extract_cross_features)
    from .features.yogini_features import extract_yogini_features
    from .features.retrograde_features import extract_retrograde_features
    from .features.sade_sati_features import extract_sade_sati_features
    from .features.d12_timeline_features import extract_d12_timeline_features
    from .features.nakshatra_features import extract_nakshatra_features
    from .features.eclipse_features import extract_eclipse_features
    from .features.sequence_features import extract_sequence_features
    from .features.combustion_features import extract_combustion_features

    tier, dscore = classify_candidate_extended(cand, chart, asc, pre)
    cand['tier'] = tier
    cand['danger_score'] = dscore
    if tier == 1:
        cand['maraka_type'] = 'primary'
    elif tier in (2, 3, 4, 5):
        cand['maraka_type'] = 'secondary'
    else:
        cand['maraka_type'] = 'none'

    f = {}
    f['tier'] = tier
    f['danger_score'] = dscore
    f['duration_days'] = cand['end_jd'] - cand['start_jd']

    # Tier features
    try:
        tier_f = extract_tier_features(cand, chart, asc, pre, multi_ref)
        f.update({f'tier_{k}': v for k, v in tier_f.items()})
    except Exception:
        pass

    # Multi-ref features
    try:
        ref_f = extract_multiref_features(cand, multi_ref, chart, asc)
        f.update({f'ref_{k}': v for k, v in ref_f.items()})
    except Exception:
        pass

    # Lord features
    try:
        lord_f = extract_lord_discrimination(cand, chart, asc)
        f.update({f'lord_{k}': v for k, v in lord_f.items()})
    except Exception:
        pass

    # D12
    try:
        d12_f = extract_d12_features(cand, chart, asc, d12_p3, d1_mk)
        f.update({f'd12_{k}': v for k, v in d12_f.items()})
    except Exception:
        pass

    # Sookshma/sandhi/identity/lord_transit
    try:
        sk_f = extract_sookshma_features(cand, asc)
        f.update({f'sk_{k}': v for k, v in sk_f.items()})
    except Exception:
        pass
    try:
        sandhi_f = extract_sandhi_features(cand, depth2, depth1)
        f.update({f'sandhi_{k}': v for k, v in sandhi_f.items()})
    except Exception:
        pass
    try:
        id_f = extract_identity_features(cand, asc)
        f.update({f'id_{k}': v for k, v in id_f.items()})
    except Exception:
        pass
    try:
        lt_f = extract_lord_transit_features(cand, chart)
        f.update({f'lt_{k}': v for k, v in lt_f.items()})
    except Exception:
        pass

    # Gochar + hierarchy
    try:
        gc_f = extract_gochar_features(cand, gc_ctx)
        f.update(gc_f)
    except Exception:
        gc_f = {}
    try:
        hi_f = extract_hierarchy_features(cand, chart, hi_ctx)
        f.update(hi_f)
    except Exception:
        hi_f = {}

    # Navamsha
    try:
        d9_f = extract_navamsha_features(cand, chart, asc, father_mk)
        f.update(d9_f)
    except Exception:
        pass

    # Multipoint + jup BAV + cross
    try:
        mp_f = extract_multipoint_features(cand, gc_ctx)
        f.update(mp_f)
    except Exception:
        pass
    try:
        jb_f = extract_jup_bav_features(cand, jup_bav)
        f.update(jb_f)
    except Exception:
        pass
    try:
        cx_f = extract_cross_features(gc_f, hi_f)
        f.update(cx_f)
    except Exception:
        pass

    # Phase 11 features
    try:
        yg_f = extract_yogini_features(cand, yogini_p3, father_mk)
        f.update(yg_f)
    except Exception:
        pass
    try:
        rt_f = extract_retrograde_features(cand, chart)
        f.update(rt_f)
    except Exception:
        pass
    try:
        ss_f = extract_sade_sati_features(cand, chart)
        f.update(ss_f)
    except Exception:
        pass
    try:
        d12t_f = extract_d12_timeline_features(cand, d12_p3, d1_mk, d12_mk)
        f.update(d12t_f)
    except Exception:
        pass
    try:
        nk_f = extract_nakshatra_features(cand, chart, father_mk)
        f.update(nk_f)
    except Exception:
        pass
    try:
        ec_f = extract_eclipse_features(cand, chart, asc)
        f.update(ec_f)
    except Exception:
        pass
    try:
        comb_f = extract_combustion_features(cand, chart)
        f.update(comb_f)
    except Exception:
        pass

    # Late-stage cross-interactions (need sade_sati + eclipse + hierarchy)
    try:
        f['cx_sade_sati_x_cascade2'] = 1.0 if (
            f.get('ss_active', 0) == 1.0
            and f.get('hi_maraka_cascade', 0) >= 2
        ) else 0.0
        f['cx_eclipse_x_maraka_lord'] = 1.0 if (
            f.get('ec_axis_on_maraka', 0) == 1.0
            and f.get('tier', 0) in (1, 2)
        ) else 0.0
    except Exception:
        pass

    # Sequence features (need pre-classified candidates)
    try:
        seq_f = extract_sequence_features(candidates, ci)
        f.update(seq_f)
    except Exception:
        pass

    return f


def build_depth_dataset(records, name, start_index, target_depth,
                         window_months=24, n_augment=1):
    """Build dataset at a specific dasha depth."""
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .astro_engine.multiref_dasha import compute_multi_reference
    from .features.dasha_window import construct_dasha_window
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

    dn = DEPTH_NAMES[target_depth]
    print(f"  Building {name} depth={target_depth} ({dn}) "
          f"({len(records)} charts x {n_augment} aug)...")

    # Need depth up to max(target_depth, 4) for sookshma features,
    # but at least 3 for existing features that use depth 1/2/3
    compute_depth = max(target_depth, 3)

    all_rows = []
    n_ok = 0
    errors = 0
    t0 = time.time()
    dur_stats = []

    for i, rec in enumerate(records):
        base_idx = start_index + i
        try:
            chart, asc = compute_chart(
                rec['birth_date'], rec['birth_time'],
                rec['lat'], rec['lon'])
            birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])
            moon_long = chart['Moon']['longitude']

            # Compute all depths (collect_all_depths only collects 2+)
            all_depth = compute_full_dasha(
                moon_long, birth_jd, max_depth=compute_depth,
                collect_all_depths=True)

            # Depth 1 needs separate call
            depth1 = compute_full_dasha(moon_long, birth_jd, max_depth=1)
            depth2 = [p for p in all_depth if p['depth'] == 2]
            depth3 = [p for p in all_depth if p['depth'] == 3]

            # Periods at target depth
            if target_depth == 1:
                target_periods = depth1
            elif target_depth == 4:
                # Need depth 4
                all_d4 = compute_full_dasha(
                    moon_long, birth_jd, max_depth=4,
                    collect_all_depths=True)
                target_periods = [p for p in all_d4 if p['depth'] == 4]
            else:
                target_periods = [p for p in all_depth
                                   if p['depth'] == target_depth]

            # Multi-ref uses depth 3 internally
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

                candidates, correct_idx, _, _ = construct_dasha_window(
                    rec['father_death_date'], target_periods,
                    window_months=window_months, seed=seed)
                if correct_idx is None:
                    continue

                # First pass: classify all (for sequence features)
                for ci, cand in enumerate(candidates):
                    tier, dscore = classify_candidate_extended(
                        cand, chart, asc, pre)
                    cand['tier'] = tier
                    cand['danger_score'] = dscore

                dur_stats.append(len(candidates))

                # Second pass: full features
                for ci, cand in enumerate(candidates):
                    f = extract_features_for_candidate(
                        cand, chart, asc, pre, multi_ref,
                        depth1, depth2, depth3,
                        d12_p3, d1_mk, gc_ctx, hi_ctx,
                        jup_bav, yogini_p3, d12_mk,
                        father_mk, candidates, ci)

                    row = {
                        'group_id': idx,
                        'cand_idx': ci,
                        'label': 1 if ci == correct_idx else 0,
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
        # Delta features: temporal derivatives across PD sequence
        _DELTA_FEATS = [
            'gc_sat_asp_str_sun', 'gc_sat_asp_str_h9',
            'gc_jup_asp_str_sun', 'gc_jup_asp_str_h9',
            'gc_mars_asp_str_sun', 'gc_mars_asp_str_h9',
            'gc_sat_dist_sun', 'gc_jup_dist_sun', 'gc_mars_dist_sun',
            'gc_merc_dist_sun', 'ec_rahu_dist_h9', 'ec_rahu_dist_sun',
            'ss_sat_moon_deg', 'gc_moon_dist_h9',
        ]
        df = df.sort_values(['group_id', 'cand_idx'])
        for feat in _DELTA_FEATS:
            if feat in df.columns:
                df[f'{feat}_delta'] = df.groupby('group_id')[feat].diff().fillna(0)
    df.to_parquet(cache_f)
    elapsed = time.time() - t0
    print(f"  {n_ok} OK, {errors} errors, {elapsed:.1f}s")
    n_groups = df['group_id'].nunique() if len(df) > 0 and 'group_id' in df.columns else 0
    print(f"  Rows: {len(df)}, Groups: {n_groups}")
    if dur_stats:
        ds = np.array(dur_stats)
        print(f"  Candidates/window: mean={ds.mean():.1f}, "
              f"median={np.median(ds):.0f}, "
              f"min={ds.min()}, max={ds.max()}")

    return df


def main():
    print("=" * 75)
    print("DASHA-NATIVE MULTI-DEPTH PREDICTION")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    META_COLS = ['group_id', 'cand_idx', 'label', 'tier', 'danger_score',
                 'duration_days']

    params_medium = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05, 'subsample': 0.8,
        'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 25,
        'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    params_tuned_v3 = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    params_xendcg = {
        'objective': 'rank_xendcg', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    # ── Build and evaluate at each depth ────────────────────────────
    depth_results = {}

    # Window sizes adapted to each depth:
    # MD spans 6-20 years, need huge window. Use 120 months (10 years).
    # AD spans ~6mo-2yr, 24 months works fine.
    # PD spans ~3-90 days, 24 months works fine.
    # SD spans ~1-10 days, 24 months gives too many candidates, use 6 months.
    DEPTH_WINDOW = {1: 120, 2: 24, 3: 24, 4: 6}
    DEPTH_AUG = {1: 3, 2: 3, 3: 5, 4: 3}

    for depth in [1, 2, 3, 4]:
        dn = DEPTH_NAMES[depth]
        wm = DEPTH_WINDOW[depth]
        aug = DEPTH_AUG[depth]
        print(f"\n{'='*75}")
        print(f"DEPTH {depth}: {dn} (window={wm}mo)")
        print(f"{'='*75}")

        df_tr = build_depth_dataset(train_recs, 'train', 0,
                                     depth, window_months=wm, n_augment=aug)
        df_va = build_depth_dataset(val_recs, 'val',
                                     len(train_recs) * 100,
                                     depth, window_months=wm, n_augment=1)

        if len(df_va) == 0 or 'group_id' not in df_va.columns:
            print(f"  SKIP: no valid data at depth {depth}")
            continue

        # Filter to groups with >= 2 candidates (needed for ranking)
        gs = df_va.groupby('group_id').size()
        valid_groups = gs[gs >= 2].index
        df_tr_f = df_tr[df_tr['group_id'].isin(
            df_tr.groupby('group_id').filter(
                lambda x: len(x) >= 2)['group_id'].unique())]
        df_va_f = df_va[df_va['group_id'].isin(valid_groups)]

        if len(df_va_f) == 0 or df_va_f['group_id'].nunique() < 5:
            print(f"  SKIP: too few valid groups at depth {depth}")
            continue

        gs_f = df_va_f.groupby('group_id').size()
        n_va = df_va_f['group_id'].nunique()

        # Feature columns
        feat_cols = [c for c in df_va_f.columns
                     if c not in META_COLS
                     and isinstance(df_va_f[c].iloc[0], (int, float,
                                                          np.integer, np.floating))]
        feat_cols = [c for c in feat_cols
                     if c in df_tr_f.columns and c in df_va_f.columns]

        print(f"  Features: {len(feat_cols)}")
        print(f"  Val groups: {n_va}")
        print(f"  Candidates/group: mean={gs_f.mean():.1f}, "
              f"median={gs_f.median():.0f}")

        gs_f = df_va_f.groupby('group_id').size()
        n_va = df_va_f['group_id'].nunique()

        # ── Random baseline ─────────────────────────────────────────
        rnd_t1 = (1.0 / gs_f.values).mean()
        rnd_t3 = np.minimum(3.0 / gs_f.values, 1.0).mean()
        rnd_t5 = np.minimum(5.0 / gs_f.values, 1.0).mean()

        # ── Duration baseline ───────────────────────────────────────
        if 'duration_days' in df_va_f.columns:
            dur_scores = df_va_f['duration_days'].values
        else:
            dur_scores = np.zeros(len(df_va_f))
        dur_topk, _ = eval_topk(df_va_f, dur_scores)

        # ── Model ───────────────────────────────────────────────────
        params = params_tuned_v3 if depth >= 3 else params_medium
        pname = 'tuned_v3' if depth >= 3 else 'medium'
        print(f"  Training LambdaRank ({pname}, 5-seed avg)...")
        models = train_seed_avg(feat_cols, df_tr_f, df_va_f, 'group_id',
                                 params)
        model_scores = predict_avg(models, df_va_f[feat_cols].values)
        model_topk, _ = eval_topk(df_va_f, model_scores)

        # ── Model without duration features ─────────────────────────
        # Exclude duration + duration proxies (movement ≈ speed * duration)
        _DUR_PROXY = {'duration_days', 'seq_duration_days', 'seq_dur_vs_mean',
                      'seq_dur_log', 'seq_danger_intensity',
                      'id_pl_planet_idx', 'id_al_planet_idx',
                      'id_lagna_planet_combo',
                      'gc_mars_movement', 'gc_merc_movement',
                      'gc_venus_movement'}
        no_dur_cols = [c for c in feat_cols
                       if 'duration' not in c and 'dur_' not in c
                       and c not in _DUR_PROXY]
        nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id',
                                    params)
        nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
        nd_topk, _ = eval_topk(df_va_f, nd_scores)

        # ── Feature importance + pruned model (PD + SD) ────────────
        if depth >= 3:
            # Average importance across 5-seed models
            imp = np.zeros(len(feat_cols))
            for m in models:
                imp += m.feature_importance(importance_type='gain')
            imp /= len(models)
            imp_pct = imp / imp.sum() * 100

            # Sort and display top 20
            idx_sorted = np.argsort(imp_pct)[::-1]
            print(f"\n  Top 20 features by importance:")
            for rank, i in enumerate(idx_sorted[:20]):
                print(f"    {rank+1:2d}. {feat_cols[i]:40s} {imp_pct[i]:5.2f}%")

            # Pruned model: keep features with >= 0.1% importance
            keep_mask = imp_pct >= 0.1
            pruned_cols = [c for c, keep in zip(feat_cols, keep_mask) if keep]
            n_pruned = len(feat_cols) - len(pruned_cols)
            print(f"\n  Feature selection: {len(feat_cols)} -> {len(pruned_cols)} "
                  f"(pruned {n_pruned} features < 0.1% importance)")

            if len(pruned_cols) < len(feat_cols):
                print(f"  Training pruned model...")
                params_safe = {**params, 'min_child_samples': 25}
                pruned_models = train_seed_avg(
                    pruned_cols, df_tr_f, df_va_f, 'group_id', params_safe)
                pruned_scores = predict_avg(
                    pruned_models, df_va_f[pruned_cols].values)
                pruned_topk, _ = eval_topk(df_va_f, pruned_scores)

                # Pruned no-dur
                pruned_nd_cols = [c for c in pruned_cols
                                  if 'duration' not in c and 'dur_' not in c
                                  and c not in _DUR_PROXY]
                pruned_nd_models = train_seed_avg(
                    pruned_nd_cols, df_tr_f, df_va_f, 'group_id', params_safe)
                pruned_nd_scores = predict_avg(
                    pruned_nd_models, df_va_f[pruned_nd_cols].values)
                pruned_nd_topk, _ = eval_topk(df_va_f, pruned_nd_scores)

                print(f"\n  Pruned SD Results ({len(pruned_cols)} features):")
                print(f"    {'Pruned full':>20s} "
                      f"{pruned_topk['top_1']:6.1%}  "
                      f"{pruned_topk['top_3']:6.1%}  "
                      f"{pruned_topk['top_5']:6.1%}  "
                      f"{pruned_topk['top_1']/max(rnd_t1,0.001):.1f}x")
                print(f"    {'Pruned no-dur':>20s} "
                      f"{pruned_nd_topk['top_1']:6.1%}  "
                      f"{pruned_nd_topk['top_3']:6.1%}  "
                      f"{pruned_nd_topk['top_5']:6.1%}  "
                      f"{pruned_nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")

        # ── No-dur hyperparameter sweep (PD only) ──────────────────
        if depth == 3:
            nd_variants = {
                'nd_A (colsamp=0.9, 2000rnd)': {
                    **params, 'colsample_bytree': 0.9,
                },
                'nd_B (low_reg)': {
                    **params, 'reg_lambda': 0.5, 'reg_alpha': 0.1,
                },
                'nd_C (leaves=64, depth=8)': {
                    **params, 'num_leaves': 64, 'max_depth': 8,
                    'min_child_samples': 20,
                },
            }
            print(f"\n  No-dur hyperparameter sweep:")
            for vname, vparams in nd_variants.items():
                v_models = train_seed_avg(
                    no_dur_cols, df_tr_f, df_va_f, 'group_id', vparams)
                v_scores = predict_avg(v_models, df_va_f[no_dur_cols].values)
                v_topk, _ = eval_topk(df_va_f, v_scores)
                print(f"    {vname:>35s} "
                      f"{v_topk['top_1']:6.1%}  "
                      f"{v_topk['top_3']:6.1%}  "
                      f"{v_topk['top_5']:6.1%}  "
                      f"{v_topk['top_1']/max(rnd_t1,0.001):.1f}x")

            # No-dur feature importance
            nd_imp = np.zeros(len(no_dur_cols))
            for m in nd_models:
                nd_imp += m.feature_importance(importance_type='gain')
            nd_imp /= len(nd_models)
            nd_imp_pct = nd_imp / nd_imp.sum() * 100
            nd_idx_sorted = np.argsort(nd_imp_pct)[::-1]
            print(f"\n  No-dur top 20 features:")
            for rank, i in enumerate(nd_idx_sorted[:20]):
                print(f"    {rank+1:2d}. {no_dur_cols[i]:40s} {nd_imp_pct[i]:5.2f}%")

        # ── rank_xendcg + binary blend (PD only) ──────────────────
        if depth == 3:
            # rank_xendcg no-dur
            print(f"\n  rank_xendcg experiments:")
            xendcg_nd_models = train_seed_avg(
                no_dur_cols, df_tr_f, df_va_f, 'group_id', params_xendcg)
            xendcg_nd_scores = predict_avg(
                xendcg_nd_models, df_va_f[no_dur_cols].values)
            xendcg_nd_topk, _ = eval_topk(df_va_f, xendcg_nd_scores)
            print(f"    {'xendcg no-dur':>25s} "
                  f"{xendcg_nd_topk['top_1']:6.1%}  "
                  f"{xendcg_nd_topk['top_3']:6.1%}  "
                  f"{xendcg_nd_topk['top_5']:6.1%}  "
                  f"{xendcg_nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")

            # Binary classifier for blend
            params_clf = {
                'objective': 'binary', 'metric': 'binary_logloss',
                'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
                'min_child_samples': 25, 'colsample_bytree': 0.8,
                'scale_pos_weight': 12.0,
                'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
            }
            clf_models = train_seed_avg(
                no_dur_cols, df_tr_f, df_va_f, 'group_id', params_clf)
            clf_scores = predict_avg(
                clf_models, df_va_f[no_dur_cols].values)

            # Blend: ranker + classifier at various weights
            for w_rank in [0.5, 0.7, 0.8]:
                w_clf = 1.0 - w_rank
                # Normalize scores to same scale
                r_norm = (nd_scores - nd_scores.mean()) / max(nd_scores.std(), 1e-6)
                c_norm = (clf_scores - clf_scores.mean()) / max(clf_scores.std(), 1e-6)
                blend = w_rank * r_norm + w_clf * c_norm
                blend_topk, _ = eval_topk(df_va_f, blend)
                print(f"    {'blend %.0f/%.0f rank/clf' % (w_rank*100, w_clf*100):>25s} "
                      f"{blend_topk['top_1']:6.1%}  "
                      f"{blend_topk['top_3']:6.1%}  "
                      f"{blend_topk['top_5']:6.1%}  "
                      f"{blend_topk['top_1']/max(rnd_t1,0.001):.1f}x")

        # ── Multi-model diversity ensemble (PD only) ──────────────
        if depth == 3:
            print(f"\n  Multi-model ensemble (no-dur):")
            ens_configs = [
                ('LR_base', params),
                ('LR_shallow', {**params, 'num_leaves': 20, 'max_depth': 4,
                                'min_child_samples': 30}),
                ('LR_deep', {**params, 'num_leaves': 64, 'max_depth': 8,
                             'min_child_samples': 25}),
                ('LR_lowreg', {**params, 'reg_lambda': 0.3, 'reg_alpha': 0.05}),
                ('LR_colsamp', {**params, 'colsample_bytree': 0.5}),
                ('xendcg', params_xendcg),
                ('binary', params_clf),
                ('LR_subsamp', {**params, 'subsample': 0.6,
                                'colsample_bytree': 0.6,
                                'min_child_samples': 25}),
            ]

            all_nd_scores = []
            for ename, eparams in ens_configs:
                try:
                    e_models = train_seed_avg(
                        no_dur_cols, df_tr_f, df_va_f, 'group_id', eparams)
                    e_scores = predict_avg(e_models, df_va_f[no_dur_cols].values)
                    e_topk, _ = eval_topk(df_va_f, e_scores)
                    all_nd_scores.append(e_scores)
                    print(f"    {ename:>15s}  {e_topk['top_1']:6.1%}  "
                          f"{e_topk['top_3']:6.1%}  {e_topk['top_5']:6.1%}")
                except Exception as ex:
                    print(f"    {ename:>15s}  FAILED: {ex}")

            if len(all_nd_scores) >= 3:
                # Normalize each model's scores to z-scores
                norm_scores = []
                for s in all_nd_scores:
                    std = max(s.std(), 1e-6)
                    norm_scores.append((s - s.mean()) / std)

                # Simple average ensemble
                avg_ens = np.mean(norm_scores, axis=0)
                avg_topk, _ = eval_topk(df_va_f, avg_ens)
                print(f"    {'avg_ensemble':>15s}  {avg_topk['top_1']:6.1%}  "
                      f"{avg_topk['top_3']:6.1%}  {avg_topk['top_5']:6.1%}  "
                      f"{avg_topk['top_1']/max(rnd_t1,0.001):.1f}x")

                # Agreement-weighted ensemble (from father_death_ml.py)
                n_ens = len(norm_scores)
                n_rows = len(df_va_f)

                # Per-group agreement scoring
                agree_scores = np.zeros(n_rows)
                borda_scores = np.zeros(n_rows)

                for gid, grp in df_va_f.groupby('group_id', sort=False):
                    idx = grp.index
                    g_size = len(idx)
                    if g_size < 2:
                        continue
                    top_k = min(3, g_size - 1)

                    # Borda + agreement per candidate in this group
                    g_borda = np.zeros(g_size)
                    g_agree = np.zeros(g_size)

                    for s in norm_scores:
                        g_s = s[idx].values if hasattr(s, 'values') else s[idx]
                        ranks = np.argsort(np.argsort(-g_s))  # 0=best
                        g_borda += (g_size - ranks) / g_size
                        g_agree += (ranks < top_k).astype(float)

                    g_borda /= n_ens
                    g_agree /= n_ens

                    # Agreement-weighted final score
                    final = g_borda * (1.0 + g_agree * 2.0)
                    agree_scores[idx] = final
                    borda_scores[idx] = g_borda

                agree_topk, _ = eval_topk(df_va_f, agree_scores)
                print(f"    {'agree_ensemble':>15s}  {agree_topk['top_1']:6.1%}  "
                      f"{agree_topk['top_3']:6.1%}  {agree_topk['top_5']:6.1%}  "
                      f"{agree_topk['top_1']/max(rnd_t1,0.001):.1f}x")

                # Agreement + binary blend
                c_norm_g = (clf_scores - clf_scores.mean()) / max(clf_scores.std(), 1e-6)
                a_norm = (agree_scores - agree_scores.mean()) / max(agree_scores.std(), 1e-6)
                for w in [0.7, 0.5]:
                    ab = w * a_norm + (1-w) * c_norm_g
                    ab_topk, _ = eval_topk(df_va_f, ab)
                    print(f"    {'agree+clf %.0f/%.0f' % (w*100,(1-w)*100):>15s}  "
                          f"{ab_topk['top_1']:6.1%}  {ab_topk['top_3']:6.1%}  "
                          f"{ab_topk['top_5']:6.1%}  "
                          f"{ab_topk['top_1']/max(rnd_t1,0.001):.1f}x")

        # ── Store results ───────────────────────────────────────────
        depth_results[depth] = {
            'name': dn,
            'n_groups': n_va,
            'mean_cands': gs.mean(),
            'median_cands': gs.median(),
            'random': {'t1': rnd_t1, 't3': rnd_t3, 't5': rnd_t5},
            'duration': dur_topk,
            'model': model_topk,
            'model_no_dur': nd_topk,
        }

        # ── Duration stats ──────────────────────────────────────────
        if 'duration_days' in df_va_f.columns:
            death_dur = df_va_f[df_va_f['label'] == 1]['duration_days']
            all_dur = df_va_f['duration_days']
            print(f"  Avg duration: all={all_dur.mean():.1f}d, "
                  f"death={death_dur.mean():.1f}d")

        print(f"\n  {dn} Results:")
        print(f"    {'Method':>20s} {'Top-1':>7s} {'Top-3':>7s} "
              f"{'Top-5':>7s} {'Lift':>6s}")
        print(f"    {'-'*48}")
        print(f"    {'Random':>20s} {rnd_t1:6.1%}  {rnd_t3:6.1%}  "
              f"{rnd_t5:6.1%}  {'1.0x':>6s}")
        print(f"    {'Duration only':>20s} "
              f"{dur_topk['top_1']:6.1%}  {dur_topk['top_3']:6.1%}  "
              f"{dur_topk['top_5']:6.1%}  "
              f"{dur_topk['top_1']/max(rnd_t1,0.001):.1f}x")
        print(f"    {'Model (no duration)':>20s} "
              f"{nd_topk['top_1']:6.1%}  {nd_topk['top_3']:6.1%}  "
              f"{nd_topk['top_5']:6.1%}  "
              f"{nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")
        print(f"    {'Full model':>20s} "
              f"{model_topk['top_1']:6.1%}  {model_topk['top_3']:6.1%}  "
              f"{model_topk['top_5']:6.1%}  "
              f"{model_topk['top_1']/max(rnd_t1,0.001):.1f}x")

    # ── Grand summary ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("GRAND SUMMARY: ALL DEPTHS")
    print("=" * 75)

    print(f"\n  {'Depth':>6s} {'Name':>4s} {'Win':>5s} {'Cands':>6s} "
          f"{'Random':>7s} {'DurOnly':>8s} {'NoDur':>7s} {'Model':>7s} "
          f"{'Lift':>6s} {'AstroLift':>10s}")
    print("-" * 80)

    for depth in [1, 2, 3, 4]:
        r = depth_results[depth]
        rnd = r['random']['t1']
        dur = r['duration']['top_1']
        nd = r['model_no_dur']['top_1']
        m = r['model']['top_1']
        lift = m / max(rnd, 0.001)
        astro_lift = nd / max(rnd, 0.001)
        wm = DEPTH_WINDOW[depth]
        print(f"  {depth:>6d} {r['name']:>4s} {wm:>4d}m {r['mean_cands']:>5.0f}  "
              f"{rnd:6.1%}  {dur:7.1%}  {nd:6.1%}  {m:6.1%}  "
              f"{lift:5.1f}x  {astro_lift:9.1f}x")

    print(f"\n  Lift = Model / Random")
    print(f"  AstroLift = Model-without-duration / Random "
          f"(pure astrological signal)")
    print("=" * 75)

    # ── Hierarchical AD→PD cascade ──────────────────────────────────
    if 2 in depth_results and 3 in depth_results:
        print("\n" + "=" * 75)
        print("HIERARCHICAL CASCADE: AD -> PD-within-AD")
        print("=" * 75)

        # Reload cached data (already built above)
        df_tr_ad = pd.read_parquet(
            DATA_DIR / f'train_d2_aug{DEPTH_AUG[2]}.parquet')
        df_va_ad = pd.read_parquet(DATA_DIR / 'val_d2_aug1.parquet')
        df_tr_pd = pd.read_parquet(
            DATA_DIR / f'train_d3_aug{DEPTH_AUG[3]}.parquet')
        df_va_pd = pd.read_parquet(DATA_DIR / 'val_d3_aug1.parquet')

        META_COLS = ['group_id', 'cand_idx', 'label', 'tier', 'danger_score',
                     'duration_days']

        feat_cols_ad = [c for c in df_va_ad.columns
                        if c not in META_COLS
                        and isinstance(df_va_ad[c].iloc[0],
                                       (int, float, np.integer, np.floating))]
        feat_cols_ad = [c for c in feat_cols_ad
                        if c in df_tr_ad.columns]

        feat_cols_pd = [c for c in df_va_pd.columns
                        if c not in META_COLS
                        and isinstance(df_va_pd[c].iloc[0],
                                       (int, float, np.integer, np.floating))]
        feat_cols_pd = [c for c in feat_cols_pd
                        if c in df_tr_pd.columns]

        nd_cols_pd = [c for c in feat_cols_pd
                      if 'duration' not in c and 'dur_' not in c
                      and c not in _DUR_PROXY]

        # Train AD model (stage 1)
        gs_ad = df_va_ad.groupby('group_id').size()
        valid_ad = gs_ad[gs_ad >= 2].index
        df_tr_ad_f = df_tr_ad[df_tr_ad['group_id'].isin(
            df_tr_ad.groupby('group_id').filter(
                lambda x: len(x) >= 2)['group_id'].unique())]
        df_va_ad_f = df_va_ad[df_va_ad['group_id'].isin(valid_ad)]

        print(f"\n  Stage 1: AD model ({df_va_ad_f['group_id'].nunique()} "
              f"val groups, {len(feat_cols_ad)} features)")
        ad_models = train_seed_avg(feat_cols_ad, df_tr_ad_f, df_va_ad_f,
                                    'group_id', params_medium)

        # ── Oracle PD-within-AD: group PDs by parent AD ────────────
        # For each val PD group, identify which AD each PD belongs to
        # PD group_id encodes chart; we need to group PDs by parent AD
        # Parent AD is identified by matching lords[:2]
        # Since we have pre-built DataFrames, use cand_idx ordering:
        # within a PD group (same group_id), PDs are ordered chronologically
        # and PDs within the same parent AD are consecutive

        # Approach: For each PD val group, partition PDs by parent AD index
        # Use the fact that cand_idx is ordered, and PDs within an AD
        # are contiguous in dasha period ordering.

        # Step 1: For each PD group, compute which "AD segment" each PD
        # belongs to. PDs sharing the same AD will have the same AD lords
        # (identified by which AD contains their midpoint).

        # We need to reload the raw dasha data to get lords.
        # Alternative: use sandhi features. When sandhi_antar_elapsed resets
        # to a low value, a new AD has started.

        # Simpler approach: build oracle groups where PDs are grouped by
        # parent AD, using sandhi_antar_elapsed to detect AD boundaries.
        # When elapsed drops significantly (< previous), new AD started.

        print(f"\n  Building oracle PD-within-AD groups...")

        oracle_groups = []
        oracle_rows = []
        new_gid = 0

        for gid, grp in df_va_pd.groupby('group_id', sort=False):
            if 'sandhi_antar_elapsed' not in grp.columns:
                continue
            grp = grp.sort_values('cand_idx')
            elapsed = grp['sandhi_antar_elapsed'].values

            # Detect AD boundaries: when elapsed drops by > 0.3
            ad_segment = 0
            segments = [0]
            for i in range(1, len(elapsed)):
                if elapsed[i] < elapsed[i-1] - 0.15:
                    ad_segment += 1
                segments.append(ad_segment)

            # Find which segment has the death
            labels = grp['label'].values
            death_idx = np.where(labels == 1)[0]
            if len(death_idx) == 0:
                continue
            death_seg = segments[death_idx[0]]

            # Build oracle group: only PDs in the death-bearing AD segment
            seg_mask = np.array(segments) == death_seg
            seg_grp = grp.iloc[seg_mask]

            if len(seg_grp) < 2:
                continue

            for _, row in seg_grp.iterrows():
                r = row.to_dict()
                r['group_id'] = new_gid
                oracle_rows.append(r)

            new_gid += 1

        df_oracle = pd.DataFrame(oracle_rows)
        if len(df_oracle) == 0:
            print("  No oracle groups built (missing sandhi features?)")
        else:
            n_oracle = df_oracle['group_id'].nunique()
            gs_oracle = df_oracle.groupby('group_id').size()
            rnd_oracle = (1.0 / gs_oracle.values).mean()
            print(f"  Oracle groups: {n_oracle}, "
                  f"mean cands: {gs_oracle.mean():.1f}, "
                  f"random: {rnd_oracle:.1%}")

            # Build oracle TRAIN data too
            oracle_tr_rows = []
            new_gid_tr = 0
            for gid, grp in df_tr_pd.groupby('group_id', sort=False):
                if 'sandhi_antar_elapsed' not in grp.columns:
                    continue
                grp = grp.sort_values('cand_idx')
                elapsed = grp['sandhi_antar_elapsed'].values
                labels = grp['label'].values

                ad_segment = 0
                segments = [0]
                for i in range(1, len(elapsed)):
                    if elapsed[i] < elapsed[i-1] - 0.15:
                        ad_segment += 1
                    segments.append(ad_segment)

                death_idx = np.where(labels == 1)[0]
                if len(death_idx) == 0:
                    continue
                death_seg = segments[death_idx[0]]
                seg_mask = np.array(segments) == death_seg
                seg_grp = grp.iloc[seg_mask]
                if len(seg_grp) < 2:
                    continue

                for _, row in seg_grp.iterrows():
                    r = row.to_dict()
                    r['group_id'] = new_gid_tr
                    oracle_tr_rows.append(r)
                new_gid_tr += 1

            df_oracle_tr = pd.DataFrame(oracle_tr_rows)
            print(f"  Oracle train groups: {df_oracle_tr['group_id'].nunique()}")

            # Train stage-2 model on oracle data
            oracle_cols = [c for c in nd_cols_pd if c in df_oracle.columns
                           and c in df_oracle_tr.columns]

            print(f"  Training oracle stage-2 ({len(oracle_cols)} no-dur features)...")

            try:
                s2_params = {**params_tuned_v3, 'min_child_samples': 25}
                s2_models = train_seed_avg(
                    oracle_cols, df_oracle_tr, df_oracle, 'group_id',
                    s2_params)
                s2_scores = predict_avg(s2_models, df_oracle[oracle_cols].values)
                s2_topk, s2_n = eval_topk(df_oracle, s2_scores)

                print(f"\n  Oracle Stage-2 Results "
                      f"({n_oracle} groups, ~{gs_oracle.mean():.0f} cands):")
                print(f"    Random:  {rnd_oracle:6.1%}")
                print(f"    NoDur:   {s2_topk['top_1']:6.1%}  "
                      f"{s2_topk['top_3']:6.1%}  {s2_topk['top_5']:6.1%}  "
                      f"{s2_topk['top_1']/max(rnd_oracle,0.001):.1f}x lift")

                # Combined cascade estimate
                ad_t1 = depth_results[2]['model_no_dur']['top_1']
                cascade_t1 = ad_t1 * s2_topk['top_1']
                print(f"\n  Cascade estimate (AD {ad_t1:.1%} * "
                      f"Stage2 {s2_topk['top_1']:.1%}):")
                print(f"    Combined top-1: {cascade_t1:.1%}")

                # Also try with nd_C params
                s2c_params = {**params_tuned_v3, 'num_leaves': 64,
                              'max_depth': 8, 'min_child_samples': 25}
                s2c_models = train_seed_avg(
                    oracle_cols, df_oracle_tr, df_oracle, 'group_id',
                    s2c_params)
                s2c_scores = predict_avg(
                    s2c_models, df_oracle[oracle_cols].values)
                s2c_topk, _ = eval_topk(df_oracle, s2c_scores)
                cascade_c = ad_t1 * s2c_topk['top_1']
                print(f"    nd_C stage2: {s2c_topk['top_1']:.1%}  "
                      f"-> cascade: {cascade_c:.1%}")

            except Exception as e:
                print(f"  Stage-2 training failed: {e}")

    print("\n" + "=" * 75)


if __name__ == '__main__':
    main()
