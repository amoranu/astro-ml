"""Yogini Dasha father death prediction pipeline.

Independent 36-year cycle with 8 Yogini-planet lords (BPHS Ch. 46).
Uses Yogini Antardasha (depth=2) as prediction unit (~45-90 day periods).
No Vimshottari features — purely Yogini-native + shared transit features.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_yogini
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
DATA_DIR = PROJECT_ROOT / 'data' / 'yogini'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


# Reuse shared training/eval infrastructure
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


def extract_features_for_candidate(cand, chart, asc, yg_ctx, gc_ctx,
                                    candidates, ci,
                                    depth1_periods, depth2_periods,
                                    jup_bav, hi_ctx):
    """Extract all features for a Yogini PD candidate."""
    from .features.yogini_native import (
        extract_yogini_native_features, extract_yogini_subperiod_features)
    from .features.gochar_features import extract_gochar_features
    from .features.sade_sati_features import extract_sade_sati_features
    from .features.eclipse_features import extract_eclipse_features
    from .features.nakshatra_features import (
        extract_nakshatra_features, _get_nakshatra_idx, NAKSHATRA_LORDS,
        NAKSHATRA_TEMPERAMENT)
    from .features.five_new_features import (
        extract_sandhi_features, extract_lord_transit_features)
    from .features.lever2_features import (
        extract_multipoint_features, extract_jup_bav_features,
        extract_cross_features)
    from .features.navamsha_features import extract_navamsha_features
    from .features.multiref_features import extract_lord_discrimination
    from .features.combustion_features import extract_combustion_features
    from .features.retrograde_features import extract_retrograde_features
    from .features.hierarchy_features import extract_hierarchy_features
    from .features.sequence_features import extract_sequence_features

    f = {}
    f['duration_days'] = cand['end_jd'] - cand['start_jd']

    # Adapter: shared modules expect lords = planet names
    adapted = {**cand, 'lords': cand['planets']}
    deepest = cand['planets'][-1]
    adapted['maraka_type'] = (
        'primary' if deepest in yg_ctx['father_marakas'] else 'none')

    # --- Yogini-native features ---
    try:
        yg_f = extract_yogini_native_features(cand, yg_ctx)
        f.update(yg_f)
    except Exception:
        pass

    # --- Yogini sub-period density ---
    try:
        ysk_f = extract_yogini_subperiod_features(cand, yg_ctx)
        f.update(ysk_f)
    except Exception:
        pass

    # --- Shared transit features at midpoint ---
    gc_f = {}
    try:
        gc_f = extract_gochar_features(cand, gc_ctx)
        f.update(gc_f)
    except Exception:
        pass

    # --- Sade Sati ---
    try:
        ss_f = extract_sade_sati_features(cand, chart)
        f.update(ss_f)
    except Exception:
        pass

    # --- Eclipse axis ---
    try:
        ec_f = extract_eclipse_features(cand, chart, asc)
        f.update(ec_f)
    except Exception:
        pass

    # --- Nakshatra of PD lord ---
    try:
        father_mk = yg_ctx['father_marakas']
        pd_planet = cand['planets'][-1]
        if pd_planet in chart and pd_planet not in ('Rahu', 'Ketu'):
            pd_lon = chart[pd_planet]['longitude']
            nak_idx = _get_nakshatra_idx(pd_lon)
            nak_lord = NAKSHATRA_LORDS[nak_idx]
            f['nk_lord_is_maraka'] = 1.0 if nak_lord in father_mk else 0.0
            f['nk_temperament'] = NAKSHATRA_TEMPERAMENT[nak_idx]
    except Exception:
        pass

    # --- Sandhi (junction proximity) ---
    try:
        sandhi_f = extract_sandhi_features(adapted, depth2_periods, depth1_periods)
        f.update({f'sandhi_{k}': v for k, v in sandhi_f.items()})
    except Exception:
        pass

    # --- Lord transit (Saturn/Jupiter on PD lord's natal degree) ---
    try:
        lt_f = extract_lord_transit_features(adapted, chart)
        f.update({f'lt_{k}': v for k, v in lt_f.items()})
    except Exception:
        pass

    # --- Multi-point transit sampling ---
    try:
        mp_f = extract_multipoint_features(adapted, gc_ctx)
        f.update(mp_f)
    except Exception:
        pass

    # --- Jupiter BAV ---
    try:
        jb_f = extract_jup_bav_features(adapted, jup_bav)
        f.update(jb_f)
    except Exception:
        pass

    # --- Hierarchy features (MD/AD/PD lord interactions) ---
    hi_f = {}
    try:
        hi_f = extract_hierarchy_features(adapted, chart, hi_ctx)
        f.update(hi_f)
    except Exception:
        pass

    # --- Cross-interaction features (transit x hierarchy) ---
    try:
        # Map Yogini hierarchy features to cross_features expected keys
        yg_hi = {
            'hi_maraka_cascade': hi_f.get('hi_maraka_cascade',
                                          f.get('yg_maraka_cascade_3', 0)),
            'hi_danger_cascade': hi_f.get('hi_danger_cascade',
                                          f.get('yg_danger_cascade_3', 0)),
            'hi_md_is_maraka': hi_f.get('hi_md_is_maraka',
                                        f.get('yg_md_is_maraka', 0)),
            'hi_md_ad_both_maraka': hi_f.get('hi_md_ad_both_maraka',
                                             f.get('yg_both_maraka', 0)),
            'hi_md_pd_combined': hi_f.get('hi_md_pd_combined', 0),
        }
        cx_f = extract_cross_features(gc_f, yg_hi)
        f.update(cx_f)
    except Exception:
        pass

    # --- Lord discrimination (PD lord natal strength) ---
    try:
        lord_f = extract_lord_discrimination(adapted, chart, asc)
        f.update({f'lord_{k}': v for k, v in lord_f.items()})
    except Exception:
        pass

    # --- Navamsha (D9 confirmation) ---
    try:
        d9_f = extract_navamsha_features(
            adapted, chart, asc, yg_ctx['father_marakas'])
        f.update(d9_f)
    except Exception:
        pass

    # --- Combustion ---
    try:
        comb_f = extract_combustion_features(adapted, chart)
        f.update({f'comb_{k}': v for k, v in comb_f.items()})
    except Exception:
        pass

    # --- Retrograde ---
    try:
        rt_f = extract_retrograde_features(adapted, chart)
        f.update(rt_f)
    except Exception:
        pass

    # --- Sequence features ---
    try:
        seq_f = extract_sequence_features(candidates, ci)
        f.update({f'seq_{k}': v for k, v in seq_f.items()})
    except Exception:
        pass

    return f


def build_yogini_dataset(records, name, start_index, n_augment=5):
    """Build Yogini PD depth=3 dataset."""
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.yogini_dasha import compute_yogini_dasha
    from .features.dasha_window import construct_dasha_window
    from .features.gochar_features import precompute_gochar_context
    from .features.yogini_native import precompute_yogini_context
    from .features.five_new_features import add_relative_features
    from .features.lever2_features import precompute_jup_bav
    from .features.hierarchy_features import precompute_hierarchy_context
    from .features.extended_maraka import (
        classify_candidate_extended, precompute_maraka_sets)

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_aug{n_augment}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    print(f"  Building {name} Yogini PD depth=3 ({len(records)} charts x {n_augment} aug)...")

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

            # Yogini periods at all depths
            yogini_d3 = compute_yogini_dasha(moon_long, birth_jd, max_depth=3)
            yogini_d2 = compute_yogini_dasha(moon_long, birth_jd, max_depth=2)
            yogini_d1 = compute_yogini_dasha(moon_long, birth_jd, max_depth=1)
            target_periods = [p for p in yogini_d3 if p['depth'] == 3]

            # Precompute contexts
            yg_ctx = precompute_yogini_context(asc, chart,
                                               moon_long=moon_long,
                                               birth_jd=birth_jd)
            gc_ctx = precompute_gochar_context(asc, chart)
            jup_bav = precompute_jup_bav(chart, asc)
            hi_ctx = precompute_hierarchy_context(asc)
            maraka_pre = precompute_maraka_sets(chart, asc)

            for aug in range(n_augment):
                seed = base_idx * 100 + aug
                idx = base_idx * 100 + aug

                candidates, correct_idx, _, _ = construct_dasha_window(
                    rec['father_death_date'], target_periods,
                    window_months=24, seed=seed)
                if correct_idx is None:
                    continue

                # Pre-classify candidates for sequence_features
                # (needs 'tier' and 'danger_score' on each candidate)
                for ci_pre, cand_pre in enumerate(candidates):
                    adapted_pre = {**cand_pre, 'lords': cand_pre['planets']}
                    try:
                        tier, dscore = classify_candidate_extended(
                            adapted_pre, chart, asc, maraka_pre)
                    except Exception:
                        # Fallback: compute from Yogini maraka status
                        deepest = cand_pre['planets'][-1]
                        is_mk = deepest in yg_ctx['father_marakas']
                        is_dg = deepest in yg_ctx['father_danger']
                        mk_count = sum(
                            p in yg_ctx['father_marakas']
                            for p in cand_pre['planets'])
                        tier = 1 if is_mk else (3 if is_dg else 0)
                        dscore = mk_count * 2.0 + (1.0 if is_dg else 0.0)
                    cand_pre['tier'] = tier
                    cand_pre['danger_score'] = dscore

                dur_stats.append(len(candidates))

                for ci, cand in enumerate(candidates):
                    f = extract_features_for_candidate(
                        cand, chart, asc, yg_ctx, gc_ctx, candidates, ci,
                        yogini_d1, yogini_d2, jup_bav, hi_ctx)
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
        # Use shared add_relative_features for 18 KEY_REL features x 4 variants
        df = add_relative_features(df, group_col='group_id')

        # Add Yogini-specific + transit zscore features (NO duration proxies)
        YOGINI_EXTRA_REL = [
            'yg_pd_uchcha', 'yg_pd_dignity',
            'lt_sat_pl_dist', 'jb_jup_bav', 'd9_dispositor_dignity',
            'yg_pd_from_birth', 'yg_cycle_pos',
            # Transit degree distances (clean, vary by midpoint)
            'gc_sat_dist_sun', 'gc_sat_dist_h9', 'gc_sat_dist_moon',
            'gc_jup_dist_sun', 'gc_jup_dist_h9',
            'gc_mars_dist_moon', 'gc_merc_dist_h9',
            'gc_venus_dist_sun',
            'ec_rahu_dist_moon',
            # Hierarchy features
            'hi_maraka_cascade', 'hi_danger_cascade',
        ]
        avail = [c for c in YOGINI_EXTRA_REL if c in df.columns]
        for feat in avail:
            df[f'{feat}_rank'] = df.groupby('group_id')[feat].rank(
                ascending=False, method='min')
            g_mean = df.groupby('group_id')[feat].transform('mean')
            g_std = df.groupby('group_id')[feat].transform('std').clip(lower=1e-6)
            df[f'{feat}_zscore'] = (df[feat] - g_mean) / g_std
            g_min = df.groupby('group_id')[feat].transform('min')
            g_max = df.groupby('group_id')[feat].transform('max')
            df[f'{feat}_is_min'] = (df[feat] == g_min).astype(float)
            df[f'{feat}_is_max'] = (df[feat] == g_max).astype(float)

    df.to_parquet(cache_f)
    elapsed = time.time() - t0
    print(f"  {n_ok} OK, {errors} errors, {elapsed:.1f}s")
    if dur_stats:
        ds = np.array(dur_stats)
        print(f"  Candidates/window: mean={ds.mean():.1f}, "
              f"median={np.median(ds):.0f}, min={ds.min()}, max={ds.max()}")
    return df


def main():
    print("=" * 75)
    print("YOGINI DASHA FATHER DEATH PREDICTION (PD level depth=3, 24mo window)")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    META_COLS = ['group_id', 'cand_idx', 'label', 'duration_days']

    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 20,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    df_tr = build_yogini_dataset(train_recs, 'train', 0, n_augment=5)
    df_va = build_yogini_dataset(val_recs, 'val', len(train_recs) * 100,
                                  n_augment=1)

    if len(df_va) == 0:
        print("  No valid data!")
        return

    gs = df_va.groupby('group_id').size()
    valid_groups = gs[gs >= 2].index
    df_tr_f = df_tr[df_tr['group_id'].isin(
        df_tr.groupby('group_id').filter(
            lambda x: len(x) >= 2)['group_id'].unique())]
    df_va_f = df_va[df_va['group_id'].isin(valid_groups)]

    feat_cols = [c for c in df_va_f.columns
                 if c not in META_COLS
                 and isinstance(df_va_f[c].iloc[0],
                                (int, float, np.integer, np.floating))]
    feat_cols = [c for c in feat_cols if c in df_tr_f.columns]

    gs_f = df_va_f.groupby('group_id').size()
    n_va = df_va_f['group_id'].nunique()
    rnd_t1 = (1.0 / gs_f.values).mean()
    rnd_t3 = np.minimum(3.0 / gs_f.values, 1.0).mean()
    rnd_t5 = np.minimum(5.0 / gs_f.values, 1.0).mean()

    print(f"\n  Features: {len(feat_cols)}")
    print(f"  Val groups: {n_va}")
    print(f"  Candidates/group: mean={gs_f.mean():.1f}, median={gs_f.median():.0f}")

    # Duration baseline
    dur_scores = df_va_f['duration_days'].values if 'duration_days' in df_va_f.columns else np.zeros(len(df_va_f))
    dur_topk, _ = eval_topk(df_va_f, dur_scores)

    # Full model
    print(f"\n  Training full model...")
    models = train_seed_avg(feat_cols, df_tr_f, df_va_f, 'group_id', params)
    model_scores = predict_avg(models, df_va_f[feat_cols].values)
    model_topk, _ = eval_topk(df_va_f, model_scores)

    # No-dur model — STRICT proxy exclusion
    # Features with |correlation| > 0.15 with duration_days are excluded
    _DUR_PROXY = {
        'duration_days', 'yg_ad_planet_idx',
        # Sequence duration features
        'seq_seq_duration_days', 'seq_seq_dur_vs_mean',
        'seq_seq_dur_log', 'seq_seq_danger_intensity',
        # Sandhi days (absolute, not fractional)
        'sandhi_days_to_antar_end',
        # Yogini nature = direct Yogini identity = duration encoding
        'yg_md_nature', 'yg_ad_nature', 'yg_pd_nature',
        'yg_nature_sum',
    }
    # Pattern-based exclusions
    _DUR_PATTERNS = ['movement', 'ingress', 'sign_change']

    no_dur_cols = [c for c in feat_cols
                   if c not in _DUR_PROXY
                   and not any(pat in c for pat in _DUR_PATTERNS)
                   and 'duration' not in c and 'dur_' not in c
                   and '_is_min' not in c and '_is_max' not in c]
    # Also exclude nature-derived relative features
    no_dur_cols = [c for c in no_dur_cols
                   if not (c.startswith('yg_pd_nature_')
                           or c.startswith('yg_ad_nature_')
                           or c.startswith('yg_md_nature_'))]
    print(f"  No-dur features: {len(no_dur_cols)}")

    nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id', params)
    nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
    nd_topk, _ = eval_topk(df_va_f, nd_scores)

    # Bias audit: check top no-dur features for duration correlation
    dur_vals = df_va_f['duration_days'].values
    print(f"\n  Duration correlation audit (top no-dur features):")
    for feat in no_dur_cols[:10]:
        try:
            from numpy import corrcoef
            r = corrcoef(df_va_f[feat].values, dur_vals)[0, 1]
            if abs(r) > 0.1:
                print(f"    WARNING: {feat:40s} r={r:+.3f}")
        except Exception:
            pass

    # No-dur hyperparameter sweep
    params_nd_C = {**params, 'num_leaves': 63, 'max_depth': 8,
                   'min_child_samples': 25}
    params_nd_D = {**params, 'num_leaves': 80, 'max_depth': 10,
                   'min_child_samples': 30}
    params_xendcg = {**params, 'objective': 'rank_xendcg',
                     'num_leaves': 63, 'max_depth': 8,
                     'min_child_samples': 25}

    nd_configs = {'nd_C': params_nd_C, 'nd_D': params_nd_D,
                  'xendcg': params_xendcg}
    nd_results = {}
    nd_best_scores = None
    nd_best_name = None
    nd_best_t1 = 0
    for nd_name, nd_params in nd_configs.items():
        try:
            nd_m = train_seed_avg(no_dur_cols, df_tr_f, df_va_f,
                                  'group_id', nd_params)
            nd_s = predict_avg(nd_m, df_va_f[no_dur_cols].values)
            nd_tk, _ = eval_topk(df_va_f, nd_s)
            nd_results[nd_name] = nd_tk
            if nd_tk['top_1'] > nd_best_t1:
                nd_best_t1 = nd_tk['top_1']
                nd_best_scores = nd_s
                nd_best_name = nd_name
        except Exception as e:
            print(f"    [WARN] {nd_name}: {e}")

    # Binary classifier blend with weight sweep (using best no-dur scores)
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 63, 'max_depth': 8, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': float(max(1, int(gs_f.mean()) - 1)),
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }
    clf_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id',
                                 params_clf)
    clf_scores = predict_avg(clf_models, df_va_f[no_dur_cols].values)

    # Use best no-dur scores for blending
    best_nd = nd_best_scores if nd_best_scores is not None else nd_scores
    r_norm = (best_nd - best_nd.mean()) / max(best_nd.std(), 1e-6)
    c_norm = (clf_scores - clf_scores.mean()) / max(clf_scores.std(), 1e-6)

    blend_results = {}
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w * r_norm + (1.0 - w) * c_norm
        btk, _ = eval_topk(df_va_f, blend)
        blend_results[f'blend_{int(w*100)}/{int((1-w)*100)}'] = btk

    # Feature importance
    nd_imp = np.zeros(len(no_dur_cols))
    for m in nd_models:
        nd_imp += m.feature_importance(importance_type='gain')
    nd_imp /= len(nd_models)
    nd_imp_pct = nd_imp / max(nd_imp.sum(), 1e-6) * 100
    idx_sorted = np.argsort(nd_imp_pct)[::-1]
    print(f"\n  No-dur top 20 features:")
    for rank, i in enumerate(idx_sorted[:20]):
        print(f"    {rank+1:2d}. {no_dur_cols[i]:40s} {nd_imp_pct[i]:5.2f}%")

    # Feature selection: top-100, top-50 pruned models
    for top_n in [100, 50]:
        pruned_cols = [no_dur_cols[i] for i in idx_sorted[:top_n]
                       if i < len(no_dur_cols)]
        try:
            pr_m = train_seed_avg(pruned_cols, df_tr_f, df_va_f,
                                   'group_id', params_nd_D)
            pr_s = predict_avg(pr_m, df_va_f[pruned_cols].values)
            pr_tk, _ = eval_topk(df_va_f, pr_s)
            nd_results[f'top_{top_n}'] = pr_tk
        except Exception as e:
            print(f"    [WARN] top_{top_n}: {e}")

    # Results
    print(f"\n  Yogini PD (depth=3) Results:")
    print(f"    {'Method':>25s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'Lift':>6s}")
    print(f"    {'-'*55}")
    print(f"    {'Random':>25s} {rnd_t1:6.1%}  {rnd_t3:6.1%}  {rnd_t5:6.1%}  {'1.0x':>6s}")
    print(f"    {'Duration only':>25s} {dur_topk['top_1']:6.1%}  {dur_topk['top_3']:6.1%}  {dur_topk['top_5']:6.1%}  {dur_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"    {'No-dur model':>25s} {nd_topk['top_1']:6.1%}  {nd_topk['top_3']:6.1%}  {nd_topk['top_5']:6.1%}  {nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")

    for nd_name, nd_tk in nd_results.items():
        print(f"    {nd_name:>25s} {nd_tk['top_1']:6.1%}  {nd_tk['top_3']:6.1%}  {nd_tk['top_5']:6.1%}  {nd_tk['top_1']/max(rnd_t1,0.001):.1f}x")

    for bl_name, bl_tk in blend_results.items():
        print(f"    {bl_name:>25s} {bl_tk['top_1']:6.1%}  {bl_tk['top_3']:6.1%}  {bl_tk['top_5']:6.1%}  {bl_tk['top_1']/max(rnd_t1,0.001):.1f}x")

    print(f"    {'Full model':>25s} {model_topk['top_1']:6.1%}  {model_topk['top_3']:6.1%}  {model_topk['top_5']:6.1%}  {model_topk['top_1']/max(rnd_t1,0.001):.1f}x")

    if 'duration_days' in df_va_f.columns:
        print(f"\n  Avg duration: all={df_va_f['duration_days'].mean():.1f}d, "
              f"death={df_va_f[df_va_f['label']==1]['duration_days'].mean():.1f}d")
    print("=" * 75)


if __name__ == '__main__':
    main()
