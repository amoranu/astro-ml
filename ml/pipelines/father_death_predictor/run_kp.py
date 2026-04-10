"""KP (Krishnamurti Paddhati) father death prediction pipeline.

Uses Vimshottari Dasha periods (same as Parashari) but with KP-native
significator chain features. Cusp sub-lord analysis replaces maraka
tier classification.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_kp
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
DATA_DIR = PROJECT_ROOT / 'data' / 'kp'

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


def extract_features_for_candidate(cand, chart, asc, kp_ctx, gc_ctx,
                                    candidates, ci):
    """Extract KP + transit features for a Vimshottari PD candidate."""
    from .features.kp_features import extract_kp_features
    from .features.gochar_features import extract_gochar_features
    from .features.sade_sati_features import extract_sade_sati_features
    from .features.eclipse_features import extract_eclipse_features

    f = {}
    f['duration_days'] = cand['end_jd'] - cand['start_jd']

    # KP significator features
    try:
        kp_f = extract_kp_features(cand, kp_ctx, chart)
        f.update(kp_f)
    except Exception:
        pass

    # Shared transit features
    try:
        gc_f = extract_gochar_features(cand, gc_ctx)
        f.update(gc_f)
    except Exception:
        pass

    try:
        ss_f = extract_sade_sati_features(cand, chart)
        f.update(ss_f)
    except Exception:
        pass

    try:
        ec_f = extract_eclipse_features(cand, chart, asc)
        f.update(ec_f)
    except Exception:
        pass

    return f


def build_kp_dataset(records, name, start_index, n_augment=5):
    """Build KP PD dataset (Vimshottari periods + KP features)."""
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .features.dasha_window import construct_dasha_window
    from .features.kp_features import precompute_kp_context
    from .features.gochar_features import precompute_gochar_context
    from .features.five_new_features import add_relative_features

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_aug{n_augment}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    print(f"  Building {name} KP PD ({len(records)} charts x {n_augment} aug)...")

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

            # Vimshottari PD (same periods as Parashari)
            all_depth = compute_full_dasha(
                moon_long, birth_jd, max_depth=3, collect_all_depths=True)
            target_periods = [p for p in all_depth if p['depth'] == 3]

            # KP context (significator chains)
            kp_ctx = precompute_kp_context(chart, asc)
            gc_ctx = precompute_gochar_context(asc, chart)

            for aug in range(n_augment):
                seed = base_idx * 100 + aug
                idx = base_idx * 100 + aug

                candidates, correct_idx, _, _ = construct_dasha_window(
                    rec['father_death_date'], target_periods,
                    window_months=24, seed=seed)
                if correct_idx is None:
                    continue

                dur_stats.append(len(candidates))

                for ci, cand in enumerate(candidates):
                    f = extract_features_for_candidate(
                        cand, chart, asc, kp_ctx, gc_ctx, candidates, ci)
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
        KEY_REL = ['kp_pd_death_count', 'kp_chain_count',
                   'gc_sat_asp_str_sun', 'gc_sat_asp_str_h9',
                   'gc_mars_dist_sun', 'ec_rahu_dist_h9']
        avail = [c for c in KEY_REL if c in df.columns]
        for feat in avail:
            df[f'{feat}_rank'] = df.groupby('group_id')[feat].rank(
                ascending=False, method='min')
            g_mean = df.groupby('group_id')[feat].transform('mean')
            g_std = df.groupby('group_id')[feat].transform('std').clip(lower=1e-6)
            df[f'{feat}_zscore'] = (df[feat] - g_mean) / g_std

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
    print("KP FATHER DEATH PREDICTION (Vimshottari PD + KP features, 24mo)")
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

    df_tr = build_kp_dataset(train_recs, 'train', 0, n_augment=5)
    df_va = build_kp_dataset(val_recs, 'val', len(train_recs) * 100,
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
    dur_scores = df_va_f['duration_days'].values
    dur_topk, _ = eval_topk(df_va_f, dur_scores)

    # Full model
    print(f"\n  Training full model...")
    models = train_seed_avg(feat_cols, df_tr_f, df_va_f, 'group_id', params)
    model_scores = predict_avg(models, df_va_f[feat_cols].values)
    model_topk, _ = eval_topk(df_va_f, model_scores)

    # No-dur model
    _DUR_PROXY = {'duration_days'}
    no_dur_cols = [c for c in feat_cols
                   if 'duration' not in c and 'dur_' not in c
                   and c not in _DUR_PROXY
                   and 'movement' not in c]
    nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id', params)
    nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
    nd_topk, _ = eval_topk(df_va_f, nd_scores)

    # Binary blend
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': float(max(1, int(gs_f.mean()) - 1)),
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }
    clf_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id',
                                 params_clf)
    clf_scores = predict_avg(clf_models, df_va_f[no_dur_cols].values)
    r_norm = (nd_scores - nd_scores.mean()) / max(nd_scores.std(), 1e-6)
    c_norm = (clf_scores - clf_scores.mean()) / max(clf_scores.std(), 1e-6)
    blend = 0.5 * r_norm + 0.5 * c_norm
    blend_topk, _ = eval_topk(df_va_f, blend)

    # Feature importance
    nd_imp = np.zeros(len(no_dur_cols))
    for m in nd_models:
        nd_imp += m.feature_importance(importance_type='gain')
    nd_imp /= len(nd_models)
    nd_imp_pct = nd_imp / max(nd_imp.sum(), 1e-6) * 100
    idx_sorted = np.argsort(nd_imp_pct)[::-1]
    print(f"\n  No-dur top 15 features:")
    for rank, idx in enumerate(idx_sorted[:15]):
        print(f"    {rank+1:2d}. {no_dur_cols[idx]:40s} {nd_imp_pct[idx]:5.2f}%")

    # Results
    print(f"\n  KP PD Results:")
    print(f"    {'Method':>25s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'Lift':>6s}")
    print(f"    {'-'*55}")
    print(f"    {'Random':>25s} {rnd_t1:6.1%}  {rnd_t3:6.1%}  {rnd_t5:6.1%}  {'1.0x':>6s}")
    print(f"    {'Duration only':>25s} {dur_topk['top_1']:6.1%}  {dur_topk['top_3']:6.1%}  {dur_topk['top_5']:6.1%}  {dur_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"    {'No-dur model':>25s} {nd_topk['top_1']:6.1%}  {nd_topk['top_3']:6.1%}  {nd_topk['top_5']:6.1%}  {nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"    {'Blend 50/50':>25s} {blend_topk['top_1']:6.1%}  {blend_topk['top_3']:6.1%}  {blend_topk['top_5']:6.1%}  {blend_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"    {'Full model':>25s} {model_topk['top_1']:6.1%}  {model_topk['top_3']:6.1%}  {model_topk['top_5']:6.1%}  {model_topk['top_1']/max(rnd_t1,0.001):.1f}x")

    if 'duration_days' in df_va_f.columns:
        print(f"\n  Avg duration: all={df_va_f['duration_days'].mean():.1f}d, "
              f"death={df_va_f[df_va_f['label']==1]['duration_days'].mean():.1f}d")
    print("=" * 75)


if __name__ == '__main__':
    main()
