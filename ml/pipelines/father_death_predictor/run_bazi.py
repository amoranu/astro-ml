"""BaZi (Four Pillars) father death prediction pipeline.

Uses monthly pillars (solar term boundaries, ~30 days each) as prediction
units. Features: stem/branch clashes, Ten God analysis, father star status.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_bazi
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'bazi'

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


def _compute_natal_bazi(birth_date_str, birth_time_str):
    """Simplified BaZi natal pillar computation.

    Returns (year_stem, year_branch, month_stem, month_branch,
             day_stem, day_branch).
    """
    from .astro_engine.bazi_monthly import (
        compute_year_stem, _FIVE_TIGERS, _find_solar_longitude_jd,
        _JIE_QI_LONGS)
    from .astro_engine.ephemeris import compute_jd

    dt = datetime.strptime(birth_date_str, '%Y-%m-%d')
    year = dt.year
    birth_jd = compute_jd(birth_date_str, birth_time_str)

    # Year pillar (changes at Lichun)
    lichun_jd = _find_solar_longitude_jd(315, birth_jd - 40)
    if birth_jd < lichun_jd:
        bazi_year = year - 1
    else:
        bazi_year = year
    year_stem = compute_year_stem(bazi_year)
    year_branch = (bazi_year - 4) % 12

    # Month pillar (find current solar month)
    # Check which Jie Qi boundary we're in
    month_num = 0
    for mn in range(12):
        jq_jd = _find_solar_longitude_jd(_JIE_QI_LONGS[mn], birth_jd - 20)
        next_jq_jd = _find_solar_longitude_jd(
            _JIE_QI_LONGS[(mn + 1) % 12], jq_jd + 25)
        if jq_jd <= birth_jd < next_jq_jd:
            month_num = mn
            break

    m1_stem = _FIVE_TIGERS[year_stem % 5]
    month_stem = (m1_stem + month_num) % 10
    month_branch = (2 + month_num) % 12  # Yin=2 for month 1

    # Day pillar (Julian Day Number based)
    # Day stem cycles every 10 days from a known reference
    # Reference: Jan 1, 2000 (JD 2451545) = Jia Zi (stem=0, branch=0)
    # Actually JD 2451550.5 = Jan 6, 2000 = Geng Wu (stem=6, branch=6)
    # Use: (JD - reference) mod 60 for sexagenary cycle
    ref_jd = 2451550.5
    ref_stem = 6  # Geng
    ref_branch = 6  # Wu
    diff = int(birth_jd - ref_jd + 0.5)
    day_stem = (ref_stem + diff) % 10
    day_branch = (ref_branch + diff) % 12

    return year_stem, year_branch, month_stem, month_branch, day_stem, day_branch


def build_bazi_dataset(records, name, start_index, n_augment=5):
    """Build BaZi monthly pillar dataset."""
    import random
    from .astro_engine.ephemeris import compute_jd
    from .astro_engine.bazi_monthly import compute_monthly_pillars, compute_year_stem
    from .features.bazi_features import precompute_bazi_context, extract_bazi_features
    from .features.gochar_features import (
        precompute_gochar_context, extract_gochar_features)
    from .features.sade_sati_features import extract_sade_sati_features
    from .features.eclipse_features import extract_eclipse_features
    from .astro_engine.ephemeris import compute_chart

    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_aug{n_augment}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    print(f"  Building {name} BaZi monthly ({len(records)} charts x {n_augment} aug)...")

    all_rows = []
    n_ok = 0
    errors = 0
    t0 = time.time()
    dur_stats = []

    AVG_DAYS_PER_MONTH = 30.44

    for i, rec in enumerate(records):
        base_idx = start_index + i
        try:
            chart, asc = compute_chart(
                rec['birth_date'], rec['birth_time'],
                rec['lat'], rec['lon'])
            birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])

            # Natal BaZi pillars
            ys, yb, ms, mb, ds, db = _compute_natal_bazi(
                rec['birth_date'], rec['birth_time'])
            bazi_ctx = precompute_bazi_context(ds, db, ys, yb, ms, mb)
            gc_ctx = precompute_gochar_context(asc, chart)

            # Death date
            death_jd = compute_jd(rec['father_death_date'], '12:00')

            for aug in range(n_augment):
                rng = random.Random(base_idx * 100 + aug)
                idx = base_idx * 100 + aug

                # Random window around death
                window_days = 24 * AVG_DAYS_PER_MONTH
                offset = rng.randint(0, int(window_days))
                win_start = death_jd - offset
                win_end = win_start + window_days

                # Compute monthly pillars in window
                year_stem = compute_year_stem(
                    int(2000 + (win_start - 2451545.0) / 365.25))
                candidates = compute_monthly_pillars(year_stem, win_start, win_end)

                if len(candidates) < 2:
                    continue

                # Find correct candidate (contains death)
                correct_idx = None
                for ci, cand in enumerate(candidates):
                    if cand['start_jd'] <= death_jd < cand['end_jd']:
                        correct_idx = ci
                        break

                if correct_idx is None:
                    continue

                dur_stats.append(len(candidates))

                for ci, cand in enumerate(candidates):
                    f = {}
                    f['duration_days'] = cand['duration_days']

                    # BaZi features
                    try:
                        bz_f = extract_bazi_features(cand, bazi_ctx)
                        f.update(bz_f)
                    except Exception:
                        pass

                    # Transit features at midpoint
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
            if errors <= 5:
                print(f"    [WARN] {base_idx}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(records)} ({(i+1)/elapsed:.0f}/s)")

    df = pd.DataFrame(all_rows)
    if len(df) > 0 and 'group_id' in df.columns:
        KEY_REL = ['gc_sat_asp_str_sun', 'gc_mars_dist_sun', 'ec_rahu_dist_h9']
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
        ds_arr = np.array(dur_stats)
        print(f"  Candidates/window: mean={ds_arr.mean():.1f}, "
              f"median={np.median(ds_arr):.0f}, "
              f"min={ds_arr.min()}, max={ds_arr.max()}")
    return df


def main():
    print("=" * 75)
    print("BAZI MONTHLY PILLAR FATHER DEATH PREDICTION (24mo window)")
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

    df_tr = build_bazi_dataset(train_recs, 'train', 0, n_augment=5)
    df_va = build_bazi_dataset(val_recs, 'val', len(train_recs) * 100,
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

    dur_scores = df_va_f['duration_days'].values
    dur_topk, _ = eval_topk(df_va_f, dur_scores)

    print(f"\n  Training full model...")
    models = train_seed_avg(feat_cols, df_tr_f, df_va_f, 'group_id', params)
    model_scores = predict_avg(models, df_va_f[feat_cols].values)
    model_topk, _ = eval_topk(df_va_f, model_scores)

    _DUR_PROXY = {'duration_days', 'bz_month_num'}
    no_dur_cols = [c for c in feat_cols
                   if 'duration' not in c and 'dur_' not in c
                   and c not in _DUR_PROXY
                   and 'movement' not in c]
    nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id', params)
    nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
    nd_topk, _ = eval_topk(df_va_f, nd_scores)

    # Blend
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

    print(f"\n  BaZi Monthly Results:")
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
