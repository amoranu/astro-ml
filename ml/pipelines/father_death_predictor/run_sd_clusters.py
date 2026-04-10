"""SD cluster prediction at ~30-day granularity (Parashari, EXP-023).

Goal: Push Parashari to ~1-month prediction units using SD-aggregated clusters.

Pipeline:
  1. Build SD-level (depth=4) dataset at 24mo with full modern feature set
  2. Group SDs into equal-count clusters (drop partial edge clusters)
  3. Aggregate features per cluster: max / mean only (no sums)
  4. Add cluster-level density features
  5. Train LambdaRank with strict bias guards + hyperparameter sweep + blend
  6. Auto-detect duration leakage (|corr(feat, cl_duration)| > 0.15)

Bias guards (CRITICAL):
  - Position bias: window construction is uniform (rng.randint(0, window_days)),
    no edge buffer, cluster_idx is NOT a feature, partial edge clusters dropped.
  - Duration bias: cl_duration excluded, all *_duration*, *_dur_*, *_movement*,
    planet-identity proxies excluded.
  - Auto leakage: features with |r(feat, cl_duration)| > 0.15 dropped from
    no-dur model.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_sd_clusters
"""

import json
import os
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'sd_clusters_v2'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

# Hard-coded duration proxies (known from EXP-007, EXP-010, EXP-012)
_DUR_PROXY_BASE = {
    'duration_days', 'seq_duration_days', 'seq_dur_vs_mean',
    'seq_dur_log', 'seq_danger_intensity',
    'id_pl_planet_idx', 'id_al_planet_idx',
    'id_lagna_planet_combo',
    'gc_mars_movement', 'gc_merc_movement', 'gc_venus_movement',
    'sandhi_days_to_antar_end', 'sandhi_days_to_maha_end',
}

# Position-bias features (must NEVER be included)
_POS_BIAS = {
    'cand_idx', 'cluster_idx',
    'seq_pos_norm', 'seq_third',
    'tier_pos_norm', 'tier_third',
}


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


def _is_proxy_name(c):
    """Hard-coded duration / position proxy by feature name."""
    if c in _DUR_PROXY_BASE or c in _POS_BIAS:
        return True
    if 'duration' in c or 'dur_' in c:
        return True
    if 'movement' in c or '_ingress' in c or 'sign_change' in c:
        return True
    if c.endswith('_pos_norm') or c.endswith('_third'):
        return True
    return False


def train_seed_avg(cols, df_tr, df_va, gcol, params, n_seeds=5,
                   num_boost_round=1500):
    models = []
    for i in range(n_seeds):
        p = {**params, 'seed': 42 + i * 17}
        td = lgb.Dataset(df_tr[cols].values, label=df_tr['label'].values,
                         group=df_tr.groupby(gcol, sort=False).size().values)
        vd = lgb.Dataset(df_va[cols].values, label=df_va['label'].values,
                         group=df_va.groupby(gcol, sort=False).size().values,
                         reference=td)
        m = lgb.train(p, td, num_boost_round=num_boost_round,
                      valid_sets=[vd],
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


# ── Cluster construction (bias-aware) ──────────────────────────────────

def build_clusters_strict(sd_df, n_per_cluster):
    """Group SDs into equal-count clusters, drop partial edge clusters.

    Bias guards:
      - Equal-count clusters → cl_n_sds = n_per_cluster constant
      - Drop charts where death falls in a partial cluster (uniform drop
        because death is uniformly placed within window)
      - Drop partial clusters entirely (don't aggregate <n_per SDs)
      - Aggregate max + mean only (no sum → avoids duration proxies)
      - Cluster-level features are count-normalised (frac, not count)

    Args:
        sd_df: SD-level DataFrame with 'group_id', 'cand_idx', 'label',
               'duration_days', and feature columns.
        n_per_cluster: number of consecutive SDs per cluster.

    Returns:
        Cluster-level DataFrame.
    """
    META = {'group_id', 'cand_idx', 'label', 'tier', 'danger_score',
            'duration_days'}

    # Identify SD-level feature columns and pre-filter proxies (saves time)
    all_cols = list(sd_df.columns)
    sd_feat_cols = []
    for c in all_cols:
        if c in META:
            continue
        if not isinstance(sd_df[c].iloc[0],
                          (int, float, np.integer, np.floating)):
            continue
        # Drop SD-level proxies BEFORE aggregation. We still keep
        # duration_days at SD level (used for cl_duration computation),
        # but it is excluded from feat_cols below.
        if _is_proxy_name(c):
            continue
        sd_feat_cols.append(c)

    all_rows = []
    for gid, grp in sd_df.groupby('group_id', sort=False):
        grp = grp.sort_values('cand_idx').reset_index(drop=True)
        n_sds = len(grp)
        n_full_clusters = n_sds // n_per_cluster
        if n_full_clusters < 2:
            # Need at least 2 clusters for ranking
            continue

        # Find which cluster (if any) contains the death
        death_cl_idx = None
        labels = grp['label'].values
        # Only consider full-cluster region
        full_region_end = n_full_clusters * n_per_cluster
        for ci in range(full_region_end):
            if labels[ci] == 1:
                death_cl_idx = ci // n_per_cluster
                break

        # Drop chart if death is outside full-cluster region (in partial)
        if death_cl_idx is None:
            continue

        for cl_idx in range(n_full_clusters):
            start = cl_idx * n_per_cluster
            end = start + n_per_cluster
            chunk = grp.iloc[start:end]

            row = {
                'group_id': gid,
                'cluster_idx_internal': cl_idx,  # NOT a feature, dropped later
                'label': 1 if cl_idx == death_cl_idx else 0,
            }

            # Cluster duration (for leakage detection only — never a feature)
            row['cl_duration'] = chunk['duration_days'].sum()

            # Aggregate SD features: max + mean only
            for c in sd_feat_cols:
                vals = chunk[c].values
                vmax = float(vals.max())
                vmean = float(vals.mean())
                row[f'{c}_max'] = vmax
                row[f'{c}_mean'] = vmean

            # Cluster-level density features (count-normalised, no raw counts)
            tiers = chunk['tier'].values if 'tier' in chunk.columns else []
            dscores = (chunk['danger_score'].values
                       if 'danger_score' in chunk.columns else [])
            n = len(chunk)

            if len(tiers) > 0:
                is_maraka = (tiers > 0) & (tiers <= 4)
                is_tier1 = tiers == 1
                row['cl_maraka_frac'] = float(is_maraka.sum()) / n
                row['cl_tier1_frac'] = float(is_tier1.sum()) / n
                row['cl_max_tier_inv'] = float(
                    1.0 / max(tiers[tiers > 0].min(), 1)
                ) if (tiers > 0).any() else 0.0

                # Longest consecutive maraka streak (count-normalised)
                longest = 0
                current = 0
                for v in is_maraka:
                    if v:
                        current += 1
                        longest = max(longest, current)
                    else:
                        current = 0
                row['cl_longest_streak_frac'] = float(longest) / n

                # First / last maraka position fraction
                first_pos = -1
                last_pos = -1
                for i, v in enumerate(is_maraka):
                    if v:
                        if first_pos == -1:
                            first_pos = i
                        last_pos = i
                row['cl_first_maraka_frac'] = (
                    (first_pos / max(n - 1, 1)) if first_pos >= 0 else -1.0)
                row['cl_last_maraka_frac'] = (
                    (last_pos / max(n - 1, 1)) if last_pos >= 0 else -1.0)
                row['cl_maraka_span_frac'] = (
                    ((last_pos - first_pos) / max(n - 1, 1))
                    if first_pos >= 0 else 0.0)

            if len(dscores) > 0:
                row['cl_max_danger'] = float(dscores.max())
                row['cl_mean_danger'] = float(dscores.mean())
                row['cl_danger_std'] = float(dscores.std())

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def add_cluster_relative_features(df, group_col='group_id'):
    """Within-window relative features at cluster level.

    For a curated set of strong cluster features, add rank, zscore, is_max.
    These help LambdaRank discriminate within a window.
    """
    KEY_CLUSTER = [
        'cl_max_danger', 'cl_mean_danger', 'cl_maraka_frac',
        'cl_tier1_frac', 'cl_longest_streak_frac',
        # Top transit features (aggregated)
        'ec_rahu_dist_h9_max', 'ec_rahu_dist_h9_mean',
        'ec_rahu_dist_sun_max', 'ec_rahu_dist_sun_mean',
        'gc_sat_asp_str_h9_max', 'gc_sat_asp_str_sun_max',
        'gc_jup_asp_str_h9_max', 'gc_mars_asp_str_sun_max',
        'gc_sat_dist_sun_max', 'gc_sat_dist_sun_mean',
        'gc_jup_dist_sun_max', 'gc_mars_dist_sun_max',
        'gc_moon_dist_h9_max',
    ]

    df = df.copy()
    available = [f for f in KEY_CLUSTER if f in df.columns]

    new_cols = {}
    for feat in available:
        new_cols[f'{feat}_rank'] = df.groupby(group_col)[feat].rank(
            ascending=False, method='min')
        g_mean = df.groupby(group_col)[feat].transform('mean')
        g_std = df.groupby(group_col)[feat].transform('std').clip(lower=1e-6)
        new_cols[f'{feat}_zscore'] = (df[feat] - g_mean) / g_std
        g_max = df.groupby(group_col)[feat].transform('max')
        new_cols[f'{feat}_is_max'] = (df[feat] == g_max).astype(float)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def get_cluster_dataset(records, name, start_index, n_per_cluster,
                        window_months=24, n_augment=1):
    """Build cluster dataset with caching."""
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_cl{n_per_cluster}_w{window_months}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    from .run_dasha_depth import build_depth_dataset
    # Distinct name to avoid colliding with 6mo SD cache used by run_dasha_depth
    sd_name = f'{name}_24mo'
    print(f"  Building SD source for {sd_name} ({window_months}mo)...")
    sd_df = build_depth_dataset(records, sd_name, start_index,
                                target_depth=4,
                                window_months=window_months,
                                n_augment=n_augment)
    print(f"  Building clusters: {n_per_cluster} SDs/cluster...")
    cl_df = build_clusters_strict(sd_df, n_per_cluster)
    print(f"  Adding cluster-relative features...")
    cl_df = add_cluster_relative_features(cl_df)
    cl_df.to_parquet(cache_f)
    return cl_df


def get_feat_cols(df, exclude_keys):
    """Numeric feature columns minus meta + proxies."""
    feat_cols = []
    for c in df.columns:
        if c in exclude_keys:
            continue
        if not isinstance(df[c].iloc[0],
                          (int, float, np.integer, np.floating)):
            continue
        if _is_proxy_name(c):
            continue
        feat_cols.append(c)
    return feat_cols


def auto_drop_leaks(df, feat_cols, target_col='cl_duration', threshold=0.15):
    """Auto-drop features with |corr(feat, cl_duration)| > threshold."""
    leaks = []
    if target_col not in df.columns:
        return feat_cols, leaks
    target = df[target_col].values
    for c in feat_cols:
        v = df[c].values
        if v.std() < 1e-9:
            continue
        try:
            r = np.corrcoef(v, target)[0, 1]
            if abs(r) > threshold:
                leaks.append((c, r))
        except Exception:
            continue
    leaks.sort(key=lambda x: -abs(x[1]))
    leak_set = {c for c, _ in leaks}
    clean_cols = [c for c in feat_cols if c not in leak_set]
    return clean_cols, leaks


def main():
    print("=" * 75)
    print("EXP-023: SD CLUSTER PREDICTION (~30 days, BIAS-CLEAN)")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    # Hyperparameter configurations from EXP-012
    params_base = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }
    params_xendcg = {**params_base, 'objective': 'rank_xendcg'}
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': 12.0,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    # Cluster size sweep — focus on ~30-day target
    cluster_sizes = [4, 5, 6]

    final_results = {}

    for n_per in cluster_sizes:
        print(f"\n{'='*75}")
        print(f"CLUSTER SIZE: {n_per} SDs/cluster")
        print(f"{'='*75}")

        cl_tr = get_cluster_dataset(train_recs, 'train', 0, n_per,
                                    window_months=24, n_augment=1)
        cl_va = get_cluster_dataset(val_recs, 'val',
                                    len(train_recs) * 100, n_per,
                                    window_months=24, n_augment=1)

        if len(cl_tr) == 0 or len(cl_va) == 0:
            print(f"  SKIP: empty dataset")
            continue

        # Filter to groups with >= 2 clusters
        cl_va = cl_va[cl_va.groupby('group_id')['group_id'].transform('count') >= 2]
        cl_tr = cl_tr[cl_tr.groupby('group_id')['group_id'].transform('count') >= 2]

        n_va = cl_va['group_id'].nunique()
        n_tr = cl_tr['group_id'].nunique()
        gs = cl_va.groupby('group_id').size()
        mean_dur = cl_va['cl_duration'].mean() if 'cl_duration' in cl_va.columns else 0
        print(f"  Train groups: {n_tr}, Val groups: {n_va}")
        print(f"  Clusters/window: mean={gs.mean():.1f}, "
              f"median={gs.median():.0f}")
        print(f"  Days/cluster: mean={mean_dur:.1f}")

        if n_va < 50:
            print(f"  SKIP: too few val groups")
            continue

        # Random baseline
        rnd_t1 = float((1.0 / gs.values).mean())
        rnd_t3 = float(np.minimum(3.0 / gs.values, 1.0).mean())
        rnd_t5 = float(np.minimum(5.0 / gs.values, 1.0).mean())
        print(f"  Random baseline: T1={rnd_t1:.1%} T3={rnd_t3:.1%} T5={rnd_t5:.1%}")

        # Feature columns
        EXCLUDE = {'group_id', 'cluster_idx_internal', 'label',
                   'cl_duration'}
        feat_cols = get_feat_cols(cl_va, EXCLUDE)
        feat_cols = [c for c in feat_cols if c in cl_tr.columns]

        # Auto-detect remaining duration leaks
        nodur_cols, leaks = auto_drop_leaks(cl_tr, feat_cols,
                                            target_col='cl_duration',
                                            threshold=0.15)
        print(f"  Features (raw):   {len(feat_cols)}")
        print(f"  Features (no-dur): {len(nodur_cols)} "
              f"(dropped {len(leaks)} leaks)")
        if leaks[:5]:
            print(f"  Top leaks:")
            for c, r in leaks[:5]:
                print(f"    {c:50s} r={r:+.3f}")

        # Duration-only baseline (uses cl_duration directly)
        dur_scores = cl_va['cl_duration'].values
        dur_topk, _ = eval_topk(cl_va, dur_scores)
        print(f"  Dur-only:  T1={dur_topk['top_1']:.1%} "
              f"T3={dur_topk['top_3']:.1%} T5={dur_topk['top_5']:.1%}")

        # ── Full model (all features minus structural proxies) ─────
        print(f"  Training full model ({len(feat_cols)} feats)...")
        full_models = train_seed_avg(feat_cols, cl_tr, cl_va, 'group_id',
                                     params_base)
        full_scores = predict_avg(full_models, cl_va[feat_cols].values)
        full_topk, _ = eval_topk(cl_va, full_scores)
        print(f"  Full:      T1={full_topk['top_1']:.1%} "
              f"T3={full_topk['top_3']:.1%} T5={full_topk['top_5']:.1%} "
              f"({full_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")

        # ── No-dur model (CRITICAL test) ──────────────────────────
        print(f"  Training no-dur model ({len(nodur_cols)} feats)...")
        nd_models = train_seed_avg(nodur_cols, cl_tr, cl_va, 'group_id',
                                   params_base)
        nd_scores = predict_avg(nd_models, cl_va[nodur_cols].values)
        nd_topk, _ = eval_topk(cl_va, nd_scores)
        print(f"  No-dur:    T1={nd_topk['top_1']:.1%} "
              f"T3={nd_topk['top_3']:.1%} T5={nd_topk['top_5']:.1%} "
              f"({nd_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")

        # ── Hyperparameter sweep on no-dur (only for primary cluster size) ──
        if n_per == 5:
            nd_variants = {
                'nd_A_colsamp09': {**params_base, 'colsample_bytree': 0.9},
                'nd_B_lowreg': {**params_base, 'reg_lambda': 0.5,
                                'reg_alpha': 0.1},
                'nd_C_leaves64': {**params_base, 'num_leaves': 64,
                                  'max_depth': 8, 'min_child_samples': 20},
            }
            print(f"\n  No-dur hyperparameter sweep:")
            best_v_topk = nd_topk
            best_v_scores = nd_scores
            best_v_name = 'nd_base'
            for vname, vparams in nd_variants.items():
                v_models = train_seed_avg(nodur_cols, cl_tr, cl_va,
                                          'group_id', vparams)
                v_scores = predict_avg(v_models, cl_va[nodur_cols].values)
                v_topk, _ = eval_topk(cl_va, v_scores)
                print(f"    {vname:>20s}: T1={v_topk['top_1']:.1%} "
                      f"T3={v_topk['top_3']:.1%} T5={v_topk['top_5']:.1%} "
                      f"({v_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")
                if v_topk['top_1'] > best_v_topk['top_1']:
                    best_v_topk = v_topk
                    best_v_scores = v_scores
                    best_v_name = vname

            # ── rank_xendcg ──
            print(f"  Training xendcg no-dur...")
            x_models = train_seed_avg(nodur_cols, cl_tr, cl_va, 'group_id',
                                      params_xendcg)
            x_scores = predict_avg(x_models, cl_va[nodur_cols].values)
            x_topk, _ = eval_topk(cl_va, x_scores)
            print(f"    {'xendcg_nodur':>20s}: T1={x_topk['top_1']:.1%} "
                  f"T3={x_topk['top_3']:.1%} T5={x_topk['top_5']:.1%}")

            # ── Binary classifier ──
            print(f"  Training binary classifier...")
            clf_models = []
            for i in range(5):
                p = {**params_clf, 'seed': 42 + i * 17}
                m = lgb.train(p, lgb.Dataset(cl_tr[nodur_cols].values,
                                              label=cl_tr['label'].values),
                              num_boost_round=500)
                clf_models.append(m)
            clf_scores = np.mean([m.predict(cl_va[nodur_cols].values)
                                  for m in clf_models], axis=0)
            clf_topk, _ = eval_topk(cl_va, clf_scores)
            print(f"    {'clf':>20s}: T1={clf_topk['top_1']:.1%} "
                  f"T3={clf_topk['top_3']:.1%} T5={clf_topk['top_5']:.1%}")

            # ── Blend ranker + classifier ──
            def _norm(s):
                r = s.max() - s.min()
                return (s - s.min()) / r if r > 1e-10 else np.zeros_like(s)
            print(f"  Blend sweep (best ranker × clf):")
            best_v_norm = _norm(best_v_scores)
            clf_norm = _norm(clf_scores)
            best_blend_topk = nd_topk
            best_blend_name = 'no-blend'
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                bs = w * best_v_norm + (1 - w) * clf_norm
                btopk, _ = eval_topk(cl_va, bs)
                tag = f'blend_{int(w*100)}/{int((1-w)*100)}'
                print(f"    {tag:>20s}: T1={btopk['top_1']:.1%} "
                      f"T3={btopk['top_3']:.1%} T5={btopk['top_5']:.1%} "
                      f"({btopk['top_1']/max(rnd_t1,1e-6):.1f}x)")
                if btopk['top_1'] > best_blend_topk['top_1']:
                    best_blend_topk = btopk
                    best_blend_name = tag

            # ── Feature importance ──
            print(f"\n  Top 20 no-dur features (by gain):")
            imp = np.zeros(len(nodur_cols))
            for m in nd_models:
                imp += m.feature_importance(importance_type='gain')
            imp /= len(nd_models)
            imp_pct = imp / max(imp.sum(), 1e-9) * 100
            idx_sort = np.argsort(imp_pct)[::-1]
            for rank, i in enumerate(idx_sort[:20]):
                print(f"    {rank+1:2d}. {nodur_cols[i]:50s} "
                      f"{imp_pct[i]:5.2f}%")

            # ── Bias verification ──
            print(f"\n  BIAS VERIFICATION:")
            full_minus_nodur = full_topk['top_1'] - nd_topk['top_1']
            if full_minus_nodur > 0.03:
                print(f"  WARNING: full ({full_topk['top_1']:.1%}) >> "
                      f"no-dur ({nd_topk['top_1']:.1%}) "
                      f"-- possible duration leak")
            else:
                print(f"  OK: full ({full_topk['top_1']:.1%}) ~= "
                      f"no-dur ({nd_topk['top_1']:.1%}) -- bias-clean")

            # Check top features for proxy patterns
            top_feats = [nodur_cols[i] for i in idx_sort[:20]]
            suspect = [f for f in top_feats
                       if any(p in f.lower() for p in
                              ['duration', 'movement', 'dur_', 'ingress'])]
            if suspect:
                print(f"  WARNING: suspected proxies in top-20: {suspect}")
            else:
                print(f"  OK: no duration/movement proxies in top-20")

            final_results[n_per] = {
                'random': rnd_t1,
                'days': mean_dur,
                'cands': float(gs.mean()),
                'dur_only': dur_topk,
                'full': full_topk,
                'nodur': nd_topk,
                'best_variant': (best_v_name, best_v_topk),
                'best_blend': (best_blend_name, best_blend_topk),
                'n_features': len(nodur_cols),
            }
        else:
            final_results[n_per] = {
                'random': rnd_t1,
                'days': mean_dur,
                'cands': float(gs.mean()),
                'dur_only': dur_topk,
                'full': full_topk,
                'nodur': nd_topk,
                'n_features': len(nodur_cols),
            }

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"EXP-023 FINAL SUMMARY")
    print(f"{'='*75}")
    print(f"\n  {'ClSize':>6s} {'Days':>6s} {'Cands':>6s} {'Rand':>6s} "
          f"{'DurOnly':>8s} {'Full':>8s} {'NoDur':>8s} {'Lift':>6s}")
    print("  " + "-" * 70)
    for n_per, r in final_results.items():
        print(f"  {n_per:>6d} {r['days']:>5.0f}d "
              f"{r['cands']:>6.1f} {r['random']:>5.1%} "
              f"{r['dur_only']['top_1']:>7.1%} "
              f"{r['full']['top_1']:>7.1%} "
              f"{r['nodur']['top_1']:>7.1%} "
              f"{r['nodur']['top_1']/max(r['random'],1e-6):>5.1f}x")

    if 5 in final_results and 'best_blend' in final_results[5]:
        bn, bt = final_results[5]['best_blend']
        bv_name, bv_topk = final_results[5]['best_variant']
        print(f"\n  Best for cl=5 (~30 days):")
        print(f"    Best variant ({bv_name}): "
              f"T1={bv_topk['top_1']:.1%} T3={bv_topk['top_3']:.1%} "
              f"T5={bv_topk['top_5']:.1%}")
        print(f"    Best blend ({bn}): "
              f"T1={bt['top_1']:.1%} T3={bt['top_3']:.1%} "
              f"T5={bt['top_5']:.1%}")
    print()


if __name__ == '__main__':
    main()
