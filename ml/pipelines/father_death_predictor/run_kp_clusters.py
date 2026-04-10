"""KP cluster prediction at ~30 days (EXP-024).

KP-native pipeline using Vimshottari Sookshma (depth=4) as base time unit,
grouped into ~30-day clusters of N consecutive sookshmas.

PURE KP — no Parashari maraka tier classification, no D12, no navamsha,
no Yogini/Jaimini/BaZi mixing. Only:
  - KP significator chains (4-level)
  - Cusp Sub-Lord theory
  - Ruling Planets (Vara, Moon Star, Moon Sub)
  - Star Lord / Sub Lord of dasha lords
  - Transit Sub-Lord positions (Saturn, Jupiter, Rahu)
  - Universal transit positions (eclipse axis, gochar, Sade Sati)

Bias guards (CRITICAL):
  - Equal-count clusters with partial-cluster filtering
  - Aggregate max + mean only (no sum)
  - No cluster_idx, no position features
  - Auto-detect duration leakage (|r| > 0.15 with cl_duration)
  - Hard-coded duration proxies excluded

Parallel processing:
  - SD-level feature extraction parallelized via multiprocessing.Pool
  - 16 worker processes (half of 32 cores)
  - Each process gets independent Swiss Ephemeris state

GPU:
  - LightGBM device='gpu' for training

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_kp_clusters
"""

import json
import os
import math
import time
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'kp_clusters'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

# Hard-coded duration / position proxy exclusions
_DUR_PROXY_BASE = {
    'duration_days', 'seq_duration_days', 'seq_dur_vs_mean',
    'seq_dur_log', 'seq_danger_intensity',
    'gc_mars_movement', 'gc_merc_movement', 'gc_venus_movement',
    'sandhi_days_to_antar_end', 'sandhi_days_to_maha_end',
}
_POS_BIAS = {
    'cand_idx', 'cluster_idx',
    'seq_pos_norm', 'seq_third',
}

# Parashari-flavored features (drop in pure-KP mode):
# - Mrityu Bhaga (BPHS death degrees) — Parashari/Lal Kitab
# - Sade Sati — primarily a Parashari/Vedic remedial concept
# - Multipoint sookshma density — Parashari sookshma
# - Eclipse axis features — used by all but rooted in Parashari Pitru Karaka
_PARASHARI_FLAVORED_PREFIXES = (
    'ss_',         # sade sati
    'mp_',         # multipoint sookshma
    'ec_',         # eclipse axis
)
_PARASHARI_FLAVORED_PATTERNS = (
    'mrityu',      # Mrityu Bhaga
    'kakshya',     # Kakshya 3.75-degree sub-divisions (Parashari)
    'jup_bav',     # Ashtakavarga (Parashari)
    'transit_uchcha',  # transit dignity (used by all but Parashari-rooted)
)


def _is_parashari_flavored(c):
    if any(c.startswith(p) for p in _PARASHARI_FLAVORED_PREFIXES):
        return True
    return any(p in c for p in _PARASHARI_FLAVORED_PATTERNS)


def _is_proxy_name(c):
    if c in _DUR_PROXY_BASE or c in _POS_BIAS:
        return True
    if 'duration' in c or 'dur_' in c:
        return True
    if 'movement' in c or '_ingress' in c or 'sign_change' in c:
        return True
    if c.endswith('_pos_norm') or c.endswith('_third'):
        return True
    return False


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


# ── Per-chart feature extraction (worker function) ───────────────────

def _extract_chart_rows(args):
    """Extract all SD rows for one (record, augmentation) pair.

    Returns a list of dicts. Used by smoke tests and the batch worker.
    """
    import swisseph as swe
    swe.set_sid_mode(swe.SIDM_LAHIRI)

    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .features.dasha_window import construct_dasha_window
    from .features.kp_native import (
        precompute_kp_native_context, extract_kp_native_features)
    from .features.gochar_features import (
        precompute_gochar_context, extract_gochar_features)
    from .features.eclipse_features import extract_eclipse_features
    from .features.sade_sati_features import extract_sade_sati_features
    from .features.combustion_features import extract_combustion_features
    from .features.retrograde_features import extract_retrograde_features
    from .features.nakshatra_features import extract_nakshatra_features
    from .features.lever2_features import (
        extract_multipoint_features, extract_jup_bav_features,
        precompute_jup_bav)

    rec_idx, rec, base_idx, n_augment, window_months = args

    try:
        chart, asc = compute_chart(
            rec['birth_date'], rec['birth_time'],
            rec['lat'], rec['lon'])
        birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])
        moon_long = chart['Moon']['longitude']

        # Vimshottari Sookshma (depth=4)
        all_d4 = compute_full_dasha(
            moon_long, birth_jd, max_depth=4, collect_all_depths=True)
        target_periods = [p for p in all_d4 if p['depth'] == 4]

        # KP-native context
        kp_ctx = precompute_kp_native_context(chart, asc, birth_jd)
        gc_ctx = precompute_gochar_context(asc, chart)
        jup_bav = precompute_jup_bav(chart, asc)

        # Father marakas needed by nakshatra_features (universal Vedic)
        from .features.hierarchy_features import precompute_hierarchy_context
        hi_ctx = precompute_hierarchy_context(asc)
        father_mk = hi_ctx['father_marakas']

        rows = []
        for aug in range(n_augment):
            seed = base_idx * 100 + aug
            idx = base_idx * 100 + aug

            candidates, correct_idx, _, _ = construct_dasha_window(
                rec['father_death_date'], target_periods,
                window_months=window_months, seed=seed)
            if correct_idx is None:
                continue

            for ci, cand in enumerate(candidates):
                f = {}
                f['duration_days'] = cand['end_jd'] - cand['start_jd']

                # KP-native (50 features)
                try:
                    f.update(extract_kp_native_features(cand, kp_ctx, chart))
                except Exception:
                    pass

                # Universal transit (Rahu, gochar, Saturn, Jupiter)
                try:
                    gc_f = extract_gochar_features(cand, gc_ctx)
                    f.update(gc_f)
                except Exception:
                    pass
                try:
                    ec_f = extract_eclipse_features(cand, chart, asc)
                    f.update(ec_f)
                except Exception:
                    pass
                try:
                    ss_f = extract_sade_sati_features(cand, chart)
                    f.update(ss_f)
                except Exception:
                    pass
                try:
                    cb_f = extract_combustion_features(cand, chart)
                    f.update(cb_f)
                except Exception:
                    pass
                try:
                    rt_f = extract_retrograde_features(cand, chart)
                    f.update(rt_f)
                except Exception:
                    pass
                try:
                    nk_f = extract_nakshatra_features(cand, chart, father_mk)
                    f.update(nk_f)
                except Exception:
                    pass
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

                row = {
                    'group_id': idx,
                    'cand_idx': ci,
                    'label': 1 if ci == correct_idx else 0,
                    **f,
                }
                rows.append(row)

        return rows
    except Exception:
        return []


def _extract_batch_to_parquet(args):
    """Worker function: extract a batch of charts and write to a temp parquet.

    Returns the path to the written parquet (or None if empty). This avoids
    pickling large dict lists across the process boundary.
    """
    batch_idx, batch_args, tmp_dir = args
    rows = []
    for chart_args in batch_args:
        rows.extend(_extract_chart_rows(chart_args))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    out_path = os.path.join(tmp_dir, f'batch_{batch_idx:05d}.parquet')
    df.to_parquet(out_path)
    return out_path


def build_sd_dataset_parallel(records, name, start_index, n_augment,
                              window_months=24, n_workers=12,
                              batch_size=40):
    """Build SD-level dataset using multiprocessing.Pool with temp files.

    Each worker processes a batch of charts and writes its rows directly
    to a temp parquet file, returning only the file path. Main process
    concatenates all parquet files at the end.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_sd_aug{n_augment}.parquet'
    if cache_f.exists():
        print(f"  Loading cached {cache_f.name}")
        return pd.read_parquet(cache_f)

    tmp_dir = str(DATA_DIR / f'_tmp_{name}_{int(time.time())}')
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"  Building {name} KP-Sookshma ({len(records)} charts x "
          f"{n_augment} aug, {n_workers} workers, batch={batch_size})...")

    chart_args_all = [
        (i, rec, start_index + i, n_augment, window_months)
        for i, rec in enumerate(records)
    ]

    # Group into batches
    batches = [
        (i, chart_args_all[i * batch_size:(i + 1) * batch_size], tmp_dir)
        for i in range(math.ceil(len(chart_args_all) / batch_size))
    ]

    t0 = time.time()
    parquet_paths = []
    n_batches_done = 0
    with mp.Pool(processes=n_workers) as pool:
        for path in pool.imap_unordered(_extract_batch_to_parquet, batches):
            if path is not None:
                parquet_paths.append(path)
            n_batches_done += 1
            if n_batches_done % 5 == 0:
                elapsed = time.time() - t0
                charts_done = n_batches_done * batch_size
                rate = charts_done / max(elapsed, 1e-3)
                print(f"    {charts_done}/{len(records)} "
                      f"({rate:.1f}/s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Built {len(parquet_paths)} batches in {elapsed:.0f}s. "
          f"Concatenating...")

    # Concat all temp parquets - read and append each
    chunks = []
    for path in parquet_paths:
        chunks.append(pd.read_parquet(path))
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    del chunks

    df.to_parquet(cache_f)
    print(f"  Cached: {cache_f.name} ({len(df)} rows, {len(df.columns)} cols)")

    # Cleanup temp files
    for path in parquet_paths:
        try:
            os.remove(path)
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return df


# ── Cluster construction (vectorized, bias-aware) ──────────────────────

def build_clusters_strict(sd_df, n_per_cluster):
    """Equal-count clusters via vectorized pandas groupby.

    Bias guards:
      - Equal-count clusters (n_per_cluster SDs each)
      - Drop incomplete edge clusters (groups where fewer than n_per_cluster
        SDs would form the last cluster)
      - Drop charts where death falls in a partial cluster
      - max + mean aggregation only (no sum)
      - cl_duration computed as sum of SD durations (used only for
        leakage detection — never a feature)
    """
    print(f"  [vectorized] Sorting {len(sd_df)} SD rows...")
    df = sd_df.sort_values(['group_id', 'cand_idx']).reset_index(drop=True)

    # Compute cluster_idx within each group
    print(f"  [vectorized] Computing within-group positions...")
    df['_pos'] = df.groupby('group_id', sort=False).cumcount()
    df['_cl_idx'] = df['_pos'] // n_per_cluster

    # Group sizes — drop SDs that would land in incomplete final clusters
    print(f"  [vectorized] Filtering incomplete edge clusters...")
    g_size = df.groupby('group_id', sort=False)['_pos'].transform('size')
    n_full = (g_size // n_per_cluster) * n_per_cluster
    df = df[df['_pos'] < n_full].copy()

    # Drop groups with < 2 full clusters (need at least 2 to rank)
    g_full_clusters = df.groupby('group_id', sort=False)['_cl_idx'].transform('nunique')
    df = df[g_full_clusters >= 2].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Identify SD-level feature columns to aggregate (skip meta + proxies)
    META = {'group_id', 'cand_idx', 'label', 'duration_days', '_pos', '_cl_idx'}
    sd_feat_cols = []
    for c in df.columns:
        if c in META:
            continue
        if not isinstance(df[c].iloc[0],
                          (int, float, np.integer, np.floating)):
            continue
        if _is_proxy_name(c):
            continue
        sd_feat_cols.append(c)

    print(f"  [vectorized] Aggregating {len(sd_feat_cols)} features over "
          f"{df['group_id'].nunique()} groups...")

    # Build a unique cluster key for groupby
    df['_cluster_key'] = (
        df['group_id'].astype(np.int64) * 100_000 + df['_cl_idx'].astype(np.int64))

    # Vectorized aggregation: max + mean for all features
    agg_dict = {c: ['max', 'mean'] for c in sd_feat_cols}
    agg_dict['duration_days'] = 'sum'  # for cl_duration
    agg_dict['label'] = 'max'  # cluster label = any SD has label 1

    # Carry group_id and _cl_idx through (use first since constant per cluster)
    agg_dict['group_id'] = 'first'
    agg_dict['_cl_idx'] = 'first'

    print(f"  [vectorized] Running pandas groupby.agg...")
    agg_df = df.groupby('_cluster_key', sort=False).agg(agg_dict)

    # Flatten multi-level columns
    print(f"  [vectorized] Flattening columns...")
    new_cols = []
    for col in agg_df.columns:
        if isinstance(col, tuple):
            base, stat = col
            if base == 'duration_days' and stat == 'sum':
                new_cols.append('cl_duration')
            elif base == 'label' and stat == 'max':
                new_cols.append('label')
            elif base == 'group_id' and stat == 'first':
                new_cols.append('group_id')
            elif base == '_cl_idx' and stat == 'first':
                new_cols.append('cluster_idx_internal')
            else:
                new_cols.append(f'{base}_{stat}')
        else:
            new_cols.append(col)
    agg_df.columns = new_cols
    agg_df = agg_df.reset_index(drop=True)

    # Cluster-level KP density features (vectorized over the SD frame)
    print(f"  [vectorized] Building cluster-level KP density features...")
    if 'kp_4chain_count' in df.columns:
        kp_chain = df.groupby('_cluster_key', sort=False)['kp_4chain_count']
        agg_df['cl_kp_max_chain'] = kp_chain.max().values
        agg_df['cl_kp_mean_chain'] = kp_chain.mean().values
        agg_df['cl_kp_3plus_frac'] = (
            df.assign(_x=(df['kp_4chain_count'] >= 3).astype(float))
              .groupby('_cluster_key', sort=False)['_x'].mean().values)
    if 'kp_composite_danger' in df.columns:
        kp_danger = df.groupby('_cluster_key', sort=False)['kp_composite_danger']
        agg_df['cl_kp_max_danger'] = kp_danger.max().values
        agg_df['cl_kp_mean_danger'] = kp_danger.mean().values
        agg_df['cl_kp_danger_std'] = kp_danger.std().fillna(0).values
    if 'kp_csl9_in_chain' in df.columns:
        agg_df['cl_kp_csl9_active_frac'] = (
            df.groupby('_cluster_key', sort=False)['kp_csl9_in_chain']
              .mean().values)
    if 'kp_rp_death_count' in df.columns:
        kp_rp = df.groupby('_cluster_key', sort=False)['kp_rp_death_count']
        agg_df['cl_kp_max_rp_death'] = kp_rp.max().values
        agg_df['cl_kp_mean_rp_death'] = kp_rp.mean().values

    # Drop the cluster_key index
    agg_df = agg_df.reset_index(drop=True) if hasattr(agg_df, 'reset_index') else agg_df

    # CRITICAL: drop groups where no cluster has label=1
    # (This happens when death fell in a partial cluster that we filtered out)
    print(f"  [vectorized] Dropping groups with no positive cluster...")
    has_positive = agg_df.groupby('group_id', sort=False)['label'].transform('max')
    agg_df = agg_df[has_positive == 1].copy()

    print(f"  [vectorized] Done. {len(agg_df)} clusters, "
          f"{agg_df['group_id'].nunique()} groups.")
    return agg_df


def add_cluster_relative_features(df, group_col='group_id'):
    """Within-window relative features at cluster level."""
    KEY_CLUSTER = [
        'cl_kp_max_chain', 'cl_kp_mean_chain', 'cl_kp_3plus_frac',
        'cl_kp_max_danger', 'cl_kp_mean_danger', 'cl_kp_csl9_active_frac',
        'cl_kp_max_rp_death',
        # KP transit-distance features (continuous, important per EXP-025)
        'kp_tpd_to_h9cusp_dist_max', 'kp_tpd_to_h9cusp_dist_mean',
        'kp_tsd_to_h9cusp_dist_max', 'kp_tsd_to_h9cusp_dist_mean',
        'kp_tsat_to_h9cusp_dist_max', 'kp_tsat_to_h9cusp_dist_mean',
        'kp_tjup_to_h9cusp_dist_max', 'kp_tjup_to_h9cusp_dist_mean',
        'kp_trahu_to_h9cusp_dist_max', 'kp_trahu_to_h9cusp_dist_mean',
        'kp_tmoon_to_h9cusp_dist_max', 'kp_tmoon_to_h9cusp_dist_mean',
        'kp_pd_to_h9cusp_dist_max', 'kp_pd_to_natal_sun_dist_max',
        'kp_t_all_death_sub_count_max', 'kp_t_all_death_sub_count_mean',
        # Universal transit aggregates
        'ec_rahu_dist_h9_max', 'ec_rahu_dist_h9_mean',
        'ec_rahu_dist_sun_max', 'ec_rahu_dist_sun_mean',
        'gc_sat_asp_str_h9_max', 'gc_sat_asp_str_sun_max',
        'gc_jup_asp_str_h9_max', 'gc_mars_asp_str_sun_max',
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
                        window_months=24, n_augment=1, n_workers=16):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_f = DATA_DIR / f'{name}_cl{n_per_cluster}_w{window_months}.parquet'
    if cache_f.exists():
        print(f"  Loading cached cluster {cache_f.name}")
        return pd.read_parquet(cache_f)

    sd_df = build_sd_dataset_parallel(records, name, start_index, n_augment,
                                      window_months, n_workers)
    print(f"  Building clusters: {n_per_cluster} SDs/cluster...")
    cl_df = build_clusters_strict(sd_df, n_per_cluster)
    print(f"  Adding cluster-relative features...")
    cl_df = add_cluster_relative_features(cl_df)
    cl_df.to_parquet(cache_f)
    return cl_df


# ── Training utilities ─────────────────────────────────────────────────

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


def get_feat_cols(df, exclude_keys, pure_kp=False):
    """Get feature columns. If pure_kp=True, also drop Parashari-flavored."""
    feat_cols = []
    for c in df.columns:
        if c in exclude_keys:
            continue
        if not isinstance(df[c].iloc[0],
                          (int, float, np.integer, np.floating)):
            continue
        if _is_proxy_name(c):
            continue
        if pure_kp and _is_parashari_flavored(c):
            continue
        # Skip constant columns (LightGBM GPU can fail on them)
        v = df[c].values
        try:
            if np.nanstd(v) < 1e-9:
                continue
        except (TypeError, ValueError):
            continue
        # Skip cols with NaN/Inf (paranoia)
        if np.isnan(v).any() or np.isinf(v).any():
            continue
        feat_cols.append(c)
    return feat_cols


def auto_drop_leaks(df, feat_cols, target_col='cl_duration', threshold=0.15):
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


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("EXP-030: KP CLUSTER cl=8 (~52d) full sweep + cl=10 exploration")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")
    print(f"  CPUs: {os.cpu_count()}, Workers: 16")

    # Use nd_B (low_reg) as the new base — it consistently outperforms
    # the default reg in EXP-026.
    params_base = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 25,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    params_xendcg = {**params_base, 'objective': 'rank_xendcg'}
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': 12.0,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }

    cluster_sizes = [8, 9, 10]
    final_results = {}

    for n_per in cluster_sizes:
        print(f"\n{'='*75}")
        print(f"CLUSTER SIZE: {n_per} SDs/cluster")
        print(f"{'='*75}")

        cl_tr = get_cluster_dataset(train_recs, 'train', 0, n_per,
                                    window_months=24, n_augment=5)
        cl_va = get_cluster_dataset(val_recs, 'val',
                                    len(train_recs) * 100, n_per,
                                    window_months=24, n_augment=1)

        if len(cl_tr) == 0 or len(cl_va) == 0:
            print(f"  SKIP: empty dataset")
            continue

        cl_va = cl_va[cl_va.groupby('group_id')['group_id'].transform('count') >= 2]
        cl_tr = cl_tr[cl_tr.groupby('group_id')['group_id'].transform('count') >= 2]

        n_va = cl_va['group_id'].nunique()
        n_tr = cl_tr['group_id'].nunique()
        gs = cl_va.groupby('group_id').size()
        mean_dur = cl_va['cl_duration'].mean() if 'cl_duration' in cl_va.columns else 0
        print(f"  Train groups: {n_tr}, Val groups: {n_va}")
        print(f"  Clusters/window: mean={gs.mean():.1f}, median={gs.median():.0f}")
        print(f"  Days/cluster: mean={mean_dur:.1f}")
        if n_va < 50:
            print(f"  SKIP: too few val groups")
            continue

        rnd_t1 = float((1.0 / gs.values).mean())
        rnd_t3 = float(np.minimum(3.0 / gs.values, 1.0).mean())
        rnd_t5 = float(np.minimum(5.0 / gs.values, 1.0).mean())
        print(f"  Random: T1={rnd_t1:.1%} T3={rnd_t3:.1%} T5={rnd_t5:.1%}")

        EXCLUDE = {'group_id', 'cluster_idx_internal', 'label', 'cl_duration'}
        feat_cols = get_feat_cols(cl_va, EXCLUDE)
        feat_cols = [c for c in feat_cols if c in cl_tr.columns]

        # Pure-KP feature set (drop Parashari-flavored at cl=5 only)
        pure_kp_cols = get_feat_cols(cl_va, EXCLUDE, pure_kp=True)
        pure_kp_cols = [c for c in pure_kp_cols if c in cl_tr.columns]

        nodur_cols, leaks = auto_drop_leaks(cl_tr, feat_cols,
                                            target_col='cl_duration',
                                            threshold=0.15)
        pure_kp_nodur, pure_leaks = auto_drop_leaks(
            cl_tr, pure_kp_cols, target_col='cl_duration', threshold=0.15)
        print(f"  Features (raw):       {len(feat_cols)}")
        print(f"  Features (no-dur):    {len(nodur_cols)} "
              f"(dropped {len(leaks)} leaks)")
        print(f"  Features (pure-KP):   {len(pure_kp_cols)}")
        print(f"  Features (pure-KP no-dur): {len(pure_kp_nodur)}")
        if leaks[:5]:
            print(f"  Top leaks:")
            for c, r in leaks[:5]:
                print(f"    {c:50s} r={r:+.3f}")

        dur_scores = cl_va['cl_duration'].values
        dur_topk, _ = eval_topk(cl_va, dur_scores)
        print(f"  Dur-only:  T1={dur_topk['top_1']:.1%} "
              f"T3={dur_topk['top_3']:.1%} T5={dur_topk['top_5']:.1%}")

        print(f"  Training full model ({len(feat_cols)} feats)...")
        full_models = train_seed_avg(feat_cols, cl_tr, cl_va, 'group_id',
                                     params_base)
        full_scores = predict_avg(full_models, cl_va[feat_cols].values)
        full_topk, _ = eval_topk(cl_va, full_scores)
        print(f"  Full:      T1={full_topk['top_1']:.1%} "
              f"T3={full_topk['top_3']:.1%} T5={full_topk['top_5']:.1%} "
              f"({full_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")

        print(f"  Training no-dur model ({len(nodur_cols)} feats)...")
        nd_models = train_seed_avg(nodur_cols, cl_tr, cl_va, 'group_id',
                                   params_base)
        nd_scores = predict_avg(nd_models, cl_va[nodur_cols].values)
        nd_topk, _ = eval_topk(cl_va, nd_scores)
        print(f"  No-dur:    T1={nd_topk['top_1']:.1%} "
              f"T3={nd_topk['top_3']:.1%} T5={nd_topk['top_5']:.1%} "
              f"({nd_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")

        if n_per in (8, 9, 10):
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
            variant_scores = {}  # store all variant scores for ensemble
            for vname, vparams in nd_variants.items():
                v_models = train_seed_avg(nodur_cols, cl_tr, cl_va,
                                          'group_id', vparams)
                v_scores = predict_avg(v_models, cl_va[nodur_cols].values)
                variant_scores[vname] = v_scores
                v_topk, _ = eval_topk(cl_va, v_scores)
                print(f"    {vname:>20s}: T1={v_topk['top_1']:.1%} "
                      f"T3={v_topk['top_3']:.1%} T5={v_topk['top_5']:.1%} "
                      f"({v_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")
                if v_topk['top_1'] > best_v_topk['top_1']:
                    best_v_topk = v_topk
                    best_v_scores = v_scores
                    best_v_name = vname

            # ── PURE-KP arm ────────────────────────────────────────
            print(f"\n  PURE-KP arm ({len(pure_kp_nodur)} features, "
                  f"no Parashari-flavored):")
            pk_models = train_seed_avg(pure_kp_nodur, cl_tr, cl_va,
                                       'group_id', params_base)
            pk_scores = predict_avg(pk_models, cl_va[pure_kp_nodur].values)
            pk_topk, _ = eval_topk(cl_va, pk_scores)
            print(f"    {'pure_kp_nodur':>20s}: T1={pk_topk['top_1']:.1%} "
                  f"T3={pk_topk['top_3']:.1%} T5={pk_topk['top_5']:.1%} "
                  f"({pk_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")

            # Pure-KP variants
            pk_best_topk = pk_topk
            pk_best_name = 'pk_base'
            for vname, vparams in nd_variants.items():
                pk_v_models = train_seed_avg(pure_kp_nodur, cl_tr, cl_va,
                                             'group_id', vparams)
                pk_v_scores = predict_avg(
                    pk_v_models, cl_va[pure_kp_nodur].values)
                pk_v_topk, _ = eval_topk(cl_va, pk_v_scores)
                print(f"    {'pk_'+vname:>20s}: T1={pk_v_topk['top_1']:.1%} "
                      f"T3={pk_v_topk['top_3']:.1%} T5={pk_v_topk['top_5']:.1%}")
                if pk_v_topk['top_1'] > pk_best_topk['top_1']:
                    pk_best_topk = pk_v_topk
                    pk_best_name = 'pk_' + vname

            # Pure-KP feature importance
            pk_imp = np.zeros(len(pure_kp_nodur))
            for m in pk_models:
                pk_imp += m.feature_importance(importance_type='gain')
            pk_imp /= len(pk_models)
            pk_imp_pct = pk_imp / max(pk_imp.sum(), 1e-9) * 100
            pk_idx = np.argsort(pk_imp_pct)[::-1]
            print(f"\n  Pure-KP top 15 features:")
            for r, i in enumerate(pk_idx[:15]):
                print(f"    {r+1:2d}. {pure_kp_nodur[i]:50s} "
                      f"{pk_imp_pct[i]:5.2f}%")

            print(f"  Training xendcg no-dur...")
            x_models = train_seed_avg(nodur_cols, cl_tr, cl_va, 'group_id',
                                      params_xendcg)
            x_scores = predict_avg(x_models, cl_va[nodur_cols].values)
            x_topk, _ = eval_topk(cl_va, x_scores)
            print(f"    {'xendcg_nodur':>20s}: T1={x_topk['top_1']:.1%} "
                  f"T3={x_topk['top_3']:.1%} T5={x_topk['top_5']:.1%}")
            if x_topk['top_1'] > best_v_topk['top_1']:
                best_v_topk = x_topk
                best_v_scores = x_scores
                best_v_name = 'xendcg'

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

            def _norm(s):
                r = s.max() - s.min()
                return (s - s.min()) / r if r > 1e-10 else np.zeros_like(s)
            print(f"  Blend sweep (best ranker x clf):")
            best_v_norm = _norm(best_v_scores)
            clf_norm = _norm(clf_scores)
            best_blend_topk = nd_topk
            best_blend_name = 'no-blend'
            best_blend_scores = best_v_scores
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
                    best_blend_scores = bs

            # ── Ensemble: average of all variants + base + xendcg + clf ──
            print(f"  Multi-ranker ensemble (reusing variants):")
            ens_avg = np.mean([
                _norm(nd_scores), _norm(x_scores),
            ] + [_norm(s) for s in variant_scores.values()], axis=0)
            ens_topk, _ = eval_topk(cl_va, ens_avg)
            print(f"    {'ensemble_rank_only':>22s}: T1={ens_topk['top_1']:.1%} "
                  f"T3={ens_topk['top_3']:.1%} T5={ens_topk['top_5']:.1%} "
                  f"({ens_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")
            # Ensemble + clf
            ens_clf = 0.7 * ens_avg + 0.3 * _norm(clf_scores)
            ens_clf_topk, _ = eval_topk(cl_va, ens_clf)
            print(f"    {'ensemble+clf_70/30':>22s}: T1={ens_clf_topk['top_1']:.1%} "
                  f"T3={ens_clf_topk['top_3']:.1%} T5={ens_clf_topk['top_5']:.1%} "
                  f"({ens_clf_topk['top_1']/max(rnd_t1,1e-6):.1f}x)")
            if ens_topk['top_1'] > best_blend_topk['top_1']:
                best_blend_topk = ens_topk
                best_blend_name = 'ensemble_rank_only'
            if ens_clf_topk['top_1'] > best_blend_topk['top_1']:
                best_blend_topk = ens_clf_topk
                best_blend_name = 'ensemble+clf_70/30'

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

            print(f"\n  BIAS VERIFICATION:")
            full_minus_nodur = full_topk['top_1'] - nd_topk['top_1']
            if full_minus_nodur > 0.03:
                print(f"  WARNING: full ({full_topk['top_1']:.1%}) >> "
                      f"no-dur ({nd_topk['top_1']:.1%}) "
                      f"-- possible duration leak")
            else:
                print(f"  OK: full ({full_topk['top_1']:.1%}) ~= "
                      f"no-dur ({nd_topk['top_1']:.1%}) -- bias-clean")
            top_feats = [nodur_cols[i] for i in idx_sort[:20]]
            suspect = [f for f in top_feats
                       if any(p in f.lower() for p in
                              ['duration', 'movement', 'dur_', 'ingress'])]
            if suspect:
                print(f"  WARNING: suspected proxies in top-20: {suspect}")
            else:
                print(f"  OK: no duration/movement proxies in top-20")

            final_results[n_per] = {
                'random': rnd_t1, 'days': mean_dur,
                'cands': float(gs.mean()),
                'dur_only': dur_topk, 'full': full_topk, 'nodur': nd_topk,
                'best_variant': (best_v_name, best_v_topk),
                'best_blend': (best_blend_name, best_blend_topk),
                'n_features': len(nodur_cols),
            }
        else:
            final_results[n_per] = {
                'random': rnd_t1, 'days': mean_dur,
                'cands': float(gs.mean()),
                'dur_only': dur_topk, 'full': full_topk, 'nodur': nd_topk,
                'n_features': len(nodur_cols),
            }

    # Final summary
    print(f"\n{'='*75}")
    print(f"EXP-024 FINAL SUMMARY (KP CLUSTERS)")
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
    mp.freeze_support()
    main()
