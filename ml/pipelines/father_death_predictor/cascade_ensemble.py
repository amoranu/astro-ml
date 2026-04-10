"""Hierarchical multi-tradition cascade ensemble.

Tests several cascade variants:
  Cascade A: Parashari AD -> KP cluster cl=5 (cross-tradition, cached data)
  Cascade B: Parashari AD -> Parashari PD (single-tradition control)
  Cascade C: Yogini PD -> KP cluster cl=5 (fine-fine, different traditions)
  Cascade D: Parashari AD top-2 -> KP cluster top-3 (looser stage 1)

For each cascade, computes Top-1, Top-3, Top-5 over the final picks.

Usage:
    python -u -m ml.pipelines.father_death_predictor.cascade_ensemble
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

_DUR_PATTERNS = ('duration', 'dur_', 'movement', '_ingress', 'sign_change')
_HARD_PROXIES = {
    'cand_idx', 'cluster_idx', 'cluster_idx_internal', 'cl_duration',
    'seq_pos_norm', 'seq_third', 'seq_duration_days', 'seq_dur_vs_mean',
    'seq_dur_log', 'seq_danger_intensity',
    'id_pl_planet_idx', 'id_al_planet_idx', 'id_lagna_planet_combo',
}


def _load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


def get_numeric_feat_cols(df, exclude_keys, drop_patterns=()):
    feat_cols = []
    for c in df.columns:
        if c in exclude_keys:
            continue
        if not isinstance(df[c].iloc[0],
                          (int, float, np.integer, np.floating)):
            continue
        if any(p in c for p in drop_patterns):
            continue
        v = df[c].values
        try:
            if np.nanstd(v) < 1e-9:
                continue
        except (TypeError, ValueError):
            continue
        if np.isnan(v).any() or np.isinf(v).any():
            continue
        feat_cols.append(c)
    return feat_cols


def auto_drop_leaks(df, feat_cols, target_col='cl_duration', threshold=0.15):
    if target_col not in df.columns:
        return feat_cols
    target = df[target_col].values
    keep = []
    for c in feat_cols:
        v = df[c].values
        if v.std() < 1e-9:
            continue
        try:
            r = np.corrcoef(v, target)[0, 1]
            if abs(r) <= threshold:
                keep.append(c)
        except Exception:
            keep.append(c)
    return keep


def train_seed_avg(cols, df_tr, df_va, params, n_seeds=5, gcol='group_id'):
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


def date_to_jd(date_str):
    from .astro_engine.ephemeris import compute_jd
    return compute_jd(date_str, '12:00')


# ── Date range computation per tradition ────────────────────────────

def compute_dates_for_chart(rec, tradition, base_idx):
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .astro_engine.yogini_dasha import compute_yogini_dasha
    from .features.dasha_window import construct_dasha_window

    chart, asc = compute_chart(
        rec['birth_date'], rec['birth_time'], rec['lat'], rec['lon'])
    birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])
    moon_long = chart['Moon']['longitude']
    seed = base_idx * 100 + 0

    if tradition == 'parashari_ad':
        # Vimshottari Antardasha at depth=2
        all_d = compute_full_dasha(
            moon_long, birth_jd, max_depth=2, collect_all_depths=True)
        ad_periods = [p for p in all_d if p['depth'] == 2]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], ad_periods, window_months=24, seed=seed)
        return [(i, c['start_jd'], c['end_jd']) for i, c in enumerate(cands)]

    if tradition == 'parashari_pd':
        # Vimshottari Pratyantardasha at depth=3
        all_d = compute_full_dasha(
            moon_long, birth_jd, max_depth=3, collect_all_depths=True)
        pd_periods = [p for p in all_d if p['depth'] == 3]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], pd_periods, window_months=24, seed=seed)
        return [(i, c['start_jd'], c['end_jd']) for i, c in enumerate(cands)]

    if tradition in ('parashari_cluster', 'kp_cluster'):
        # Sookshma cluster of 5
        all_d4 = compute_full_dasha(
            moon_long, birth_jd, max_depth=4, collect_all_depths=True)
        sd_periods = [p for p in all_d4 if p['depth'] == 4]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], sd_periods, window_months=24, seed=seed)
        n_full = (len(cands) // 5) * 5
        out = []
        for cl_idx in range(n_full // 5):
            chunk = cands[cl_idx * 5:(cl_idx + 1) * 5]
            out.append((cl_idx, chunk[0]['start_jd'], chunk[-1]['end_jd']))
        return out

    if tradition == 'yogini_pd':
        all_yg = compute_yogini_dasha(moon_long, birth_jd, max_depth=3)
        pds = [p for p in all_yg if p.get('depth') == 3]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], pds, window_months=24, seed=seed)
        return [(i, c['start_jd'], c['end_jd']) for i, c in enumerate(cands)]

    raise ValueError(f"Unknown tradition: {tradition}")


# ── Per-tradition trained scores ─────────────────────────────────────

def train_parashari_ad():
    """Train Parashari Antardasha (depth=2) model."""
    print("[Parashari AD] Loading cache...")
    df_tr = pd.read_parquet(DATA_DIR / 'dasha_depth' / 'train_d2_aug3.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'dasha_depth' / 'val_d2_aug1.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Features: {len(feat_cols)}")
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05, 'subsample': 0.8,
        'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 15,
        'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    # Filter groups with >= 2 candidates (lambdarank needs that)
    df_tr_f = df_tr[df_tr.groupby('group_id')['group_id'].transform('count') >= 2]
    df_va_f = df_va[df_va.groupby('group_id')['group_id'].transform('count') >= 2]
    models = train_seed_avg(feat_cols, df_tr_f, df_va_f, params)
    scores = predict_avg(models, df_va_f[feat_cols].values)
    out = df_va_f[['group_id', 'cand_idx', 'label']].copy()
    out['score'] = scores
    return out


def train_parashari_pd():
    """Train Parashari Pratyantardasha (depth=3) model."""
    print("[Parashari PD] Loading cache...")
    df_tr = pd.read_parquet(DATA_DIR / 'dasha_depth' / 'train_d3_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'dasha_depth' / 'val_d3_aug1.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Features: {len(feat_cols)}")
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(feat_cols, df_tr, df_va, params)
    scores = predict_avg(models, df_va[feat_cols].values)
    out = df_va[['group_id', 'cand_idx', 'label']].copy()
    out['score'] = scores
    return out


def train_kp_cluster():
    """Train KP cluster cl=5 (~33 days)."""
    print("[KP cluster cl=5] Loading cache...")
    cl_tr = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    print(f"  Features: {len(nodur_cols)}")
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 25,
        'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(nodur_cols, cl_tr, cl_va, params)
    scores = predict_avg(models, cl_va[nodur_cols].values)
    out = cl_va[['group_id', 'cluster_idx_internal', 'label']].copy()
    out = out.rename(columns={'cluster_idx_internal': 'cand_idx'})
    out['score'] = scores
    return out


def train_yogini_pd():
    """Train Yogini PD d=3 (rank+clf blend)."""
    print("[Yogini PD] Loading cache...")
    df_tr = pd.read_parquet(DATA_DIR / 'yogini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'yogini' / 'val_aug1.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Features: {len(feat_cols)}")
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    rank_models = train_seed_avg(feat_cols, df_tr, df_va, params)
    rank_scores = predict_avg(rank_models, df_va[feat_cols].values)
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': 30.0,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    clf_models = []
    for i in range(5):
        p = {**params_clf, 'seed': 42 + i * 17}
        m = lgb.train(p, lgb.Dataset(df_tr[feat_cols].values,
                                     label=df_tr['label'].values),
                      num_boost_round=500)
        clf_models.append(m)
    clf_scores = np.mean([m.predict(df_va[feat_cols].values)
                          for m in clf_models], axis=0)

    def _norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 1e-10 else np.zeros_like(s)
    blend = 0.5 * _norm(rank_scores) + 0.5 * _norm(clf_scores)
    out = df_va[['group_id', 'cand_idx', 'label']].copy()
    out['score'] = blend
    return out


# ── Cascade orchestration ──────────────────────────────────────────

def get_topk_dates(scores_df, dates_dict, k):
    """For each gid, return list of (start_jd, end_jd) for top-k by score."""
    out = {}
    for gid, grp in scores_df.groupby('group_id', sort=False):
        gid = int(gid)
        if gid not in dates_dict:
            continue
        sorted_g = grp.sort_values('score', ascending=False)
        topk_idxs = sorted_g['cand_idx'].head(k).astype(int).tolist()
        ranges = [dates_dict[gid].get(ci) for ci in topk_idxs
                  if ci in dates_dict[gid]]
        out[gid] = [r for r in ranges if r is not None]
    return out


def date_overlaps(ranges_a, range_b, min_overlap_days=1):
    """Does range_b overlap any of ranges_a by at least min_overlap_days?"""
    s2, e2 = range_b
    for s1, e1 in ranges_a:
        ov = min(e1, e2) - max(s1, s2)
        if ov >= min_overlap_days:
            return True
    return False


def cascade_evaluate(stage1_topk_dates, stage2_scores, stage2_dates,
                     death_jd_dict, final_k=3, min_overlap=1):
    """Two-stage cascade.

    Args:
        stage1_topk_dates: {gid: list of (start_jd, end_jd) for stage 1 top-K}
        stage2_scores: DataFrame with group_id, cand_idx, label, score
        stage2_dates: {gid: {cand_idx: (start_jd, end_jd)}}
        death_jd_dict: {gid: death_jd}
        final_k: number of top stage-2 candidates to keep
        min_overlap: minimum days of overlap to count as "in stage 1"

    Returns:
        dict with top1, top3, top5, n_eval, stage1_recall, etc.
    """
    n = 0
    stage1_recall_hits = 0
    final_top1 = 0
    final_top3 = 0
    final_top5 = 0
    no_overlap_charts = 0

    stage2_grouped = {gid: g for gid, g in stage2_scores.groupby('group_id', sort=False)}

    for gid, ranges_a in stage1_topk_dates.items():
        d_jd = death_jd_dict.get(gid)
        if d_jd is None:
            continue
        if gid not in stage2_dates or gid not in stage2_grouped:
            continue
        n += 1

        # Stage 1 recall: does any stage-1 range contain death?
        s1_hit = any(s <= d_jd <= e for s, e in ranges_a)
        if s1_hit:
            stage1_recall_hits += 1

        # Filter stage 2 candidates that overlap stage 1's union
        s2_grp = stage2_grouped[gid]
        filtered = []
        for _, row in s2_grp.iterrows():
            ci = int(row['cand_idx'])
            cand_range = stage2_dates[gid].get(ci)
            if cand_range is None:
                continue
            if date_overlaps(ranges_a, cand_range, min_overlap):
                filtered.append({
                    'cand_idx': ci,
                    'score': float(row['score']),
                    'label': int(row['label']),
                    'range': cand_range,
                })

        if not filtered:
            no_overlap_charts += 1
            continue

        # Sort by score
        filtered.sort(key=lambda x: -x['score'])
        # Top-K
        topk_labels = [c['label'] for c in filtered[:5]]
        if len(topk_labels) >= 1 and topk_labels[0] == 1:
            final_top1 += 1
        if 1 in topk_labels[:3]:
            final_top3 += 1
        if 1 in topk_labels[:5]:
            final_top5 += 1

    return {
        'n_eval': n,
        'stage1_recall': stage1_recall_hits / max(n, 1),
        'top1': final_top1 / max(n, 1),
        'top3': final_top3 / max(n, 1),
        'top5': final_top5 / max(n, 1),
        'no_overlap': no_overlap_charts / max(n, 1),
    }


def per_tradition_topk(scores_df, k=5):
    """Standalone Top-K accuracy for one tradition."""
    n_correct = {1: 0, 3: 0, 5: 0}
    n = 0
    for gid, grp in scores_df.groupby('group_id', sort=False):
        if grp['label'].max() == 0:
            continue
        n += 1
        sorted_g = grp.sort_values('score', ascending=False)
        labels = sorted_g['label'].head(5).values
        if len(labels) >= 1 and labels[0] == 1:
            n_correct[1] += 1
        if 1 in labels[:3]:
            n_correct[3] += 1
        if 1 in labels[:5]:
            n_correct[5] += 1
    return {k: c / max(n, 1) for k, c in n_correct.items()}, n


def main():
    print("=" * 75)
    print("CASCADE ENSEMBLE — multi-tradition hierarchical filtering")
    print("=" * 75)

    train_recs = [r for r in _load_json(str(V2_JSON)) + _load_json(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load_json(str(VAL_JSON)) if _valid(r)]
    n_train = len(train_recs)
    start_index = n_train * 100
    print(f"  Train: {n_train}, Val: {len(val_recs)}")

    # ── Train all models ──────────────────────────────────────
    print("\n[1] Training models...")
    par_ad = train_parashari_ad()
    par_pd = train_parashari_pd()
    kp_cl = train_kp_cluster()
    yog_pd = train_yogini_pd()

    # ── Compute date ranges per chart ─────────────────────────
    print("\n[2] Computing date ranges per chart per tradition...")
    t0 = time.time()
    par_ad_dates = {}
    par_pd_dates = {}
    kp_cl_dates = {}
    yog_pd_dates = {}
    for i, rec in enumerate(val_recs):
        gid = (start_index + i) * 100
        try:
            par_ad_dates[gid] = {
                ci: (s, e) for ci, s, e in
                compute_dates_for_chart(rec, 'parashari_ad', start_index + i)
            }
            par_pd_dates[gid] = {
                ci: (s, e) for ci, s, e in
                compute_dates_for_chart(rec, 'parashari_pd', start_index + i)
            }
            kp_cl_dates[gid] = {
                ci: (s, e) for ci, s, e in
                compute_dates_for_chart(rec, 'parashari_cluster', start_index + i)
            }
            yog_pd_dates[gid] = {
                ci: (s, e) for ci, s, e in
                compute_dates_for_chart(rec, 'yogini_pd', start_index + i)
            }
        except Exception:
            pass
    print(f"  Done in {time.time()-t0:.0f}s")

    death_jd = {(start_index + i) * 100: date_to_jd(rec['father_death_date'])
                for i, rec in enumerate(val_recs)}

    # ── Per-tradition baselines ───────────────────────────────
    print("\n" + "=" * 75)
    print("PER-TRADITION BASELINES (Top-1 / Top-3 / Top-5)")
    print("=" * 75)
    for name, scores in [('Parashari AD', par_ad), ('Parashari PD', par_pd),
                          ('KP cluster cl=5', kp_cl), ('Yogini PD', yog_pd)]:
        topk, n = per_tradition_topk(scores)
        print(f"  {name:>20s}: T1={topk[1]:.1%} T3={topk[3]:.1%} "
              f"T5={topk[5]:.1%}  (n={n})")

    # ── Cascade A: Parashari AD top-1 -> KP cluster cl=5 ──────
    print("\n" + "=" * 75)
    print("CASCADE A: Parashari AD top-1 -> KP cluster cl=5 top-3")
    print("=" * 75)
    s1 = get_topk_dates(par_ad, par_ad_dates, k=1)
    res = cascade_evaluate(s1, kp_cl, kp_cl_dates, death_jd, final_k=5)
    print(f"  Stage 1 recall: {res['stage1_recall']:.1%}")
    print(f"  No overlap fall-thru: {res['no_overlap']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── Cascade A2: Parashari AD top-2 -> KP cluster ──────────
    print("\n" + "=" * 75)
    print("CASCADE A2: Parashari AD top-2 -> KP cluster cl=5 top-3")
    print("=" * 75)
    s1 = get_topk_dates(par_ad, par_ad_dates, k=2)
    res = cascade_evaluate(s1, kp_cl, kp_cl_dates, death_jd, final_k=5)
    print(f"  Stage 1 recall: {res['stage1_recall']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── Cascade B: Parashari AD top-2 -> Parashari PD top-3 ────
    print("\n" + "=" * 75)
    print("CASCADE B: Parashari AD top-2 -> Parashari PD top-3 (single-trad)")
    print("=" * 75)
    s1 = get_topk_dates(par_ad, par_ad_dates, k=2)
    res = cascade_evaluate(s1, par_pd, par_pd_dates, death_jd, final_k=5)
    print(f"  Stage 1 recall: {res['stage1_recall']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── Cascade C: Yogini PD top-5 -> KP cluster cl=5 top-3 ────
    print("\n" + "=" * 75)
    print("CASCADE C: Yogini PD top-5 -> KP cluster cl=5 top-3 (cross-trad)")
    print("=" * 75)
    s1 = get_topk_dates(yog_pd, yog_pd_dates, k=5)
    res = cascade_evaluate(s1, kp_cl, kp_cl_dates, death_jd, final_k=5)
    print(f"  Stage 1 recall: {res['stage1_recall']:.1%}")
    print(f"  No overlap fall-thru: {res['no_overlap']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── Cascade D: Parashari PD top-3 -> KP cluster cl=5 top-3 ──
    print("\n" + "=" * 75)
    print("CASCADE D: Parashari PD top-3 -> KP cluster cl=5 top-3")
    print("=" * 75)
    s1 = get_topk_dates(par_pd, par_pd_dates, k=3)
    res = cascade_evaluate(s1, kp_cl, kp_cl_dates, death_jd, final_k=5)
    print(f"  Stage 1 recall: {res['stage1_recall']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── Cascade E: 3-stage Parashari AD -> Parashari PD -> KP cluster ──
    print("\n" + "=" * 75)
    print("CASCADE E: 3-stage Par AD top-2 -> Par PD top-3 -> KP top-3")
    print("=" * 75)

    def three_stage(par_ad_k=2, par_pd_k=3):
        s1 = get_topk_dates(par_ad, par_ad_dates, k=par_ad_k)
        # Stage 2: Parashari PD restricted to s1
        s2_dates = {}  # gid -> list of (start, end) for top par_pd_k
        par_pd_grp = {gid: g for gid, g in par_pd.groupby('group_id', sort=False)}
        for gid, ranges_a in s1.items():
            if gid not in par_pd_grp or gid not in par_pd_dates:
                continue
            grp = par_pd_grp[gid]
            filtered = []
            for _, row in grp.iterrows():
                ci = int(row['cand_idx'])
                d = par_pd_dates[gid].get(ci)
                if d is None:
                    continue
                if date_overlaps(ranges_a, d):
                    filtered.append({'ci': ci, 'score': row['score'], 'range': d})
            filtered.sort(key=lambda x: -x['score'])
            s2_dates[gid] = [c['range'] for c in filtered[:par_pd_k]]
        # Stage 3: KP cluster restricted to s2
        return cascade_evaluate(s2_dates, kp_cl, kp_cl_dates, death_jd)

    res = three_stage(par_ad_k=2, par_pd_k=3)
    print(f"  Stage 1+2 cumulative recall (proxy): "
          f"{res['stage1_recall']:.1%}")
    print(f"  Top-1: {res['top1']:.1%}")
    print(f"  Top-3: {res['top3']:.1%}")
    print(f"  Top-5: {res['top5']:.1%}")

    # ── SOFT CASCADES: multiply Stage 2 score by Stage 1 prior ─
    print("\n" + "=" * 75)
    print("SOFT CASCADES (multiply Stage 2 score by Stage 1 score-of-parent)")
    print("=" * 75)

    def _build_score_lookup(scores_df, dates_dict):
        """For each gid, return list of (start, end, score) for all candidates."""
        out = {}
        for gid, grp in scores_df.groupby('group_id', sort=False):
            gid = int(gid)
            if gid not in dates_dict:
                continue
            ranges = []
            for _, row in grp.iterrows():
                ci = int(row['cand_idx'])
                d = dates_dict[gid].get(ci)
                if d is None:
                    continue
                ranges.append((d[0], d[1], float(row['score'])))
            out[gid] = ranges
        return out

    def _zscore_within(out_dict):
        """Z-score the scores within each group."""
        new = {}
        for gid, items in out_dict.items():
            scores = np.array([x[2] for x in items])
            if scores.std() < 1e-9:
                new[gid] = items
                continue
            zs = (scores - scores.mean()) / scores.std()
            new[gid] = [(x[0], x[1], float(z)) for x, z in zip(items, zs)]
        return new

    def _stage1_score_for_range(stage1_items, range_b):
        """Find the highest stage-1 z-score among items overlapping range_b."""
        s2, e2 = range_b
        best = -10.0  # very negative default
        for s1, e1, sc1 in stage1_items:
            if min(e1, e2) - max(s1, s2) > 0:
                if sc1 > best:
                    best = sc1
        return best if best > -10.0 else 0.0  # default neutral if no overlap

    def soft_cascade_eval(stage1_lookup, stage2_lookup, death_jd_dict,
                          alpha=0.5):
        """Multiply or add stage 1 score as a prior on stage 2 candidates."""
        n = 0
        t1 = 0
        t3 = 0
        t5 = 0
        for gid in stage2_lookup:
            d_jd = death_jd_dict.get(gid)
            if d_jd is None:
                continue
            if gid not in stage1_lookup:
                continue
            n += 1
            s1_items = stage1_lookup[gid]
            s2_items = stage2_lookup[gid]
            # Score each stage-2 candidate with combined score
            scored = []
            for s, e, sc2 in s2_items:
                sc1 = _stage1_score_for_range(s1_items, (s, e))
                combined = alpha * sc2 + (1 - alpha) * sc1
                hit = int(s <= d_jd <= e)
                scored.append((combined, hit))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][1]:
                t1 += 1
            if any(h for _, h in scored[:3]):
                t3 += 1
            if any(h for _, h in scored[:5]):
                t5 += 1
        return {
            'n_eval': n,
            'top1': t1 / max(n, 1),
            'top3': t3 / max(n, 1),
            'top5': t5 / max(n, 1),
        }

    # Build z-scored lookups
    par_ad_lookup = _zscore_within(_build_score_lookup(par_ad, par_ad_dates))
    par_pd_lookup = _zscore_within(_build_score_lookup(par_pd, par_pd_dates))
    kp_cl_lookup = _zscore_within(_build_score_lookup(kp_cl, kp_cl_dates))
    yog_pd_lookup = _zscore_within(_build_score_lookup(yog_pd, yog_pd_dates))

    print("\n  Soft Cascade SA: Par AD prior on KP cl=5 (alpha=0.5)")
    res = soft_cascade_eval(par_ad_lookup, kp_cl_lookup, death_jd, alpha=0.5)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    print("\n  Soft Cascade SA: Par AD prior on KP cl=5 (alpha=0.7 — more KP)")
    res = soft_cascade_eval(par_ad_lookup, kp_cl_lookup, death_jd, alpha=0.7)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    print("\n  Soft Cascade SA: Par AD prior on KP cl=5 (alpha=0.3 — more AD)")
    res = soft_cascade_eval(par_ad_lookup, kp_cl_lookup, death_jd, alpha=0.3)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    print("\n  Soft Cascade SB: Par AD prior on Par PD (alpha=0.5)")
    res = soft_cascade_eval(par_ad_lookup, par_pd_lookup, death_jd, alpha=0.5)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    print("\n  Soft Cascade SC: Par AD prior on Par PD (alpha=0.7)")
    res = soft_cascade_eval(par_ad_lookup, par_pd_lookup, death_jd, alpha=0.7)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    print("\n  Soft Cascade SD: Yogini PD prior on KP cl=5 (alpha=0.5)")
    res = soft_cascade_eval(yog_pd_lookup, kp_cl_lookup, death_jd, alpha=0.5)
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    # ── 3-WAY soft fusion ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("THREE-WAY SOFT FUSION (KP + Par_PD + Par_AD prior, equal weights)")
    print("=" * 75)

    def three_way_fusion(weights):
        """Fuse Par AD, Par PD, KP cluster scores at the cluster level.
        For each KP cluster, sum weighted z-scores from each tradition.
        """
        n = 0
        t1 = 0
        t3 = 0
        t5 = 0
        w_par_ad, w_par_pd, w_kp = weights
        for gid in kp_cl_lookup:
            d_jd = death_jd.get(gid)
            if d_jd is None:
                continue
            n += 1
            s1 = par_ad_lookup.get(gid, [])
            s2 = par_pd_lookup.get(gid, [])
            kp_items = kp_cl_lookup[gid]
            scored = []
            for s, e, sc_kp in kp_items:
                sc_ad = _stage1_score_for_range(s1, (s, e)) if s1 else 0.0
                sc_pd = _stage1_score_for_range(s2, (s, e)) if s2 else 0.0
                combined = w_kp * sc_kp + w_par_pd * sc_pd + w_par_ad * sc_ad
                hit = int(s <= d_jd <= e)
                scored.append((combined, hit))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][1]:
                t1 += 1
            if any(h for _, h in scored[:3]):
                t3 += 1
            if any(h for _, h in scored[:5]):
                t5 += 1
        return {
            'n_eval': n,
            'top1': t1 / max(n, 1),
            'top3': t3 / max(n, 1),
            'top5': t5 / max(n, 1),
        }

    print("\n  Equal weights (1/3 each):")
    res = three_way_fusion((1/3, 1/3, 1/3))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  KP-heavy (0.5 KP + 0.3 Par_PD + 0.2 Par_AD):")
    res = three_way_fusion((0.2, 0.3, 0.5))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  Par_PD-heavy (0.5 Par_PD + 0.3 KP + 0.2 Par_AD):")
    res = three_way_fusion((0.2, 0.5, 0.3))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  Par_AD-heavy (0.5 Par_AD + 0.3 Par_PD + 0.2 KP):")
    res = three_way_fusion((0.5, 0.3, 0.2))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    # ── 4-WAY: add Yogini PD ─────────────────────────────────────
    print("\n" + "=" * 75)
    print("FOUR-WAY SOFT FUSION (+Yogini PD)")
    print("=" * 75)

    def four_way_fusion(weights):
        n = 0
        t1 = 0
        t3 = 0
        t5 = 0
        w_par_ad, w_par_pd, w_kp, w_yog = weights
        for gid in kp_cl_lookup:
            d_jd = death_jd.get(gid)
            if d_jd is None:
                continue
            n += 1
            s1 = par_ad_lookup.get(gid, [])
            s2 = par_pd_lookup.get(gid, [])
            s3 = yog_pd_lookup.get(gid, [])
            kp_items = kp_cl_lookup[gid]
            scored = []
            for s, e, sc_kp in kp_items:
                sc_ad = _stage1_score_for_range(s1, (s, e)) if s1 else 0.0
                sc_pd = _stage1_score_for_range(s2, (s, e)) if s2 else 0.0
                sc_yog = _stage1_score_for_range(s3, (s, e)) if s3 else 0.0
                combined = (w_kp * sc_kp + w_par_pd * sc_pd +
                            w_par_ad * sc_ad + w_yog * sc_yog)
                hit = int(s <= d_jd <= e)
                scored.append((combined, hit))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][1]:
                t1 += 1
            if any(h for _, h in scored[:3]):
                t3 += 1
            if any(h for _, h in scored[:5]):
                t5 += 1
        return {
            'n_eval': n,
            'top1': t1 / max(n, 1),
            'top3': t3 / max(n, 1),
            'top5': t5 / max(n, 1),
        }

    print("\n  Equal 4-way (0.25 each):")
    res = four_way_fusion((0.25, 0.25, 0.25, 0.25))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  Par_AD-prior heavy (0.4, 0.3, 0.2, 0.1):")
    res = four_way_fusion((0.4, 0.3, 0.2, 0.1))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  Par_PD-heavy (0.2, 0.4, 0.25, 0.15):")
    res = four_way_fusion((0.2, 0.4, 0.25, 0.15))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")
    print("\n  Par_PD + KP balanced (0.15, 0.35, 0.35, 0.15):")
    res = four_way_fusion((0.15, 0.35, 0.35, 0.15))
    print(f"    Top-1: {res['top1']:.1%}  Top-3: {res['top3']:.1%}  "
          f"Top-5: {res['top5']:.1%}")

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print("CASCADE SUMMARY")
    print("=" * 75)


if __name__ == '__main__':
    main()
