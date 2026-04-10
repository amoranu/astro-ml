"""Multi-tradition ensemble — Paths A (date-overlap), B (day-vote), E (abstain).

Uses cached val parquets + retrains models per tradition (Parashari KP=cl=5,
Yogini PD d=3) to get full per-candidate scores. Then computes per-chart
date ranges by re-running the dasha calculations.

Output:
  - Path E: abstention curves
  - Path A: date-overlap voting (top-1 of each tradition)
  - Path B: day-level vote across all candidates of all traditions
  - Comparison vs single-tradition baselines and oracle ceiling

Usage:
    python -u -m ml.pipelines.father_death_predictor.ensemble_full
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


# ── Date-range computation per tradition ──────────────────────────────

def compute_dates_for_chart(rec, tradition, base_idx):
    """Return list of (cand_idx, start_jd, end_jd) for one chart in one tradition.

    base_idx = start_index + i, where start_index = n_train * 100.
    The seed used in pipelines is base_idx * 100 + aug.
    """
    from .astro_engine.ephemeris import compute_chart, compute_jd
    from .astro_engine.dasha import compute_full_dasha
    from .astro_engine.yogini_dasha import compute_yogini_dasha
    from .features.dasha_window import construct_dasha_window

    chart, asc = compute_chart(
        rec['birth_date'], rec['birth_time'], rec['lat'], rec['lon'])
    birth_jd = compute_jd(rec['birth_date'], rec['birth_time'])
    moon_long = chart['Moon']['longitude']

    seed = base_idx * 100 + 0  # n_augment=1, aug=0

    if tradition in ('parashari', 'kp'):
        # Vimshottari Sookshma at depth=4
        all_d4 = compute_full_dasha(
            moon_long, birth_jd, max_depth=4, collect_all_depths=True)
        sd_periods = [p for p in all_d4 if p['depth'] == 4]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], sd_periods, window_months=24, seed=seed)
        # Cluster every 5 SDs (drop partial last cluster)
        n_full = (len(cands) // 5) * 5
        out = []
        for cl_idx in range(n_full // 5):
            chunk = cands[cl_idx * 5:(cl_idx + 1) * 5]
            out.append((cl_idx, chunk[0]['start_jd'], chunk[-1]['end_jd']))
        return out

    if tradition == 'yogini':
        # Yogini PD depth=3
        all_yg = compute_yogini_dasha(moon_long, birth_jd, max_depth=3)
        # depth=3 = PD
        pds = [p for p in all_yg if p.get('depth') == 3]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], pds, window_months=24, seed=seed)
        return [(i, c['start_jd'], c['end_jd']) for i, c in enumerate(cands)]

    raise ValueError(f"Unknown tradition: {tradition}")


def date_to_jd(date_str):
    from .astro_engine.ephemeris import compute_jd
    return compute_jd(date_str, '12:00')


# ── Per-tradition trained scores ─────────────────────────────────────

def get_parashari_scores():
    """Returns DataFrame with group_id, cand_idx (cluster_idx_internal),
    score, label, and trained from sd_clusters_v2 cl5."""
    cl_tr = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    print(f"  Parashari features: {len(nodur_cols)}")
    params = {
        'objective': 'rank_xendcg', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(nodur_cols, cl_tr, cl_va, params)
    scores = predict_avg(models, cl_va[nodur_cols].values)
    out = cl_va[['group_id', 'cluster_idx_internal', 'label']].copy()
    out['score'] = scores
    out = out.rename(columns={'cluster_idx_internal': 'cand_idx'})
    return out


def get_yogini_scores():
    df_tr = pd.read_parquet(DATA_DIR / 'yogini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'yogini' / 'val_aug1.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Yogini features: {len(feat_cols)}")
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


def get_kp_scores():
    cl_tr = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    print(f"  KP features: {len(nodur_cols)}")
    params_A = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 25,
        'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(nodur_cols, cl_tr, cl_va, params_A)
    scores = predict_avg(models, cl_va[nodur_cols].values)
    out = cl_va[['group_id', 'cluster_idx_internal', 'label']].copy()
    out['score'] = scores
    out = out.rename(columns={'cluster_idx_internal': 'cand_idx'})
    return out


def per_chart_top1(scores_df):
    """Group by group_id, return top1 cand_idx + correctness + scores."""
    rows = []
    for gid, grp in scores_df.groupby('group_id', sort=False):
        if grp['label'].max() == 0:
            continue
        sorted_g = grp.sort_values('score', ascending=False)
        top = sorted_g.iloc[0]
        # also save score margin
        s = sorted_g['score'].values
        margin = float(s[0] - s[1]) if len(s) > 1 else float(s[0])
        ex = np.exp(s - s.max())
        prob = ex / ex.sum()
        rows.append({
            'group_id': gid,
            'top1_cand_idx': int(top['cand_idx']),
            'top1_correct': int(top['label'] == 1),
            'top1_score': float(top['score']),
            'margin': margin,
            'top1_prob': float(prob[0]),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 75)
    print("MULTI-TRADITION ENSEMBLE — Paths A, B, E")
    print("=" * 75)

    # Load val records and compute base_idx mapping
    train_recs = [r for r in _load_json(str(V2_JSON)) + _load_json(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load_json(str(VAL_JSON)) if _valid(r)]
    n_train = len(train_recs)
    print(f"  Train: {n_train}, Val: {len(val_recs)}")

    # group_id mapping. The pipelines use:
    #   start_index = n_train * 100 (passed by main() to dataset builders)
    #   base_idx_per_chart = start_index + i = n_train * 100 + i
    #   seed = base_idx * 100 + aug
    #   group_id = base_idx * 100 + aug = (n_train * 100 + i) * 100 + aug
    # For n_augment=1, aug=0, so gid = (n_train * 100 + i) * 100
    start_index = n_train * 100
    gid_to_recidx = {(start_index + i) * 100: i for i in range(len(val_recs))}

    # ── Train all 3 models, get per-candidate scores ────────────────
    print("\n[1] Training Parashari (xendcg)...")
    par_scores = get_parashari_scores()
    print("\n[2] Training Yogini (rank+clf blend)...")
    yog_scores = get_yogini_scores()
    print("\n[3] Training KP (nd_A)...")
    kp_scores = get_kp_scores()

    # ── Per-chart top-1 ────────────────────────────────────────────
    par_top = per_chart_top1(par_scores)
    yog_top = per_chart_top1(yog_scores)
    kp_top = per_chart_top1(kp_scores)
    print(f"\n  Parashari top-1 charts: {len(par_top)}, "
          f"correct: {par_top['top1_correct'].sum()}")
    print(f"  Yogini top-1 charts:    {len(yog_top)}, "
          f"correct: {yog_top['top1_correct'].sum()}")
    print(f"  KP top-1 charts:        {len(kp_top)}, "
          f"correct: {kp_top['top1_correct'].sum()}")

    # ── Compute date ranges per chart per tradition ─────────────────
    print("\n[4] Computing date ranges per chart per tradition...")
    t0 = time.time()
    par_dates = {}  # gid -> {cand_idx: (start_jd, end_jd)}
    yog_dates = {}
    kp_dates = {}
    n_done = 0
    for gid, recidx in gid_to_recidx.items():
        rec = val_recs[recidx]
        base = start_index + recidx
        try:
            par_dates[int(gid)] = {
                ci: (s, e)
                for ci, s, e in compute_dates_for_chart(rec, 'parashari', base)
            }
            kp_dates[int(gid)] = par_dates[int(gid)]  # same Vimshottari
            yog_dates[int(gid)] = {
                ci: (s, e)
                for ci, s, e in compute_dates_for_chart(rec, 'yogini', base)
            }
            n_done += 1
        except Exception as e:
            pass
    print(f"  Date ranges computed for {n_done} charts in {time.time()-t0:.0f}s")

    # ── Augment per-chart top-1 with date ranges ──────────────────
    def _add_dates(top_df, dates_dict):
        out = top_df.copy()
        starts = []
        ends = []
        for _, row in out.iterrows():
            gid = int(row['group_id'])
            ci = int(row['top1_cand_idx'])
            sd = dates_dict.get(gid, {}).get(ci, (None, None))
            starts.append(sd[0])
            ends.append(sd[1])
        out['start_jd'] = starts
        out['end_jd'] = ends
        return out

    par_top = _add_dates(par_top, par_dates)
    yog_top = _add_dates(yog_top, yog_dates)
    kp_top = _add_dates(kp_top, kp_dates)

    # ── Death JD per chart ────────────────────────────────────────
    death_jd = {int((start_index + i) * 100): date_to_jd(rec['father_death_date'])
                for i, rec in enumerate(val_recs) if _valid(rec)}

    # ── Inner join all three ─────────────────────────────────────
    df = par_top.rename(columns={
        'top1_cand_idx': 'par_cand_idx',
        'top1_correct': 'par_correct',
        'top1_score': 'par_score',
        'margin': 'par_margin',
        'top1_prob': 'par_prob',
        'start_jd': 'par_start',
        'end_jd': 'par_end',
    })
    df = df.merge(
        yog_top.rename(columns={
            'top1_cand_idx': 'yog_cand_idx',
            'top1_correct': 'yog_correct',
            'top1_score': 'yog_score',
            'margin': 'yog_margin',
            'top1_prob': 'yog_prob',
            'start_jd': 'yog_start',
            'end_jd': 'yog_end',
        }),
        on='group_id',
    )
    df = df.merge(
        kp_top.rename(columns={
            'top1_cand_idx': 'kp_cand_idx',
            'top1_correct': 'kp_correct',
            'top1_score': 'kp_score',
            'margin': 'kp_margin',
            'top1_prob': 'kp_prob',
            'start_jd': 'kp_start',
            'end_jd': 'kp_end',
        }),
        on='group_id',
    )
    df['death_jd'] = df['group_id'].astype(int).map(death_jd)
    df = df.dropna(subset=['par_start', 'yog_start', 'kp_start', 'death_jd'])
    n = len(df)
    print(f"\n  Common charts with full data: {n}")

    par_acc = df['par_correct'].mean()
    yog_acc = df['yog_correct'].mean()
    kp_acc = df['kp_correct'].mean()
    union = ((df['par_correct'] == 1) | (df['yog_correct'] == 1) |
             (df['kp_correct'] == 1)).mean()
    print(f"  Parashari: {par_acc:.1%}, Yogini: {yog_acc:.1%}, "
          f"KP: {kp_acc:.1%}")
    print(f"  Oracle ceiling (union): {union:.1%}")

    # ──────────────────────────────────────────────────────────────
    # Path E: Abstention curves
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PATH E — Abstention curves (commit only on confident charts)")
    print("=" * 75)
    for trad, prob_col, correct_col in [
        ('Parashari', 'par_prob', 'par_correct'),
        ('Yogini',    'yog_prob', 'yog_correct'),
        ('KP',        'kp_prob',  'kp_correct'),
    ]:
        d = df.sort_values(prob_col, ascending=False).reset_index(drop=True)
        print(f"\n  {trad}:")
        print(f"    {'%kept':>7s} {'N':>5s} {'#correct':>10s} {'accuracy':>10s}")
        for pct in [10, 20, 30, 50, 100]:
            k = max(1, int(n * pct / 100))
            sub = d.iloc[:k]
            acc = sub[correct_col].mean()
            print(f"    {pct:>5d}% {k:>5d} "
                  f"{int(sub[correct_col].sum()):>10d} {acc:>9.1%}")

    # ──────────────────────────────────────────────────────────────
    # Path A: Date-overlap voting
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PATH A — Date-overlap voting")
    print("=" * 75)

    def _overlap(a_s, a_e, b_s, b_e):
        return max(0.0, min(a_e, b_e) - max(a_s, b_s))

    def _path_a_predict(row):
        # Compute pairwise overlaps
        pairs = [
            ('par', 'yog'),
            ('par', 'kp'),
            ('yog', 'kp'),
        ]
        pair_overlaps = {}
        for a, b in pairs:
            ov = _overlap(row[f'{a}_start'], row[f'{a}_end'],
                          row[f'{b}_start'], row[f'{b}_end'])
            pair_overlaps[(a, b)] = ov

        max_overlap = max(pair_overlaps.values())
        if max_overlap > 0:
            # Use the midpoint of the overlap as prediction
            best_pair = max(pair_overlaps, key=pair_overlaps.get)
            a, b = best_pair
            mid = (max(row[f'{a}_start'], row[f'{b}_start']) +
                   min(row[f'{a}_end'], row[f'{b}_end'])) / 2
            # Correct if death within the overlap
            ov_start = max(row[f'{a}_start'], row[f'{b}_start'])
            ov_end = min(row[f'{a}_end'], row[f'{b}_end'])
            return int(ov_start <= row['death_jd'] <= ov_end)

        # No overlap → fall back to highest-confidence single tradition
        scores = {
            'par': (row['par_prob'], row['par_correct']),
            'yog': (row['yog_prob'] * 0.5, row['yog_correct']),
            'kp':  (row['kp_prob'], row['kp_correct']),
        }
        winner = max(scores, key=lambda k: scores[k][0])
        return int(scores[winner][1])

    df['path_a'] = df.apply(_path_a_predict, axis=1)
    pa_acc = df['path_a'].mean()
    print(f"  Path A accuracy: {pa_acc:.1%}")

    # Pairwise overlap stats
    print(f"\n  Pairwise top-1 overlap distribution:")
    for a, b in [('par', 'yog'), ('par', 'kp'), ('yog', 'kp')]:
        overlaps = df.apply(
            lambda r: _overlap(r[f'{a}_start'], r[f'{a}_end'],
                               r[f'{b}_start'], r[f'{b}_end']),
            axis=1)
        n_overlap = (overlaps > 0).sum()
        print(f"    {a}-{b}: {n_overlap}/{n} charts have non-zero overlap "
              f"({n_overlap/n:.0%})")

    # Path A accuracy when there IS overlap vs when there isn't
    df['has_overlap'] = df.apply(
        lambda r: max(_overlap(r['par_start'], r['par_end'],
                                r['yog_start'], r['yog_end']),
                       _overlap(r['par_start'], r['par_end'],
                                r['kp_start'], r['kp_end']),
                       _overlap(r['yog_start'], r['yog_end'],
                                r['kp_start'], r['kp_end'])) > 0,
        axis=1)
    n_with = df['has_overlap'].sum()
    n_without = (~df['has_overlap']).sum()
    if n_with > 0:
        acc_with = df[df['has_overlap']]['path_a'].mean()
        print(f"\n  When 2+ traditions overlap: {n_with} charts, "
              f"acc {acc_with:.1%}")
    if n_without > 0:
        acc_without = df[~df['has_overlap']]['path_a'].mean()
        print(f"  When NO overlap (fallback):  {n_without} charts, "
              f"acc {acc_without:.1%}")

    # ──────────────────────────────────────────────────────────────
    # Path B: Day-level voting across all candidates
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PATH B — Day-level voting across all candidates")
    print("=" * 75)
    print(f"  Building per-chart day grids...")

    # Need full per-candidate scores + date ranges
    # Annotate par_scores / yog_scores / kp_scores with date ranges
    def _annotate_with_dates(scores_df, dates_dict):
        starts, ends = [], []
        for _, row in scores_df.iterrows():
            d = dates_dict.get(int(row['group_id']), {}).get(
                int(row['cand_idx']), (None, None))
            starts.append(d[0])
            ends.append(d[1])
        out = scores_df.copy()
        out['start_jd'] = starts
        out['end_jd'] = ends
        return out

    par_full = _annotate_with_dates(par_scores, par_dates).dropna(
        subset=['start_jd'])
    yog_full = _annotate_with_dates(yog_scores, yog_dates).dropna(
        subset=['start_jd'])
    kp_full = _annotate_with_dates(kp_scores, kp_dates).dropna(
        subset=['start_jd'])

    # Normalize each tradition's scores within group (z-score)
    def _zscore_within_group(d):
        d = d.copy()
        g = d.groupby('group_id', sort=False)['score']
        mean = g.transform('mean')
        std = g.transform('std').clip(lower=1e-9)
        d['score_z'] = (d['score'] - mean) / std
        return d

    par_full = _zscore_within_group(par_full)
    yog_full = _zscore_within_group(yog_full)
    kp_full = _zscore_within_group(kp_full)

    # For each chart, build a day-level vote map
    DAY_GRID_RES = 1.0  # 1 day
    correct_count = 0
    eval_count = 0
    for gid in df['group_id'].unique():
        gid = int(gid)
        d_jd = death_jd.get(gid)
        if d_jd is None:
            continue
        # Window bounds = death ± 12 months
        win_start = d_jd - 365  # ~12 months before
        win_end = d_jd + 365
        n_days = int(win_end - win_start) + 1
        votes = np.zeros(n_days)

        for trad_df, weight in [
            (par_full, 1.0),
            (yog_full, 0.5),
            (kp_full, 1.0),
        ]:
            grp = trad_df[trad_df['group_id'] == gid]
            for _, row in grp.iterrows():
                s = max(row['start_jd'], win_start)
                e = min(row['end_jd'], win_end)
                if e <= s:
                    continue
                i_s = max(0, int(s - win_start))
                i_e = min(n_days, int(e - win_start) + 1)
                votes[i_s:i_e] += weight * row['score_z']

        if votes.max() == 0:
            continue
        best_day = int(np.argmax(votes))
        best_jd = win_start + best_day
        # Correct if best_jd is within ±15 days of actual death
        if abs(best_jd - d_jd) <= 15:
            correct_count += 1
        eval_count += 1

    pb_acc = correct_count / max(eval_count, 1)
    print(f"  Path B accuracy (best day within ±15 days): "
          f"{pb_acc:.1%} ({correct_count}/{eval_count})")

    # ── Final comparison ──────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)
    print(f"\n  Random baseline:                  ~5.9%")
    print(f"  Best single tradition (Par/KP):   ~13.1%")
    print(f"  Yogini:                            {yog_acc:.1%}")
    print(f"  Oracle ceiling (any correct):     {union:.1%}")
    print()
    print(f"  Rule 2 (highest margin) [prior]:  15.1%")
    print(f"  Path A (date-overlap voting):     {pa_acc:.1%}")
    print(f"  Path B (day-level voting ±15d):   {pb_acc:.1%}")

    # Save full table
    out = PROJECT_ROOT / 'ensemble_full_results.csv'
    df.to_csv(out, index=False)
    print(f"\n  Per-chart results saved: {out}")


if __name__ == '__main__':
    main()
