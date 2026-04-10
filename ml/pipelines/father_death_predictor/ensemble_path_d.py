"""Path D: Joint candidate ranking via score-weighted date intersection.

For each chart, take ALL candidates from all 3 traditions with their
date ranges and z-scored model scores. Walk the calendar day-by-day
in the death window and pick the day that has the highest sum of
overlapping z-scored scores from across the 3 traditions.

This is the most aggressive ensemble — it uses the FULL ranking
distribution from each tradition, not just the top-1.

Key difference from Path B: we use weighted z-scores at the candidate
level (not the day level), and we pick the day with maximum sum.

Usage:
    python -u -m ml.pipelines.father_death_predictor.ensemble_path_d
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

    if tradition in ('parashari', 'kp'):
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

    if tradition == 'yogini':
        all_yg = compute_yogini_dasha(moon_long, birth_jd, max_depth=3)
        pds = [p for p in all_yg if p.get('depth') == 3]
        cands, _, _, _ = construct_dasha_window(
            rec['father_death_date'], pds, window_months=24, seed=seed)
        return [(i, c['start_jd'], c['end_jd']) for i, c in enumerate(cands)]


def date_to_jd(date_str):
    from .astro_engine.ephemeris import compute_jd
    return compute_jd(date_str, '12:00')


def main():
    print("=" * 75)
    print("PATH D - Joint candidate ranking (date-weighted score sum)")
    print("=" * 75)

    train_recs = [r for r in _load_json(str(V2_JSON)) + _load_json(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load_json(str(VAL_JSON)) if _valid(r)]
    n_train = len(train_recs)
    start_index = n_train * 100
    print(f"  Train: {n_train}, Val: {len(val_recs)}")

    # ── Get full per-candidate scores from each tradition ─────
    print("\n[1] Training Parashari (xendcg)...")
    cl_tr = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    par_models = train_seed_avg(nodur_cols, cl_tr, cl_va, {
        'objective': 'rank_xendcg', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    })
    par_scores = predict_avg(par_models, cl_va[nodur_cols].values)
    par_full = cl_va[['group_id', 'cluster_idx_internal', 'label']].copy()
    par_full = par_full.rename(columns={'cluster_idx_internal': 'cand_idx'})
    par_full['score'] = par_scores

    print("\n[2] Training Yogini...")
    df_tr = pd.read_parquet(DATA_DIR / 'yogini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'yogini' / 'val_aug1.parquet')
    yog_feat = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    yog_feat = [c for c in yog_feat if c in df_tr.columns]
    yog_models = train_seed_avg(yog_feat, df_tr, df_va, {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    })
    yog_scores = predict_avg(yog_models, df_va[yog_feat].values)
    yog_full = df_va[['group_id', 'cand_idx', 'label']].copy()
    yog_full['score'] = yog_scores

    print("\n[3] Training KP...")
    cl_tr_kp = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'train_cl5_w24.parquet')
    cl_va_kp = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'val_cl5_w24.parquet')
    kp_feat = get_numeric_feat_cols(cl_va_kp, EXCLUDE, _DUR_PATTERNS)
    kp_feat = [c for c in kp_feat if c in cl_tr_kp.columns]
    kp_nodur = auto_drop_leaks(cl_tr_kp, kp_feat, 'cl_duration', 0.15)
    kp_models = train_seed_avg(kp_nodur, cl_tr_kp, cl_va_kp, {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 25,
        'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    })
    kp_scores_arr = predict_avg(kp_models, cl_va_kp[kp_nodur].values)
    kp_full = cl_va_kp[['group_id', 'cluster_idx_internal', 'label']].copy()
    kp_full = kp_full.rename(columns={'cluster_idx_internal': 'cand_idx'})
    kp_full['score'] = kp_scores_arr

    # Z-score within each chart group (so scales are comparable)
    def _zwithin(d):
        d = d.copy()
        g = d.groupby('group_id', sort=False)['score']
        d['score_z'] = (d['score'] - g.transform('mean')) / g.transform('std').clip(lower=1e-9)
        return d

    par_full = _zwithin(par_full)
    yog_full = _zwithin(yog_full)
    kp_full = _zwithin(kp_full)

    # ── Compute date ranges ────────────────────────────────────
    print("\n[4] Computing date ranges per chart...")
    t0 = time.time()
    par_dates = {}
    yog_dates = {}
    for i, rec in enumerate(val_recs):
        gid = (start_index + i) * 100
        try:
            par_dates[gid] = {
                ci: (s, e)
                for ci, s, e in compute_dates_for_chart(rec, 'parashari',
                                                         start_index + i)
            }
            yog_dates[gid] = {
                ci: (s, e)
                for ci, s, e in compute_dates_for_chart(rec, 'yogini',
                                                         start_index + i)
            }
        except Exception:
            pass
    print(f"  Done in {time.time()-t0:.0f}s")

    death_jd = {(start_index + i) * 100: date_to_jd(rec['father_death_date'])
                for i, rec in enumerate(val_recs)}

    # ── Path D: per-chart day-level vote with z-scores ────────
    print("\n[5] Running joint candidate ranking...")

    par_grouped = {gid: g for gid, g in par_full.groupby('group_id', sort=False)}
    yog_grouped = {gid: g for gid, g in yog_full.groupby('group_id', sort=False)}
    kp_grouped = {gid: g for gid, g in kp_full.groupby('group_id', sort=False)}

    correct_in_30 = 0
    correct_in_15 = 0
    correct_in_60 = 0
    n_eval = 0
    for gid in par_grouped:
        gid = int(gid)
        d_jd = death_jd.get(gid)
        if d_jd is None:
            continue
        if gid not in par_dates or gid not in yog_dates:
            continue
        win_start = d_jd - 365
        win_end = d_jd + 365
        n_days = int(win_end - win_start) + 1
        votes = np.zeros(n_days)

        # Parashari (weight 1.0)
        par_g = par_grouped[gid]
        for _, r in par_g.iterrows():
            ci = int(r['cand_idx'])
            d = par_dates[gid].get(ci)
            if d is None:
                continue
            s, e = d
            i_s = max(0, int(s - win_start))
            i_e = min(n_days, int(e - win_start) + 1)
            if i_e > i_s:
                votes[i_s:i_e] += 1.0 * r['score_z']

        # Yogini (weight 0.5 - anti-cal)
        yog_g = yog_grouped.get(gid)
        if yog_g is not None:
            for _, r in yog_g.iterrows():
                ci = int(r['cand_idx'])
                d = yog_dates[gid].get(ci)
                if d is None:
                    continue
                s, e = d
                i_s = max(0, int(s - win_start))
                i_e = min(n_days, int(e - win_start) + 1)
                if i_e > i_s:
                    votes[i_s:i_e] += 0.5 * r['score_z']

        # KP (weight 1.0, uses same Vimshottari dates as Parashari)
        kp_g = kp_grouped.get(gid)
        if kp_g is not None:
            for _, r in kp_g.iterrows():
                ci = int(r['cand_idx'])
                d = par_dates[gid].get(ci)  # same Vimshottari clusters
                if d is None:
                    continue
                s, e = d
                i_s = max(0, int(s - win_start))
                i_e = min(n_days, int(e - win_start) + 1)
                if i_e > i_s:
                    votes[i_s:i_e] += 1.0 * r['score_z']

        if votes.max() == votes.min():
            continue
        best_day = int(np.argmax(votes))
        best_jd = win_start + best_day
        diff = abs(best_jd - d_jd)
        if diff <= 15:
            correct_in_15 += 1
        if diff <= 30:
            correct_in_30 += 1
        if diff <= 60:
            correct_in_60 += 1
        n_eval += 1

    print(f"\n  Path D accuracy at various tolerances:")
    print(f"    +/- 15 days: {correct_in_15}/{n_eval} = {correct_in_15/max(n_eval,1):.1%}")
    print(f"    +/- 30 days: {correct_in_30}/{n_eval} = {correct_in_30/max(n_eval,1):.1%}")
    print(f"    +/- 60 days: {correct_in_60}/{n_eval} = {correct_in_60/max(n_eval,1):.1%}")
    print(f"\n  For comparison, baseline accuracy at +/-30 days "
          f"(random in 24mo window):")
    # Random baseline: (60 days / 730 days) ~= 8.2%
    print(f"    expected random: ~8.2%")


if __name__ == '__main__':
    main()
