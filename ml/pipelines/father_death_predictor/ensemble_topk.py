"""Top-3 and Top-5 metrics for the best ensemble strategies.

Computes:
  1. Per-tradition Top-3 / Top-5 baselines
  2. Rule 2 (highest margin) — uses chosen tradition's own Top-K
  3. Combined ranking — merges all candidates by z-scored score,
     picks Top-K from the union, "correct" = death in any of them

Usage:
    python -u -m ml.pipelines.father_death_predictor.ensemble_topk
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


def per_chart_topk(scores_df, k):
    """Return dict {gid: list of top-k cand_idxs sorted by score desc}."""
    out = {}
    for gid, grp in scores_df.groupby('group_id', sort=False):
        sorted_g = grp.sort_values('score', ascending=False)
        out[int(gid)] = sorted_g['cand_idx'].head(k).astype(int).tolist()
    return out


def per_chart_topk_with_label(scores_df, k):
    """Return dict {gid: 1 if any of top-k has label==1}."""
    out = {}
    for gid, grp in scores_df.groupby('group_id', sort=False):
        if grp['label'].max() == 0:
            continue
        sorted_g = grp.sort_values('score', ascending=False)
        topk = sorted_g['label'].head(k).values
        out[int(gid)] = int(topk.max() == 1)
    return out


def main():
    print("=" * 75)
    print("ENSEMBLE TOP-3 AND TOP-5 METRICS")
    print("=" * 75)

    train_recs = [r for r in _load_json(str(V2_JSON)) + _load_json(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load_json(str(VAL_JSON)) if _valid(r)]
    n_train = len(train_recs)
    start_index = n_train * 100

    # ── Train all 3 models ────────────────────────────────────
    print("\n[1] Training Parashari (xendcg)...")
    cl_tr = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    par_nodur = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    par_models = train_seed_avg(par_nodur, cl_tr, cl_va, {
        'objective': 'rank_xendcg', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    })
    par_full = cl_va[['group_id', 'cluster_idx_internal', 'label']].copy()
    par_full = par_full.rename(columns={'cluster_idx_internal': 'cand_idx'})
    par_full['score'] = predict_avg(par_models, cl_va[par_nodur].values)

    print("\n[2] Training Yogini (rank+clf blend)...")
    df_tr = pd.read_parquet(DATA_DIR / 'yogini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'yogini' / 'val_aug1.parquet')
    yog_feat = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    yog_feat = [c for c in yog_feat if c in df_tr.columns]
    yog_rank_models = train_seed_avg(yog_feat, df_tr, df_va, {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    })
    yog_rank_scores = predict_avg(yog_rank_models, df_va[yog_feat].values)
    params_clf = {
        'objective': 'binary', 'metric': 'binary_logloss',
        'num_leaves': 40, 'max_depth': 6, 'learning_rate': 0.03,
        'min_child_samples': 25, 'colsample_bytree': 0.8,
        'scale_pos_weight': 30.0,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    yog_clf_models = []
    for i in range(5):
        p = {**params_clf, 'seed': 42 + i * 17}
        m = lgb.train(p, lgb.Dataset(df_tr[yog_feat].values,
                                     label=df_tr['label'].values),
                      num_boost_round=500)
        yog_clf_models.append(m)
    yog_clf_scores = np.mean([m.predict(df_va[yog_feat].values)
                               for m in yog_clf_models], axis=0)

    def _norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 1e-10 else np.zeros_like(s)
    yog_blend = 0.5 * _norm(yog_rank_scores) + 0.5 * _norm(yog_clf_scores)
    yog_full = df_va[['group_id', 'cand_idx', 'label']].copy()
    yog_full['score'] = yog_blend

    print("\n[3] Training KP (nd_A)...")
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
    kp_full = cl_va_kp[['group_id', 'cluster_idx_internal', 'label']].copy()
    kp_full = kp_full.rename(columns={'cluster_idx_internal': 'cand_idx'})
    kp_full['score'] = predict_avg(kp_models, cl_va_kp[kp_nodur].values)

    # ── Per-tradition Top-K baselines ─────────────────────────
    print("\n" + "=" * 75)
    print("PER-TRADITION TOP-K BASELINES")
    print("=" * 75)
    for k in [1, 3, 5]:
        for name, scores in [('Parashari', par_full), ('Yogini', yog_full),
                              ('KP', kp_full)]:
            d = per_chart_topk_with_label(scores, k)
            n = len(d)
            n_correct = sum(d.values())
            print(f"  {name:>10s} Top-{k}: {n_correct}/{n} = {n_correct/n:.1%}")
        print()

    # ── Compute date ranges per chart ─────────────────────────
    print("[4] Computing date ranges per chart...")
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

    # ── Per-chart Top-1 + margin for each tradition ───────────
    def _per_chart_top1_margin(scores_df):
        out = {}
        for gid, grp in scores_df.groupby('group_id', sort=False):
            if grp['label'].max() == 0:
                continue
            sorted_g = grp.sort_values('score', ascending=False).reset_index(drop=True)
            s = sorted_g['score'].values
            margin = float(s[0] - s[1]) if len(s) > 1 else 0.0
            out[int(gid)] = {
                'top1_cand': int(sorted_g['cand_idx'].iloc[0]),
                'top1_correct': int(sorted_g['label'].iloc[0] == 1),
                'top3_correct': int(sorted_g['label'].head(3).max() == 1),
                'top5_correct': int(sorted_g['label'].head(5).max() == 1),
                'margin': margin,
            }
        return out

    par_per = _per_chart_top1_margin(par_full)
    yog_per = _per_chart_top1_margin(yog_full)
    kp_per = _per_chart_top1_margin(kp_full)

    common = set(par_per.keys()) & set(yog_per.keys()) & set(kp_per.keys())
    n = len(common)
    print(f"\n  Common charts: {n}")

    # ── Rule 2: highest margin wins, use that tradition's Top-K ──
    print("\n" + "=" * 75)
    print("RULE 2 — pick tradition with highest margin, use ITS Top-K")
    print("=" * 75)
    rule2_t1 = 0
    rule2_t3 = 0
    rule2_t5 = 0
    rule2_choices = {'par': 0, 'yog': 0, 'kp': 0}
    for gid in common:
        margins = {
            'par': par_per[gid]['margin'],
            'yog': yog_per[gid]['margin'],
            'kp': kp_per[gid]['margin'],
        }
        winner = max(margins, key=margins.get)
        rule2_choices[winner] += 1
        chosen_per = {'par': par_per, 'yog': yog_per, 'kp': kp_per}[winner]
        rule2_t1 += chosen_per[gid]['top1_correct']
        rule2_t3 += chosen_per[gid]['top3_correct']
        rule2_t5 += chosen_per[gid]['top5_correct']

    print(f"  Top-1: {rule2_t1}/{n} = {rule2_t1/n:.1%}")
    print(f"  Top-3: {rule2_t3}/{n} = {rule2_t3/n:.1%}")
    print(f"  Top-5: {rule2_t5}/{n} = {rule2_t5/n:.1%}")
    print(f"  Winner distribution: {rule2_choices}")

    # ── Combined ranking: union of all 3 tradition's candidates ──
    print("\n" + "=" * 75)
    print("COMBINED RANKING — merge candidates from all 3 by z-scored score")
    print("=" * 75)

    def _zwithin(d):
        d = d.copy()
        g = d.groupby('group_id', sort=False)['score']
        d['score_z'] = (d['score'] - g.transform('mean')) / g.transform('std').clip(lower=1e-9)
        return d

    par_z = _zwithin(par_full)
    yog_z = _zwithin(yog_full)
    kp_z = _zwithin(kp_full)

    par_g = {gid: g for gid, g in par_z.groupby('group_id', sort=False)}
    yog_g = {gid: g for gid, g in yog_z.groupby('group_id', sort=False)}
    kp_g = {gid: g for gid, g in kp_z.groupby('group_id', sort=False)}

    combined_t1 = 0
    combined_t3 = 0
    combined_t5 = 0
    n_eval = 0

    for gid in common:
        d_jd = death_jd.get(gid)
        if d_jd is None:
            continue
        # Build candidate list: (score_z, start_jd, end_jd, source)
        cands_list = []

        # Parashari (weight 1.0)
        if gid in par_g and gid in par_dates:
            for _, r in par_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in par_dates[gid]:
                    s, e = par_dates[gid][ci]
                    cands_list.append((1.0 * r['score_z'], s, e, 'par'))

        # Yogini (weight 0.5 — anti-cal)
        if gid in yog_g and gid in yog_dates:
            for _, r in yog_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in yog_dates[gid]:
                    s, e = yog_dates[gid][ci]
                    cands_list.append((0.5 * r['score_z'], s, e, 'yog'))

        # KP (weight 1.0, same dates as Parashari)
        if gid in kp_g and gid in par_dates:
            for _, r in kp_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in par_dates[gid]:
                    s, e = par_dates[gid][ci]
                    cands_list.append((1.0 * r['score_z'], s, e, 'kp'))

        if not cands_list:
            continue

        # Sort by score desc, then check top-K for death containment
        cands_list.sort(key=lambda x: -x[0])

        def _hit_in_topk(k):
            for sc, s, e, src in cands_list[:k]:
                if s <= d_jd <= e:
                    return 1
            return 0

        combined_t1 += _hit_in_topk(1)
        combined_t3 += _hit_in_topk(3)
        combined_t5 += _hit_in_topk(5)
        n_eval += 1

    print(f"  Charts evaluated: {n_eval}")
    print(f"  Top-1: {combined_t1}/{n_eval} = {combined_t1/n_eval:.1%}")
    print(f"  Top-3: {combined_t3}/{n_eval} = {combined_t3/n_eval:.1%}")
    print(f"  Top-5: {combined_t5}/{n_eval} = {combined_t5/n_eval:.1%}")

    # ── Combined: equal weight (no Yogini downweight) ─────────
    print("\n" + "=" * 75)
    print("COMBINED RANKING — EQUAL weights (no Yogini downweight)")
    print("=" * 75)
    combined_eq_t1 = 0
    combined_eq_t3 = 0
    combined_eq_t5 = 0
    n_eval2 = 0

    for gid in common:
        d_jd = death_jd.get(gid)
        if d_jd is None:
            continue
        cands_list = []
        if gid in par_g and gid in par_dates:
            for _, r in par_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in par_dates[gid]:
                    s, e = par_dates[gid][ci]
                    cands_list.append((r['score_z'], s, e))
        if gid in yog_g and gid in yog_dates:
            for _, r in yog_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in yog_dates[gid]:
                    s, e = yog_dates[gid][ci]
                    cands_list.append((r['score_z'], s, e))
        if gid in kp_g and gid in par_dates:
            for _, r in kp_g[gid].iterrows():
                ci = int(r['cand_idx'])
                if ci in par_dates[gid]:
                    s, e = par_dates[gid][ci]
                    cands_list.append((r['score_z'], s, e))
        if not cands_list:
            continue
        cands_list.sort(key=lambda x: -x[0])

        def _hit(k):
            for sc, s, e in cands_list[:k]:
                if s <= d_jd <= e:
                    return 1
            return 0

        combined_eq_t1 += _hit(1)
        combined_eq_t3 += _hit(3)
        combined_eq_t5 += _hit(5)
        n_eval2 += 1

    print(f"  Top-1: {combined_eq_t1}/{n_eval2} = {combined_eq_t1/n_eval2:.1%}")
    print(f"  Top-3: {combined_eq_t3}/{n_eval2} = {combined_eq_t3/n_eval2:.1%}")
    print(f"  Top-5: {combined_eq_t5}/{n_eval2} = {combined_eq_t5/n_eval2:.1%}")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL TOP-K SUMMARY")
    print("=" * 75)
    print(f"\n  {'Method':<35s} {'Top-1':>8s} {'Top-3':>8s} {'Top-5':>8s}")
    print("  " + "-" * 60)
    par_t1 = sum(1 for g in par_per.values() if g['top1_correct'])
    par_t3 = sum(1 for g in par_per.values() if g['top3_correct'])
    par_t5 = sum(1 for g in par_per.values() if g['top5_correct'])
    n_par = len(par_per)
    yog_t1 = sum(1 for g in yog_per.values() if g['top1_correct'])
    yog_t3 = sum(1 for g in yog_per.values() if g['top3_correct'])
    yog_t5 = sum(1 for g in yog_per.values() if g['top5_correct'])
    n_yog = len(yog_per)
    kp_t1 = sum(1 for g in kp_per.values() if g['top1_correct'])
    kp_t3 = sum(1 for g in kp_per.values() if g['top3_correct'])
    kp_t5 = sum(1 for g in kp_per.values() if g['top5_correct'])
    n_kp = len(kp_per)
    print(f"  {'Parashari (single)':<35s} "
          f"{par_t1/n_par:>7.1%} {par_t3/n_par:>7.1%} {par_t5/n_par:>7.1%}")
    print(f"  {'Yogini (single)':<35s} "
          f"{yog_t1/n_yog:>7.1%} {yog_t3/n_yog:>7.1%} {yog_t5/n_yog:>7.1%}")
    print(f"  {'KP (single)':<35s} "
          f"{kp_t1/n_kp:>7.1%} {kp_t3/n_kp:>7.1%} {kp_t5/n_kp:>7.1%}")
    print()
    print(f"  {'Rule 2 (highest margin pick)':<35s} "
          f"{rule2_t1/n:>7.1%} {rule2_t3/n:>7.1%} {rule2_t5/n:>7.1%}")
    print(f"  {'Combined (Yog x0.5)':<35s} "
          f"{combined_t1/n_eval:>7.1%} {combined_t3/n_eval:>7.1%} "
          f"{combined_t5/n_eval:>7.1%}")
    print(f"  {'Combined (equal)':<35s} "
          f"{combined_eq_t1/n_eval2:>7.1%} {combined_eq_t3/n_eval2:>7.1%} "
          f"{combined_eq_t5/n_eval2:>7.1%}")


if __name__ == '__main__':
    main()
