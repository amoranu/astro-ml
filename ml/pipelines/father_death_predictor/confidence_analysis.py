"""Confidence calibration analysis for the best Parashari model.

For each chart, compute:
  - score of top-1 candidate
  - score of top-2 candidate
  - margin = top1 - top2 (raw confidence)
  - normalized margin (top1 - top2) / (top1 - bottom)
  - is_correct (1 if top-1 = death cluster)

Then:
  - Sort charts by confidence (margin)
  - Report accuracy at top-X% confident subsets
  - Report accuracy in confidence bins (low/mid/high)

If the model is calibrated, high-confidence picks should be more accurate
than the overall average (13.1% Top-1).

Usage:
    python -u -m ml.pipelines.father_death_predictor.confidence_analysis
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'

_DUR_PATTERNS = (
    'duration', 'dur_', 'movement', '_ingress', 'sign_change',
)
_HARD_PROXIES = {
    'cand_idx', 'cluster_idx', 'cluster_idx_internal', 'cl_duration',
    'seq_pos_norm', 'seq_third', 'seq_duration_days', 'seq_dur_vs_mean',
    'seq_dur_log', 'seq_danger_intensity',
    'id_pl_planet_idx', 'id_al_planet_idx', 'id_lagna_planet_combo',
}


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


def per_chart_confidence(df, scores, gcol='group_id'):
    """Return per-chart records with confidence + correctness."""
    df = df.copy()
    df['_score'] = scores
    rows = []
    for gid, grp in df.groupby(gcol, sort=False):
        if grp['label'].max() == 0:
            continue
        sorted_g = grp.sort_values('_score', ascending=False)
        labels = sorted_g['label'].values
        s = sorted_g['_score'].values

        top1_score = float(s[0])
        top2_score = float(s[1]) if len(s) > 1 else float(s[0])
        bottom = float(s[-1])

        margin_abs = top1_score - top2_score
        spread = max(top1_score - bottom, 1e-9)
        margin_norm = margin_abs / spread

        # Softmax-style confidence (assumes scores are roughly logit-like)
        ex = np.exp(s - s.max())
        prob = ex / ex.sum()
        top1_prob = float(prob[0])

        rows.append({
            'chart_id': gid,
            'top1_correct': int(labels[0] == 1),
            'top1_score': top1_score,
            'top2_score': top2_score,
            'margin_abs': margin_abs,
            'margin_norm': margin_norm,
            'top1_prob': top1_prob,
            'n_candidates': len(s),
        })
    return pd.DataFrame(rows)


def report(name, df):
    """Print confidence calibration report for one tradition."""
    n = len(df)
    overall = df['top1_correct'].mean()
    print(f"\n{'='*75}")
    print(f"{name}  (n={n}, overall Top-1={overall:.1%})")
    print(f"{'='*75}")

    # Top-X% confident
    df_sorted = df.sort_values('margin_abs', ascending=False).reset_index(drop=True)
    print(f"\n  Top-X% most confident charts (sorted by margin_abs):")
    print(f"  {'%X':>5s} {'N':>5s} {'#correct':>10s} {'accuracy':>10s} "
          f"{'lift':>7s}")
    print("  " + "-" * 45)
    for pct in [5, 10, 20, 30, 50, 100]:
        n_take = max(1, int(n * pct / 100))
        sub = df_sorted.iloc[:n_take]
        acc = sub['top1_correct'].mean()
        lift = acc / max(overall, 1e-9)
        print(f"  {pct:>4d}% {n_take:>5d} "
              f"{int(sub['top1_correct'].sum()):>10d} "
              f"{acc:>9.1%} {lift:>6.2f}x")

    # Margin quartiles
    print(f"\n  Margin_abs quartile accuracy:")
    df = df.copy()
    df['q'] = pd.qcut(df['margin_abs'], q=4, duplicates='drop',
                       labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    qstats = df.groupby('q', observed=True).agg(
        n=('chart_id', 'size'),
        n_correct=('top1_correct', 'sum'),
        accuracy=('top1_correct', 'mean'),
        avg_margin=('margin_abs', 'mean'),
    )
    print(qstats.to_string())

    # Top1_prob quartiles
    print(f"\n  Top1_prob (softmax) quartile accuracy:")
    df['qp'] = pd.qcut(df['top1_prob'], q=4, duplicates='drop',
                        labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    qpstats = df.groupby('qp', observed=True).agg(
        n=('chart_id', 'size'),
        n_correct=('top1_correct', 'sum'),
        accuracy=('top1_correct', 'mean'),
        avg_prob=('top1_prob', 'mean'),
    )
    print(qpstats.to_string())

    # Summary insights
    top10 = df_sorted.iloc[:max(1, n // 10)]
    top10_acc = top10['top1_correct'].mean()
    bot25_q = df_sorted.iloc[3 * n // 4:]
    bot25_acc = bot25_q['top1_correct'].mean()
    print(f"\n  Most-confident 10%: {top10_acc:.1%} "
          f"({top10_acc/max(overall,1e-9):.1f}x)")
    print(f"  Least-confident 25%: {bot25_acc:.1%} "
          f"({bot25_acc/max(overall,1e-9):.1f}x)")
    return df


def eval_parashari():
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
    return per_chart_confidence(cl_va, scores)


def eval_yogini():
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
    return per_chart_confidence(df_va, blend)


def eval_jaimini():
    df_tr = pd.read_parquet(DATA_DIR / 'jaimini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'jaimini' / 'val_aug1.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Jaimini features: {len(feat_cols)}")
    params_C = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 64, 'max_depth': 8, 'min_child_samples': 20,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(feat_cols, df_tr, df_va, params_C)
    scores = predict_avg(models, df_va[feat_cols].values)
    return per_chart_confidence(df_va, scores)


def eval_kp():
    cl_tr = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'kp_clusters' / 'val_cl5_w24.parquet')
    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    print(f"  KP features: {len(nodur_cols)}")
    # EXP-028 best variant: nd_A_colsamp09
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
    return per_chart_confidence(cl_va, scores)


def main():
    print("=" * 75)
    print("CONFIDENCE CALIBRATION — all traditions at ~30-day granularity")
    print("=" * 75)

    print("\n[1/4] PARASHARI cl=5 (xendcg)...")
    par = eval_parashari()
    par_out = report("PARASHARI cl=5 (~33 days)", par)
    par_out.to_csv(PROJECT_ROOT / 'parashari_confidence.csv', index=False)

    print("\n[2/4] YOGINI PD d=3 (blend)...")
    yog = eval_yogini()
    yog_out = report("YOGINI PD d=3 (~27 days)", yog)
    yog_out.to_csv(PROJECT_ROOT / 'yogini_confidence.csv', index=False)

    print("\n[3/4] JAIMINI Chara PD d=3...")
    jai = eval_jaimini()
    jai_out = report("JAIMINI Chara PD d=3 (~15 days)", jai)
    jai_out.to_csv(PROJECT_ROOT / 'jaimini_confidence.csv', index=False)

    print("\n[4/4] KP cluster cl=5 (nd_A)...")
    kp = eval_kp()
    kp_out = report("KP cluster cl=5 (~33 days)", kp)
    kp_out.to_csv(PROJECT_ROOT / 'kp_confidence.csv', index=False)

    # ── Cross-tradition summary ───────────────────────────────────
    print(f"\n{'='*75}")
    print(f"CROSS-TRADITION CONFIDENCE COMPARISON")
    print(f"{'='*75}")
    summary_rows = []
    for name, df in [('Parashari', par), ('Yogini', yog),
                      ('Jaimini', jai), ('KP', kp)]:
        n = len(df)
        overall = df['top1_correct'].mean()
        df_sorted = df.sort_values('margin_abs', ascending=False).reset_index(drop=True)
        top10 = df_sorted.iloc[:max(1, n // 10)]['top1_correct'].mean()
        top25 = df_sorted.iloc[:max(1, n // 4)]['top1_correct'].mean()
        bot25 = df_sorted.iloc[3 * n // 4:]['top1_correct'].mean()
        summary_rows.append({
            'tradition': name,
            'n': n,
            'overall': overall,
            'top10pct_acc': top10,
            'top25pct_acc': top25,
            'bot25pct_acc': bot25,
            'top10_lift': top10 / max(overall, 1e-9),
            'spread_top10_vs_bot25': top10 - bot25,
        })
    summary_df = pd.DataFrame(summary_rows)
    print()
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
