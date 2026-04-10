"""SD (Sookshma) on 24-month window.

Usage:
    python -u -m ml.pipelines.father_death_predictor.run_sd_24mo
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'

META_COLS = ['group_id', 'cand_idx', 'label', 'tier', 'danger_score',
             'duration_days']


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


def main():
    print("=" * 75)
    print("SD (SOOKSHMA) ON 24-MONTH WINDOW")
    print("=" * 75)

    train_recs = [r for r in _load(str(V2_JSON)) + _load(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load(str(VAL_JSON)) if _valid(r)]
    print(f"  Train: {len(train_recs)}, Val: {len(val_recs)}")

    from .run_dasha_depth import build_depth_dataset

    print("\n=== FEATURE EXTRACTION ===")
    df_tr = build_depth_dataset(train_recs, 'train_24mo', 0,
                                 target_depth=4, window_months=24, n_augment=1)
    df_va = build_depth_dataset(val_recs, 'val_24mo',
                                 len(train_recs) * 100,
                                 target_depth=4, window_months=24, n_augment=1)

    if len(df_va) == 0 or 'group_id' not in df_va.columns:
        print("  No data!")
        return

    gs = df_va.groupby('group_id').size()
    valid = gs[gs >= 2].index
    df_tr_f = df_tr[df_tr['group_id'].isin(
        df_tr.groupby('group_id').filter(
            lambda x: len(x) >= 2)['group_id'].unique())]
    df_va_f = df_va[df_va['group_id'].isin(valid)]
    gs_f = df_va_f.groupby('group_id').size()
    n_va = df_va_f['group_id'].nunique()

    feat_cols = [c for c in df_va_f.columns
                 if c not in META_COLS
                 and isinstance(df_va_f[c].iloc[0],
                                 (int, float, np.integer, np.floating))]
    feat_cols = [c for c in feat_cols
                 if c in df_tr_f.columns and c in df_va_f.columns]

    no_dur_cols = [c for c in feat_cols
                   if 'duration' not in c and 'dur_' not in c
                   and c not in ('seq_duration_days', 'seq_dur_vs_mean',
                                 'seq_dur_log', 'seq_danger_intensity',
                                 'duration_days')]

    print(f"\n  Val groups: {n_va}")
    print(f"  Candidates/group: mean={gs_f.mean():.1f}, "
          f"median={gs_f.median():.0f}, min={gs_f.min()}, max={gs_f.max()}")
    print(f"  Features: {len(feat_cols)} (no-dur: {len(no_dur_cols)})")

    # Duration stats
    all_dur = df_va_f['duration_days']
    death_dur = df_va_f[df_va_f['label'] == 1]['duration_days']
    print(f"  Avg duration: all={all_dur.mean():.1f}d, "
          f"death={death_dur.mean():.1f}d")

    # Random baseline
    rnd_t1 = (1.0 / gs_f.values).mean()
    rnd_t3 = np.minimum(3.0 / gs_f.values, 1.0).mean()
    rnd_t5 = np.minimum(5.0 / gs_f.values, 1.0).mean()

    # Duration baseline
    dur_topk, _ = eval_topk(df_va_f, df_va_f['duration_days'].values)

    # Model training
    params = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.05, 'subsample': 0.8,
        'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 25,
        'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
        'verbose': -1, 'device': 'gpu', 'gpu_use_dp': False,
    }

    print("\n  Training full model...")
    models = train_seed_avg(feat_cols, df_tr_f, df_va_f, 'group_id', params)
    model_scores = predict_avg(models, df_va_f[feat_cols].values)
    model_topk, _ = eval_topk(df_va_f, model_scores)

    print("  Training no-duration model...")
    nd_models = train_seed_avg(no_dur_cols, df_tr_f, df_va_f, 'group_id', params)
    nd_scores = predict_avg(nd_models, df_va_f[no_dur_cols].values)
    nd_topk, _ = eval_topk(df_va_f, nd_scores)

    # Results
    print(f"\n{'='*75}")
    print(f"SD 24-MONTH RESULTS")
    print(f"{'='*75}")
    print(f"  {'Method':>25s} {'Top-1':>7s} {'Top-3':>7s} "
          f"{'Top-5':>7s} {'Lift':>6s}")
    print(f"  {'-'*55}")
    print(f"  {'Random':>25s} {rnd_t1:6.1%}  {rnd_t3:6.1%}  "
          f"{rnd_t5:6.1%}  {'1.0x':>6s}")
    print(f"  {'Duration only':>25s} "
          f"{dur_topk['top_1']:6.1%}  {dur_topk['top_3']:6.1%}  "
          f"{dur_topk['top_5']:6.1%}  "
          f"{dur_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"  {'Model (no duration)':>25s} "
          f"{nd_topk['top_1']:6.1%}  {nd_topk['top_3']:6.1%}  "
          f"{nd_topk['top_5']:6.1%}  "
          f"{nd_topk['top_1']/max(rnd_t1,0.001):.1f}x")
    print(f"  {'Full model':>25s} "
          f"{model_topk['top_1']:6.1%}  {model_topk['top_3']:6.1%}  "
          f"{model_topk['top_5']:6.1%}  "
          f"{model_topk['top_1']/max(rnd_t1,0.001):.1f}x")

    print(f"\n  Comparison:")
    print(f"    SD 6mo:  23.1% top-1, 4.9x lift (30 cands)")
    print(f"    SD 24mo: {model_topk['top_1']:.1%} top-1, "
          f"{model_topk['top_1']/max(rnd_t1,0.001):.1f}x lift "
          f"({gs_f.mean():.0f} cands)")
    print("=" * 75)


if __name__ == '__main__':
    main()
