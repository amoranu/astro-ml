"""Compare per-chart Top-1 correctness across Parashari, Yogini, Jaimini.

For each tradition:
  1. Load cached val + train datasets
  2. Train the same model used in the best honest result
  3. Predict Top-1 per chart
  4. Mark chart as "correct" if the predicted top-1 contains the death

Then compute the overlap matrix:
  - Charts where ALL 3 traditions are correct
  - Charts where exactly 2 / 1 / 0 are correct
  - Pairwise agreement (Jaccard)

Usage:
    python -u -m ml.pipelines.father_death_predictor.compare_top1_overlap
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'


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


def per_chart_top1_correct(df, scores, gcol='group_id'):
    """Return dict {group_id: 1 if top-1 prediction is the correct cluster}."""
    df = df.copy()
    df['_score'] = scores
    out = {}
    for gid, grp in df.groupby(gcol, sort=False):
        if grp['label'].max() == 0:
            continue  # No positive in this group (skip)
        ranked = grp.sort_values('_score', ascending=False)
        top_label = ranked['label'].iloc[0]
        out[gid] = int(top_label == 1)
    return out


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


# Standard duration / proxy exclusion patterns
_DUR_PATTERNS = (
    'duration', 'dur_', 'movement', '_ingress', 'sign_change',
)
_HARD_PROXIES = {
    'cand_idx', 'cluster_idx', 'cluster_idx_internal', 'cl_duration',
    'seq_pos_norm', 'seq_third', 'seq_duration_days', 'seq_dur_vs_mean',
    'seq_dur_log', 'seq_danger_intensity',
    'id_pl_planet_idx', 'id_al_planet_idx', 'id_lagna_planet_combo',
}


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


# ── Tradition-specific eval ────────────────────────────────────────────

def eval_parashari():
    """Parashari SD-cluster cl=5 (xendcg variant, ~33 days)."""
    print("=" * 75)
    print("PARASHARI SD-cluster cl=5 (xendcg, ~33 days)")
    print("=" * 75)

    cl_tr = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'train_cl5_w24.parquet')
    cl_va = pd.read_parquet(DATA_DIR / 'sd_clusters_v2' / 'val_cl5_w24.parquet')

    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(cl_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in cl_tr.columns]
    nodur_cols = auto_drop_leaks(cl_tr, feat_cols, 'cl_duration', 0.15)
    print(f"  Features no-dur: {len(nodur_cols)}")

    params_xendcg = {
        'objective': 'rank_xendcg', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    models = train_seed_avg(nodur_cols, cl_tr, cl_va, params_xendcg)
    scores = predict_avg(models, cl_va[nodur_cols].values)

    per_chart = per_chart_top1_correct(cl_va, scores)
    n = len(per_chart)
    n_correct = sum(per_chart.values())
    print(f"  Charts: {n}, top-1 correct: {n_correct} ({n_correct/n:.1%})")
    return per_chart


def eval_yogini():
    """Yogini PD d=3 (~27 days)."""
    print("=" * 75)
    print("YOGINI PD d=3 (~27 days)")
    print("=" * 75)

    df_tr = pd.read_parquet(DATA_DIR / 'yogini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'yogini' / 'val_aug1.parquet')

    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Features no-dur: {len(feat_cols)}")

    params_base = {
        'objective': 'lambdarank', 'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.03, 'subsample': 0.8,
        'num_leaves': 40, 'max_depth': 6, 'min_child_samples': 10,
        'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.5,
        'verbose': -1, 'device': 'cpu', 'num_threads': 16,
    }
    rank_models = train_seed_avg(feat_cols, df_tr, df_va, params_base)
    rank_scores = predict_avg(rank_models, df_va[feat_cols].values)

    # Binary classifier blend (the EXP-017 best is blend)
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

    per_chart = per_chart_top1_correct(df_va, blend)
    n = len(per_chart)
    n_correct = sum(per_chart.values())
    print(f"  Charts: {n}, top-1 correct: {n_correct} ({n_correct/n:.1%})")
    return per_chart


def eval_jaimini():
    """Jaimini Chara PD d=3 (~15 days)."""
    print("=" * 75)
    print("JAIMINI Chara PD d=3 (~15 days)")
    print("=" * 75)

    df_tr = pd.read_parquet(DATA_DIR / 'jaimini' / 'train_aug5.parquet')
    df_va = pd.read_parquet(DATA_DIR / 'jaimini' / 'val_aug1.parquet')

    EXCLUDE = _HARD_PROXIES | {'group_id', 'label'}
    feat_cols = get_numeric_feat_cols(df_va, EXCLUDE, _DUR_PATTERNS)
    feat_cols = [c for c in feat_cols if c in df_tr.columns]
    print(f"  Features no-dur: {len(feat_cols)}")

    # EXP-022 best: nd_C variant (lv=64, depth=8)
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

    per_chart = per_chart_top1_correct(df_va, scores)
    n = len(per_chart)
    n_correct = sum(per_chart.values())
    print(f"  Charts: {n}, top-1 correct: {n_correct} ({n_correct/n:.1%})")
    return per_chart


def main():
    par = eval_parashari()
    yog = eval_yogini()
    jai = eval_jaimini()

    # Inner join on chart ids
    common = set(par.keys()) & set(yog.keys()) & set(jai.keys())
    print(f"\nCommon charts (inner join): {len(common)}")

    rows = []
    for cid in sorted(common):
        rows.append({
            'chart_id': cid,
            'parashari': par[cid],
            'yogini': yog[cid],
            'jaimini': jai[cid],
        })
    df = pd.DataFrame(rows)

    # Save IMMEDIATELY in case the printing crashes on Unicode
    out_path = PROJECT_ROOT / 'top1_overlap.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  [saved] {out_path}")

    n = len(df)
    print(f"\n{'='*75}")
    print(f"OVERLAP ANALYSIS — {n} charts")
    print(f"{'='*75}")

    # Per-tradition recall on common set
    print(f"\n  Per-tradition Top-1 accuracy on common set:")
    for col in ['parashari', 'yogini', 'jaimini']:
        c = df[col].sum()
        print(f"    {col:>12s}: {c}/{n} = {c/n:.1%}")

    # All correct, exactly 2, exactly 1, none
    df['n_correct'] = df[['parashari', 'yogini', 'jaimini']].sum(axis=1)
    print(f"\n  Distribution of agreement:")
    for k in [3, 2, 1, 0]:
        nk = (df['n_correct'] == k).sum()
        print(f"    {k} correct: {nk}/{n} = {nk/n:.1%}")

    # Pairwise overlap
    print(f"\n  Pairwise (charts where BOTH are correct):")
    pairs = [('parashari', 'yogini'), ('parashari', 'jaimini'),
             ('yogini', 'jaimini')]
    for a, b in pairs:
        both = ((df[a] == 1) & (df[b] == 1)).sum()
        either = ((df[a] == 1) | (df[b] == 1)).sum()
        a_only = ((df[a] == 1) & (df[b] == 0)).sum()
        b_only = ((df[a] == 0) & (df[b] == 1)).sum()
        jaccard = both / max(either, 1)
        print(f"    {a:>10s} AND {b:>10s}: both={both}, "
              f"{a}_only={a_only}, {b}_only={b_only}, "
              f"jaccard={jaccard:.2f}")

    # Independence test: are correct picks independent?
    print(f"\n  Independence check (expected vs observed if independent):")
    for a, b in pairs:
        pa = df[a].mean()
        pb = df[b].mean()
        observed_both = ((df[a] == 1) & (df[b] == 1)).mean()
        expected_both = pa * pb
        ratio = observed_both / max(expected_both, 1e-9)
        print(f"    P({a}=1, {b}=1): observed={observed_both:.3f}, "
              f"expected={expected_both:.3f}, ratio={ratio:.2f} "
              f"({'INDEPENDENT' if abs(ratio-1) < 0.3 else 'CORRELATED'})")

    # Union (charts where AT LEAST ONE is correct)
    union = (df['n_correct'] >= 1).sum()
    print(f"\n  Union (at least 1 tradition correct): "
          f"{union}/{n} = {union/n:.1%}")
    print(f"  Intersection (all 3 correct): "
          f"{(df['n_correct'] == 3).sum()}/{n} = "
          f"{(df['n_correct'] == 3).mean():.1%}")

    # Save per-chart table
    out_path = PROJECT_ROOT / 'top1_overlap.csv'
    df.to_csv(out_path, index=False)
    print(f"\n  Per-chart results saved to: {out_path}")


if __name__ == '__main__':
    main()
