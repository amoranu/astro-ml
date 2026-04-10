"""Multi-tradition ensemble for Top-1 prediction.

Uses the saved per-chart confidence CSVs from confidence_analysis.py:
  parashari_confidence.csv, yogini_confidence.csv, kp_confidence.csv

For each chart, evaluates several ensemble rules:
  1. Best single tradition (oracle-by-confidence within the BEST tradition)
  2. Confidence-weighted selection: pick the prediction from whichever
     tradition has the highest top1_prob (or margin)
  3. Per-tradition reliability weighting (Parashari > KP > Yogini per
     calibration analysis)
  4. Oracle ceiling: if any tradition is right, ensemble is right (= union)

Reports each strategy's Top-1 on the common chart set.

Usage:
    python -u -m ml.pipelines.father_death_predictor.ensemble_top1
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    par = pd.read_csv(PROJECT_ROOT / 'parashari_confidence.csv')
    yog = pd.read_csv(PROJECT_ROOT / 'yogini_confidence.csv')
    kp = pd.read_csv(PROJECT_ROOT / 'kp_confidence.csv')

    print(f"  Loaded: Parashari={len(par)}, Yogini={len(yog)}, KP={len(kp)}")

    # Inner-join on chart_id
    df = par.merge(yog, on='chart_id', suffixes=('_par', '_yog'))
    df = df.merge(kp, on='chart_id')
    df = df.rename(columns={
        'top1_correct': 'top1_correct_kp',
        'top1_prob': 'top1_prob_kp',
        'margin_abs': 'margin_abs_kp',
        'margin_norm': 'margin_norm_kp',
        'top1_score': 'top1_score_kp',
        'top2_score': 'top2_score_kp',
        'n_candidates': 'n_candidates_kp',
    })
    n = len(df)
    print(f"  Common charts: {n}\n")

    # ── Per-tradition baselines ────────────────────────────────────
    print("=" * 75)
    print("Per-tradition Top-1 (on common set)")
    print("=" * 75)
    par_acc = df['top1_correct_par'].mean()
    yog_acc = df['top1_correct_yog'].mean()
    kp_acc = df['top1_correct_kp'].mean()
    print(f"  Parashari:  {par_acc:.1%}")
    print(f"  Yogini:     {yog_acc:.1%}")
    print(f"  KP:         {kp_acc:.1%}")

    # ── Oracle ceilings ────────────────────────────────────────────
    df['any_correct'] = (
        (df['top1_correct_par'] == 1) |
        (df['top1_correct_yog'] == 1) |
        (df['top1_correct_kp'] == 1)
    ).astype(int)
    df['all_correct'] = (
        (df['top1_correct_par'] == 1) &
        (df['top1_correct_yog'] == 1) &
        (df['top1_correct_kp'] == 1)
    ).astype(int)
    df['n_correct'] = (
        df['top1_correct_par'] +
        df['top1_correct_yog'] +
        df['top1_correct_kp']
    )

    print(f"\n  Oracle ceiling (any correct): {df['any_correct'].mean():.1%}")
    print(f"  All-3 agree (intersection):   {df['all_correct'].mean():.1%}")
    print(f"  Distribution of n_correct (out of 3):")
    for k in [3, 2, 1, 0]:
        nk = (df['n_correct'] == k).sum()
        print(f"    {k}: {nk}/{n} = {nk/n:.1%}")

    # ── Ensemble Rule 1: highest top1_prob wins ────────────────────
    print("\n" + "=" * 75)
    print("Ensemble Rule 1: pick tradition with highest top1_prob")
    print("=" * 75)

    def _highest_prob_winner(row):
        probs = {
            'par': row['top1_prob_par'],
            'yog': row['top1_prob_yog'],
            'kp': row['top1_prob_kp'],
        }
        winner = max(probs, key=probs.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule1 = df.apply(_highest_prob_winner, axis=1)
    rule1_acc = rule1['correct'].mean()
    print(f"  Top-1 accuracy: {rule1_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule1['winner'].value_counts().items():
        sub_acc = rule1[rule1['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 2: highest margin_abs wins ───────────────────
    print("\n" + "=" * 75)
    print("Ensemble Rule 2: pick tradition with highest margin_abs")
    print("=" * 75)

    def _highest_margin_winner(row):
        margins = {
            'par': row['margin_abs_par'],
            'yog': row['margin_abs_yog'],
            'kp': row['margin_abs_kp'],
        }
        winner = max(margins, key=margins.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule2 = df.apply(_highest_margin_winner, axis=1)
    rule2_acc = rule2['correct'].mean()
    print(f"  Top-1 accuracy: {rule2_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule2['winner'].value_counts().items():
        sub_acc = rule2[rule2['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 3: confidence-weighted with reliability prior ──
    # Parashari and KP are well-calibrated; Yogini is anti-calibrated
    # -> DOWN-weight Yogini's confidence
    print("\n" + "=" * 75)
    print("Ensemble Rule 3: prob_weighted with reliability prior")
    print("(Parashari weight 1.0, KP 1.0, Yogini 0.5 — Yogini was anti-calibrated)")
    print("=" * 75)

    def _weighted_winner(row):
        scores = {
            'par': row['top1_prob_par'] * 1.0,
            'yog': row['top1_prob_yog'] * 0.5,
            'kp': row['top1_prob_kp'] * 1.0,
        }
        winner = max(scores, key=scores.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule3 = df.apply(_weighted_winner, axis=1)
    rule3_acc = rule3['correct'].mean()
    print(f"  Top-1 accuracy: {rule3_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule3['winner'].value_counts().items():
        sub_acc = rule3[rule3['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 4: cascade — try Parashari, fall back to KP, then Yogini ──
    # Use each tradition only if its confidence is in its top-30%
    print("\n" + "=" * 75)
    print("Ensemble Rule 4: cascade (high-conf Parashari -> high-conf KP -> Yogini)")
    print("=" * 75)
    par_thresh = df['top1_prob_par'].quantile(0.7)
    kp_thresh = df['top1_prob_kp'].quantile(0.7)

    def _cascade(row):
        if row['top1_prob_par'] >= par_thresh:
            return pd.Series({'winner': 'par', 'correct': int(row['top1_correct_par'])})
        if row['top1_prob_kp'] >= kp_thresh:
            return pd.Series({'winner': 'kp', 'correct': int(row['top1_correct_kp'])})
        # Fallback: highest of par or kp (skip Yogini due to anti-cal)
        scores = {'par': row['top1_prob_par'], 'kp': row['top1_prob_kp']}
        winner = max(scores, key=scores.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule4 = df.apply(_cascade, axis=1)
    rule4_acc = rule4['correct'].mean()
    print(f"  Top-1 accuracy: {rule4_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule4['winner'].value_counts().items():
        sub_acc = rule4[rule4['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 5: Parashari + KP only (drop Yogini) ─────────
    print("\n" + "=" * 75)
    print("Ensemble Rule 5: Parashari + KP only (drop anti-calibrated Yogini)")
    print("=" * 75)
    df['par_kp_any'] = (
        (df['top1_correct_par'] == 1) | (df['top1_correct_kp'] == 1)
    ).astype(int)

    def _par_kp_highest(row):
        if row['top1_prob_par'] > row['top1_prob_kp']:
            return pd.Series({'winner': 'par',
                              'correct': int(row['top1_correct_par'])})
        else:
            return pd.Series({'winner': 'kp',
                              'correct': int(row['top1_correct_kp'])})

    rule5 = df.apply(_par_kp_highest, axis=1)
    rule5_acc = rule5['correct'].mean()
    par_kp_oracle = df['par_kp_any'].mean()
    print(f"  Top-1 accuracy (Par+KP highest-conf): {rule5_acc:.1%}")
    print(f"  Par+KP oracle (any correct):          {par_kp_oracle:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule5['winner'].value_counts().items():
        sub_acc = rule5[rule5['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 6: Platt-scaled probabilities ────────────────
    # For each tradition, fit a logistic regression on (margin_abs, top1_prob)
    # to predict actual P(correct), then pick the tradition with highest
    # calibrated probability per chart.
    print("\n" + "=" * 75)
    print("Ensemble Rule 6: Platt-calibrated confidence (LOOCV)")
    print("=" * 75)
    from sklearn.linear_model import LogisticRegression

    def _platt_loocv(conf_col, correct_col, margin_col):
        """LOOCV-fit Platt scaling: returns calibrated probs for each row."""
        X_full = df[[conf_col, margin_col]].values
        y_full = df[correct_col].values
        cal = np.zeros(len(df))
        for i in range(len(df)):
            mask = np.ones(len(df), dtype=bool)
            mask[i] = False
            X_tr = X_full[mask]
            y_tr = y_full[mask]
            try:
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit(X_tr, y_tr)
                cal[i] = lr.predict_proba(X_full[i:i+1])[0, 1]
            except Exception:
                cal[i] = X_full[i, 0]
        return cal

    print("  Fitting LOOCV Platt scaling for each tradition (slow)...")
    par_cal = _platt_loocv('top1_prob_par', 'top1_correct_par', 'margin_abs_par')
    yog_cal = _platt_loocv('top1_prob_yog', 'top1_correct_yog', 'margin_abs_yog')
    kp_cal = _platt_loocv('top1_prob_kp', 'top1_correct_kp', 'margin_abs_kp')
    df['par_cal'] = par_cal
    df['yog_cal'] = yog_cal
    df['kp_cal'] = kp_cal

    def _highest_cal(row):
        scores = {
            'par': row['par_cal'],
            'yog': row['yog_cal'],
            'kp': row['kp_cal'],
        }
        winner = max(scores, key=scores.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule6 = df.apply(_highest_cal, axis=1)
    rule6_acc = rule6['correct'].mean()
    print(f"  Top-1 accuracy: {rule6_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule6['winner'].value_counts().items():
        sub_acc = rule6[rule6['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")
    print(f"  Avg calibrated probs: par={df['par_cal'].mean():.3f}, "
          f"yog={df['yog_cal'].mean():.3f}, kp={df['kp_cal'].mean():.3f}")

    # ── Ensemble Rule 7: Meta-learner via LOOCV ─────────────────────
    # Train logreg on (par_prob, par_margin, yog_prob, yog_margin, kp_prob, kp_margin)
    # to predict P(correct) per (chart, tradition). Pick best tradition per chart.
    print("\n" + "=" * 75)
    print("Ensemble Rule 7: LOOCV meta-learner per tradition")
    print("=" * 75)

    feat_cols = ['top1_prob_par', 'margin_abs_par',
                 'top1_prob_yog', 'margin_abs_yog',
                 'top1_prob_kp', 'margin_abs_kp']
    X_full = df[feat_cols].values
    correct_cols = {
        'par': df['top1_correct_par'].values,
        'yog': df['top1_correct_yog'].values,
        'kp': df['top1_correct_kp'].values,
    }

    print(f"  Fitting LOOCV meta-learner (3 models x {n} folds)...")
    meta_pred = {trad: np.zeros(n) for trad in ['par', 'yog', 'kp']}
    for trad, y in correct_cols.items():
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            try:
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit(X_full[mask], y[mask])
                meta_pred[trad][i] = lr.predict_proba(X_full[i:i+1])[0, 1]
            except Exception:
                meta_pred[trad][i] = 0

    df['meta_par'] = meta_pred['par']
    df['meta_yog'] = meta_pred['yog']
    df['meta_kp'] = meta_pred['kp']

    def _highest_meta(row):
        scores = {
            'par': row['meta_par'],
            'yog': row['meta_yog'],
            'kp': row['meta_kp'],
        }
        winner = max(scores, key=scores.get)
        correct_col = f'top1_correct_{winner}'
        return pd.Series({'winner': winner, 'correct': int(row[correct_col])})

    rule7 = df.apply(_highest_meta, axis=1)
    rule7_acc = rule7['correct'].mean()
    print(f"  Top-1 accuracy: {rule7_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in rule7['winner'].value_counts().items():
        sub_acc = rule7[rule7['winner'] == w]['correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), acc when chosen={sub_acc:.1%}")

    # ── Ensemble Rule 8: confidence-thresholded with abstain ───────
    # Only commit a prediction when at least one tradition's calibrated
    # prob > 0.20. On those charts, what's the accuracy?
    print("\n" + "=" * 75)
    print("Ensemble Rule 8: confident-only subset (calibrated prob > 0.18)")
    print("=" * 75)
    threshold = 0.18
    df['max_meta'] = df[['meta_par', 'meta_yog', 'meta_kp']].max(axis=1)
    confident_mask = df['max_meta'] > threshold
    n_conf = confident_mask.sum()
    if n_conf > 0:
        conf_subset = rule7[confident_mask.values]
        acc = conf_subset['correct'].mean()
        print(f"  Charts above threshold: {n_conf}/{n} ({n_conf/n:.1%})")
        print(f"  Accuracy on confident subset: {acc:.1%}")
    else:
        print(f"  No charts above threshold {threshold}")

    # ── Final summary ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("ENSEMBLE SUMMARY")
    print("=" * 75)
    print()
    print(f"  Random baseline (cl=5):                  ~5.9%")
    print(f"  Best single tradition (Parashari/KP):    {max(par_acc, kp_acc):.1%}")
    print(f"  Yogini (anti-calibrated):                {yog_acc:.1%}")
    print()
    print(f"  Rule 1: highest top1_prob              -> {rule1_acc:.1%}")
    print(f"  Rule 2: highest margin_abs             -> {rule2_acc:.1%}")
    print(f"  Rule 3: prob_weighted (Yogini x 0.5)   -> {rule3_acc:.1%}")
    print(f"  Rule 4: cascade (Par -> KP -> fallback)-> {rule4_acc:.1%}")
    print(f"  Rule 5: Par+KP highest_conf only       -> {rule5_acc:.1%}")
    print(f"  Rule 6: LOOCV Platt-calibrated         -> {rule6_acc:.1%}")
    print(f"  Rule 7: LOOCV meta-learner             -> {rule7_acc:.1%}")
    if 'max_meta' in df.columns and (df['max_meta'] > threshold).sum() > 0:
        print(f"  Rule 8: confident-only subset          -> "
              f"{rule7[(df['max_meta'] > threshold).values]['correct'].mean():.1%} "
              f"({(df['max_meta'] > threshold).sum()}/{n})")
    print()
    print(f"  Par+KP oracle ceiling (any correct):     {par_kp_oracle:.1%}")
    print(f"  3-tradition oracle ceiling (any correct):{df['any_correct'].mean():.1%}")
    print()
    print(f"  Best ensemble lift over best single: "
          f"{max(rule1_acc, rule2_acc, rule3_acc, rule4_acc, rule5_acc) / max(par_acc, kp_acc):.2f}x")


if __name__ == '__main__':
    main()
