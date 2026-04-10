"""Path C: stacking meta-learner with natal-chart features.

For each val chart, extract:
  - 3 confidence scores (par, yog, kp)
  - 3 margins
  - Natal features: lagna sign, moon sign, moon nakshatra, sun sign,
    9th lord, 8th lord, dignity scores

Train per-tradition logistic regression with 5-fold CV to predict
P(top1_correct | confidence + margin + natal). Pick tradition with
highest predicted probability.

Usage:
    python -u -m ml.pipelines.father_death_predictor.ensemble_path_c
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

V2_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v2_clean.json'
V3_JSON = REPO_ROOT / 'ml' / 'father_passing_date_v3_clean.json'
VAL_JSON = REPO_ROOT / 'ml' / 'father_passing_date_clean.json'


def _load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _valid(r):
    try:
        r['father_death_date'].split('-')
        return True
    except Exception:
        return False


def extract_natal_features(rec):
    """Compute simple natal-chart features for one chart."""
    from .astro_engine.ephemeris import compute_chart
    from .astro_engine.houses import get_sign, get_sign_lord, get_house_number
    from .astro_engine.dasha import NAKSHATRA_LORDS

    PLANET_IDX = {
        'Sun': 0, 'Moon': 1, 'Mars': 2, 'Mercury': 3, 'Jupiter': 4,
        'Venus': 5, 'Saturn': 6, 'Rahu': 7, 'Ketu': 8,
    }
    NAK_SPAN = 360.0 / 27.0

    chart, asc = compute_chart(
        rec['birth_date'], rec['birth_time'], rec['lat'], rec['lon'])
    asc_sign = get_sign(asc)

    f = {}
    f['nat_lagna_sign'] = float(asc_sign)
    f['nat_moon_sign'] = float(get_sign(chart['Moon']['longitude']))
    f['nat_sun_sign'] = float(get_sign(chart['Sun']['longitude']))
    moon_nak = int(chart['Moon']['longitude'] / NAK_SPAN) % 27
    f['nat_moon_nak'] = float(moon_nak)
    f['nat_moon_nak_lord'] = float(PLANET_IDX.get(NAKSHATRA_LORDS[moon_nak], 0))

    # House lords
    h9_sign = (asc_sign + 8) % 12
    h9_lord = get_sign_lord(h9_sign)
    f['nat_h9_lord'] = float(PLANET_IDX.get(h9_lord, 0))
    h8_sign = (asc_sign + 7) % 12
    h8_lord = get_sign_lord(h8_sign)
    f['nat_h8_lord'] = float(PLANET_IDX.get(h8_lord, 0))

    # 9th lord placement
    if h9_lord in chart:
        h9l_house = get_house_number(chart[h9_lord]['longitude'], asc)
        f['nat_h9_lord_house'] = float(h9l_house)
    else:
        f['nat_h9_lord_house'] = 0.0

    # Sun in which house
    sun_house = get_house_number(chart['Sun']['longitude'], asc)
    f['nat_sun_house'] = float(sun_house)
    moon_house = get_house_number(chart['Moon']['longitude'], asc)
    f['nat_moon_house'] = float(moon_house)

    # Sat / Mars / Rahu houses (malefics in death houses)
    for p in ('Saturn', 'Mars', 'Rahu', 'Ketu'):
        if p in chart:
            h = get_house_number(chart[p]['longitude'], asc)
            f[f'nat_{p.lower()}_house'] = float(h)

    return f


def main():
    print("=" * 75)
    print("PATH C — Stacking meta-learner with natal features")
    print("=" * 75)

    # Load existing per-chart confidence CSVs
    par = pd.read_csv(PROJECT_ROOT / 'parashari_confidence.csv')
    yog = pd.read_csv(PROJECT_ROOT / 'yogini_confidence.csv')
    kp = pd.read_csv(PROJECT_ROOT / 'kp_confidence.csv')
    print(f"  Loaded confidences: par={len(par)}, yog={len(yog)}, kp={len(kp)}")

    df = par.merge(yog, on='chart_id', suffixes=('_par', '_yog'))
    df = df.merge(kp, on='chart_id')
    df = df.rename(columns={
        'top1_correct': 'top1_correct_kp',
        'top1_prob': 'top1_prob_kp',
        'margin_abs': 'margin_abs_kp',
    })
    n = len(df)
    print(f"  Common: {n}")

    # ── Extract natal features per chart ───────────────────────
    print(f"\n  Extracting natal features for {n} charts...")
    train_recs = [r for r in _load_json(str(V2_JSON)) + _load_json(str(V3_JSON))
                  if _valid(r)]
    val_recs = [r for r in _load_json(str(VAL_JSON)) if _valid(r)]
    n_train = len(train_recs)
    start_index = n_train * 100
    # gid = (start_index + i) * 100
    gid_to_recidx = {(start_index + i) * 100: i for i in range(len(val_recs))}

    natal_rows = []
    t0 = time.time()
    for gid in df['chart_id']:
        recidx = gid_to_recidx.get(int(gid))
        if recidx is None:
            natal_rows.append({})
            continue
        try:
            f = extract_natal_features(val_recs[recidx])
            f['chart_id'] = int(gid)
            natal_rows.append(f)
        except Exception:
            natal_rows.append({'chart_id': int(gid)})
    natal_df = pd.DataFrame(natal_rows)
    print(f"  Done in {time.time()-t0:.0f}s, "
          f"{len(natal_df.columns)-1} natal features")

    df = df.merge(natal_df, left_on='chart_id', right_on='chart_id')
    df = df.dropna()
    n = len(df)
    print(f"  Final n with natal features: {n}")

    par_acc = df['top1_correct_par'].mean()
    yog_acc = df['top1_correct_yog'].mean()
    kp_acc = df['top1_correct_kp'].mean()
    union = ((df['top1_correct_par'] == 1) | (df['top1_correct_yog'] == 1) |
             (df['top1_correct_kp'] == 1)).mean()
    print(f"  Single-tradition baselines: par={par_acc:.1%} "
          f"yog={yog_acc:.1%} kp={kp_acc:.1%}")
    print(f"  Oracle ceiling: {union:.1%}")

    natal_cols = [c for c in df.columns if c.startswith('nat_')]

    # ── Per-tradition stacking model with 5-fold CV ──────────
    def _stacked_pred(trad_prefix, correct_col):
        feat_cols = (
            [f'top1_prob_{trad_prefix}', f'margin_abs_{trad_prefix}'] +
            natal_cols
        )
        X = df[feat_cols].values
        y = df[correct_col].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(n)
        for tr_idx, te_idx in kf.split(X):
            try:
                lr = LogisticRegression(max_iter=500, C=0.5)
                lr.fit(X[tr_idx], y[tr_idx])
                preds[te_idx] = lr.predict_proba(X[te_idx])[:, 1]
            except Exception:
                preds[te_idx] = y[tr_idx].mean()
        return preds

    print(f"\n  Training stacking models (5-fold CV per tradition)...")
    df['stack_par'] = _stacked_pred('par', 'top1_correct_par')
    df['stack_yog'] = _stacked_pred('yog', 'top1_correct_yog')
    df['stack_kp'] = _stacked_pred('kp', 'top1_correct_kp')

    print(f"  Avg stacked probs: par={df['stack_par'].mean():.3f}, "
          f"yog={df['stack_yog'].mean():.3f}, kp={df['stack_kp'].mean():.3f}")
    print(f"  Std of stacked probs: par={df['stack_par'].std():.3f}, "
          f"yog={df['stack_yog'].std():.3f}, kp={df['stack_kp'].std():.3f}")

    def _pick_highest(row):
        scores = {
            'par': row['stack_par'],
            'yog': row['stack_yog'],
            'kp': row['stack_kp'],
        }
        winner = max(scores, key=scores.get)
        return int(row[f'top1_correct_{winner}']), winner

    picks = df.apply(_pick_highest, axis=1)
    df['stack_correct'] = picks.apply(lambda x: x[0])
    df['stack_winner'] = picks.apply(lambda x: x[1])
    stack_acc = df['stack_correct'].mean()
    print(f"\n  Path C accuracy: {stack_acc:.1%}")
    print(f"  Winner distribution:")
    for w, count in df['stack_winner'].value_counts().items():
        sub_acc = df[df['stack_winner'] == w]['stack_correct'].mean()
        print(f"    {w}: {count}/{n} ({count/n:.0%}), "
              f"acc when chosen={sub_acc:.1%}")

    # ── Stacked + abstention combo ────────────────────────────
    print(f"\n  Path C + abstention (top X% by max stacked prob):")
    df['max_stack'] = df[['stack_par', 'stack_yog', 'stack_kp']].max(axis=1)
    df_sorted = df.sort_values('max_stack', ascending=False)
    print(f"    {'%X':>5s} {'N':>5s} {'#correct':>10s} {'accuracy':>10s}")
    for pct in [5, 10, 20, 30, 50, 100]:
        k = max(1, int(n * pct / 100))
        sub = df_sorted.iloc[:k]
        acc = sub['stack_correct'].mean()
        print(f"    {pct:>4d}% {k:>5d} "
              f"{int(sub['stack_correct'].sum()):>10d} {acc:>9.1%}")

    # ── Compare with confidence-only meta-learner (no natal) ──
    print(f"\n  Comparison: confidence-only vs +natal features")
    def _stacked_no_natal(trad_prefix, correct_col):
        feat_cols = [f'top1_prob_{trad_prefix}', f'margin_abs_{trad_prefix}']
        X = df[feat_cols].values
        y = df[correct_col].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros(n)
        for tr_idx, te_idx in kf.split(X):
            try:
                lr = LogisticRegression(max_iter=500, C=0.5)
                lr.fit(X[tr_idx], y[tr_idx])
                preds[te_idx] = lr.predict_proba(X[te_idx])[:, 1]
            except Exception:
                preds[te_idx] = y[tr_idx].mean()
        return preds

    par_nn = _stacked_no_natal('par', 'top1_correct_par')
    yog_nn = _stacked_no_natal('yog', 'top1_correct_yog')
    kp_nn = _stacked_no_natal('kp', 'top1_correct_kp')

    nn_correct = []
    for i in range(n):
        scores = {'par': par_nn[i], 'yog': yog_nn[i], 'kp': kp_nn[i]}
        winner = max(scores, key=scores.get)
        nn_correct.append(int(df[f'top1_correct_{winner}'].iloc[i]))
    nn_acc = np.mean(nn_correct)
    print(f"  Conf-only meta-learner: {nn_acc:.1%}")
    print(f"  +Natal stacking:        {stack_acc:.1%}")

    print(f"\n  -- FINAL --")
    print(f"  Random:           5.9%")
    print(f"  Best single:      13.1%")
    print(f"  Path C accuracy:  {stack_acc:.1%}")
    print(f"  Oracle ceiling:   {union:.1%}")


if __name__ == '__main__':
    main()
