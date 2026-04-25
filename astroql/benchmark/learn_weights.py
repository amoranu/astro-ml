"""Learn per-rule weights from the rule-firing matrix.

Logistic regression on (X, y):
    X[i, j] = strength of rule j on candidate window i
    y[i]   = 1 if window i contains true death date, else 0

Within each chart there is exactly 1 (or 0) positive window and ~700
negative windows — this is heavily class-imbalanced. We:
  1. Down-sample negatives within each chart to a fixed cap (default 50)
     to keep the training set balanced and tractable.
  2. Train logistic regression with L2 regularization.
  3. Save learned coefficients keyed by rule_id, plus the bias term.

The learned weights replace the noisy-OR aggregator at inference time:
    score(window) = sigmoid(bias + sum_j w_j * X[window, j])

Usage:
    python -u -m astroql.benchmark.learn_weights \\
        --matrix ml/firing_matrix_train.npz \\
        --out astroql/rules/learned_weights.yaml \\
        --neg-per-chart 50

The output YAML can be read by a new aggregator path that sums learned
weights instead of running noisy-OR.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def _down_sample_per_chart(
    keys: np.ndarray, y: np.ndarray, neg_per_chart: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Keep ALL positives + at most `neg_per_chart` negatives per chart.

    Returns boolean mask over rows.
    """
    chart_ids = keys[:, 0]
    mask = np.zeros(len(y), dtype=bool)
    for cid in np.unique(chart_ids):
        chart_rows = np.where(chart_ids == cid)[0]
        chart_pos = chart_rows[y[chart_rows] == 1]
        chart_neg = chart_rows[y[chart_rows] == 0]
        mask[chart_pos] = True
        if len(chart_neg) > neg_per_chart:
            chosen = rng.choice(
                chart_neg, size=neg_per_chart, replace=False,
            )
            mask[chosen] = True
        else:
            mask[chart_neg] = True
    return mask


def _logistic_train(
    X: np.ndarray, y: np.ndarray,
    *,
    n_iter: int = 200,
    lr: float = 0.1,
    l2: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """L2-regularized logistic regression via sklearn (lbfgs solver).

    Falls back to plain GD if sklearn unavailable.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        # C is inverse of L2; balanced class_weight handles 2% positive rate.
        clf = LogisticRegression(
            C=1.0 / max(l2, 1e-6),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=max(n_iter, 1000),
            tol=1e-5,
        )
        clf.fit(X, y)
        return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    except ImportError:
        pass
    # Fallback: plain GD.
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    w_pos = neg / max(1, pos + neg)
    w_neg = pos / max(1, pos + neg)
    sample_w = np.where(y == 1, w_pos, w_neg)
    for it in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) * sample_w
        grad_w = X.T @ err / n + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        if it % 20 == 0:
            eps = 1e-9
            ll = -(
                sample_w * (
                    y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
                )
            ).mean()
            print(f"  iter {it:4d}  loss={ll:.4f}  ||w||={np.linalg.norm(w):.3f}")
    return w, float(b)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True, help=".npz from firing_matrix.py")
    p.add_argument("--out", required=True, help="learned_weights.yaml output")
    p.add_argument("--neg-per-chart", type=int, default=50,
                   help="negatives sampled per chart (default 50)")
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    z = np.load(args.matrix, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int8)
    rule_ids = list(z["rule_ids"])
    keys = z["keys"]
    polarities = list(z["polarities"])

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"Loaded matrix: {X.shape}, pos={n_pos}, neg={n_neg}")

    rng = np.random.default_rng(args.seed)
    mask = _down_sample_per_chart(keys, y, args.neg_per_chart, rng)
    Xt = X[mask].astype(np.float64)
    yt = y[mask].astype(np.float64)
    print(f"After per-chart down-sampling: {Xt.shape}, "
          f"pos={int(yt.sum())}, neg={int((yt==0).sum())}")

    print("\nTraining logistic regression...")
    w, b = _logistic_train(
        Xt, yt, n_iter=args.n_iter, lr=args.lr, l2=args.l2,
    )

    # Evaluate on full (training-only-subsetted) matrix for sanity.
    z_pred = X.astype(np.float64) @ w + b
    p_pred = 1.0 / (1.0 + np.exp(-np.clip(z_pred, -30, 30)))
    print(f"\nFull matrix eval: mean_p={p_pred.mean():.4f}  "
          f"mean_p_at_pos={p_pred[y==1].mean():.4f}  "
          f"mean_p_at_neg={p_pred[y==0].mean():.4f}")

    # Per-chart Hit@K with learned weights.
    chart_ids = keys[:, 0]
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0}
    n_charts = 0
    for cid in np.unique(chart_ids):
        rows = np.where(chart_ids == cid)[0]
        if y[rows].sum() == 0:
            continue
        scores = p_pred[rows]
        order = np.argsort(-scores)
        true_local = np.where(y[rows] == 1)[0]
        # Rank of the true window in this chart's local ordering.
        for k in hits_at:
            if any(np.where(order == ti)[0][0] < k for ti in true_local):
                hits_at[k] += 1
        n_charts += 1
    print(f"\nLearned-weights in-sample Hit@K (n_charts={n_charts}):")
    for k, h in hits_at.items():
        print(f"  Hit@{k} = {h/max(1,n_charts):.3f} ({h}/{n_charts})")

    # Output sorted weights for inspection.
    weights = sorted(
        zip(rule_ids, w.tolist(), polarities),
        key=lambda r: -abs(r[1]),
    )
    print("\nTop 20 most-impactful rules (|weight| desc):")
    for rid, weight, pol in weights[:20]:
        print(f"  {weight:+0.4f}  ({pol:8})  {rid}")

    out = {
        "meta": {
            "n_train_rows": int(mask.sum()),
            "n_features": len(rule_ids),
            "bias": float(b),
            "neg_per_chart": args.neg_per_chart,
            "n_iter": args.n_iter,
            "l2": args.l2,
        },
        # Cast keys to plain str (numpy str_ can't yaml-serialize).
        "weights": {str(rid): float(w_i) for rid, w_i in zip(rule_ids, w)},
    }
    Path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
