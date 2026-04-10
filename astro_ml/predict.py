"""Prediction — single chart and batch inference.

Usage:
  # Single chart prediction
  python -m astro_ml.predict --single chart.json --model_dir ./models

  # Batch prediction
  python -m astro_ml.predict --input_dir ./data/test --model_dir ./models
"""
import os, sys, json, argparse
import numpy as np
import lightgbm as lgb

from astro_ml.compute import compute
from astro_ml.features import extract_monthly_windows, N_FEATURES, FEATURE_NAMES
from astro_ml.advanced_features import (
    add_advanced_death_features, ADVANCED_FEATURE_NAMES, N_TOTAL,
)
from astro_ml.data_prep import load_charts, extract_all, build_dataset
from astro_ml.train import (
    compute_topk_accuracy, compute_topk_with_tolerance, compute_mrr,
    ALL_FEATURE_NAMES,
)
from astro_ml.config import domain_fathers_death as cfg

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load_models(model_dir=None):
    """Load classifier and ranker models."""
    if model_dir is None:
        model_dir = MODEL_DIR

    clf_path = os.path.join(model_dir, "classifier.txt")
    ranker_path = os.path.join(model_dir, "ranker.txt")

    clf = lgb.Booster(model_file=clf_path)
    ranker = lgb.Booster(model_file=ranker_path)

    return clf, ranker


def predict_single(chart, clf, ranker, top_k=5):
    """Predict for a single chart.

    Args:
        chart: dict with birth data
        clf: LightGBM classifier booster
        ranker: LightGBM ranker booster
        top_k: number of top predictions to return

    Returns list of dicts with rank, month, score, tier, dasha info.
    """
    payload = compute(chart)
    windows = extract_monthly_windows(payload)
    windows = add_advanced_death_features(windows, payload)

    if not windows:
        return []

    X = np.stack([w["full_vector"] for w in windows])

    # Get scores from both models
    clf_scores = clf.predict(X)
    ranker_scores = ranker.predict(X)

    # Normalize
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-10)

    # Tier scores
    tier_idx = FEATURE_NAMES.index("deterministic_tier") if "deterministic_tier" in FEATURE_NAMES else -1
    if tier_idx >= 0:
        tier_scores = X[:, tier_idx] / 5.0
    else:
        tier_scores = np.zeros(len(X))

    # Ensemble
    ensemble_scores = 0.4 * _norm(clf_scores) + 0.5 * _norm(ranker_scores) + 0.1 * tier_scores

    # Rank
    order = np.argsort(-ensemble_scores)

    tier_names = {5: "S", 4: "A", 3: "B", 2: "C", 1: "D", 0: "F"}
    results = []
    for rank, idx in enumerate(order[:top_k], 1):
        w = windows[idx]
        tier_val = int(w["feature_vector"][tier_idx]) if tier_idx >= 0 else 0
        results.append({
            "rank": rank,
            "month": w["month"],
            "score": round(float(ensemble_scores[idx]), 4),
            "tier": tier_names.get(tier_val, "F"),
            "dasha_period": f"{w['md']}-{w['ad']}-{w['pd']}",
            "dasha_quality_score": round(float(w["features"].get("dasha_quality_score", 0)), 2),
            "primary_cusps_active": int(w["features"].get("n_primary_cusps_active", 0)),
            "has_dasha_lock": bool(w["features"].get("has_primary_dasha_lock", 0)),
            "death_karaka_count": int(
                w["features"].get("karaka_active_count", 0)
            ),
            "saturn_transit": bool(w["features"].get("saturn_transit_active", 0)),
        })

    return results


def predict_batch(charts, clf, ranker, top_k=5):
    """Batch prediction for multiple charts.

    Returns list of (chart_id, predictions, event_month) tuples.
    """
    results = []
    for i, chart in enumerate(charts):
        if (i + 1) % 50 == 0:
            print(f"  Predicted {i + 1}/{len(charts)}...")
        try:
            preds = predict_single(chart, clf, ranker, top_k)
            chart_id = chart.get("id", chart.get("name", f"chart_{i}"))
            event_month = chart.get("father_death_date", "")[:7]
            results.append((chart_id, preds, event_month))
        except Exception as e:
            print(f"  Error predicting {chart.get('name', 'unknown')}: {e}")
    return results


def evaluate_batch(batch_results, tolerance=1):
    """Evaluate batch predictions."""
    correct = {1: 0, 3: 0, 5: 0}
    correct_tol = {1: 0, 3: 0, 5: 0}
    total = 0
    rr_sum = 0

    for chart_id, preds, event_month in batch_results:
        if not event_month or not preds:
            continue
        total += 1

        # Check if event month is in top-K
        for k in [1, 3, 5]:
            top_months = [p["month"] for p in preds[:k]]
            if event_month in top_months:
                correct[k] += 1

            # With tolerance
            import datetime
            ep = event_month.split("-")
            try:
                event_dt = datetime.date(int(ep[0]), int(ep[1]), 1)
            except (ValueError, IndexError):
                continue
            acceptable = set()
            for delta in range(-tolerance, tolerance + 1):
                m = event_dt.month + delta
                y = event_dt.year
                while m < 1: m += 12; y -= 1
                while m > 12: m -= 12; y += 1
                acceptable.add(f"{y}-{m:02d}")
            if any(m in acceptable for m in top_months):
                correct_tol[k] += 1

        # MRR
        for rank, p in enumerate(preds, 1):
            if p["month"] == event_month:
                rr_sum += 1.0 / rank
                break

    print(f"\nBatch Evaluation ({total} charts):")
    for k in [1, 3, 5]:
        print(f"  Top-{k}: {correct[k]/max(total,1):.1%} (exact)  "
              f"{correct_tol[k]/max(total,1):.1%} (+-{tolerance}mo)")
    print(f"  MRR: {rr_sum/max(total,1):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Father's death prediction")
    parser.add_argument("--single", type=str, help="Path to single chart JSON")
    parser.add_argument("--input_path", type=str, help="Path to charts JSON file for batch")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    clf, ranker = load_models(args.model_dir)

    if args.single:
        with open(args.single, "r") as f:
            chart = json.load(f)
        results = predict_single(chart, clf, ranker, args.top_k)
        print(json.dumps(results, indent=2))
    elif args.input_path:
        charts = load_charts(args.input_path)
        batch = predict_batch(charts, clf, ranker, args.top_k)
        evaluate_batch(batch)
    else:
        print("Provide --single or --input_path")
        sys.exit(1)


if __name__ == "__main__":
    main()
