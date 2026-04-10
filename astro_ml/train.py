"""Training pipeline — V1: binary classifier + LambdaMART ranker + evaluation.

Two-stage approach:
  Stage 1: LightGBM binary classifier trained with soft labels
  Stage 2: LightGBM LambdaMART ranker trained with ranking labels
  Ensemble: Weighted combination of classifier + ranker + tier scores

Usage:
  python -m astro_ml.train [train_dir] [test_dir] [model_dir]
"""
import os, sys, json, time, datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from collections import defaultdict

from astro_ml.data_prep import (
    load_charts, extract_all, build_dataset, build_ranking_dataset,
    TRAIN_PATH, TEST_PATH,
)
from astro_ml.features import FEATURE_NAMES, N_FEATURES
from astro_ml.advanced_features import ADVANCED_FEATURE_NAMES, N_ADVANCED, N_TOTAL

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALL_FEATURE_NAMES = FEATURE_NAMES + ADVANCED_FEATURE_NAMES


# ── Evaluation metrics ───────────────────────────────────────────────────

def compute_topk_accuracy(scores, groups, y_true, k_values=(1, 3, 5)):
    """Compute top-K accuracy: fraction of charts where true event is in top K.

    Args:
        scores: np.array of predicted scores (higher = more likely)
        groups: np.array of chart indices
        y_true: np.array of hard labels (1.0 = event month)
        k_values: tuple of K values

    Returns dict mapping K -> accuracy.
    """
    unique_groups = sorted(set(groups))
    results = {k: 0 for k in k_values}
    n_charts = 0

    for g in unique_groups:
        mask = groups == g
        g_scores = scores[mask]
        g_labels = y_true[mask]

        if g_labels.sum() == 0:
            continue  # no positive label
        n_charts += 1

        # Rank by score (descending)
        order = np.argsort(-g_scores)
        ranked_labels = g_labels[order]

        for k in k_values:
            if ranked_labels[:k].max() > 0:
                results[k] += 1

    for k in k_values:
        results[k] = results[k] / max(n_charts, 1)
    return results


def compute_topk_with_tolerance(scores, groups, info, k_values=(1, 3, 5), tolerance_months=1):
    """Top-K accuracy with +/-tolerance month tolerance.

    A prediction is correct if it falls within tolerance_months of the true event.
    """
    unique_groups = sorted(set(groups))
    results = {k: 0 for k in k_values}
    n_charts = 0

    for g in unique_groups:
        mask = groups == g
        g_scores = scores[mask]
        g_info = [info[i] for i in range(len(info)) if mask[i]]

        event_month = g_info[0]["event_month"] if g_info else ""
        if not event_month:
            continue
        n_charts += 1

        # Parse event date
        try:
            ep = event_month.split("-")
            event_dt = datetime.date(int(ep[0]), int(ep[1]), 1)
        except (ValueError, IndexError):
            continue

        # Build set of acceptable months
        acceptable = set()
        for delta in range(-tolerance_months, tolerance_months + 1):
            m = event_dt.month + delta
            y = event_dt.year
            while m < 1:
                m += 12; y -= 1
            while m > 12:
                m -= 12; y += 1
            acceptable.add(f"{y}-{m:02d}")

        # Rank by score
        order = np.argsort(-g_scores)
        for k in k_values:
            top_months = [g_info[order[j]]["month"] for j in range(min(k, len(order)))]
            if any(m in acceptable for m in top_months):
                results[k] += 1

    for k in k_values:
        results[k] = results[k] / max(n_charts, 1)
    return results


def compute_mrr(scores, groups, y_true):
    """Mean Reciprocal Rank."""
    unique_groups = sorted(set(groups))
    rr_sum = 0
    n = 0
    for g in unique_groups:
        mask = groups == g
        g_scores = scores[mask]
        g_labels = y_true[mask]
        if g_labels.sum() == 0:
            continue
        n += 1
        order = np.argsort(-g_scores)
        for rank, idx in enumerate(order, 1):
            if g_labels[idx] > 0:
                rr_sum += 1.0 / rank
                break
    return rr_sum / max(n, 1)


# ── Stage 1: Binary Classifier ──────────────────────────────────────────

def train_classifier(X_train, y_soft, X_val=None, y_val_hard=None,
                     groups_val=None, feature_names=None):
    """Train LightGBM regressor with soft labels (0.0/0.5/1.0).

    Using regression instead of classification so we can train on continuous
    soft labels. predict() returns scores directly (higher = more likely).
    """
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES[:X_train.shape[1]]

    model = lgb.LGBMRegressor(**params)

    callbacks = [lgb.log_evaluation(period=100)]
    if X_val is not None:
        callbacks.append(lgb.early_stopping(50, verbose=True))
        model.fit(X_train, y_soft,
                  eval_set=[(X_val, y_val_hard)],
                  callbacks=callbacks,
                  feature_name=feature_names)
    else:
        model.fit(X_train, y_soft, callbacks=callbacks, feature_name=feature_names)

    return model


# ── Stage 2: LambdaMART Ranker ──────────────────────────────────────────

def train_ranker(X_train, y_rank, group_sizes_train,
                 X_val=None, y_rank_val=None, group_sizes_val=None,
                 feature_names=None):
    """Train LightGBM LambdaMART ranker."""
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "label_gain": [0, 1, 3],
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES[:X_train.shape[1]]

    train_ds = lgb.Dataset(X_train, label=y_rank, group=group_sizes_train,
                           feature_name=feature_names)

    eval_sets = [train_ds]
    eval_names = ["train"]
    if X_val is not None:
        val_ds = lgb.Dataset(X_val, label=y_rank_val, group=group_sizes_val,
                             feature_name=feature_names, reference=train_ds)
        eval_sets.append(val_ds)
        eval_names.append("val")

    callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(50, verbose=True)]

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=500,
        valid_sets=eval_sets,
        valid_names=eval_names,
        callbacks=callbacks,
    )

    return model


# ── Feature importance ───────────────────────────────────────────────────

def print_feature_importance(model, feature_names=None, top_k=25, model_type="classifier"):
    """Print top features by gain."""
    if model_type == "classifier":
        importances = model.feature_importances_
    else:
        importances = model.feature_importance(importance_type="gain")

    if feature_names is None:
        feature_names = ALL_FEATURE_NAMES[:len(importances)]

    indices = np.argsort(-importances)
    print(f"\nTop {top_k} features by gain ({model_type}):")
    print("-" * 60)

    # Expected top features for father's death
    expected = {
        "dasha_quality_score", "has_primary_dasha_lock", "n_primary_cusps_active",
        "death_signal_composite", "saturn_dasha_level", "maraka_score",
        "n_confirmed_kills", "deterministic_tier",
    }

    for r in range(min(top_k, len(indices))):
        i = indices[r]
        name = feature_names[i] if i < len(feature_names) else f"f_{i}"
        flag = " <-- EXPECTED" if name in expected else ""
        print(f"  {r + 1:3d}. {name:45s} gain={importances[i]:.1f}{flag}")


# ── Cross-validation ─────────────────────────────────────────────────────

def cross_validate(X, y_hard, y_soft, y_rank, groups, info, n_folds=5):
    """GroupKFold cross-validation. Returns per-fold and mean metrics."""
    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_hard, groups)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_soft_tr = y_soft[train_idx]
        y_hard_val = y_hard[val_idx]
        y_rank_tr, y_rank_val = y_rank[train_idx], y_rank[val_idx]
        groups_tr, groups_val = groups[train_idx], groups[val_idx]

        # Group sizes for ranker
        def _group_sizes(g):
            sizes = []
            for ug in sorted(set(g)):
                sizes.append(int((g == ug).sum()))
            return sizes

        gs_tr = _group_sizes(groups_tr)
        gs_val = _group_sizes(groups_val)

        # Train classifier
        clf = train_classifier(X_tr, y_soft_tr, X_val, y_hard_val, groups_val)
        clf_scores = clf.predict(X_val)

        # Train ranker
        ranker = train_ranker(X_tr, y_rank_tr, gs_tr, X_val, y_rank_val, gs_val)
        ranker_scores = ranker.predict(X_val)

        # Evaluate both
        info_val = [info[i] for i in val_idx]

        for name, scores in [("Classifier", clf_scores), ("Ranker", ranker_scores)]:
            topk = compute_topk_accuracy(scores, groups_val, y_hard_val, (1, 3, 5))
            topk_tol = compute_topk_with_tolerance(scores, groups_val, info_val, (1, 3, 5))
            mrr = compute_mrr(scores, groups_val, y_hard_val)
            print(f"\n  {name}:")
            print(f"    Top-1: {topk[1]:.1%}  Top-3: {topk[3]:.1%}  Top-5: {topk[5]:.1%}")
            print(f"    Top-1(+-1): {topk_tol[1]:.1%}  Top-3(+-1): {topk_tol[3]:.1%}  Top-5(+-1): {topk_tol[5]:.1%}")
            print(f"    MRR: {mrr:.4f}")

        # Ensemble: 0.4 * clf + 0.5 * ranker + 0.1 * tier
        tier_idx = ALL_FEATURE_NAMES.index("deterministic_tier") if "deterministic_tier" in ALL_FEATURE_NAMES else -1
        if tier_idx >= 0 and tier_idx < X_val.shape[1]:
            tier_scores = X_val[:, tier_idx] / 5.0  # normalize to [0,1]
        else:
            tier_scores = np.zeros(len(X_val))

        # Normalize scores to [0,1]
        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-10)

        ensemble_scores = 0.4 * _norm(clf_scores) + 0.5 * _norm(ranker_scores) + 0.1 * tier_scores

        topk_ens = compute_topk_accuracy(ensemble_scores, groups_val, y_hard_val, (1, 3, 5))
        topk_ens_tol = compute_topk_with_tolerance(ensemble_scores, groups_val, info_val, (1, 3, 5))
        mrr_ens = compute_mrr(ensemble_scores, groups_val, y_hard_val)
        print(f"\n  Ensemble (0.4*clf + 0.5*rank + 0.1*tier):")
        print(f"    Top-1: {topk_ens[1]:.1%}  Top-3: {topk_ens[3]:.1%}  Top-5: {topk_ens[5]:.1%}")
        print(f"    Top-1(+-1): {topk_ens_tol[1]:.1%}  Top-3(+-1): {topk_ens_tol[3]:.1%}  Top-5(+-1): {topk_ens_tol[5]:.1%}")
        print(f"    MRR: {mrr_ens:.4f}")

        fold_metrics.append({
            "fold": fold + 1,
            "clf_topk": topk, "clf_topk_tol": topk_tol, "clf_mrr": mrr,
            "ranker_topk": compute_topk_accuracy(ranker_scores, groups_val, y_hard_val),
            "ranker_mrr": compute_mrr(ranker_scores, groups_val, y_hard_val),
            "ensemble_topk": topk_ens, "ensemble_topk_tol": topk_ens_tol, "ensemble_mrr": mrr_ens,
        })

    # Summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for metric_name in ["clf", "ranker", "ensemble"]:
        mrrs = [f[f"{metric_name}_mrr"] for f in fold_metrics]
        top3s = [f[f"{metric_name}_topk"][3] for f in fold_metrics]
        print(f"  {metric_name.capitalize():12s}: MRR={np.mean(mrrs):.4f}+-{np.std(mrrs):.4f}  "
              f"Top-3={np.mean(top3s):.1%}+-{np.std(top3s):.1%}")

    return fold_metrics


# ── Full training pipeline ───────────────────────────────────────────────

def train_full(train_path=None, test_path=None, model_dir=None):
    """Full training pipeline: extract, train, evaluate, save."""
    if model_dir is None:
        model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    t0 = time.time()

    # Load and extract
    print("Loading training data...")
    train_charts = load_charts(train_path or TRAIN_PATH)
    print("Loading test data...")
    test_charts = load_charts(test_path or TEST_PATH)

    print("\nExtracting training features...")
    train_results = extract_all(train_charts)
    print("\nExtracting test features...")
    test_results = extract_all(test_charts)

    # Build datasets
    print("\nBuilding datasets...")
    X_train, y_hard_train, y_soft_train, y_rank_train, groups_train, info_train = build_dataset(train_results)
    X_test, y_hard_test, y_soft_test, y_rank_test, groups_test, info_test = build_dataset(test_results)

    if X_train.shape[0] == 0:
        print("ERROR: No training data extracted!")
        return

    feature_names = ALL_FEATURE_NAMES[:X_train.shape[1]]

    # Cross-validation
    print("\n--- Cross-Validation ---")
    cv_metrics = cross_validate(X_train, y_hard_train, y_soft_train, y_rank_train,
                                groups_train, info_train)

    # Train final models on full training set
    print(f"\n{'='*60}")
    print("Training final models on full training set...")
    print(f"{'='*60}")

    # Classifier
    clf = train_classifier(X_train, y_soft_train, feature_names=feature_names)
    clf.booster_.save_model(os.path.join(model_dir, "classifier.txt"))

    # Ranker
    _, y_rank_all, groups_all, gs_all, _ = build_ranking_dataset(train_results)
    ranker = train_ranker(X_train, y_rank_train, gs_all, feature_names=feature_names)
    ranker.save_model(os.path.join(model_dir, "ranker.txt"))

    # Feature importance
    print_feature_importance(clf, feature_names, model_type="classifier")
    print_feature_importance(ranker, feature_names, model_type="ranker")

    # Evaluate on test set
    if X_test.shape[0] > 0:
        print(f"\n{'='*60}")
        print("Test Set Evaluation")
        print(f"{'='*60}")

        clf_scores = clf.predict(X_test)
        ranker_scores = ranker.predict(X_test)

        tier_idx = ALL_FEATURE_NAMES.index("deterministic_tier") if "deterministic_tier" in ALL_FEATURE_NAMES else -1
        if tier_idx >= 0 and tier_idx < X_test.shape[1]:
            tier_scores = X_test[:, tier_idx] / 5.0
        else:
            tier_scores = np.zeros(len(X_test))

        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-10)

        ensemble_scores = 0.4 * _norm(clf_scores) + 0.5 * _norm(ranker_scores) + 0.1 * tier_scores

        for name, scores in [("Classifier", clf_scores), ("Ranker", ranker_scores),
                              ("Ensemble", ensemble_scores)]:
            topk = compute_topk_accuracy(scores, groups_test, y_hard_test, (1, 3, 5))
            topk_tol = compute_topk_with_tolerance(scores, groups_test, info_test, (1, 3, 5))
            mrr = compute_mrr(scores, groups_test, y_hard_test)
            print(f"\n  {name}:")
            print(f"    Top-1: {topk[1]:.1%}  Top-3: {topk[3]:.1%}  Top-5: {topk[5]:.1%}")
            print(f"    Top-1(+-1): {topk_tol[1]:.1%}  Top-3(+-1): {topk_tol[3]:.1%}  Top-5(+-1): {topk_tol[5]:.1%}")
            print(f"    MRR: {mrr:.4f}")

    # Save summary
    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "train_charts": len(train_charts),
        "test_charts": len(test_charts),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "cv_metrics": cv_metrics,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Models saved to {model_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    train_path = args[0] if len(args) > 0 else None
    test_path = args[1] if len(args) > 1 else None
    model_dir = args[2] if len(args) > 2 else None
    train_full(train_path, test_path, model_dir)
