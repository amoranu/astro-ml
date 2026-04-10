"""Training pipeline — V3: continuous orb + temporal features.

Aggregates feature names from all 4 layers:
  1. Base features (features.py)          ~99 features
  2. Advanced features (advanced_features.py) ~49 features
  3. Orb reconstruction (orb_reconstruction.py) 44 features
  4. Temporal features (temporal_features.py)   ~95 features

Two-stage approach:
  Stage 1: LightGBM regressor with soft labels
  Stage 2: LightGBM LambdaMART ranker
  Ensemble: 0.4*clf + 0.5*ranker + 0.1*tier

Usage:
  python -m astro_ml.train_v3
"""
import os, sys, json, time, datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from astro_ml.data_prep import (
    load_charts, extract_all, build_dataset, build_ranking_dataset,
    TRAIN_PATH, TEST_PATH,
)
from astro_ml.features import FEATURE_NAMES
from astro_ml.advanced_features import ADVANCED_FEATURE_NAMES
from astro_ml.orb_reconstruction import ORB_RECON_FEATURE_NAMES
from astro_ml.temporal_features import TEMPORAL_FEATURE_NAMES

# Import evaluation metrics from train.py (reuse, don't duplicate)
from astro_ml.train import (
    compute_topk_accuracy, compute_topk_with_tolerance, compute_mrr,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_v3")
os.makedirs(MODEL_DIR, exist_ok=True)

# V3 feature names: base + advanced + orb_recon + temporal
ALL_V3_FEATURE_NAMES = (
    FEATURE_NAMES
    + ADVANCED_FEATURE_NAMES
    + ORB_RECON_FEATURE_NAMES
    + TEMPORAL_FEATURE_NAMES
)
N_V3_FEATURES = len(ALL_V3_FEATURE_NAMES)


# ── Stage 1: Regressor with soft labels ─────────────────────────────────

def train_classifier(X_train, y_soft, X_val=None, y_val_hard=None,
                     groups_val=None, feature_names=None):
    """Train LightGBM regressor with soft labels (0.0/0.5/1.0)."""
    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 800,
        "max_depth": 7,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    if feature_names is None:
        feature_names = ALL_V3_FEATURE_NAMES[:X_train.shape[1]]

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
        "n_estimators": 800,
        "max_depth": 7,
        "num_leaves": 63,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    if feature_names is None:
        feature_names = ALL_V3_FEATURE_NAMES[:X_train.shape[1]]

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
        num_boost_round=800,
        valid_sets=eval_sets,
        valid_names=eval_names,
        callbacks=callbacks,
    )

    return model


# ── Feature importance ───────────────────────────────────────────────────

def print_feature_importance(model, feature_names=None, top_k=30, model_type="classifier"):
    """Print top features by gain. Flag V3 temporal/orb features."""
    if model_type == "classifier":
        importances = model.feature_importances_
    else:
        importances = model.feature_importance(importance_type="gain")

    if feature_names is None:
        feature_names = ALL_V3_FEATURE_NAMES[:len(importances)]

    indices = np.argsort(-importances)
    print(f"\nTop {top_k} features by gain ({model_type}):")
    print("-" * 70)

    # V3 key features — should appear in top 20 if V3 is working
    v3_expected = {
        "convergence_proxy", "tightest_orb_inv_primary", "transit_freshness",
        "net_transit_signal", "trigger_score", "total_signal",
        "peak_composite", "env_x_trigger", "convergence_transition_score",
        "double_transit_orb_inv", "saturn_best_orb_inv",
    }
    # V1/V2 expected
    v2_expected = {
        "dasha_quality_score", "has_primary_dasha_lock", "n_primary_cusps_active",
        "death_signal_composite", "saturn_dasha_level", "maraka_score",
        "deterministic_tier",
    }

    v3_in_top20 = 0
    for r in range(min(top_k, len(indices))):
        i = indices[r]
        name = feature_names[i] if i < len(feature_names) else f"f_{i}"
        if name in v3_expected:
            flag = " *** V3-NEW"
            if r < 20:
                v3_in_top20 += 1
        elif name in v2_expected:
            flag = " <-- BASE"
        else:
            flag = ""
        print(f"  {r + 1:3d}. {name:45s} gain={importances[i]:.1f}{flag}")

    print(f"\n  V3 temporal/orb features in top-20: {v3_in_top20}")
    return indices, importances


# ── Cross-validation ─────────────────────────────────────────────────────

def cross_validate(X, y_hard, y_soft, y_rank, groups, info, n_folds=5):
    """GroupKFold cross-validation with V3 features."""
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

        def _group_sizes(g):
            return [int((g == ug).sum()) for ug in sorted(set(g))]

        gs_tr = _group_sizes(groups_tr)
        gs_val = _group_sizes(groups_val)

        # Train classifier
        clf = train_classifier(X_tr, y_soft_tr, X_val, y_hard_val, groups_val)
        clf_scores = clf.predict(X_val)

        # Train ranker
        ranker = train_ranker(X_tr, y_rank_tr, gs_tr, X_val, y_rank_val, gs_val)
        ranker_scores = ranker.predict(X_val)

        info_val = [info[i] for i in val_idx]

        # Ensemble
        tier_idx = (ALL_V3_FEATURE_NAMES.index("deterministic_tier")
                    if "deterministic_tier" in ALL_V3_FEATURE_NAMES else -1)
        if 0 <= tier_idx < X_val.shape[1]:
            tier_scores = X_val[:, tier_idx] / 5.0
        else:
            tier_scores = np.zeros(len(X_val))

        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-10)

        ensemble_scores = 0.4 * _norm(clf_scores) + 0.5 * _norm(ranker_scores) + 0.1 * tier_scores

        for name, scores in [("Classifier", clf_scores), ("Ranker", ranker_scores),
                              ("Ensemble", ensemble_scores)]:
            topk = compute_topk_accuracy(scores, groups_val, y_hard_val, (1, 3, 5))
            topk_tol = compute_topk_with_tolerance(scores, groups_val, info_val, (1, 3, 5))
            mrr = compute_mrr(scores, groups_val, y_hard_val)
            print(f"\n  {name}:")
            print(f"    Top-1: {topk[1]:.1%}  Top-3: {topk[3]:.1%}  Top-5: {topk[5]:.1%}")
            print(f"    Top-1(+-1): {topk_tol[1]:.1%}  Top-3(+-1): {topk_tol[3]:.1%}  Top-5(+-1): {topk_tol[5]:.1%}")
            print(f"    MRR: {mrr:.4f}")

        fold_metrics.append({
            "fold": fold + 1,
            "ensemble_topk": compute_topk_accuracy(ensemble_scores, groups_val, y_hard_val),
            "ensemble_topk_tol": compute_topk_with_tolerance(ensemble_scores, groups_val, info_val),
            "ensemble_mrr": compute_mrr(ensemble_scores, groups_val, y_hard_val),
            "clf_mrr": compute_mrr(clf_scores, groups_val, y_hard_val),
            "ranker_mrr": compute_mrr(ranker_scores, groups_val, y_hard_val),
        })

    # Summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary (V3)")
    print(f"{'='*60}")
    for key in ["clf", "ranker", "ensemble"]:
        mrrs = [f[f"{key}_mrr"] for f in fold_metrics]
        print(f"  {key.capitalize():12s}: MRR={np.mean(mrrs):.4f}+-{np.std(mrrs):.4f}")
    ens_top3s = [f["ensemble_topk"][3] for f in fold_metrics]
    ens_top3_tols = [f["ensemble_topk_tol"][3] for f in fold_metrics]
    print(f"  Ensemble Top-3:      {np.mean(ens_top3s):.1%}+-{np.std(ens_top3s):.1%}")
    print(f"  Ensemble Top-3(+-1): {np.mean(ens_top3_tols):.1%}+-{np.std(ens_top3_tols):.1%}")

    return fold_metrics


# ── Full training pipeline ───────────────────────────────────────────────

def train_full(train_path=None, test_path=None, model_dir=None):
    """Full V3 training pipeline: extract, train, evaluate, save."""
    if model_dir is None:
        model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    t0 = time.time()

    # Load and extract
    print(f"V3 Pipeline — {N_V3_FEATURES} total features")
    print(f"  Base: {len(FEATURE_NAMES)}, Advanced: {len(ADVANCED_FEATURE_NAMES)}, "
          f"Orb-Recon: {len(ORB_RECON_FEATURE_NAMES)}, Temporal: {len(TEMPORAL_FEATURE_NAMES)}")

    print("\nLoading training data...")
    train_charts = load_charts(train_path or TRAIN_PATH)
    print("Loading test data...")
    test_charts = load_charts(test_path or TEST_PATH)

    print("\nExtracting training features (V3 — 4 layers)...")
    train_results = extract_all(train_charts)
    print("\nExtracting test features (V3 — 4 layers)...")
    test_results = extract_all(test_charts)

    # Build datasets (with random position slicing)
    print("\nBuilding V3 datasets (random sub-window slicing)...")
    X_train, y_hard_train, y_soft_train, y_rank_train, groups_train, info_train = build_dataset(train_results)
    X_test, y_hard_test, y_soft_test, y_rank_test, groups_test, info_test = build_dataset(test_results, randomize_position=False)

    if X_train.shape[0] == 0:
        print("ERROR: No training data extracted!")
        return

    print(f"\nFeature vector size: {X_train.shape[1]} (expected {N_V3_FEATURES})")
    if X_train.shape[1] != N_V3_FEATURES:
        print(f"WARNING: Feature count mismatch! Got {X_train.shape[1]}, expected {N_V3_FEATURES}")

    feature_names = ALL_V3_FEATURE_NAMES[:X_train.shape[1]]

    # Cross-validation
    print("\n--- Cross-Validation (V3) ---")
    cv_metrics = cross_validate(X_train, y_hard_train, y_soft_train, y_rank_train,
                                groups_train, info_train)

    # Train final models on full training set
    print(f"\n{'='*60}")
    print("Training final V3 models on full training set...")
    print(f"{'='*60}")

    clf = train_classifier(X_train, y_soft_train, feature_names=feature_names)
    clf.booster_.save_model(os.path.join(model_dir, "classifier_v3.txt"))

    _, y_rank_all, groups_all, gs_all, _ = build_ranking_dataset(train_results)
    ranker = train_ranker(X_train, y_rank_train, gs_all, feature_names=feature_names)
    ranker.save_model(os.path.join(model_dir, "ranker_v3.txt"))

    # Feature importance
    print_feature_importance(clf, feature_names, top_k=30, model_type="classifier")
    print_feature_importance(ranker, feature_names, top_k=30, model_type="ranker")

    # Evaluate on test set
    if X_test.shape[0] > 0:
        print(f"\n{'='*60}")
        print("Test Set Evaluation (V3)")
        print(f"{'='*60}")

        clf_scores = clf.predict(X_test)
        ranker_scores = ranker.predict(X_test)

        tier_idx = (ALL_V3_FEATURE_NAMES.index("deterministic_tier")
                    if "deterministic_tier" in ALL_V3_FEATURE_NAMES else -1)
        if 0 <= tier_idx < X_test.shape[1]:
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
        "version": "V3",
        "timestamp": datetime.datetime.now().isoformat(),
        "train_charts": len(train_charts),
        "test_charts": len(test_charts),
        "n_features": X_train.shape[1],
        "feature_breakdown": {
            "base": len(FEATURE_NAMES),
            "advanced": len(ADVANCED_FEATURE_NAMES),
            "orb_recon": len(ORB_RECON_FEATURE_NAMES),
            "temporal": len(TEMPORAL_FEATURE_NAMES),
        },
        "feature_names": feature_names,
        "cv_metrics": cv_metrics,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(model_dir, "summary_v3.json"), "w") as f:
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
