"""Training pipeline — V6: multi-slice augmentation + feature pruning on V4 base.

Key changes from V4:
  1. Multi-slice augmentation: N_SLICES (default 5) random window slices per chart
     during training → 5× more diverse training data, reduces position bias
  2. Two-pass training: first pass identifies top-K features by importance,
     second pass trains on pruned feature set for better generalization
  3. Even stronger regularization to fight the ~20-year era shift between
     train (mean death year 1957) and test (mean death year 1937)

Usage:
  python -m astro_ml.train_v6
"""
import os, sys, json, time, datetime
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from astro_ml.data_prep import load_charts, TRAIN_PATH, TEST_PATH
from astro_ml.data_prep_v4 import (
    extract_all_v4, build_dataset_v4, build_ranking_dataset_v4,
)
from astro_ml.features_v4 import V4_FEATURE_NAMES, N_V4_FEATURES
from astro_ml.train import (
    compute_topk_accuracy, compute_topk_with_tolerance, compute_mrr,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_v6")
os.makedirs(MODEL_DIR, exist_ok=True)

N_SLICES = 5          # slices per chart during training
TOP_K_FEATURES = 30   # prune to top-K after first pass


def build_dataset_v4_multislice(extracted_results, n_slices=N_SLICES):
    """Build V4 dataset with N random slices per chart.

    Each chart contributes N different random window slices, each with
    the death month at a different position. This gives the model
    more diverse views of each chart's temporal context.
    """
    from astro_ml.data_prep import generate_labels, TRAIN_WINDOW_MONTHS
    from astro_ml.data_prep_v4 import _slice_random_window
    from astro_ml.temporal_features import compute_temporal_features
    from astro_ml.features_v4 import rank_normalize_chart, assemble_v4_vector

    all_X = []
    all_y_hard = []
    all_y_soft = []
    all_y_rank = []
    all_groups = []
    all_info = []

    for group_idx, (chart_id, windows, event_month) in enumerate(extracted_results):
        if not windows:
            continue

        for slice_i in range(n_slices):
            if len(windows) > TRAIN_WINDOW_MONTHS:
                windows_slice = _slice_random_window(windows, event_month)
            else:
                windows_slice = list(windows)

            # Recompute context-dependent features on the slice
            windows_slice = compute_temporal_features(windows_slice)
            windows_slice = rank_normalize_chart(windows_slice)

            for w in windows_slice:
                w["v4_vector"] = assemble_v4_vector(w)

            y_hard, y_soft, y_rank = generate_labels(windows_slice, event_month)

            for i, w in enumerate(windows_slice):
                all_X.append(w["v4_vector"])
                all_y_hard.append(y_hard[i])
                all_y_soft.append(y_soft[i])
                all_y_rank.append(y_rank[i])
                # Same group_idx for all slices of same chart (prevents leakage in GroupKFold)
                all_groups.append(group_idx)
                all_info.append({
                    "chart_id": chart_id,
                    "month": w["month"],
                    "event_month": event_month,
                    "md": w.get("md", ""),
                    "ad": w.get("ad", ""),
                    "pd": w.get("pd", ""),
                })

    X = np.stack(all_X) if all_X else np.zeros((0, 0), dtype=np.float32)
    y_hard = np.array(all_y_hard, dtype=np.float32)
    y_soft = np.array(all_y_soft, dtype=np.float32)
    y_rank = np.array(all_y_rank, dtype=np.int32)
    groups = np.array(all_groups, dtype=np.int32)

    print(f"V6 Dataset ({n_slices} slices): {X.shape[0]} windows, "
          f"{len(set(all_groups))} charts, {X.shape[1]} features")
    print(f"  Positive (hard): {int(y_hard.sum())} windows")
    print(f"  Positive (soft): {int((y_soft > 0).sum())} windows")
    return X, y_hard, y_soft, y_rank, groups, all_info


def _lgb_params_clf():
    return {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "max_depth": 4,           # same as V4 — multi-slice is the regularizer
        "num_leaves": 15,         # same as V4
        "learning_rate": 0.02,    # same as V4
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,  # same as V4
        "min_child_weight": 5.0,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "verbose": -1,
        "n_jobs": -1,
    }


def _lgb_params_ranker():
    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5],
        "label_gain": [0, 1, 3],
        "boosting_type": "gbdt",
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "min_child_weight": 5.0,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "verbose": -1,
        "n_jobs": -1,
    }


def train_classifier(X_train, y_soft, X_val=None, y_val_hard=None,
                     feature_names=None):
    params = _lgb_params_clf()
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.log_evaluation(period=100)]
    if X_val is not None:
        callbacks.append(lgb.early_stopping(60, verbose=True))
        model.fit(X_train, y_soft,
                  eval_set=[(X_val, y_val_hard)],
                  callbacks=callbacks,
                  feature_name=feature_names)
    else:
        model.fit(X_train, y_soft, callbacks=callbacks, feature_name=feature_names)
    return model


def train_ranker(X_train, y_rank, group_sizes_train,
                 X_val=None, y_rank_val=None, group_sizes_val=None,
                 feature_names=None):
    params = _lgb_params_ranker()
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    train_ds = lgb.Dataset(X_train, label=y_rank, group=group_sizes_train,
                           feature_name=feature_names)
    eval_sets = [train_ds]
    eval_names = ["train"]
    if X_val is not None:
        val_ds = lgb.Dataset(X_val, label=y_rank_val, group=group_sizes_val,
                             feature_name=feature_names, reference=train_ds)
        eval_sets.append(val_ds)
        eval_names.append("val")

    callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(60, verbose=True)]
    model = lgb.train(params, train_ds, num_boost_round=600,
                      valid_sets=eval_sets, valid_names=eval_names,
                      callbacks=callbacks)
    return model


def get_feature_importance(model, n_features, model_type="classifier"):
    """Get feature importance array."""
    if model_type == "classifier":
        return model.feature_importances_
    else:
        return model.feature_importance(importance_type="gain")


def select_top_features(X, feature_names, importance, top_k=TOP_K_FEATURES):
    """Select top-K features by importance. Returns pruned X, names, indices."""
    indices = np.argsort(-importance)[:top_k]
    indices = np.sort(indices)  # keep original order
    X_pruned = X[:, indices]
    names_pruned = [feature_names[i] for i in indices]
    return X_pruned, names_pruned, indices


def print_feature_importance(model, feature_names, top_k=30, model_type="classifier"):
    imp = get_feature_importance(model, len(feature_names), model_type)
    indices = np.argsort(-imp)
    print(f"\nTop {top_k} features by gain ({model_type}):")
    print("-" * 70)
    for r in range(min(top_k, len(indices))):
        i = indices[r]
        name = feature_names[i] if i < len(feature_names) else f"f_{i}"
        flag = ""
        if name.startswith("cusp"):
            flag = " [PER-CUSP]"
        elif name.startswith("rank_"):
            flag = " [RANK]"
        print(f"  {r + 1:3d}. {name:45s} gain={imp[i]:.1f}{flag}")

    total_gain = imp.sum()
    top1_share = imp[indices[0]] / total_gain if total_gain > 0 else 0
    print(f"\n  Top-1 share: {top1_share:.1%}")


def _group_sizes(g):
    return [int((g == ug).sum()) for ug in sorted(set(g))]


def cross_validate(X, y_hard, y_soft, y_rank, groups, info,
                   feature_names, n_folds=5, pruned_indices=None):
    """GroupKFold CV. If pruned_indices is set, uses subset of features."""
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

        if pruned_indices is not None:
            X_tr = X_tr[:, pruned_indices]
            X_val = X_val[:, pruned_indices]
            fn = [feature_names[i] for i in pruned_indices]
        else:
            fn = feature_names

        clf = train_classifier(X_tr, y_soft_tr, X_val, y_hard_val, fn)
        clf_scores = clf.predict(X_val)

        ranker = train_ranker(X_tr, y_rank_tr, _group_sizes(groups_tr),
                              X_val, y_rank_val, _group_sizes(groups_val), fn)
        ranker_scores = ranker.predict(X_val)

        info_val = [info[i] for i in val_idx]

        # Deterministic tier for ensemble
        tier_name = "deterministic_tier"
        if pruned_indices is not None:
            fn_full = feature_names
            if tier_name in fn_full:
                full_idx = fn_full.index(tier_name)
                if full_idx in pruned_indices:
                    local_idx = list(pruned_indices).index(full_idx)
                    tier_scores = X_val[:, local_idx] / 5.0
                else:
                    tier_scores = np.zeros(len(X_val))
            else:
                tier_scores = np.zeros(len(X_val))
        else:
            if tier_name in feature_names:
                tier_idx = feature_names.index(tier_name)
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

    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for key in ["clf", "ranker", "ensemble"]:
        mrrs = [f[f"{key}_mrr"] for f in fold_metrics]
        print(f"  {key.capitalize():12s}: MRR={np.mean(mrrs):.4f}+-{np.std(mrrs):.4f}")
    ens_top3s = [f["ensemble_topk"][3] for f in fold_metrics]
    ens_top3_tols = [f["ensemble_topk_tol"][3] for f in fold_metrics]
    print(f"  Ensemble Top-3:      {np.mean(ens_top3s):.1%}+-{np.std(ens_top3s):.1%}")
    print(f"  Ensemble Top-3(+-1): {np.mean(ens_top3_tols):.1%}+-{np.std(ens_top3_tols):.1%}")

    return fold_metrics


def train_full(train_path=None, test_path=None, model_dir=None):
    if model_dir is None:
        model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    t0 = time.time()

    print(f"V6 Pipeline — {N_V4_FEATURES} V4 features, {N_SLICES} slices/chart, "
          f"prune to top-{TOP_K_FEATURES}")

    print("\nLoading data...")
    train_charts = load_charts(train_path or TRAIN_PATH)
    test_charts = load_charts(test_path or TEST_PATH)

    print("\nExtracting V4 features (cached)...")
    train_results = extract_all_v4(train_charts)
    test_results = extract_all_v4(test_charts)

    # ── Pass 1: Multi-slice augmented dataset ────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pass 1: Multi-slice training ({N_SLICES} slices/chart)")
    print(f"{'='*60}")

    X_train_aug, y_hard_aug, y_soft_aug, y_rank_aug, groups_aug, info_aug = \
        build_dataset_v4_multislice(train_results, n_slices=N_SLICES)

    # Single-slice test (no augmentation)
    X_test, y_hard_test, y_soft_test, y_rank_test, groups_test, info_test = \
        build_dataset_v4(test_results, randomize_position=False)

    feature_names = V4_FEATURE_NAMES[:X_train_aug.shape[1]]

    if X_train_aug.shape[0] == 0:
        print("ERROR: No training data!")
        return

    # Pass 1 CV (multi-slice)
    print("\n--- Pass 1 CV (full features, multi-slice) ---")
    cv1_metrics = cross_validate(
        X_train_aug, y_hard_aug, y_soft_aug, y_rank_aug, groups_aug, info_aug,
        feature_names
    )

    # Train a classifier on full augmented data to get feature importance
    print("\nTraining Pass 1 classifier for feature selection...")
    clf1 = train_classifier(X_train_aug, y_soft_aug, feature_names=feature_names)
    imp_clf = get_feature_importance(clf1, len(feature_names), "classifier")

    # Also train ranker for importance
    gs_aug = _group_sizes(groups_aug)
    ranker1 = train_ranker(X_train_aug, y_rank_aug, gs_aug, feature_names=feature_names)
    imp_ranker = get_feature_importance(ranker1, len(feature_names), "ranker")

    # Combined importance (average of normalized importances)
    imp_combined = imp_clf / (imp_clf.sum() + 1e-10) + imp_ranker / (imp_ranker.sum() + 1e-10)

    print_feature_importance(clf1, feature_names, top_k=30, model_type="classifier")

    # ── Pass 2: Pruned features ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pass 2: Pruned to top-{TOP_K_FEATURES} features")
    print(f"{'='*60}")

    _, pruned_names, pruned_indices = select_top_features(
        X_train_aug, feature_names, imp_combined, top_k=TOP_K_FEATURES
    )
    print(f"Selected features: {pruned_names}")

    # CV on pruned features
    print("\n--- Pass 2 CV (pruned features, multi-slice) ---")
    cv2_metrics = cross_validate(
        X_train_aug, y_hard_aug, y_soft_aug, y_rank_aug, groups_aug, info_aug,
        feature_names, pruned_indices=pruned_indices
    )

    # ── Final models on pruned features ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Training final V6 models (pruned, multi-slice)...")
    print(f"{'='*60}")

    X_train_pruned = X_train_aug[:, pruned_indices]
    X_test_pruned = X_test[:, pruned_indices]

    clf = train_classifier(X_train_pruned, y_soft_aug, feature_names=pruned_names)
    clf.booster_.save_model(os.path.join(model_dir, "classifier_v6.txt"))

    gs_all = _group_sizes(groups_aug)
    ranker = train_ranker(X_train_pruned, y_rank_aug, gs_all, feature_names=pruned_names)
    ranker.save_model(os.path.join(model_dir, "ranker_v6.txt"))

    print_feature_importance(clf, pruned_names, top_k=TOP_K_FEATURES, model_type="classifier")
    print_feature_importance(ranker, pruned_names, top_k=TOP_K_FEATURES, model_type="ranker")

    # ── Test evaluation ──────────────────────────────────────────────────
    if X_test_pruned.shape[0] > 0:
        print(f"\n{'='*60}")
        print("Test Set Evaluation (V6)")
        print(f"{'='*60}")

        clf_scores = clf.predict(X_test_pruned)
        ranker_scores = ranker.predict(X_test_pruned)

        if "deterministic_tier" in pruned_names:
            tier_idx = pruned_names.index("deterministic_tier")
            tier_scores = X_test_pruned[:, tier_idx] / 5.0
        else:
            tier_scores = np.zeros(len(X_test_pruned))

        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-10)

        ensemble_scores = 0.4 * _norm(clf_scores) + 0.5 * _norm(ranker_scores) + 0.1 * tier_scores

        cv_top3_tol = np.mean([f["ensemble_topk_tol"][3] for f in cv2_metrics])

        for name, scores in [("Classifier", clf_scores), ("Ranker", ranker_scores),
                              ("Ensemble", ensemble_scores)]:
            topk = compute_topk_accuracy(scores, groups_test, y_hard_test, (1, 3, 5))
            topk_tol = compute_topk_with_tolerance(scores, groups_test, info_test, (1, 3, 5))
            mrr = compute_mrr(scores, groups_test, y_hard_test)
            print(f"\n  {name}:")
            print(f"    Top-1: {topk[1]:.1%}  Top-3: {topk[3]:.1%}  Top-5: {topk[5]:.1%}")
            print(f"    Top-1(+-1): {topk_tol[1]:.1%}  Top-3(+-1): {topk_tol[3]:.1%}  Top-5(+-1): {topk_tol[5]:.1%}")
            print(f"    MRR: {mrr:.4f}")

        test_top3_tol = compute_topk_with_tolerance(ensemble_scores, groups_test, info_test)[3]
        gap = cv_top3_tol - test_top3_tol
        print(f"\n  CV-Test Gap (Top-3 +-1): {gap:.1%}")

    # ── Comparison ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("V6 vs V4 Comparison")
    print(f"{'='*60}")
    cv1_top3 = np.mean([f["ensemble_topk_tol"][3] for f in cv1_metrics])
    cv2_top3 = np.mean([f["ensemble_topk_tol"][3] for f in cv2_metrics])
    print(f"  Pass 1 (full features, multi-slice) CV Top-3(+-1): {cv1_top3:.1%}")
    print(f"  Pass 2 (pruned, multi-slice) CV Top-3(+-1):        {cv2_top3:.1%}")
    print(f"  V4 baseline CV Top-3(+-1):                          ~32.2%")

    elapsed = time.time() - t0
    summary = {
        "version": "V6",
        "timestamp": datetime.datetime.now().isoformat(),
        "n_slices": N_SLICES,
        "n_features_full": X_train_aug.shape[1],
        "n_features_pruned": len(pruned_names),
        "pruned_features": pruned_names,
        "pruned_indices": pruned_indices.tolist(),
        "cv1_metrics": cv1_metrics,
        "cv2_metrics": cv2_metrics,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(model_dir, "summary_v6.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    args = sys.argv[1:]
    train_full(args[0] if args else None, args[1] if len(args) > 1 else None,
               args[2] if len(args) > 2 else None)
