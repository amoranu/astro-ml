"""V5 Data Preparation — V3 extraction + V4 per-cusp + rule features + relative features.

No hard shortlisting. Rule scorer provides features, not filters.
"""
import os, datetime, pickle, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from astro_ml.data_prep import (
    load_charts, generate_labels, TRAIN_PATH, TEST_PATH,
    EXTRACT_WINDOW_MONTHS, TRAIN_WINDOW_MONTHS, CACHE_DIR,
)
from astro_ml.features_v4 import (
    compute_per_cusp_features, rank_normalize_chart,
)
from astro_ml.features_v5 import (
    compute_rule_features, compute_relative_features,
    assemble_v5_vector, V5_FEATURE_NAMES, N_V5_FEATURES,
)
from astro_ml.temporal_features import compute_temporal_features
from astro_ml.rule_scorer import score_chart

CACHE_VERSION_V5 = "astro_ml_v5"


def _compute_one_chart_v5(chart):
    """Full V5 extraction for one chart.

    Pipeline: compute → base features → advanced → orb_recon → temporal
              → per-cusp → rule scoring → rank normalize → relative features
              → assemble v5_vector
    """
    from astro_ml.compute import compute
    from astro_ml.features import extract_monthly_windows
    from astro_ml.advanced_features import add_advanced_death_features
    from astro_ml.orb_reconstruction import reconstruct_orb_features

    try:
        fd = chart["father_death_date"]
        parts = fd.split("-")
        death_dt = datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        death_month = f"{death_dt.year}-{death_dt.month:02d}"

        start_dt = death_dt - datetime.timedelta(days=EXTRACT_WINDOW_MONTHS // 2 * 30)
        end_dt = death_dt + datetime.timedelta(days=EXTRACT_WINDOW_MONTHS // 2 * 30)

        payload = compute(chart, start_date=start_dt, end_date=end_dt)

        # V3 pipeline
        windows = extract_monthly_windows(payload)
        windows = add_advanced_death_features(windows, payload)
        windows = reconstruct_orb_features(windows, payload)
        windows = compute_temporal_features(windows)

        # V4: per-cusp features
        windows = compute_per_cusp_features(windows, payload)

        # V5: rule scoring
        scored_windows = score_chart(payload)

        # V4: rank normalize
        windows = rank_normalize_chart(windows)

        # V5: rule features + relative features
        windows = compute_rule_features(windows, scored_windows)
        windows = compute_relative_features(windows, scored_windows)

        # Assemble v5 vectors
        for w in windows:
            w["v5_vector"] = assemble_v5_vector(w)

        chart_id = chart.get("id", chart.get("name", "unknown"))
        return chart_id, windows, death_month
    except Exception as e:
        name = chart.get("name", "unknown")
        print(f"  ERROR extracting {name}: {e}")
        traceback.print_exc()
        return None


def extract_all_v5(charts, max_workers=8, use_cache=True):
    """Extract V5 features for all charts in parallel."""
    cache_path = os.path.join(CACHE_DIR, f"{CACHE_VERSION_V5}_{len(charts)}.pkl")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached V5 features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_one_chart_v5, c): i for i, c in enumerate(charts)}
        for i, fut in enumerate(as_completed(futures)):
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i + 1}/{len(charts)}...")
            result = fut.result()
            if result is not None:
                results.append(result)
            else:
                failed += 1

    print(f"Extracted {len(results)} charts ({failed} failed)")

    if use_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Cached to {cache_path}")

    return results


def _slice_random_window(windows, event_month, window_size=TRAIN_WINDOW_MONTHS):
    """Slice a random sub-window. Death lands at random position."""
    months = [w["month"] for w in windows]
    if event_month not in months:
        start = max(0, len(windows) // 2 - window_size // 2)
        return windows[start:start + window_size]

    death_idx = months.index(event_month)
    earliest_start = max(0, death_idx - window_size + 1)
    latest_start = min(death_idx, len(windows) - window_size)
    latest_start = max(earliest_start, latest_start)

    import random
    start = random.randint(earliest_start, latest_start)
    return windows[start:start + window_size]


def build_dataset_v5(extracted_results, randomize_position=True):
    """Build V5 dataset.

    After slicing, recomputes temporal features, rank normalization,
    and relative features (all depend on window context).
    """
    all_X = []
    all_y_hard = []
    all_y_soft = []
    all_y_rank = []
    all_groups = []
    all_info = []

    for group_idx, (chart_id, windows, event_month) in enumerate(extracted_results):
        if not windows:
            continue

        # Slice random sub-window
        if randomize_position and len(windows) > TRAIN_WINDOW_MONTHS:
            windows_slice = _slice_random_window(windows, event_month)
        else:
            windows_slice = list(windows)

        # Recompute context-dependent features on the slice
        windows_slice = compute_temporal_features(windows_slice)
        windows_slice = rank_normalize_chart(windows_slice)

        # Recompute relative features on the slice
        # (need scored_windows scoped to this slice's months)
        from astro_ml.rule_scorer import score_chart as _sc  # already computed in extraction
        # Use the rule_features already on the windows, just recompute relative
        # Build mini scored_windows from existing rule_features
        scored_mini = []
        for w in windows_slice:
            rf = w.get("rule_features", {})
            scored_mini.append({
                "month_str": w["month"],
                "rule_score": rf.get("rule_score_normalized", 0) * 100,  # approximate
                "tier_score": int(rf.get("rule_tier_score", 0)),
                "components": {
                    "primary_active": [],
                    "primary_locks": [],
                    "primary_double": [],
                    "secondary_active": [],
                    "negative_active": [],
                    "total_quality": rf.get("rule_total_quality", 0),
                    "yogakaraka_count": int(rf.get("rule_yogakaraka_count", 0)),
                    "death_karaka_active": int(rf.get("rule_death_karaka_active", 0)),
                    "sun_active": rf.get("rule_sun_active", 0) > 0,
                    "karaka_count": 0,
                },
            })
        windows_slice = compute_relative_features(windows_slice, scored_mini)

        # Rebuild v5 vectors
        for w in windows_slice:
            w["v5_vector"] = assemble_v5_vector(w)

        y_hard, y_soft, y_rank = generate_labels(windows_slice, event_month)

        for i, w in enumerate(windows_slice):
            all_X.append(w["v5_vector"])
            all_y_hard.append(y_hard[i])
            all_y_soft.append(y_soft[i])
            all_y_rank.append(y_rank[i])
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

    print(f"V5 Dataset: {X.shape[0]} windows, {len(set(all_groups))} charts, {X.shape[1]} features")
    print(f"  Positive (hard): {int(y_hard.sum())} windows")
    print(f"  Positive (soft): {int((y_soft > 0).sum())} windows")
    return X, y_hard, y_soft, y_rank, groups, all_info


def build_ranking_dataset_v5(extracted_results):
    """Build dataset formatted for LambdaMART."""
    X, _, _, y_rank, groups, info = build_dataset_v5(extracted_results)
    unique_groups = sorted(set(groups))
    group_sizes = [int((groups == g).sum()) for g in unique_groups]
    return X, y_rank, groups, group_sizes, info
