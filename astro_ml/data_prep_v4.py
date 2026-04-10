"""V4 Data Preparation — extraction with per-cusp features + rank normalization.

Uses V3 extraction pipeline internally, then adds V4 features.
Key difference from data_prep.py: keeps the payload around for per-cusp features.
"""
import os, sys, json, datetime, hashlib, pickle, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from astro_ml.data_prep import (
    load_charts, generate_labels, TRAIN_PATH, TEST_PATH,
    EXTRACT_WINDOW_MONTHS, TRAIN_WINDOW_MONTHS, CACHE_DIR,
)
from astro_ml.features_v4 import (
    extract_features_v4, rank_normalize_chart, V4_FEATURE_NAMES, N_V4_FEATURES,
)
from astro_ml.temporal_features import compute_temporal_features

CACHE_VERSION_V4 = "astro_ml_v4"


def _compute_one_chart_v4(chart):
    """Compute V3 features + V4 per-cusp features for one chart.

    Returns (chart_id, windows, event_month) or None.
    Each window has v4_vector populated.
    """
    from astro_ml.compute import compute
    from astro_ml.features import extract_monthly_windows
    from astro_ml.advanced_features import add_advanced_death_features
    from astro_ml.orb_reconstruction import reconstruct_orb_features
    from astro_ml.temporal_features import compute_temporal_features

    try:
        fd = chart["father_death_date"]
        parts = fd.split("-")
        death_dt = datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        death_month = f"{death_dt.year}-{death_dt.month:02d}"

        # Extract WIDE window: 24 months before + 24 months after death
        start_dt = death_dt - datetime.timedelta(days=EXTRACT_WINDOW_MONTHS // 2 * 30)
        end_dt = death_dt + datetime.timedelta(days=EXTRACT_WINDOW_MONTHS // 2 * 30)

        payload = compute(chart, start_date=start_dt, end_date=end_dt)

        # V3 pipeline: base → advanced → orb_recon → temporal
        windows = extract_monthly_windows(payload)
        windows = add_advanced_death_features(windows, payload)
        windows = reconstruct_orb_features(windows, payload)
        windows = compute_temporal_features(windows)

        # V4: per-cusp + rank normalize + assemble v4_vector
        windows = extract_features_v4(windows, payload)

        chart_id = chart.get("id", chart.get("name", "unknown"))
        return chart_id, windows, death_month
    except Exception as e:
        name = chart.get("name", "unknown")
        print(f"  ERROR extracting {name}: {e}")
        traceback.print_exc()
        return None


def extract_all_v4(charts, max_workers=8, use_cache=True):
    """Extract V4 features for all charts in parallel."""
    cache_path = os.path.join(CACHE_DIR, f"{CACHE_VERSION_V4}_{len(charts)}.pkl")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached V4 features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_one_chart_v4, c): i for i, c in enumerate(charts)}
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


def build_dataset_v4(extracted_results, randomize_position=True):
    """Build V4 dataset from extracted results.

    After slicing, recomputes temporal features AND rank normalization
    on the slice (both depend on sequence context).
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

        # Recompute temporal features on the slice
        windows_slice = compute_temporal_features(windows_slice)

        # Recompute rank normalization on the slice (ranks depend on window context)
        windows_slice = rank_normalize_chart(windows_slice)

        # Rebuild v4_vector after recomputation
        from astro_ml.features_v4 import assemble_v4_vector
        for w in windows_slice:
            w["v4_vector"] = assemble_v4_vector(w)

        y_hard, y_soft, y_rank = generate_labels(windows_slice, event_month)

        for i, w in enumerate(windows_slice):
            all_X.append(w["v4_vector"])
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

    print(f"V4 Dataset: {X.shape[0]} windows, {len(set(all_groups))} charts, {X.shape[1]} features")
    print(f"  Positive (hard): {int(y_hard.sum())} windows")
    print(f"  Positive (soft): {int((y_soft > 0).sum())} windows")
    return X, y_hard, y_soft, y_rank, groups, all_info


def build_ranking_dataset_v4(extracted_results):
    """Build dataset formatted for LambdaMART ranking."""
    X, _, _, y_rank, groups, info = build_dataset_v4(extracted_results)
    unique_groups = sorted(set(groups))
    group_sizes = [int((groups == g).sum()) for g in unique_groups]
    return X, y_rank, groups, group_sizes, info
