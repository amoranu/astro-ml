"""Data preparation — loading, label generation, dataset building.

Loads chart data from JSON files, computes payloads via compute.py,
extracts features, and builds train/val/test datasets with labels.
"""
import os, sys, json, datetime, hashlib, pickle, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "ml")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "father_passing_date_v2_clean.json")
TEST_PATH = os.path.join(DATA_DIR, "father_passing_date_clean.json")

# Window config
EXTRACT_WINDOW_MONTHS = 48  # wide extraction window (death centered)
TRAIN_WINDOW_MONTHS = 24    # 2-year training/eval window
CACHE_VERSION = "astro_ml_v3"


def load_charts(path=None):
    """Load chart list from JSON file."""
    if path is None:
        path = TRAIN_PATH
    with open(path, "r", encoding="utf-8") as f:
        charts = json.load(f)
    # Filter to those with father_death_date
    valid = [c for c in charts if c.get("father_death_date")]
    print(f"Loaded {len(valid)} charts with father_death_date from {os.path.basename(path)}")
    return valid


def _chart_hash(chart):
    """Stable hash for cache keying."""
    key = f"{chart.get('name', '')}__{chart.get('birth_date', '')}__{chart.get('birth_time', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _compute_one_chart(chart):
    """Compute payload + V3 features for one chart.

    Extracts a wide 48-month window centered on death, applies ALL feature layers:
      1. Base features (features.py)
      2. Advanced features (advanced_features.py)
      3. Orb reconstruction (orb_reconstruction.py)
      4. Temporal features (temporal_features.py) — MUST be last (depends on 1-3)

    Returns (chart_id, windows, event_month, payload) or None.
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

        # Step 1: Base features
        windows = extract_monthly_windows(payload)
        # Step 2: Advanced features
        windows = add_advanced_death_features(windows, payload)
        # Step 3: Orb reconstruction features
        windows = reconstruct_orb_features(windows, payload)
        # Step 4: Temporal features (deltas, peaks, composites — depends on 1-3)
        windows = compute_temporal_features(windows)

        # Build V3 full vector: base + advanced + orb_recon + temporal
        for w in windows:
            parts = [w["feature_vector"], w.get("advanced_vector", np.array([], dtype=np.float32))]
            if "orb_recon_vector" in w:
                parts.append(w["orb_recon_vector"])
            if "temporal_vector" in w:
                parts.append(w["temporal_vector"])
            w["v3_vector"] = np.concatenate(parts)

        chart_id = chart.get("id", chart.get("name", "unknown"))
        return chart_id, windows, death_month
    except Exception as e:
        name = chart.get("name", "unknown")
        print(f"  ERROR extracting {name}: {e}")
        traceback.print_exc()
        return None


def extract_all(charts, max_workers=8, use_cache=True):
    """Extract features for all charts in parallel.

    Returns list of (chart_id, windows, event_month) tuples.
    """
    cache_path = os.path.join(CACHE_DIR, f"{CACHE_VERSION}_{len(charts)}.pkl")

    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = []
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_one_chart, c): i for i, c in enumerate(charts)}
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


# ── Label generation ─────────────────────────────────────────────────────

def generate_labels(windows, event_month):
    """Generate hard, soft, and ranking labels for a set of windows.

    Args:
        windows: list of window dicts from extract_monthly_windows
        event_month: "YYYY-MM" string of the actual event

    Returns:
        y_hard: np.array — 1.0 for exact month, 0.0 otherwise
        y_soft: np.array — 1.0 exact, 0.5 for +/-1 month, 0.0 otherwise
        y_rank: np.array — 2 for exact, 1 for +/-1 month, 0 otherwise
    """
    n = len(windows)
    y_hard = np.zeros(n, dtype=np.float32)
    y_soft = np.zeros(n, dtype=np.float32)
    y_rank = np.zeros(n, dtype=np.int32)

    # Parse event month
    try:
        ep = event_month.split("-")
        event_dt = datetime.date(int(ep[0]), int(ep[1]), 1)
    except (ValueError, IndexError):
        return y_hard, y_soft, y_rank

    # Adjacent months
    if event_dt.month == 1:
        prev_month = f"{event_dt.year - 1}-12"
    else:
        prev_month = f"{event_dt.year}-{event_dt.month - 1:02d}"
    if event_dt.month == 12:
        next_month = f"{event_dt.year + 1}-01"
    else:
        next_month = f"{event_dt.year}-{event_dt.month + 1:02d}"

    for i, w in enumerate(windows):
        m = w["month"]
        if m == event_month:
            y_hard[i] = 1.0
            y_soft[i] = 1.0
            y_rank[i] = 2
        elif m in (prev_month, next_month):
            y_soft[i] = 0.5
            y_rank[i] = 1

    return y_hard, y_soft, y_rank


# ── Dataset building ─────────────────────────────────────────────────────

def _slice_random_window(windows, event_month, window_size=TRAIN_WINDOW_MONTHS):
    """Slice a random TRAIN_WINDOW_MONTHS sub-window from the wide extraction.

    Death lands at a random position within the sub-window, preventing
    position bias. Returns the sliced windows list.
    """
    months = [w["month"] for w in windows]
    if event_month not in months:
        # Death not in extraction — use the middle chunk
        start = max(0, len(windows) // 2 - window_size // 2)
        return windows[start:start + window_size]

    death_idx = months.index(event_month)

    # Death must land somewhere in the sub-window [0, window_size-1]
    # So start can range from (death_idx - window_size + 1) to death_idx
    earliest_start = max(0, death_idx - window_size + 1)
    latest_start = min(death_idx, len(windows) - window_size)
    latest_start = max(earliest_start, latest_start)

    import random
    start = random.randint(earliest_start, latest_start)
    return windows[start:start + window_size]


def build_dataset(extracted_results, use_advanced=True, randomize_position=True):
    """Build V3 dataset from extracted results.

    When randomize_position=True, slices a random 24-month sub-window from
    each chart's 48-month extraction so death lands at different positions.
    After slicing, recomputes temporal features (deltas/peaks depend on context).

    Uses v3_vector (base + advanced + orb_recon + temporal) by default.
    """
    from astro_ml.temporal_features import compute_temporal_features

    all_X = []
    all_y_hard = []
    all_y_soft = []
    all_y_rank = []
    all_groups = []
    all_info = []

    for group_idx, (chart_id, windows, event_month) in enumerate(extracted_results):
        if not windows:
            continue

        # Slice random sub-window to randomize death position
        if randomize_position and len(windows) > TRAIN_WINDOW_MONTHS:
            windows_slice = _slice_random_window(windows, event_month)
        else:
            windows_slice = list(windows)

        # Recompute temporal features on the slice (deltas/peaks depend on sequence)
        windows_slice = compute_temporal_features(windows_slice)

        # Rebuild v3_vector after temporal recomputation
        for w in windows_slice:
            parts = [w["feature_vector"], w.get("advanced_vector", np.array([], dtype=np.float32))]
            if "orb_recon_vector" in w:
                parts.append(w["orb_recon_vector"])
            if "temporal_vector" in w:
                parts.append(w["temporal_vector"])
            w["v3_vector"] = np.concatenate(parts)

        y_hard, y_soft, y_rank = generate_labels(windows_slice, event_month)

        for i, w in enumerate(windows_slice):
            vec = w.get("v3_vector", w.get("full_vector", w["feature_vector"]))
            all_X.append(vec)
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

    print(f"Dataset: {X.shape[0]} windows, {len(set(all_groups))} charts, {X.shape[1]} features")
    print(f"  Positive (hard): {int(y_hard.sum())} windows")
    print(f"  Positive (soft): {int((y_soft > 0).sum())} windows")
    return X, y_hard, y_soft, y_rank, groups, all_info


def build_ranking_dataset(extracted_results, use_advanced=True):
    """Build dataset formatted for LambdaMART ranking.

    Returns:
        X, y_relevance, groups, group_sizes, info
    Where group_sizes[i] = number of windows in group i.
    """
    X, _, _, y_rank, groups, info = build_dataset(extracted_results, use_advanced)

    # Compute group sizes
    unique_groups = sorted(set(groups))
    group_sizes = []
    for g in unique_groups:
        group_sizes.append(int((groups == g).sum()))

    return X, y_rank, groups, group_sizes, info


def prepare_splits(train_path=None, test_path=None, max_workers=8, use_cache=True):
    """Prepare train and test datasets.

    Returns dict with 'train' and 'test' keys, each containing
    (X, y_hard, y_soft, y_rank, groups, info).
    """
    train_charts = load_charts(train_path or TRAIN_PATH)
    test_charts = load_charts(test_path or TEST_PATH)

    print("\n--- Extracting train set ---")
    train_results = extract_all(train_charts, max_workers, use_cache)
    print("\n--- Extracting test set ---")
    test_results = extract_all(test_charts, max_workers, use_cache)

    print("\n--- Building train dataset ---")
    train_data = build_dataset(train_results)
    print("\n--- Building test dataset ---")
    test_data = build_dataset(test_results)

    return {
        "train": train_data,
        "test": test_data,
        "train_results": train_results,
        "test_results": test_results,
    }
