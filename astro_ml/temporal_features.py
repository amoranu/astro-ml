"""Temporal Features — V3 core fix: transitions, deltas, peaks, composites.

These features create PEAKS at exact event months instead of flat plateaus.

Group B: Dasha transition features (PD/AD changes within month)
Group C: Delta features (month-over-month changes)
Group D: Rolling & peak detection
Group E: Composite temporal scores

MUST be computed AFTER base + advanced + orb_reconstruction features.
"""
import numpy as np

from astro_ml.config import domain_fathers_death as cfg

# ── Feature names ────────────────────────────────────────────────────────

TEMPORAL_FEATURE_NAMES = []

# Features we take deltas of (from base + advanced + orb_recon)
_DELTA_SOURCE_FEATURES = [
    "n_primary_cusps_active",
    "n_primary_cusps_transit",
    "n_secondary_cusps_active",
    "n_negative_cusps_active",
    "n_dasha_locks_primary",
    "n_double_activations_primary",
    "net_activation_score",
    "dasha_quality_score",
    "deterministic_tier",
    "death_signal_composite",
    "maraka_score",
    "negative_event_strength",
    # From orb_recon
    "convergence_proxy",
    "tightest_orb_inv_primary",
    "transit_freshness",
    "net_transit_signal",
    "saturn_best_orb_inv",
    "jupiter_best_orb_inv",
]

# Features we compute rolling windows and peaks over
_ROLLING_SOURCE_FEATURES = [
    "convergence_proxy",
    "tightest_orb_inv_primary",
    "n_primary_cusps_active",
    "net_activation_score",
    "net_transit_signal",
    "transit_freshness",
]


def _build_temporal_names():
    names = []
    # Group B: Dasha transitions (8)
    names.extend([
        "is_pd_transition",          # PD lord changed this month
        "is_ad_transition",          # AD lord changed this month
        "is_md_transition",          # MD lord changed this month (rare)
        "is_any_dasha_transition",   # any dasha level changed
        "pd_tenure_months",          # how many months current PD has been running
        "is_new_pd",                 # PD tenure == 1 (just started)
        "is_new_ad",                 # AD tenure == 1
        "dasha_transition_count",    # count of transitions (0-3)
    ])
    # Group C: Deltas (len(_DELTA_SOURCE_FEATURES) + 2)
    for feat in _DELTA_SOURCE_FEATURES:
        names.append(f"delta_{feat}")
    names.extend(["delta_pd_lord_changed", "delta_ad_lord_changed"])
    # Group D: Rolling & peaks
    for n in [2, 3]:
        for feat in _ROLLING_SOURCE_FEATURES:
            names.append(f"roll{n}_max_{feat}")
            names.append(f"roll{n}_mean_{feat}")
    for feat in _ROLLING_SOURCE_FEATURES:
        names.append(f"is_peak_{feat}")
        names.append(f"is_global_max_{feat}")
    # Group E: Composites (5)
    names.extend([
        "trigger_score",
        "convergence_transition_score",
        "peak_composite",
        "env_x_trigger",
        "total_signal",
    ])
    return names


TEMPORAL_FEATURE_NAMES = _build_temporal_names()
N_TEMPORAL = len(TEMPORAL_FEATURE_NAMES)


def _get_feature_value(w, feat_name):
    """Get a feature value from any of the feature dicts in a window."""
    # Check base features
    v = w.get("features", {}).get(feat_name)
    if v is not None:
        return float(v)
    # Check advanced
    v = w.get("advanced_features", {}).get(feat_name)
    if v is not None:
        return float(v)
    # Check orb_recon
    v = w.get("orb_recon_features", {}).get(feat_name)
    if v is not None:
        return float(v)
    return 0.0


def compute_temporal_features(windows):
    """Compute all temporal features (Groups B-E) on the full window list.

    MUST be called AFTER base, advanced, and orb_recon features are computed.
    Modifies windows in-place.

    Returns windows.
    """
    n = len(windows)
    if n == 0:
        return windows

    # ── Group B: Dasha transitions ───────────────────────────────────────
    # Detect PD/AD/MD changes between consecutive months
    pd_lords = [w.get("pd", "") for w in windows]
    ad_lords = [w.get("ad", "") for w in windows]
    md_lords = [w.get("md", "") for w in windows]

    is_pd_transition = np.zeros(n, dtype=np.float32)
    is_ad_transition = np.zeros(n, dtype=np.float32)
    is_md_transition = np.zeros(n, dtype=np.float32)
    pd_tenure = np.ones(n, dtype=np.float32)

    for i in range(1, n):
        if pd_lords[i] != pd_lords[i - 1]:
            is_pd_transition[i] = 1.0
            pd_tenure[i] = 1.0
        else:
            pd_tenure[i] = pd_tenure[i - 1] + 1.0
        if ad_lords[i] != ad_lords[i - 1]:
            is_ad_transition[i] = 1.0
        if md_lords[i] != md_lords[i - 1]:
            is_md_transition[i] = 1.0

    # ── Collect source feature arrays for deltas/rolling ─────────────────
    source_arrays = {}
    for feat in _DELTA_SOURCE_FEATURES:
        arr = np.array([_get_feature_value(w, feat) for w in windows], dtype=np.float32)
        source_arrays[feat] = arr

    rolling_arrays = {}
    for feat in _ROLLING_SOURCE_FEATURES:
        if feat in source_arrays:
            rolling_arrays[feat] = source_arrays[feat]
        else:
            rolling_arrays[feat] = np.array([_get_feature_value(w, feat) for w in windows],
                                            dtype=np.float32)

    # ── Assemble into per-window vectors ─────────────────────────────────
    for i, w in enumerate(windows):
        tv = np.zeros(N_TEMPORAL, dtype=np.float32)
        idx = 0

        # Group B: Dasha transitions (8)
        tv[idx] = is_pd_transition[i]
        tv[idx + 1] = is_ad_transition[i]
        tv[idx + 2] = is_md_transition[i]
        tv[idx + 3] = 1.0 if (is_pd_transition[i] or is_ad_transition[i] or is_md_transition[i]) else 0.0
        tv[idx + 4] = pd_tenure[i]
        tv[idx + 5] = 1.0 if pd_tenure[i] == 1.0 else 0.0
        tv[idx + 6] = 1.0 if (i > 0 and ad_lords[i] != ad_lords[i - 1]) and i <= 1 or (i > 1 and ad_lords[i] != ad_lords[i - 1]) else is_ad_transition[i]  # is_new_ad
        tv[idx + 7] = is_pd_transition[i] + is_ad_transition[i] + is_md_transition[i]
        idx += 8

        # Group C: Deltas (len + 2)
        for fi, feat in enumerate(_DELTA_SOURCE_FEATURES):
            if i > 0:
                tv[idx] = source_arrays[feat][i] - source_arrays[feat][i - 1]
            idx += 1
        # delta_pd_lord_changed, delta_ad_lord_changed
        tv[idx] = is_pd_transition[i]
        tv[idx + 1] = is_ad_transition[i]
        idx += 2

        # Group D: Rolling (2 window sizes × features × 2 stats) + peaks
        for n_roll in [2, 3]:
            for feat in _ROLLING_SOURCE_FEATURES:
                arr = rolling_arrays[feat]
                start = max(0, i - n_roll + 1)
                window_vals = arr[start:i + 1]
                tv[idx] = float(window_vals.max())
                tv[idx + 1] = float(window_vals.mean())
                idx += 2

        # Peak detection
        for feat in _ROLLING_SOURCE_FEATURES:
            arr = rolling_arrays[feat]
            val = arr[i]
            # is_peak: value >= both neighbors AND > 0
            if i > 0 and i < n - 1:
                is_peak = (val >= arr[i - 1] and val >= arr[i + 1] and val > 0)
            elif i == 0:
                is_peak = (val > 0 and (n == 1 or val >= arr[i + 1]))
            else:
                is_peak = (val > 0 and val >= arr[i - 1])
            tv[idx] = 1.0 if is_peak else 0.0

            # is_global_max
            tv[idx + 1] = 1.0 if (val > 0 and val >= arr.max()) else 0.0
            idx += 2

        # Group E: Composites (5)
        # Gather needed values
        conv_proxy = _get_feature_value(w, "convergence_proxy")
        n_ing_pri = _get_feature_value(w, "n_ingresses_primary")
        any_pri_ing = _get_feature_value(w, "any_primary_ingress")
        freshness = _get_feature_value(w, "transit_freshness")
        sat_pri_exact = _get_feature_value(w, "saturn_primary_exact")
        sat_pri_ing = _get_feature_value(w, "saturn_primary_ingress")
        quality = _get_feature_value(w, "dasha_quality_score")
        has_lock = _get_feature_value(w, "has_primary_dasha_lock")
        death_karaka = _get_feature_value(w, "karaka_active_count")
        death_composite = _get_feature_value(w, "death_signal_composite")
        delta_pri = tv[8 + _DELTA_SOURCE_FEATURES.index("n_primary_cusps_active")] if "n_primary_cusps_active" in _DELTA_SOURCE_FEATURES else 0.0

        # trigger_score
        trigger = (conv_proxy * 2
                   + n_ing_pri * 3
                   + any_pri_ing * 5
                   + is_pd_transition[i] * 4
                   + is_ad_transition[i] * 6
                   + delta_pri * 3
                   + sat_pri_exact * 5
                   + sat_pri_ing * 4)
        tv[idx] = trigger

        # convergence_transition_score
        tv[idx + 1] = conv_proxy * (1 + is_pd_transition[i] * 2) * (1 + is_ad_transition[i] * 3)

        # peak_composite: weighted sum of is_peak signals
        peak_sum = 0.0
        peak_start = 8 + len(_DELTA_SOURCE_FEATURES) + 2 + len(_ROLLING_SOURCE_FEATURES) * 2 * 2  # after rolling
        for fi in range(len(_ROLLING_SOURCE_FEATURES)):
            peak_sum += tv[peak_start + fi * 2] * 2.0       # is_peak weight
            peak_sum += tv[peak_start + fi * 2 + 1] * 3.0   # is_global_max weight
        tv[idx + 2] = peak_sum

        # env_x_trigger
        env = quality + has_lock * 3 + death_karaka * 2
        tv[idx + 3] = env * max(trigger, 0.1)

        # total_signal
        tv[idx + 4] = death_composite * 0.3 + trigger * 0.4 + tv[idx + 2] * 0.3

        w["temporal_vector"] = tv
        w["temporal_features"] = {TEMPORAL_FEATURE_NAMES[j]: float(tv[j])
                                  for j in range(N_TEMPORAL)}

    return windows
