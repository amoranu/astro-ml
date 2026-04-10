"""V4 Feature Engineering — per-cusp disaggregation + rank normalization.

Three key fixes over V3:
  1. Per-cusp features for PRIMARY cusps (3,4,10) and SECONDARY (8,12)
  2. Within-chart rank normalization for continuous features
  3. Aggressive feature selection (~55 features)

Calls V3 modules internally, then curates output.
"""
import numpy as np
from scipy.stats import rankdata

from astro_ml.config import domain_fathers_death as cfg

# ── V4 Feature Names ─────────────────────────────────────────────────────

# Group A: Per-cusp features (25)
_PER_CUSP_NAMES = []
for _c in cfg.PRIMARY_HOUSES:  # [3, 4, 10]
    _PER_CUSP_NAMES.extend([
        f"cusp{_c}_dasha_active",
        f"cusp{_c}_lock_level",
        f"cusp{_c}_has_lock",
        f"cusp{_c}_transit_active",
        f"cusp{_c}_transit_ingress",
        f"cusp{_c}_transit_orb_inv",
        f"cusp{_c}_double_active",
    ])
for _c in cfg.SECONDARY_HOUSES:  # [8, 12]
    _PER_CUSP_NAMES.extend([
        f"cusp{_c}_active",
        f"cusp{_c}_transit_orb_inv",
    ])

# Group B: Rank-normalized continuous features (15)
FEATURES_TO_RANK = [
    "dasha_quality_score",
    "convergence_proxy",
    "trigger_score",
    "net_activation_score",
    "death_signal_composite",
    "maraka_score",
    "transit_freshness",
    "tightest_orb_inv_primary",
    "n_ingresses_primary",
    "env_x_trigger",
    "total_signal",
    "negative_event_strength",
    "saturn_transit_density",
    "activation_density",
    "net_transit_signal",
]
_RANK_NAMES = [f"rank_{f}" for f in FEATURES_TO_RANK]

# Group C: Binary/categorical (15)
_BINARY_NAMES = [
    "is_pd_transition",
    "is_ad_transition",
    "is_new_pd",
    "yogakaraka_count",
    "death_karaka_count",
    "karaka_active_count",
    "any_primary_ingress",
    "all_three_heavy_active",
    "saturn_is_md",
    "saturn_dasha_level",
    "both_marakas_active",
    "is_peak_convergence_proxy",
    "is_global_max_convergence_proxy",
    "dasha_transition_count",
    "deterministic_tier",
]

V4_FEATURE_NAMES = _PER_CUSP_NAMES + _RANK_NAMES + _BINARY_NAMES
N_V4_FEATURES = len(V4_FEATURE_NAMES)


# ── Per-cusp feature computation ─────────────────────────────────────────

def _prev_month_str(month_str):
    """'2020-03' -> '2020-02'. Handle year boundary."""
    parts = month_str.split("-")
    y, m = int(parts[0]), int(parts[1])
    m -= 1
    if m < 1:
        m = 12
        y -= 1
    return f"{y}-{m:02d}"


def compute_per_cusp_features(windows, payload):
    """Add per-cusp features to each window dict.

    For PRIMARY cusps (3,4,10): 7 features each.
    For SECONDARY cusps (8,12): 2 features each.

    Reads directly from Calculated_Triggers (which has per-cusp Macro_Hits,
    Dasha_Lock, Dasha_Lock_Level) and Planet_Significations.
    """
    triggers = payload.get("Calculated_Triggers", {})
    planet_sigs = payload.get("Planet_Significations", {})

    for w in windows:
        month_str = w["month"]
        md, ad, pd = w.get("md", ""), w.get("ad", ""), w.get("pd", "")
        month_data = triggers.get(month_str, {})
        prev_month_data = triggers.get(_prev_month_str(month_str), {})

        # Which cusps each dasha lord signifies
        cusps_md = set(planet_sigs.get(md, [])) if md else set()
        cusps_ad = set(planet_sigs.get(ad, [])) if ad else set()
        cusps_pd = set(planet_sigs.get(pd, [])) if pd else set()
        dasha_cusps = cusps_md | cusps_ad | cusps_pd

        pcf = {}

        # PRIMARY cusps: 7 features each
        for cusp_num in cfg.PRIMARY_HOUSES:
            cusp_key = f"Cusp_{cusp_num}"
            cdata = month_data.get(cusp_key, {})
            prev_cdata = prev_month_data.get(cusp_key, {})

            # Dasha active: any dasha lord signifies this cusp
            dasha_active = cusp_num in dasha_cusps

            # Dasha lock: sub-lord of this cusp IS a dasha lord
            lock_level = 0
            if cdata.get("Dasha_Lock"):
                ll = cdata.get("Dasha_Lock_Level", "")
                if ll == "MD":
                    lock_level = 3
                elif ll == "AD":
                    lock_level = 2
                elif ll == "PD":
                    lock_level = 1

            # Transit: any Macro_Hit on this cusp this month
            hits = cdata.get("Macro_Hits", [])
            transit_active = len(hits) > 0

            # Transit ingress: active now but NOT in previous month
            prev_hits = prev_cdata.get("Macro_Hits", [])
            transit_ingress = transit_active and len(prev_hits) == 0

            # Transit orb inverse: 1/(min_orb + 0.1), peaks when exact
            if hits:
                min_orb = min(h.get("orb", 5.0) for h in hits)
                transit_orb_inv = 1.0 / (min_orb + 0.1)
            else:
                transit_orb_inv = 0.0

            # Double active: both dasha AND transit
            double_active = dasha_active and transit_active

            pcf[f"cusp{cusp_num}_dasha_active"] = float(dasha_active)
            pcf[f"cusp{cusp_num}_lock_level"] = float(lock_level)
            pcf[f"cusp{cusp_num}_has_lock"] = float(lock_level > 0)
            pcf[f"cusp{cusp_num}_transit_active"] = float(transit_active)
            pcf[f"cusp{cusp_num}_transit_ingress"] = float(transit_ingress)
            pcf[f"cusp{cusp_num}_transit_orb_inv"] = transit_orb_inv
            pcf[f"cusp{cusp_num}_double_active"] = float(double_active)

        # SECONDARY cusps: 2 features each
        for cusp_num in cfg.SECONDARY_HOUSES:
            cusp_key = f"Cusp_{cusp_num}"
            cdata = month_data.get(cusp_key, {})

            hits = cdata.get("Macro_Hits", [])
            transit_active = len(hits) > 0
            dasha_active = cusp_num in dasha_cusps
            active = dasha_active or transit_active

            if hits:
                min_orb = min(h.get("orb", 5.0) for h in hits)
                transit_orb_inv = 1.0 / (min_orb + 0.1)
            else:
                transit_orb_inv = 0.0

            pcf[f"cusp{cusp_num}_active"] = float(active)
            pcf[f"cusp{cusp_num}_transit_orb_inv"] = transit_orb_inv

        # Store per-cusp features in window
        if "per_cusp_features" not in w:
            w["per_cusp_features"] = {}
        w["per_cusp_features"].update(pcf)

    return windows


# ── Within-chart rank normalization ───────────────────────────────────────

def _get_v3_feature(w, feat_name):
    """Get a feature value from any of the V3 feature dicts."""
    for key in ("features", "advanced_features", "orb_recon_features", "temporal_features"):
        d = w.get(key, {})
        if feat_name in d:
            return float(d[feat_name])
    return 0.0


def rank_normalize_chart(windows):
    """Rank-normalize selected continuous features within a chart's windows.

    Converts absolute values to [0,1] ranks so the model learns
    'best month in THIS chart' regardless of absolute scale.
    """
    n = len(windows)
    if n <= 1:
        for w in windows:
            for feat in FEATURES_TO_RANK:
                w.setdefault("rank_features", {})[f"rank_{feat}"] = 0.5
        return windows

    for feat in FEATURES_TO_RANK:
        values = np.array([_get_v3_feature(w, feat) for w in windows], dtype=np.float64)
        if np.std(values) < 1e-10:
            ranks = np.full(n, 0.5)
        else:
            ranks = rankdata(values, method='average') / n
        for i in range(n):
            windows[i].setdefault("rank_features", {})[f"rank_{feat}"] = float(ranks[i])

    return windows


# ── V4 vector assembly ────────────────────────────────────────────────────

def assemble_v4_vector(w):
    """Assemble the final V4 feature vector from a single window dict.

    Expects per_cusp_features, rank_features, and V3 feature dicts
    to already be populated on the window.
    """
    vec = np.zeros(N_V4_FEATURES, dtype=np.float32)

    idx = 0
    # Group A: Per-cusp (25)
    pcf = w.get("per_cusp_features", {})
    for name in _PER_CUSP_NAMES:
        vec[idx] = pcf.get(name, 0.0)
        idx += 1

    # Group B: Rank-normalized (15)
    rf = w.get("rank_features", {})
    for name in _RANK_NAMES:
        vec[idx] = rf.get(name, 0.5)
        idx += 1

    # Group C: Binary/categorical (15)
    for name in _BINARY_NAMES:
        vec[idx] = _get_v3_feature(w, name)
        idx += 1

    return vec


def extract_features_v4(windows, payload):
    """Full V4 feature extraction on pre-computed V3 windows.

    Args:
        windows: list of window dicts (already have V3 features computed)
        payload: the raw compute() payload (needed for per-cusp features)

    Returns:
        windows with v4_vector added to each window dict
    """
    # Step 1: Per-cusp features
    windows = compute_per_cusp_features(windows, payload)

    # Step 2: Rank normalize within chart
    windows = rank_normalize_chart(windows)

    # Step 3: Assemble V4 vectors
    for w in windows:
        w["v4_vector"] = assemble_v4_vector(w)

    return windows
