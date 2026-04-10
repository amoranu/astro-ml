"""V5 Feature Engineering — V4 base + rule components + relative features.

V5 adapts the two-stage plan: instead of hard shortlisting (which fails
because 84% of windows are B+ tier), the rule scorer provides FEATURES
to the ML model, and relative features compare each window to others
in the same chart.

Feature groups:
  A. Per-cusp features (25) — from V4
  B. Rank-normalized (15) — from V4, within-chart relative ranking
  C. Rule components (12) — NEW: domain logic from rule_scorer
  D. Relative/comparative (12) — NEW: margin-from-best, unique flags
  E. Binary/categorical (10) — subset of V4

Total: ~74 features
"""
import numpy as np
from scipy.stats import rankdata

from astro_ml.config import domain_fathers_death as cfg
from astro_ml.features_v4 import (
    _PER_CUSP_NAMES, FEATURES_TO_RANK, _RANK_NAMES,
    compute_per_cusp_features, _get_v3_feature,
)


# ── V5 Feature Names ─────────────────────────────────────────────────────

# Group C: Rule components (12)
_RULE_NAMES = [
    "rule_tier_score",
    "rule_score_normalized",      # rule_score / max(rule_scores_in_chart)
    "rule_n_primary_active",
    "rule_n_primary_locks",
    "rule_n_primary_double",
    "rule_n_secondary_active",
    "rule_n_negative_active",
    "rule_total_quality",
    "rule_yogakaraka_count",
    "rule_death_karaka_active",
    "rule_sun_active",
    "rule_cusp4_lock_bonus",      # 1 if cusp 4 has lock, else 0
]

# Group D: Relative features (12)
# For key discriminating features: margin from chart-best, is_unique_best
_RELATIVE_SOURCE = [
    "convergence_proxy",
    "trigger_score",
    "net_transit_signal",
    "transit_freshness",
]
_RELATIVE_NAMES = []
for _f in _RELATIVE_SOURCE:
    _RELATIVE_NAMES.extend([
        f"margin_from_best_{_f}",    # this_value - best_value (0 if this IS best)
        f"is_unique_best_{_f}",      # 1 if this window is the sole max
    ])
_RELATIVE_NAMES.extend([
    "n_unique_best_flags",           # count of is_unique_best across all sources
    "rule_score_margin_from_best",   # rule_score gap from top
    "rank_composite_mean",           # mean of all rank features (composite signal)
    "rank_composite_max_count",      # how many rank features are >= 0.9
])

# Group E: Binary/categorical (10) — trimmed from V4's 15
_BINARY_NAMES_V5 = [
    "is_pd_transition",
    "is_ad_transition",
    "is_new_pd",
    "any_primary_ingress",
    "all_three_heavy_active",
    "saturn_is_md",
    "saturn_dasha_level",
    "both_marakas_active",
    "dasha_transition_count",
    "deterministic_tier",
]

V5_FEATURE_NAMES = _PER_CUSP_NAMES + _RANK_NAMES + _RULE_NAMES + _RELATIVE_NAMES + _BINARY_NAMES_V5
N_V5_FEATURES = len(V5_FEATURE_NAMES)


# ── Rule component extraction ────────────────────────────────────────────

def compute_rule_features(windows, scored_windows):
    """Add rule-derived features to each window.

    Args:
        windows: list of V3-enriched window dicts
        scored_windows: output of rule_scorer.score_chart() for same chart
    """
    # Index scored windows by month
    scored_by_month = {sw["month_str"]: sw for sw in scored_windows}

    # Max rule_score in chart for normalization
    max_rule_score = max((sw["rule_score"] for sw in scored_windows), default=1.0)
    if max_rule_score <= 0:
        max_rule_score = 1.0

    for w in windows:
        month = w["month"]
        sw = scored_by_month.get(month)

        rf = {}
        if sw:
            comp = sw["components"]
            rf["rule_tier_score"] = float(sw["tier_score"])
            rf["rule_score_normalized"] = sw["rule_score"] / max_rule_score
            rf["rule_n_primary_active"] = float(len(comp["primary_active"]))
            rf["rule_n_primary_locks"] = float(len(comp["primary_locks"]))
            rf["rule_n_primary_double"] = float(len(comp["primary_double"]))
            rf["rule_n_secondary_active"] = float(len(comp["secondary_active"]))
            rf["rule_n_negative_active"] = float(len(comp["negative_active"]))
            rf["rule_total_quality"] = comp["total_quality"]
            rf["rule_yogakaraka_count"] = float(comp["yogakaraka_count"])
            rf["rule_death_karaka_active"] = float(comp["death_karaka_active"])
            rf["rule_sun_active"] = 1.0 if comp["sun_active"] else 0.0
            rf["rule_cusp4_lock_bonus"] = 1.0 if any(
                c == 4 for c, _, _ in comp["primary_locks"]) else 0.0
        else:
            # Window not in scored (shouldn't happen, but fallback)
            for name in _RULE_NAMES:
                rf[name] = 0.0

        w["rule_features"] = rf

    return windows


# ── Relative features ─────────────────────────────────────────────────────

def compute_relative_features(windows, scored_windows):
    """Add relative/comparative features to each window.

    Compares each window to others in the chart on key dimensions.
    """
    n = len(windows)
    if n <= 1:
        for w in windows:
            w["relative_features"] = {name: 0.0 for name in _RELATIVE_NAMES}
        return windows

    # Collect source arrays
    source_arrays = {}
    for feat in _RELATIVE_SOURCE:
        source_arrays[feat] = np.array(
            [_get_v3_feature(w, feat) for w in windows], dtype=np.float64
        )

    # Rule scores
    scored_by_month = {sw["month_str"]: sw for sw in scored_windows}
    rule_scores = np.array(
        [scored_by_month.get(w["month"], {}).get("rule_score", 0.0) for w in windows],
        dtype=np.float64
    )

    # Rank features for composite
    rank_features_per_window = []
    for w in windows:
        ranks = [w.get("rank_features", {}).get(f"rank_{f}", 0.5) for f in FEATURES_TO_RANK]
        rank_features_per_window.append(ranks)

    for i, w in enumerate(windows):
        rel = {}

        n_unique_best = 0
        for feat in _RELATIVE_SOURCE:
            arr = source_arrays[feat]
            val = arr[i]
            best_val = arr.max()
            n_at_best = int((arr == best_val).sum())

            # Margin from best (0 if this IS the best, negative otherwise)
            rel[f"margin_from_best_{feat}"] = val - best_val

            # Is unique best
            is_ub = 1.0 if (val == best_val and n_at_best == 1) else 0.0
            rel[f"is_unique_best_{feat}"] = is_ub
            n_unique_best += int(is_ub)

        rel["n_unique_best_flags"] = float(n_unique_best)

        # Rule score margin
        best_rule = rule_scores.max()
        rel["rule_score_margin_from_best"] = rule_scores[i] - best_rule

        # Rank composite
        ranks = rank_features_per_window[i]
        rel["rank_composite_mean"] = float(np.mean(ranks))
        rel["rank_composite_max_count"] = float(sum(1 for r in ranks if r >= 0.9))

        w["relative_features"] = rel

    return windows


# ── V5 vector assembly ────────────────────────────────────────────────────

def assemble_v5_vector(w):
    """Assemble the V5 feature vector from a window dict."""
    vec = np.zeros(N_V5_FEATURES, dtype=np.float32)
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

    # Group C: Rule components (12)
    rule_f = w.get("rule_features", {})
    for name in _RULE_NAMES:
        vec[idx] = rule_f.get(name, 0.0)
        idx += 1

    # Group D: Relative features (12)
    rel_f = w.get("relative_features", {})
    for name in _RELATIVE_NAMES:
        vec[idx] = rel_f.get(name, 0.0)
        idx += 1

    # Group E: Binary/categorical (10)
    for name in _BINARY_NAMES_V5:
        vec[idx] = _get_v3_feature(w, name)
        idx += 1

    return vec
