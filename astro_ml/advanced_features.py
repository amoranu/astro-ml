"""Advanced death-specific features — ~53 additional features per window.

Call add_advanced_death_features(windows, payload) AFTER extract_monthly_windows().
Adds features in-place to each window dict.

Feature groups:
  - Maraka features: ~10
  - Saturn features: ~6
  - Rahu-Ketu features: ~5
  - Interaction features: ~8
  - Temporal features: ~4
  - Father house features: ~7
  - Transit pattern features: ~4
  - Composite features: ~5
  Total: ~49 named + extras = ~53
"""
import math
import numpy as np

from astro_ml.config import domain_fathers_death as cfg
from astro_ml.features import (
    _encode_func_nature, _encode_dignity, get_planet_cusp_significations, N_FEATURES,
)

# Feature names for the advanced block
ADVANCED_FEATURE_NAMES = []


def _build_advanced_names():
    names = []
    # Maraka (10)
    names.extend([
        "maraka10_lord_active", "maraka10_lord_is_md", "maraka10_lord_is_ad",
        "maraka10_lord_is_pd", "maraka3_lord_active",
        "both_marakas_active", "maraka10_lord_dignity", "maraka10_lord_strong",
        "maraka10_sublord_active", "maraka3_sublord_active",
    ])
    # Saturn (6)
    names.extend([
        "saturn_dignity", "saturn_is_malefic", "saturn_is_yogakaraka",
        "saturn_dasha_level", "saturn_transit_density",
        "saturn_is_md",
    ])
    # Rahu-Ketu (5)
    names.extend([
        "rahu_active", "ketu_active", "nodes_active",
        "saturn_rahu_combo", "rahu_sign_lord_malefic",
    ])
    # Interaction (8)
    names.extend([
        "all_lords_malefic", "any_lord_malefic", "mixed_malefic_count",
        "md_ad_same", "ad_pd_same", "all_same",
        "cusp_sig_overlap_md_ad", "triple_activation_count",
    ])
    # Temporal (4)
    names.extend(["window_position", "month_of_year", "month_sin", "month_cos"])
    # Father house (7)
    names.extend([
        "cusp9_sub_is_malefic", "cusp9_sub_dignity", "cusp9_sub_weak",
        "cusp9_sub_active_and_weak",
        "cusp4_sub_is_malefic", "cusp4_sub_active", "cusp4_sub_malefic_and_active",
    ])
    # Transit pattern (4)
    names.extend([
        "transit_density_target", "transit_retro_ratio",
        "n_distinct_transit_planets_adv", "all_heavy_transiting",
    ])
    # Composite (5)
    names.extend([
        "death_signal_composite", "activation_density",
        "quality_weighted_activation", "maraka_score", "negative_event_strength",
    ])
    return names


ADVANCED_FEATURE_NAMES = _build_advanced_names()
N_ADVANCED = len(ADVANCED_FEATURE_NAMES)
N_TOTAL = N_FEATURES + N_ADVANCED


def _get_cusp_sign_lord(payload, cusp_num):
    """Get the lord of the sign on a cusp."""
    kp_cusps = payload.get("KP_Cusps", {})
    cdata = kp_cusps.get(f"Cusp_{cusp_num}", {})
    sign_lord = cdata.get("sign_lord", "")
    if not sign_lord:
        rashi = cdata.get("rashi", "")
        sn = cfg.SIGN_TO_NUM.get(rashi, 0)
        sign_lord = cfg.SIGN_LORDS.get(sn, "")
    return sign_lord


def add_advanced_death_features(windows, payload):
    """Add ~53 advanced death-specific features to each window.

    Modifies windows in-place. Each window gets:
      - "advanced_features": dict of name -> value
      - "advanced_vector": np.array of shape (N_ADVANCED,)
      - "full_vector": np.array of shape (N_TOTAL,) — base + advanced
    """
    fn_map = payload.get("Functional_Nature", {})
    dig_map = payload.get("Planetary_Dignity", {})
    kp_cusps = payload.get("KP_Cusps", {})
    flags_map = payload.get("Natal_Flags", {})
    triggers = payload.get("Calculated_Triggers", {})

    # Pre-compute maraka lords
    maraka10_lord = _get_cusp_sign_lord(payload, 10)
    maraka3_lord = _get_cusp_sign_lord(payload, 3)
    cusp9_sublord = kp_cusps.get("Cusp_9", {}).get("sub_lord", "")
    cusp4_sublord = kp_cusps.get("Cusp_4", {}).get("sub_lord", "")

    # Maraka sub-lords
    maraka10_sublord = kp_cusps.get("Cusp_10", {}).get("sub_lord", "")
    maraka3_sublord = kp_cusps.get("Cusp_3", {}).get("sub_lord", "")

    # Saturn nature
    saturn_fn = fn_map.get("Saturn", "Neutral")
    saturn_fn_score = _encode_func_nature(saturn_fn, payload, "Saturn")
    saturn_dig = dig_map.get("Saturn", "Neutral")

    # Rahu sign lord malefic?
    rahu_data = payload.get("KP_Planets", {}).get("Rahu", {})
    rahu_rashi = rahu_data.get("rashi", "")
    rahu_sn = cfg.SIGN_TO_NUM.get(rahu_rashi, 0)
    rahu_sign_lord = cfg.SIGN_LORDS.get(rahu_sn, "")
    rahu_sl_malefic = 1.0 if rahu_sign_lord in cfg.MALEFICS else 0.0

    n_windows = len(windows)

    for wi, w in enumerate(windows):
        av = np.zeros(N_ADVANCED, dtype=np.float32)
        md, ad, pd = w["md"], w["ad"], w["pd"]
        month_str = w["month"]
        bf = w["features"]  # base features dict
        month_data = triggers.get(month_str, {})

        idx = 0

        # === Maraka features (10) ===
        m10_active = 1.0 if maraka10_lord in (md, ad, pd) else 0.0
        m3_active = 1.0 if maraka3_lord in (md, ad, pd) else 0.0
        av[idx] = m10_active
        av[idx + 1] = 1.0 if maraka10_lord == md else 0.0
        av[idx + 2] = 1.0 if maraka10_lord == ad else 0.0
        av[idx + 3] = 1.0 if maraka10_lord == pd else 0.0
        av[idx + 4] = m3_active
        av[idx + 5] = 1.0 if m10_active and m3_active else 0.0
        m10_dig = _encode_dignity(dig_map.get(maraka10_lord, "Neutral"))
        av[idx + 6] = m10_dig
        av[idx + 7] = 1.0 if m10_dig >= 4 else 0.0  # EXALTED or OWN_SIGN
        av[idx + 8] = 1.0 if maraka10_sublord in (md, ad, pd) else 0.0
        av[idx + 9] = 1.0 if maraka3_sublord in (md, ad, pd) else 0.0
        idx += 10

        # === Saturn features (6) ===
        av[idx] = _encode_dignity(saturn_dig)
        av[idx + 1] = 1.0 if saturn_fn_score < 0 else 0.0
        av[idx + 2] = 1.0 if saturn_fn_score == 3 else 0.0
        sat_level = 0
        if "Saturn" == md: sat_level = 3
        elif "Saturn" == ad: sat_level = 2
        elif "Saturn" == pd: sat_level = 1
        av[idx + 3] = sat_level

        # Saturn transit density
        sat_hits = 0
        for cusp_num in cfg.TARGET_CUSPS:
            cusp_key = f"Cusp_{cusp_num}"
            cdata = month_data.get(cusp_key, {})
            for h in cdata.get("Macro_Hits", []):
                if h.get("planet") == "Saturn":
                    sat_hits += 1
        av[idx + 4] = sat_hits
        av[idx + 5] = 1.0 if "Saturn" == md else 0.0
        idx += 6

        # === Rahu-Ketu features (5) ===
        rahu_active = 1.0 if "Rahu" in (md, ad, pd) else 0.0
        ketu_active = 1.0 if "Ketu" in (md, ad, pd) else 0.0
        av[idx] = rahu_active
        av[idx + 1] = ketu_active
        av[idx + 2] = 1.0 if rahu_active or ketu_active else 0.0
        av[idx + 3] = 1.0 if sat_level > 0 and rahu_active else 0.0
        av[idx + 4] = rahu_sl_malefic
        idx += 5

        # === Interaction features (8) ===
        lords = [md, ad, pd]
        fn_scores = [_encode_func_nature(fn_map.get(l, "Neutral"), payload, l) if l else 0 for l in lords]
        av[idx] = 1.0 if all(s < 0 for s in fn_scores if s is not None) else 0.0
        av[idx + 1] = 1.0 if any(s < 0 for s in fn_scores) else 0.0
        av[idx + 2] = sum(1 for s in fn_scores if s < 0)
        av[idx + 3] = 1.0 if md == ad and md else 0.0
        av[idx + 4] = 1.0 if ad == pd and ad else 0.0
        av[idx + 5] = 1.0 if md == ad == pd and md else 0.0

        # Cusp signification overlap between MD and AD
        md_cusps = get_planet_cusp_significations(md, payload) if md else set()
        ad_cusps = get_planet_cusp_significations(ad, payload) if ad else set()
        target_set = set(cfg.TARGET_CUSPS)
        overlap = md_cusps & ad_cusps & target_set
        av[idx + 6] = len(overlap)

        # Triple activation: cusps where all 3 lords signify same target cusp
        pd_cusps = get_planet_cusp_significations(pd, payload) if pd else set()
        triple = md_cusps & ad_cusps & pd_cusps & target_set
        av[idx + 7] = len(triple)
        idx += 8

        # === Temporal features (4) ===
        av[idx] = 0.0  # window_position DISABLED — causes position bias leakage
        try:
            parts = month_str.split("-")
            month_num = int(parts[1])
        except (IndexError, ValueError):
            month_num = 1
        av[idx + 1] = month_num
        av[idx + 2] = math.sin(2 * math.pi * month_num / 12)
        av[idx + 3] = math.cos(2 * math.pi * month_num / 12)
        idx += 4

        # === Father house features (7) ===
        c9_malefic = 1.0 if cusp9_sublord in cfg.MALEFICS else 0.0
        c9_dig = _encode_dignity(dig_map.get(cusp9_sublord, "Neutral")) if cusp9_sublord else 2
        c9_weak = 1.0 if c9_dig <= 1 else 0.0  # ENEMY or DEBILITATED
        c9_active = 1.0 if cusp9_sublord in (md, ad, pd) else 0.0
        av[idx] = c9_malefic
        av[idx + 1] = c9_dig
        av[idx + 2] = c9_weak
        av[idx + 3] = 1.0 if c9_active and c9_weak else 0.0

        c4_malefic = 1.0 if cusp4_sublord in cfg.MALEFICS else 0.0
        c4_active = 1.0 if cusp4_sublord in (md, ad, pd) else 0.0
        av[idx + 4] = c4_malefic
        av[idx + 5] = c4_active
        av[idx + 6] = 1.0 if c4_malefic and c4_active else 0.0
        idx += 7

        # === Transit pattern features (4) ===
        total_hits = 0
        retro_hits = 0
        transit_planets_seen = set()
        for cusp_num in cfg.TARGET_CUSPS:
            cusp_key = f"Cusp_{cusp_num}"
            cdata = month_data.get(cusp_key, {})
            for h in cdata.get("Macro_Hits", []):
                total_hits += 1
                transit_planets_seen.add(h.get("planet", ""))
                # Retrograde info would need transit data — approximate
        av[idx] = total_hits
        av[idx + 1] = 0.0  # transit_retro_ratio — needs transit planet data
        av[idx + 2] = len(transit_planets_seen)
        av[idx + 3] = 1.0 if len(transit_planets_seen) >= 3 else 0.0
        idx += 4

        # === Composite features (5) ===
        death_karaka_count = sum(1 for l in lords if l in cfg.DEATH_KARAKAS)
        has_lock = bf.get("has_primary_dasha_lock", 0)
        has_double = bf.get("has_double_activation_primary", 0)
        n_neg = bf.get("n_negative_cusps_dasha", 0)
        n_pri = bf.get("n_primary_cusps_active", 0)
        n_sec = bf.get("n_secondary_cusps_active", 0)

        # death_signal_composite
        composite = (death_karaka_count * 2 + has_lock * 3 + sat_level * 1.5
                     + av[5] * 2  # both_marakas_active
                     + has_double * 2.5 - n_neg * 1)
        av[idx] = composite

        # activation_density
        density = n_pri + n_sec * 0.5 + total_hits * 0.3 + len(triple) * 2
        av[idx + 1] = density

        # quality_weighted_activation
        quality = bf.get("dasha_quality_score", 0)
        net = n_pri * 2 + n_sec - n_neg
        av[idx + 2] = net * (1 + quality / 10.0)

        # maraka_score
        m_score = (m10_active * 2 + m3_active * 1.5
                   + av[8] * 3  # maraka10_sublord_active
                   + av[9] * 2  # maraka3_sublord_active
                   + av[7] * 1)  # maraka10_lord_strong
        av[idx + 3] = m_score

        # negative_event_strength
        neg_strength = (av[idx - 4 - 7 - 4 - 8 + 2]  # mixed_malefic_count (interaction[2])
                        + sat_level * 1.5
                        + (rahu_active + ketu_active)
                        + av[idx - 4 - 7 + 6] * 2)  # cusp4_malefic_and_active
        av[idx + 4] = neg_strength
        idx += 5

        # Store in window
        w["advanced_features"] = {ADVANCED_FEATURE_NAMES[i]: float(av[i]) for i in range(N_ADVANCED)}
        w["advanced_vector"] = av
        w["full_vector"] = np.concatenate([w["feature_vector"], av])

    return windows
