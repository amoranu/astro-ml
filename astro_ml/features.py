"""Base feature engineering — ~100 features per monthly window.

Input: raw payload from compute().
Output: list of window dicts, each with feature vector + metadata.

Feature groups:
  - Per dasha lord (MD, AD, PD): 16 features each = 48 total
  - Dasha quality: 7
  - Domain cusp activation: ~20
  - Transit: 6
  - Karaka: 7
  - Main cusp sub-lord: 7
  - Sun (pitrukaraka): 3
  - 9th house: 3
  - Deterministic tier: 1
  Total: ~102 features
"""
import datetime
import numpy as np

from astro_ml.config import domain_fathers_death as cfg


# ── Encoding helpers ─────────────────────────────────────────────────────

def _encode_func_nature(fn_str, payload=None, planet=None):
    """Encode functional nature string to numeric. Handle shadow planets."""
    val = cfg.FUNC_NATURE_ENCODING.get(fn_str)
    if val is not None:
        return val
    # Shadow planet — use sign lord's nature
    if planet and planet in cfg.SHADOW_PLANETS and payload:
        pdata = payload.get("KP_Planets", {}).get(planet, {})
        rashi = pdata.get("rashi", "")
        sn = cfg.SIGN_TO_NUM.get(rashi, 0)
        sign_lord = cfg.SIGN_LORDS.get(sn, "")
        if sign_lord:
            sl_fn = payload.get("Functional_Nature", {}).get(sign_lord, "Neutral")
            return cfg.FUNC_NATURE_ENCODING.get(sl_fn, 0)
    return 0


def _encode_dignity(dig_str):
    """Encode dignity string to ordinal 0-5."""
    return cfg.DIGNITY_ENCODING.get(dig_str, 2)  # default Neutral


def _encode_dignity_factor(dig_str):
    """Get dignity multiplier factor."""
    return cfg.DIGNITY_FACTOR.get(dig_str, 1.0)


def _encode_nav_dignity(nav_dig_str):
    """Encode navamsha dignity string to ordinal."""
    if not nav_dig_str:
        return 1
    s = nav_dig_str.upper()
    # Try exact match first
    val = cfg.NAV_DIGNITY_ENCODING.get(s)
    if val is not None:
        return val
    # Try partial matches
    if "VARGOTTAMA" in s and "EXALTED" in s:
        return 7
    if "VARGOTTAMA" in s and "OWN" in s:
        return 6
    if "VARGOTTAMA" in s:
        return 4
    if "EXALTED" in s:
        return 5
    if "OWN" in s:
        return 3
    if "FRIENDLY" in s or "FRIEND" in s:
        return 2
    if "DEBILITATED" in s:
        return 0
    return 1


# ── Helper functions ─────────────────────────────────────────────────────

def get_planet_cusp_significations(planet, payload):
    """Return set of cusp numbers the planet signifies."""
    sigs = payload.get("Planet_Significations", {}).get(planet, [])
    if isinstance(sigs, (list, tuple)):
        return set(int(x) for x in sigs)
    return set()


def get_dasha_activated_cusps(md, ad, pd, payload):
    """Union of all cusps activated by the 3 dasha lords."""
    activated = set()
    for lord in [md, ad, pd]:
        if lord:
            activated |= get_planet_cusp_significations(lord, payload)
    return activated


def get_dasha_locks(md, ad, pd, payload, target_cusps):
    """Find cusps where sub-lord = one of the dasha lords.

    Returns list of (cusp_num, lord, level) tuples.
    """
    locks = []
    kp_cusps = payload.get("KP_Cusps", {})
    for cusp_num in target_cusps:
        cdata = kp_cusps.get(f"Cusp_{cusp_num}", {})
        sublord = cdata.get("sub_lord", "")
        if sublord == md and md:
            locks.append((cusp_num, md, "MD"))
        elif sublord == ad and ad:
            locks.append((cusp_num, ad, "AD"))
        elif sublord == pd and pd:
            locks.append((cusp_num, pd, "PD"))
    return locks


def get_transit_activations_for_month(month_str, payload, target_cusps):
    """Return set of cusps activated by transit this month."""
    triggers = payload.get("Calculated_Triggers", {}).get(month_str, {})
    activated = set()
    for cusp_num in target_cusps:
        cusp_key = f"Cusp_{cusp_num}"
        cdata = triggers.get(cusp_key, {})
        if cdata.get("Macro_Hits"):
            activated.add(cusp_num)
    return activated


def compute_dasha_quality_score(md, ad, pd, payload):
    """Compute dasha quality score and per-lord breakdown.

    Returns (total_score, md_score, ad_score, pd_score).
    """
    fn_map = payload.get("Functional_Nature", {})
    dig_map = payload.get("Planetary_Dignity", {})

    scores = []
    for lord in [md, ad, pd]:
        if not lord:
            scores.append(0.0)
            continue
        fn = fn_map.get(lord, "Neutral")
        base = _encode_func_nature(fn, payload, lord)
        dig = dig_map.get(lord, "Neutral")
        factor = _encode_dignity_factor(dig)
        scores.append(base * factor)

    return sum(scores), scores[0], scores[1], scores[2]


def _get_badhaka_house(lagna_sign_num):
    """Compute badhaka house based on 9th cusp sign mobility."""
    mob = cfg.SIGN_MOBILITY.get(lagna_sign_num, "dual")
    return cfg.BADHAKA_MAP.get(mob, 7)


# ── Deterministic tier scoring ───────────────────────────────────────────

def compute_deterministic_tier(feats):
    """Compute convergence tier (S/A/B/C/D/F) from feature dict.

    Returns tier score 0-5.
    """
    n_primary = feats.get("n_primary_cusps_active", 0)
    n_secondary = feats.get("n_secondary_cusps_active", 0)
    has_lock = feats.get("has_primary_dasha_lock", 0)
    has_yogakaraka = feats.get("yogakaraka_count", 0) > 0
    has_double = feats.get("has_double_activation_primary", 0)
    quality = feats.get("dasha_quality_score", 0)
    n_negative = feats.get("n_negative_cusps_dasha", 0)

    # S-TIER: all 3 primary + lock + transit + quality >= 6.0
    if n_primary >= 3 and has_lock and has_double and quality >= 6.0:
        return 5
    # A-TIER: lock on primary OR 2+ primary w/ yogakaraka OR double activation; quality >= 3.0
    if quality >= 3.0 and (has_lock or (n_primary >= 2 and has_yogakaraka) or has_double):
        return 4
    # B-TIER: 2+ primary OR 1 primary + yogakaraka
    if n_primary >= 2 or (n_primary >= 1 and has_yogakaraka):
        return 3
    # C-TIER: 1 primary, no lock, no yogakaraka
    if n_primary >= 1:
        return 2
    # D-TIER: only secondary
    if n_secondary >= 1:
        return 1
    # F-TIER
    return 0


# ── Main extraction ──────────────────────────────────────────────────────

# Feature name list for consistent ordering
FEATURE_NAMES = []


def _build_feature_names():
    """Build the ordered feature name list."""
    names = []
    # Per-lord features (48)
    for prefix in ["md", "ad", "pd"]:
        names.extend([
            f"{prefix}_func_nature_score",
            f"{prefix}_is_yogakaraka",
            f"{prefix}_is_benefic",
            f"{prefix}_is_malefic",
            f"{prefix}_is_mixed",
            f"{prefix}_dignity",
            f"{prefix}_is_exalted",
            f"{prefix}_is_debilitated",
            f"{prefix}_is_own_sign",
            f"{prefix}_is_enemy",
            f"{prefix}_is_retrograde",
            f"{prefix}_is_combust",
            f"{prefix}_nav_dignity",
            f"{prefix}_is_vargottama",
            f"{prefix}_is_natural_karaka",
            f"{prefix}_is_shadow",
        ])
    # Dasha quality (7)
    names.extend([
        "dasha_quality_score",
        "md_weighted_score", "ad_weighted_score", "pd_weighted_score",
        "yogakaraka_count", "malefic_count", "benefic_count",
    ])
    # Domain cusp activation (~20)
    names.extend([
        "n_primary_cusps_dasha", "n_secondary_cusps_dasha", "n_negative_cusps_dasha",
        "n_primary_cusps_transit", "n_secondary_cusps_transit", "n_negative_cusps_transit",
        "n_primary_cusps_active", "n_secondary_cusps_active", "n_negative_cusps_active",
        "n_dasha_locks_primary", "n_dasha_locks_total",
        "has_primary_dasha_lock", "primary_lock_is_yogakaraka", "primary_lock_is_karaka",
        "n_double_activations_primary", "has_double_activation_primary",
        "net_activation_score",
    ])
    # Transit (6)
    names.extend([
        "n_total_transits",
        "jupiter_transit_active", "saturn_transit_active", "rahu_transit_active",
        "n_moon_micro_hits_primary",
        "n_distinct_transit_planets",
    ])
    # Karaka (7)
    names.extend([
        "karaka_at_md", "karaka_at_ad", "karaka_at_pd",
        "karaka_active_count",
        "death_karaka_at_md", "death_karaka_at_ad", "death_karaka_at_pd",
    ])
    # Main cusp sub-lord (7)
    names.extend([
        "main_cusp_sublord_dignity", "main_cusp_sublord_nav_dignity",
        "main_cusp_sublord_vargottama", "main_cusp_sublord_retrograde",
        "main_cusp_sublord_combust",
        "main_cusp_sublord_is_malefic", "main_cusp_sublord_is_karaka",
    ])
    # Sun pitrukaraka (3)
    names.extend(["sun_dignity", "sun_is_malefic", "sun_at_any_level"])
    # 9th house (3)
    names.extend(["cusp9_sublord_active", "cusp9_in_dasha_activated", "cusp9_in_transit_activated"])
    # Deterministic tier (1)
    names.append("deterministic_tier")
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


def _extract_lord_features(lord, payload):
    """Extract 16 features for one dasha lord."""
    feats = np.zeros(16, dtype=np.float32)
    if not lord:
        return feats

    fn_map = payload.get("Functional_Nature", {})
    dig_map = payload.get("Planetary_Dignity", {})
    nav_map = payload.get("Navamsha_Positions", {})
    flags_map = payload.get("Natal_Flags", {})

    fn = fn_map.get(lord, "Neutral")
    fn_score = _encode_func_nature(fn, payload, lord)
    dig = dig_map.get(lord, "Neutral")
    nav_data = nav_map.get(lord, {})
    nav_dig_str = nav_data.get("nav_dignity_string", "NEUTRAL")
    flags = flags_map.get(lord, {})

    feats[0] = fn_score
    feats[1] = 1.0 if fn_score == 3 else 0.0  # is_yogakaraka
    feats[2] = 1.0 if fn_score >= 2 else 0.0   # is_benefic (YOGAKARAKA or BENEFIC)
    feats[3] = 1.0 if fn_score < 0 else 0.0    # is_malefic
    feats[4] = 1.0 if fn_score == 1 else 0.0   # is_mixed
    feats[5] = _encode_dignity(dig)
    feats[6] = 1.0 if dig in ("Exalted", "EXALTED") else 0.0
    feats[7] = 1.0 if dig in ("Debilitated", "DEBILITATED") else 0.0
    feats[8] = 1.0 if dig in ("Own", "OWN_SIGN", "Own Sign") else 0.0
    feats[9] = 1.0 if dig in ("Enemy", "ENEMY") else 0.0
    feats[10] = 1.0 if flags.get("is_retrograde", False) else 0.0
    feats[11] = 1.0 if flags.get("is_combust", False) else 0.0
    feats[12] = _encode_nav_dignity(nav_dig_str)
    feats[13] = 1.0 if nav_data.get("is_vargottama", False) else 0.0
    feats[14] = 1.0 if lord in cfg.ALL_KARAKAS else 0.0
    feats[15] = 1.0 if lord in cfg.SHADOW_PLANETS else 0.0

    return feats


def extract_monthly_windows(payload):
    """Extract feature vectors for every month in the payload's trigger data.

    Returns list of dicts, each with:
      - "month": "YYYY-MM"
      - "features": dict mapping feature_name -> value
      - "feature_vector": np.array of shape (N_FEATURES,)
      - "md", "ad", "pd": dasha lord names
    """
    triggers = payload.get("Calculated_Triggers", {})
    fn_map = payload.get("Functional_Nature", {})
    dig_map = payload.get("Planetary_Dignity", {})
    kp_cusps = payload.get("KP_Cusps", {})

    # Sort months chronologically
    months = sorted(k for k in triggers.keys() if not k.startswith("_"))
    if not months:
        return []

    windows = []
    for month_str in months:
        month_data = triggers.get(month_str, {})
        dasha_state = month_data.get("_dasha", {})
        md = dasha_state.get("md", "")
        ad = dasha_state.get("ad", "")
        pd = dasha_state.get("pd", "")

        fv = np.zeros(N_FEATURES, dtype=np.float32)
        feats = {}  # name -> value for tier computation

        # --- Per-lord features (48) ---
        for i, (prefix, lord) in enumerate([("md", md), ("ad", ad), ("pd", pd)]):
            lord_feats = _extract_lord_features(lord, payload)
            fv[i * 16:(i + 1) * 16] = lord_feats

        # --- Dasha quality (7) ---
        total_q, md_q, ad_q, pd_q = compute_dasha_quality_score(md, ad, pd, payload)
        idx = 48
        fv[idx] = total_q; feats["dasha_quality_score"] = total_q
        fv[idx + 1] = md_q
        fv[idx + 2] = ad_q
        fv[idx + 3] = pd_q

        yogakaraka_count = sum(1 for lord in [md, ad, pd]
                               if lord and _encode_func_nature(fn_map.get(lord, "Neutral"), payload, lord) == 3)
        malefic_count = sum(1 for lord in [md, ad, pd]
                            if lord and _encode_func_nature(fn_map.get(lord, "Neutral"), payload, lord) < 0)
        benefic_count = sum(1 for lord in [md, ad, pd]
                            if lord and _encode_func_nature(fn_map.get(lord, "Neutral"), payload, lord) >= 2)
        fv[idx + 4] = yogakaraka_count; feats["yogakaraka_count"] = yogakaraka_count
        fv[idx + 5] = malefic_count
        fv[idx + 6] = benefic_count
        idx += 7

        # --- Domain cusp activation (~17) ---
        dasha_cusps = get_dasha_activated_cusps(md, ad, pd, payload)
        transit_cusps = get_transit_activations_for_month(month_str, payload, list(range(1, 13)))
        all_cusps = dasha_cusps | transit_cusps

        primary_set = set(cfg.PRIMARY_HOUSES)
        secondary_set = set(cfg.SECONDARY_HOUSES)
        negative_set = set(cfg.NEGATIVE_HOUSES)

        n_pri_d = len(dasha_cusps & primary_set)
        n_sec_d = len(dasha_cusps & secondary_set)
        n_neg_d = len(dasha_cusps & negative_set)
        n_pri_t = len(transit_cusps & primary_set)
        n_sec_t = len(transit_cusps & secondary_set)
        n_neg_t = len(transit_cusps & negative_set)
        n_pri_a = len(all_cusps & primary_set)
        n_sec_a = len(all_cusps & secondary_set)
        n_neg_a = len(all_cusps & negative_set)

        fv[idx] = n_pri_d; fv[idx + 1] = n_sec_d; fv[idx + 2] = n_neg_d
        fv[idx + 3] = n_pri_t; fv[idx + 4] = n_sec_t; fv[idx + 5] = n_neg_t
        fv[idx + 6] = n_pri_a; fv[idx + 7] = n_sec_a; fv[idx + 8] = n_neg_a
        feats["n_primary_cusps_active"] = n_pri_a
        feats["n_secondary_cusps_active"] = n_sec_a
        feats["n_negative_cusps_dasha"] = n_neg_d

        # Dasha locks
        locks_primary = get_dasha_locks(md, ad, pd, payload, cfg.PRIMARY_HOUSES)
        locks_all = get_dasha_locks(md, ad, pd, payload, cfg.TARGET_CUSPS)
        n_locks_primary = len(locks_primary)
        n_locks_total = len(locks_all)
        has_primary_lock = 1.0 if n_locks_primary > 0 else 0.0

        # Check if primary lock lord is yogakaraka or karaka
        primary_lock_yogakaraka = 0.0
        primary_lock_karaka = 0.0
        for (cusp_num, lord, level) in locks_primary:
            fn = fn_map.get(lord, "Neutral")
            if _encode_func_nature(fn, payload, lord) == 3:
                primary_lock_yogakaraka = 1.0
            if lord in cfg.ALL_KARAKAS:
                primary_lock_karaka = 1.0

        fv[idx + 9] = n_locks_primary; fv[idx + 10] = n_locks_total
        fv[idx + 11] = has_primary_lock; feats["has_primary_dasha_lock"] = has_primary_lock
        fv[idx + 12] = primary_lock_yogakaraka
        fv[idx + 13] = primary_lock_karaka

        # Double activations
        double_primary = len((dasha_cusps & primary_set) & (transit_cusps & primary_set))
        fv[idx + 14] = double_primary
        fv[idx + 15] = 1.0 if double_primary > 0 else 0.0
        feats["has_double_activation_primary"] = fv[idx + 15]

        # Net activation score
        net = n_pri_a * 2 + n_sec_a * 1 - n_neg_a * 1
        fv[idx + 16] = net
        idx += 17

        # --- Transit features (6) ---
        # Count total macro hits across all cusps this month
        all_hits = []
        transit_planets_active = set()
        for cusp_num in cfg.TARGET_CUSPS:
            cusp_key = f"Cusp_{cusp_num}"
            cdata = month_data.get(cusp_key, {})
            hits = cdata.get("Macro_Hits", [])
            all_hits.extend(hits)
            for h in hits:
                transit_planets_active.add(h.get("planet", ""))

        fv[idx] = len(all_hits)
        fv[idx + 1] = 1.0 if "Jupiter" in transit_planets_active else 0.0
        fv[idx + 2] = 1.0 if "Saturn" in transit_planets_active else 0.0
        fv[idx + 3] = 1.0 if "Rahu" in transit_planets_active else 0.0
        fv[idx + 4] = 0.0  # Moon micro hits — not yet computed in triggers
        fv[idx + 5] = len(transit_planets_active)
        idx += 6

        # --- Karaka features (7) ---
        fv[idx] = 1.0 if md in cfg.NATURAL_KARAKAS else 0.0
        fv[idx + 1] = 1.0 if ad in cfg.NATURAL_KARAKAS else 0.0
        fv[idx + 2] = 1.0 if pd in cfg.NATURAL_KARAKAS else 0.0
        fv[idx + 3] = sum(1 for l in [md, ad, pd] if l in cfg.NATURAL_KARAKAS)
        fv[idx + 4] = 1.0 if md in cfg.DEATH_KARAKAS else 0.0
        fv[idx + 5] = 1.0 if ad in cfg.DEATH_KARAKAS else 0.0
        fv[idx + 6] = 1.0 if pd in cfg.DEATH_KARAKAS else 0.0
        idx += 7

        # --- Main cusp sub-lord features (7) ---
        main_cusp_key = f"Cusp_{cfg.MAIN_CUSP}"
        main_sublord = kp_cusps.get(main_cusp_key, {}).get("sub_lord", "")
        if main_sublord:
            main_dig = dig_map.get(main_sublord, "Neutral")
            nav_data = payload.get("Navamsha_Positions", {}).get(main_sublord, {})
            flags = payload.get("Natal_Flags", {}).get(main_sublord, {})
            fv[idx] = _encode_dignity(main_dig)
            fv[idx + 1] = _encode_nav_dignity(nav_data.get("nav_dignity_string", ""))
            fv[idx + 2] = 1.0 if nav_data.get("is_vargottama", False) else 0.0
            fv[idx + 3] = 1.0 if flags.get("is_retrograde", False) else 0.0
            fv[idx + 4] = 1.0 if flags.get("is_combust", False) else 0.0
            fv[idx + 5] = 1.0 if main_sublord in cfg.MALEFICS else 0.0
            fv[idx + 6] = 1.0 if main_sublord in cfg.ALL_KARAKAS else 0.0
        idx += 7

        # --- Sun pitrukaraka features (3) ---
        sun_dig = dig_map.get("Sun", "Neutral")
        sun_fn = fn_map.get("Sun", "Neutral")
        fv[idx] = _encode_dignity(sun_dig)
        fv[idx + 1] = 1.0 if _encode_func_nature(sun_fn, payload, "Sun") < 0 else 0.0
        fv[idx + 2] = 1.0 if "Sun" in (md, ad, pd) else 0.0
        idx += 3

        # --- 9th house features (3) ---
        cusp9_sublord = kp_cusps.get("Cusp_9", {}).get("sub_lord", "")
        fv[idx] = 1.0 if cusp9_sublord in (md, ad, pd) else 0.0
        fv[idx + 1] = 1.0 if 9 in dasha_cusps else 0.0
        fv[idx + 2] = 1.0 if 9 in transit_cusps else 0.0
        idx += 3

        # --- Deterministic tier (1) ---
        tier = compute_deterministic_tier(feats)
        fv[idx] = tier
        idx += 1

        windows.append({
            "month": month_str,
            "features": {FEATURE_NAMES[i]: float(fv[i]) for i in range(N_FEATURES)},
            "feature_vector": fv,
            "md": md, "ad": ad, "pd": pd,
        })

    return windows


def windows_to_feature_matrix(windows, feature_names=None):
    """Convert list of window dicts to numpy feature matrix.

    Returns (X, months) where X is shape (n_windows, n_features).
    """
    if not windows:
        return np.zeros((0, N_FEATURES), dtype=np.float32), []

    if feature_names is None:
        feature_names = FEATURE_NAMES

    X = np.stack([w["feature_vector"] for w in windows])
    months = [w["month"] for w in windows]
    return X, months
