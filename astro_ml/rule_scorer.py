"""Rule-Based Scorer — deterministic V38 convergence rubric.

Stage 1 of the two-stage V5 pipeline. No ML.
Scores every monthly window using astrological rules, assigns tiers,
then shortlists top candidates for Stage 2 (ML reranker).
"""
import datetime
import numpy as np

from astro_ml.config import domain_fathers_death as cfg


# ── Helpers ───────────────────────────────────────────────────────────────

def _prev_month_str(month_str):
    """'2020-03' -> '2020-02'."""
    parts = month_str.split("-")
    y, m = int(parts[0]), int(parts[1])
    m -= 1
    if m < 1:
        m = 12
        y -= 1
    return f"{y}-{m:02d}"


def _get_dasha_for_month(payload, month_str):
    """Get MD/AD/PD lords for a given month from Calculated_Triggers."""
    triggers = payload.get("Calculated_Triggers", {})
    month_data = triggers.get(month_str, {})
    dasha = month_data.get("_dasha", {})
    return dasha.get("md", ""), dasha.get("ad", ""), dasha.get("pd", "")


def _compute_quality_score(lord, payload):
    """Compute dasha quality score for one lord."""
    if not lord:
        return 0.0

    fn_str = payload.get("Functional_Nature", {}).get(lord, "Neutral")
    dig_str = payload.get("Planetary_Dignity", {}).get(lord, "Neutral")

    # Shadow planets (Rahu, Ketu) use sign lord's nature
    if lord in ("Rahu", "Ketu"):
        # Get sign lord from KP_Planets
        kp_data = payload.get("KP_Planets", {}).get(lord, {})
        rashi = kp_data.get("rashi", "")
        sign_num = cfg.SIGN_TO_NUM.get(rashi, 0)
        sign_lord = cfg.SIGN_LORDS.get(sign_num, "")
        if sign_lord:
            fn_str = payload.get("Functional_Nature", {}).get(sign_lord, "Neutral")
        base = cfg.FUNC_NATURE_ENCODING.get(fn_str, 0)
        return float(base)  # factor=1.0 for shadow planets

    base = cfg.FUNC_NATURE_ENCODING.get(fn_str, 0)
    factor = cfg.DIGNITY_FACTOR.get(dig_str, 1.0)
    return float(base) * factor


# ── Per-Cusp Analysis ─────────────────────────────────────────────────────

def _analyze_cusp(cusp_num, md, ad, pd, payload, month_str):
    """Analyze one cusp for dasha activation, locks, and transits.

    Returns dict with cusp analysis results.
    """
    triggers = payload.get("Calculated_Triggers", {})
    kp_cusps = payload.get("KP_Cusps", {})
    planet_sigs = payload.get("Planet_Significations", {})

    cusp_key = f"Cusp_{cusp_num}"
    month_data = triggers.get(month_str, {})
    cusp_data = month_data.get(cusp_key, {})

    # Sub-lord of this cusp
    sub_lord = kp_cusps.get(cusp_key, {}).get("sub_lord", "")

    # Dasha Lock: sub-lord IS a running dasha lord
    lock_level = 0
    lock_lord = ""
    if sub_lord:
        if sub_lord == md:
            lock_level = 3
            lock_lord = md
        elif sub_lord == ad:
            lock_level = 2
            lock_lord = ad
        elif sub_lord == pd:
            lock_level = 1
            lock_lord = pd

    # Dasha signification: any dasha lord signifies this cusp
    cusps_md = set(planet_sigs.get(md, [])) if md else set()
    cusps_ad = set(planet_sigs.get(ad, [])) if ad else set()
    cusps_pd = set(planet_sigs.get(pd, [])) if pd else set()
    dasha_active = cusp_num in (cusps_md | cusps_ad | cusps_pd)

    # Transit: Macro_Hits on this cusp this month
    hits = cusp_data.get("Macro_Hits", [])
    transit_active = len(hits) > 0

    # Transit orb
    if hits:
        min_orb = min(h.get("orb", 5.0) for h in hits)
    else:
        min_orb = 5.0

    # Transit ingress: active now but not previous month
    prev_month = _prev_month_str(month_str)
    prev_data = triggers.get(prev_month, {}).get(cusp_key, {})
    prev_hits = prev_data.get("Macro_Hits", [])
    transit_ingress = transit_active and len(prev_hits) == 0

    # Double activation
    double = dasha_active and transit_active

    return {
        "cusp_num": cusp_num,
        "sub_lord": sub_lord,
        "dasha_active": dasha_active,
        "lock_level": lock_level,
        "lock_lord": lock_lord,
        "transit_active": transit_active,
        "transit_orb": min_orb,
        "transit_ingress": transit_ingress,
        "double": double,
    }


# ── Score Chart ───────────────────────────────────────────────────────────

def score_chart(payload):
    """Score every monthly window using deterministic astrological rules.

    Returns list of scored window dicts sorted by rule_score descending.
    """
    triggers = payload.get("Calculated_Triggers", {})
    months = sorted(k for k in triggers.keys() if not k.startswith("_"))

    scored_windows = []

    for month_str in months:
        md, ad, pd = _get_dasha_for_month(payload, month_str)
        if not md:
            continue

        # Analyze all cusps
        cusp_results = {}
        for cusp_num in cfg.PRIMARY_HOUSES + cfg.SECONDARY_HOUSES + cfg.NEGATIVE_HOUSES:
            cusp_results[cusp_num] = _analyze_cusp(cusp_num, md, ad, pd, payload, month_str)

        # Classify activations
        primary_active = [c for c in cfg.PRIMARY_HOUSES
                          if cusp_results[c]["dasha_active"] or cusp_results[c]["transit_active"]]
        primary_dasha = [c for c in cfg.PRIMARY_HOUSES if cusp_results[c]["dasha_active"]]
        primary_transit = [c for c in cfg.PRIMARY_HOUSES if cusp_results[c]["transit_active"]]
        primary_locks = [(c, cusp_results[c]["lock_lord"], cusp_results[c]["lock_level"])
                         for c in cfg.PRIMARY_HOUSES if cusp_results[c]["lock_level"] > 0]
        primary_double = [c for c in cfg.PRIMARY_HOUSES if cusp_results[c]["double"]]
        secondary_active = [c for c in cfg.SECONDARY_HOUSES
                            if cusp_results[c]["dasha_active"] or cusp_results[c]["transit_active"]]
        negative_active = [c for c in cfg.NEGATIVE_HOUSES
                           if cusp_results[c]["dasha_active"] or cusp_results[c]["transit_active"]]

        # Dasha quality
        total_quality = sum(_compute_quality_score(l, payload) for l in [md, ad, pd])

        # Karaka checks
        lords = [md, ad, pd]
        yogakaraka_count = sum(1 for l in lords
                               if payload.get("Functional_Nature", {}).get(l, "") == "Yogakaraka")
        death_karaka_active = sum(1 for l in lords if l in cfg.DEATH_KARAKAS)
        sun_active = "Sun" in lords
        karaka_count = sum(1 for l in lords if l in cfg.ALL_KARAKAS)

        # Dignity check
        has_exalted = any(
            payload.get("Planetary_Dignity", {}).get(l, "") == "Exalted"
            for l in lords if l
        )

        # ── Tier assignment (V38 rubric) ────────────────────────────
        n_primary = len(primary_active)
        has_lock = len(primary_locks) > 0
        has_double = len(primary_double) > 0
        n_transit = len(primary_transit)

        if (n_primary >= 3 and has_lock and n_transit >= 1
                and (yogakaraka_count >= 1 or has_exalted) and total_quality >= 6.0):
            tier, tier_score = "S", 5
        elif has_lock and total_quality >= 3.0:
            tier, tier_score = "A", 4
        elif n_primary >= 2 and yogakaraka_count >= 1 and total_quality >= 3.0:
            tier, tier_score = "A", 4
        elif has_double and total_quality >= 3.0:
            tier, tier_score = "A", 4
        elif n_primary >= 2:
            tier, tier_score = "B", 3
        elif n_primary >= 1 and yogakaraka_count >= 1:
            tier, tier_score = "B", 3
        elif n_primary >= 1:
            tier, tier_score = "C", 2
        elif len(secondary_active) >= 1:
            tier, tier_score = "D", 1
        else:
            tier, tier_score = "F", 0

        # ── Continuous rule_score for fine-grained ranking ──────────
        rule_score = (
            tier_score * 100
            + len(primary_locks) * 15
            + len(primary_double) * 12
            + n_primary * 8
            + len(secondary_active) * 3
            + total_quality * 2
            + yogakaraka_count * 5
            + death_karaka_active * 4
            + (3 if sun_active else 0)
            - len(negative_active) * 4
            # Cusp-4 specific (8th from 9th = most important)
            + (10 if 4 in [c for c, _, _ in primary_locks] else 0)
            + (5 if 4 in primary_double else 0)
        )

        scored_windows.append({
            "month_str": month_str,
            "tier": tier,
            "tier_score": tier_score,
            "rule_score": rule_score,
            "md": md, "ad": ad, "pd": pd,
            "dasha_period": f"{md}-{ad}-{pd}",
            "components": {
                "primary_active": primary_active,
                "primary_dasha": primary_dasha,
                "primary_transit": primary_transit,
                "primary_locks": primary_locks,
                "primary_double": primary_double,
                "secondary_active": secondary_active,
                "negative_active": negative_active,
                "total_quality": total_quality,
                "yogakaraka_count": yogakaraka_count,
                "death_karaka_active": death_karaka_active,
                "sun_active": sun_active,
                "karaka_count": karaka_count,
            },
            "cusp_results": cusp_results,
        })

    scored_windows.sort(key=lambda x: x["rule_score"], reverse=True)
    return scored_windows


# ── Shortlisting ──────────────────────────────────────────────────────────

def shortlist(scored_windows, min_tier_score=3, max_candidates=8, min_candidates=3):
    """Select candidate windows for Stage 2 reranker.

    Rules:
    - Keep all B-tier (score>=3) and above
    - If fewer than min_candidates, add top C-tier windows
    - Cap at max_candidates (by rule_score)
    - Always keep at least min_candidates
    """
    candidates = [w for w in scored_windows if w["tier_score"] >= min_tier_score]

    if len(candidates) < min_candidates:
        c_tier = [w for w in scored_windows
                  if w["tier_score"] == min_tier_score - 1]
        c_tier.sort(key=lambda x: x["rule_score"], reverse=True)
        candidates.extend(c_tier[:min_candidates - len(candidates)])

    if len(candidates) < min_candidates:
        # Still not enough — add D-tier
        d_tier = [w for w in scored_windows
                  if w["tier_score"] == min_tier_score - 2
                  and w not in candidates]
        d_tier.sort(key=lambda x: x["rule_score"], reverse=True)
        candidates.extend(d_tier[:min_candidates - len(candidates)])

    candidates.sort(key=lambda x: x["rule_score"], reverse=True)
    return candidates[:max_candidates]


# ── Stage 1 Recall Evaluation ─────────────────────────────────────────────

def evaluate_stage1_recall(extracted_results, tolerance=1,
                           min_tier_score=3, max_candidates=8, min_candidates=3):
    """Evaluate Stage 1 recall: what fraction of event months are retained?

    Args:
        extracted_results: list of (chart_id, windows, event_month) from V4 extraction.
            Each window must have the payload accessible OR we re-score from triggers.
        tolerance: months of tolerance (1 = ±1 month)

    Returns:
        recall, avg_shortlist_size, tier_distribution
    """
    # We need to score from payloads, but extracted_results don't have them.
    # Instead, we use the already-computed cusp data from Calculated_Triggers.
    # This function expects payloads directly — see evaluate_from_payloads().
    raise NotImplementedError("Use evaluate_from_payloads() with raw payloads")


def evaluate_from_payloads(chart_payloads, tolerance=1,
                           min_tier_score=3, max_candidates=8, min_candidates=3):
    """Evaluate Stage 1 recall from list of (payload, event_month) tuples.

    Returns (recall, avg_size, tier_dist).
    """
    retained = 0
    total = 0
    shortlist_sizes = []
    tier_dist = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}

    for payload, event_month in chart_payloads:
        if not event_month:
            continue
        total += 1

        scored = score_chart(payload)
        candidates = shortlist(scored, min_tier_score, max_candidates, min_candidates)
        shortlist_sizes.append(len(candidates))

        # Count tier distribution for the event month window
        for sw in scored:
            if sw["month_str"] == event_month:
                tier_dist[sw["tier"]] = tier_dist.get(sw["tier"], 0) + 1
                break

        # Check if event month (±tolerance) is in shortlist
        try:
            ep = event_month.split("-")
            event_ym = int(ep[0]) * 12 + int(ep[1])
        except (ValueError, IndexError):
            continue

        hit = any(
            abs(int(m["month_str"][:4]) * 12 + int(m["month_str"][5:7]) - event_ym) <= tolerance
            for m in candidates
        )
        if hit:
            retained += 1

    recall = retained / max(total, 1)
    avg_size = sum(shortlist_sizes) / max(len(shortlist_sizes), 1)

    print(f"Stage 1 Recall (±{tolerance}mo): {recall:.1%} ({retained}/{total})")
    print(f"Average shortlist size: {avg_size:.1f}")
    print(f"Event month tier distribution: {tier_dist}")
    return recall, avg_size, tier_dist
