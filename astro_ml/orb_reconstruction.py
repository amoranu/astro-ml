"""Orb Reconstruction — derive temporal transit signals from Macro_Hits.

Parses the per-month Macro_Hits data to reconstruct:
  1. Transit runs (contiguous months where a planet aspects a target)
  2. Per-month temporal properties (ingress, egress, position in run, freshness)
  3. Aggregated per-month features for the model

This creates PEAKS at exact transit months instead of flat binary flags.
"""
import numpy as np
from collections import defaultdict

from astro_ml.config import domain_fathers_death as cfg

# Feature names for orb reconstruction block
ORB_RECON_FEATURE_NAMES = []

# Per-planet features (Jupiter, Saturn, Rahu) × features
_TRANSIT_PLANETS = ["Jupiter", "Saturn", "Rahu"]
_PRIMARY_CUSPS = set(cfg.PRIMARY_HOUSES)
_SECONDARY_CUSPS = set(cfg.SECONDARY_HOUSES)
_NEGATIVE_CUSPS = set(cfg.NEGATIVE_HOUSES)
_TARGET_CUSPS = set(cfg.TARGET_CUSPS)


def _build_orb_recon_names():
    names = []
    # Per planet (3 × 8 = 24)
    for p in _TRANSIT_PLANETS:
        names.extend([
            f"{p.lower()}_best_orb_inv",      # 1/(tightest_orb + 0.1) — peaks at exact
            f"{p.lower()}_primary_orb_inv",    # same but only primary cusps
            f"{p.lower()}_is_ingress",         # first month in a transit run
            f"{p.lower()}_is_egress",          # last month in a transit run
            f"{p.lower()}_months_active",      # position in run (1, 2, 3...)
            f"{p.lower()}_run_freshness",      # 1/(months_active + 0.5)
            f"{p.lower()}_n_hits",             # number of hits this month
            f"{p.lower()}_primary_active",     # any hit on primary cusp
        ])
    # Aggregated (20)
    names.extend([
        "tightest_orb_inv_any",         # best across all planets
        "tightest_orb_inv_primary",     # best on primary cusps
        "mean_orb_inv_target",          # mean 1/(orb+0.1) across all target hits
        "n_transits_within_1deg",       # count of hits with orb < 1°
        "n_transits_within_2deg",       # count of hits with orb < 2°
        "n_exact_hits",                 # count of hits with orb < 0.5°
        "n_ingresses_target",           # number of planet-target pairs starting this month
        "n_ingresses_primary",          # ingresses on primary cusps
        "any_primary_ingress",          # binary: any primary cusp ingress
        "n_egresses_target",            # number ending this month
        "n_distinct_planets_active",    # how many of Jup/Sat/Rahu have any hit
        "all_three_heavy_active",       # all 3 transit planets hitting something
        "transit_freshness",            # sum of 1/(months_active+0.5), primary weighted 2×
        "convergence_proxy",            # n_primary_active_planets × (1 + freshness)
        "n_retro_hits",                 # not available from our data, placeholder
        "net_transit_signal",           # primary×2 + primary_ingress×3 + freshness - negative×1.5
        "saturn_primary_exact",         # Saturn orb < 1° on primary
        "saturn_primary_ingress",       # Saturn starting transit on primary
        "jupiter_primary_ingress",      # Jupiter starting transit on primary
        "double_transit_orb_inv",       # Saturn × Jupiter combined orb inverse (peaks when both exact)
    ])
    return names


ORB_RECON_FEATURE_NAMES = _build_orb_recon_names()
N_ORB_RECON = len(ORB_RECON_FEATURE_NAMES)


def _cusp_num_from_key(cusp_key):
    """Extract cusp number from 'Cusp_3' -> 3."""
    try:
        return int(cusp_key.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _build_transit_runs(triggers, months):
    """Build transit runs from Macro_Hits across all months.

    A "run" is a contiguous sequence of months where a specific
    (planet, target_degree, cusp) triple has hits.

    Returns dict: (planet, target_key) -> list of runs
    where each run = {"start_idx": int, "end_idx": int, "months": [idx...], "cusps": set}
    """
    # Collect per-(planet, target_degree_rounded) activity by month index
    activity = defaultdict(list)  # (planet, target_key) -> [(month_idx, cusp_num, orb)]

    for mi, month_str in enumerate(months):
        month_data = triggers.get(month_str, {})
        for cusp_key, cdata in month_data.items():
            if cusp_key.startswith("_"):
                continue
            cusp_num = _cusp_num_from_key(cusp_key)
            for hit in cdata.get("Macro_Hits", []):
                planet = hit.get("planet", "")
                target = hit.get("target_degree", 0)
                orb = hit.get("orb", 5.0)
                # Group by planet + target (round to 1° to handle minor variations)
                target_key = round(target, 0)
                activity[(planet, target_key)].append((mi, cusp_num, orb))

    # Build contiguous runs
    runs = {}
    for key, entries in activity.items():
        entries.sort(key=lambda x: x[0])
        month_indices = sorted(set(e[0] for e in entries))
        cusps_by_month = defaultdict(set)
        orbs_by_month = defaultdict(list)
        for mi, cn, orb in entries:
            cusps_by_month[mi].add(cn)
            orbs_by_month[mi].append(orb)

        # Split into contiguous runs (allow 1-month gap for slow planets)
        run_list = []
        current_run = None
        for mi in month_indices:
            if current_run is None:
                current_run = {"start_idx": mi, "end_idx": mi, "months": [mi],
                               "cusps": cusps_by_month[mi].copy(),
                               "orbs": {mi: orbs_by_month[mi]}}
            elif mi <= current_run["end_idx"] + 2:  # allow 1-month gap
                current_run["end_idx"] = mi
                current_run["months"].append(mi)
                current_run["cusps"] |= cusps_by_month[mi]
                current_run["orbs"][mi] = orbs_by_month[mi]
            else:
                run_list.append(current_run)
                current_run = {"start_idx": mi, "end_idx": mi, "months": [mi],
                               "cusps": cusps_by_month[mi].copy(),
                               "orbs": {mi: orbs_by_month[mi]}}
        if current_run:
            run_list.append(current_run)

        runs[key] = run_list

    return runs


def reconstruct_orb_features(windows, payload):
    """Add orb reconstruction features to each window.

    Modifies windows in-place, adding:
      - "orb_recon_vector": np.array of shape (N_ORB_RECON,)
      - "orb_recon_features": dict of name -> value

    Returns windows.
    """
    triggers = payload.get("Calculated_Triggers", {})
    months = [w["month"] for w in windows]
    n_months = len(months)
    month_to_idx = {m: i for i, m in enumerate(months)}

    # Build transit runs
    runs = _build_transit_runs(triggers, months)

    # Pre-compute per-month run membership
    # For each month, for each planet: which run is it in, position, etc.
    month_run_info = defaultdict(dict)  # month_idx -> planet -> {run_info}

    for (planet, target_key), run_list in runs.items():
        for run in run_list:
            for pos, mi in enumerate(run["months"]):
                is_ingress = (pos == 0)
                is_egress = (pos == len(run["months"]) - 1)
                months_active = pos + 1
                run_duration = len(run["months"])
                best_orb = min(run["orbs"].get(mi, [5.0]))
                cusps_hit = run["cusps"]

                if planet not in month_run_info[mi]:
                    month_run_info[mi][planet] = []
                month_run_info[mi][planet].append({
                    "is_ingress": is_ingress,
                    "is_egress": is_egress,
                    "months_active": months_active,
                    "run_duration": run_duration,
                    "best_orb": best_orb,
                    "cusps": cusps_hit,
                    "target_key": target_key,
                })

    # Now compute features for each month
    for mi, w in enumerate(windows):
        fv = np.zeros(N_ORB_RECON, dtype=np.float32)
        month_str = w["month"]
        month_data = triggers.get(month_str, {})

        # Collect all hits for this month with orb info
        all_hits = []
        for cusp_key, cdata in month_data.items():
            if cusp_key.startswith("_"):
                continue
            cusp_num = _cusp_num_from_key(cusp_key)
            for hit in cdata.get("Macro_Hits", []):
                all_hits.append({
                    "planet": hit.get("planet", ""),
                    "orb": hit.get("orb", 5.0),
                    "cusp": cusp_num,
                    "aspect": hit.get("aspect", 0),
                })

        idx = 0
        saturn_primary_best_orb = 5.0
        jupiter_primary_best_orb = 5.0

        # Per-planet features (3 × 8 = 24)
        for pi, planet in enumerate(_TRANSIT_PLANETS):
            p_hits = [h for h in all_hits if h["planet"] == planet]
            p_primary_hits = [h for h in p_hits if h["cusp"] in _PRIMARY_CUSPS]
            run_infos = month_run_info.get(mi, {}).get(planet, [])

            # Best orb inverse
            if p_hits:
                best_orb = min(h["orb"] for h in p_hits)
                fv[idx] = 1.0 / (best_orb + 0.1)
            else:
                fv[idx] = 0.0

            # Primary orb inverse
            if p_primary_hits:
                best_pri_orb = min(h["orb"] for h in p_primary_hits)
                fv[idx + 1] = 1.0 / (best_pri_orb + 0.1)
                if planet == "Saturn":
                    saturn_primary_best_orb = best_pri_orb
                elif planet == "Jupiter":
                    jupiter_primary_best_orb = best_pri_orb
            else:
                fv[idx + 1] = 0.0

            # Ingress / egress from run info
            any_ingress = any(ri["is_ingress"] for ri in run_infos)
            any_egress = any(ri["is_egress"] for ri in run_infos)
            fv[idx + 2] = 1.0 if any_ingress else 0.0
            fv[idx + 3] = 1.0 if any_egress else 0.0

            # Months active (max across runs)
            if run_infos:
                fv[idx + 4] = max(ri["months_active"] for ri in run_infos)
                fv[idx + 5] = max(1.0 / (ri["months_active"] + 0.5) for ri in run_infos)
            else:
                fv[idx + 4] = 0.0
                fv[idx + 5] = 0.0

            fv[idx + 6] = len(p_hits)
            fv[idx + 7] = 1.0 if p_primary_hits else 0.0

            idx += 8

        # Aggregated features (20)
        if all_hits:
            all_orbs = [h["orb"] for h in all_hits]
            primary_hits = [h for h in all_hits if h["cusp"] in _PRIMARY_CUSPS]
            primary_orbs = [h["orb"] for h in primary_hits] if primary_hits else [5.0]
            negative_hits = [h for h in all_hits if h["cusp"] in _NEGATIVE_CUSPS]

            fv[idx] = 1.0 / (min(all_orbs) + 0.1)          # tightest_orb_inv_any
            fv[idx + 1] = 1.0 / (min(primary_orbs) + 0.1)  # tightest_orb_inv_primary
            fv[idx + 2] = np.mean([1.0 / (o + 0.1) for o in all_orbs])  # mean_orb_inv
            fv[idx + 3] = sum(1 for o in all_orbs if o < 1.0)   # within 1°
            fv[idx + 4] = sum(1 for o in all_orbs if o < 2.0)   # within 2°
            fv[idx + 5] = sum(1 for o in all_orbs if o < 0.5)   # exact
        idx += 6

        # Ingress counts
        all_run_infos = month_run_info.get(mi, {})
        n_ingresses = sum(1 for p_infos in all_run_infos.values()
                          for ri in p_infos if ri["is_ingress"])
        n_primary_ingresses = sum(1 for p_infos in all_run_infos.values()
                                  for ri in p_infos
                                  if ri["is_ingress"] and ri["cusps"] & _PRIMARY_CUSPS)
        n_egresses = sum(1 for p_infos in all_run_infos.values()
                         for ri in p_infos if ri["is_egress"])

        fv[idx] = n_ingresses                                # n_ingresses_target
        fv[idx + 1] = n_primary_ingresses                    # n_ingresses_primary
        fv[idx + 2] = 1.0 if n_primary_ingresses > 0 else 0.0  # any_primary_ingress
        fv[idx + 3] = n_egresses                             # n_egresses_target
        idx += 4

        # Planet diversity
        active_planets = set(h["planet"] for h in all_hits) if all_hits else set()
        fv[idx] = len(active_planets)                        # n_distinct_planets_active
        fv[idx + 1] = 1.0 if len(active_planets) >= 3 else 0.0  # all_three_heavy_active
        idx += 2

        # Transit freshness
        freshness = 0.0
        for p_infos in all_run_infos.values():
            for ri in p_infos:
                weight = 2.0 if ri["cusps"] & _PRIMARY_CUSPS else 1.0
                freshness += weight / (ri["months_active"] + 0.5)
        fv[idx] = freshness                                  # transit_freshness
        idx += 1

        # Convergence proxy
        n_primary_active = sum(1 for p in _TRANSIT_PLANETS
                               if any(h["planet"] == p and h["cusp"] in _PRIMARY_CUSPS
                                      for h in all_hits))
        fv[idx] = n_primary_active * (1.0 + freshness)      # convergence_proxy
        idx += 1

        fv[idx] = 0.0  # n_retro_hits placeholder
        idx += 1

        # Net transit signal
        n_neg = len([h for h in all_hits if h["cusp"] in _NEGATIVE_CUSPS]) if all_hits else 0
        n_pri = len([h for h in all_hits if h["cusp"] in _PRIMARY_CUSPS]) if all_hits else 0
        fv[idx] = n_pri * 2 + n_primary_ingresses * 3 + freshness - n_neg * 1.5
        idx += 1

        # Saturn/Jupiter specifics
        fv[idx] = 1.0 if saturn_primary_best_orb < 1.0 else 0.0       # saturn_primary_exact
        fv[idx + 1] = 1.0 if any(ri["is_ingress"] and ri["cusps"] & _PRIMARY_CUSPS
                                  for ri in all_run_infos.get("Saturn", [])) else 0.0
        fv[idx + 2] = 1.0 if any(ri["is_ingress"] and ri["cusps"] & _PRIMARY_CUSPS
                                  for ri in all_run_infos.get("Jupiter", [])) else 0.0
        idx += 3

        # Double transit: both Saturn AND Jupiter hitting primary
        sat_inv = 1.0 / (saturn_primary_best_orb + 0.1) if saturn_primary_best_orb < 5.0 else 0.0
        jup_inv = 1.0 / (jupiter_primary_best_orb + 0.1) if jupiter_primary_best_orb < 5.0 else 0.0
        fv[idx] = sat_inv * jup_inv                          # double_transit_orb_inv
        idx += 1

        w["orb_recon_vector"] = fv
        w["orb_recon_features"] = {ORB_RECON_FEATURE_NAMES[i]: float(fv[i])
                                   for i in range(N_ORB_RECON)}

    return windows
