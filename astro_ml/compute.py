"""Compute bridge — generates the standardized payload from astro_engine.

This replaces the Colab compute() function. For each chart, it produces a
JSON-serializable payload with the same structure the plan expects:
  - Vimshottari_Timeline, KP_Cusps, KP_Planets, Functional_Nature,
    Planetary_Dignity, Navamsha_Positions, Natal_Flags, Calculated_Triggers,
    Planet_Significations, House_Lordships, Cross_Chart_Data
"""
import sys, os, datetime, threading
import pytz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from astro_engine import AstroEngine
from astro_ml.config import domain_fathers_death as cfg

_thread_local = threading.local()


def _get_engine():
    if not hasattr(_thread_local, "engine"):
        _thread_local.engine = AstroEngine()
    return _thread_local.engine


def _sign_num(rashi_name):
    """Convert sign name to 1-12."""
    return cfg.SIGN_TO_NUM.get(rashi_name, 0)


def compute(chart, start_date=None, end_date=None):
    """Generate a standardized payload for one chart.

    Args:
        chart: dict with keys birth_date, birth_time, lat, lon, tz, gender, etc.
        start_date: datetime — start of analysis window (default: death - 12 months)
        end_date: datetime — end of analysis window (default: death + 12 months)

    Returns:
        dict: payload with all computed chart data
    """
    engine = _get_engine()

    # --- Parse birth data ---
    bd = chart["birth_date"].split("-")
    bt = chart["birth_time"].split(":")
    tz_str = chart.get("tz", "UTC")
    tz = pytz.timezone(tz_str)

    birth_dt = tz.localize(datetime.datetime(
        int(bd[0]), int(bd[1]), int(bd[2]),
        int(bt[0]), int(bt[1]),
        int(bt[2]) if len(bt) > 2 else 0
    ))
    lat, lon = float(chart["lat"]), float(chart["lon"])

    # --- Natal planets ---
    natal = engine.calculate_planetary_positions(birth_dt, lat, lon)

    # --- Lagna ---
    lagna = engine.calculate_lagna(birth_dt, lat, lon)
    lagna_sign = lagna.get("rashi", "Aries")
    lagna_sign_num = _sign_num(lagna_sign)

    # --- KP Cusps (Placidus) ---
    kp_cusps_raw = engine.calculate_placidus_cusps(birth_dt, lat, lon)
    kp_cusps = {}
    for house_num, data in kp_cusps_raw.items():
        kp_cusps[f"Cusp_{house_num}"] = {
            "degree": data["longitude"],
            "rashi": data.get("rashi", ""),
            "degree_in_sign": data.get("degree", 0),
            "star_lord": data.get("star_lord", ""),
            "sub_lord": data.get("sub_lord", ""),
            "sign_lord": data.get("sign_lord", cfg.SIGN_LORDS.get(
                _sign_num(data.get("rashi", "")), ""
            )),
        }

    # --- KP Planet details (4-level chain) ---
    kp_details_raw = engine.calculate_kp_details(natal)
    kp_planets = {}
    for planet, pdata in natal.items():
        kp_info = kp_details_raw.get(planet, {})
        kp_planets[planet] = {
            "degree": pdata.get("longitude", 0),
            "rashi": pdata.get("rashi", ""),
            "degree_in_sign": pdata.get("degree", 0),
            "nakshatra": pdata.get("nakshatra", ""),
            "star_lord": kp_info.get("star_lord", ""),
            "sub_lord": kp_info.get("sub_lord", ""),
            "sub_lord_star_lord": kp_info.get("sub_lord_star_lord", ""),
        }

    # --- Planet Significations (which cusps each planet signifies) ---
    cusp_lons = [kp_cusps_raw[h]["longitude"] for h in range(1, 13)]
    kp_sigs = engine.calculate_kp_significators(natal, kp_cusps_raw, kp_details_raw)
    planet_significations = {}
    for planet, houses in kp_sigs.items():
        planet_significations[planet] = sorted(houses) if isinstance(houses, (list, set)) else houses

    # --- House Lordships ---
    house_lordships = {}
    for h in range(1, 13):
        rashi = kp_cusps_raw.get(h, {}).get("rashi", "")
        lord = cfg.SIGN_LORDS.get(_sign_num(rashi), "")
        house_lordships[str(h)] = lord

    # --- Functional Nature ---
    functional_nature = {}
    for planet in cfg.PLANETS:
        fn = engine.get_functional_nature(planet, lagna_sign)
        functional_nature[planet] = fn

    # --- Planetary Dignity ---
    planetary_dignity = {}
    for planet in cfg.PLANETS:
        pdata = natal.get(planet, {})
        sn = _sign_num(pdata.get("rashi", ""))
        if sn > 0:
            dig = engine.get_planet_dignity(planet, sn)
            planetary_dignity[planet] = dig if dig else "Neutral"
        else:
            planetary_dignity[planet] = "Neutral"

    # --- Navamsha Positions ---
    d9 = engine.get_divisional_chart(natal, 9)
    navamsha_positions = {}
    for planet, d9data in d9.items():
        d9_sign_num = d9data.get("sign_num", 0)
        natal_sign_num = _sign_num(natal.get(planet, {}).get("rashi", ""))
        is_vargottama = (d9_sign_num == natal_sign_num) if natal_sign_num > 0 else False
        d9_dig = engine.get_planet_dignity(planet, d9_sign_num) if d9_sign_num > 0 else "Neutral"
        nav_dig_str = f"VARGOTTAMA+{d9_dig.upper()}" if is_vargottama else d9_dig.upper()
        navamsha_positions[planet] = {
            "rashi": d9data.get("rashi", ""),
            "sign_num": d9_sign_num,
            "dignity": d9_dig,
            "is_vargottama": is_vargottama,
            "nav_dignity_string": nav_dig_str,
        }

    # --- Natal Flags ---
    natal_flags = {}
    for planet in cfg.PLANETS:
        pdata = natal.get(planet, {})
        natal_flags[planet] = {
            "is_retrograde": pdata.get("is_retrograde", False),
            "is_combust": pdata.get("is_combust", False),
        }

    # --- Vimshottari Timeline ---
    moon_lon = natal.get("Moon", {}).get("longitude", 0)

    # Determine scan window
    if start_date and end_date:
        scan_start, scan_end = start_date, end_date
    else:
        # Default: build a wide window around father's death
        fd = chart.get("father_death_date", "")
        if fd:
            parts = fd.split("-")
            death_dt = datetime.datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            death_dt = datetime.datetime(2020, 1, 1)  # fallback
        scan_start = death_dt - datetime.timedelta(days=365 * 2)
        scan_end = death_dt + datetime.timedelta(days=365 * 2)

    scan_start_tz = tz.localize(scan_start) if scan_start.tzinfo is None else scan_start
    scan_end_tz = tz.localize(scan_end) if scan_end.tzinfo is None else scan_end

    dasha_seq = engine.calculate_dasha_sequence(moon_lon, birth_dt, scan_start_tz, scan_end_tz)
    vimshottari_timeline = []
    for entry in dasha_seq:
        vimshottari_timeline.append({
            "lord": entry.get("lord", ""),
            "sub_lord": entry.get("sub_lord"),
            "type": entry.get("type", "MD"),
            "start": entry.get("start", ""),
            "end": entry.get("end", ""),
        })

    # --- Calculated Triggers (transit hits per month) ---
    # We compute transit positions for each month in the window and check
    # if Jupiter/Saturn/Rahu aspect natal cusp sub-lord degrees
    calculated_triggers = _compute_monthly_triggers(
        engine, tz, lat, lon, natal, kp_cusps_raw, kp_details_raw,
        scan_start, scan_end, moon_lon, birth_dt
    )

    # --- Cross Chart Data ---
    cross_chart = {
        "lagna_lord": cfg.SIGN_LORDS.get(lagna_sign_num, ""),
        "lagna_sign": lagna_sign,
        "lagna_sign_num": lagna_sign_num,
        "lagna_degree": lagna.get("longitude", 0),
        "moon_sign": natal.get("Moon", {}).get("rashi", ""),
        "moon_sign_num": _sign_num(natal.get("Moon", {}).get("rashi", "")),
        "moon_lon": moon_lon,
        "sun_sign": natal.get("Sun", {}).get("rashi", ""),
        "9th_cusp_sign": kp_cusps_raw.get(9, {}).get("rashi", ""),
        "9th_cusp_sign_num": _sign_num(kp_cusps_raw.get(9, {}).get("rashi", "")),
    }

    return {
        "Vimshottari_Timeline": vimshottari_timeline,
        "KP_Cusps": kp_cusps,
        "KP_Planets": kp_planets,
        "Functional_Nature": functional_nature,
        "Planetary_Dignity": planetary_dignity,
        "Navamsha_Positions": navamsha_positions,
        "Natal_Flags": natal_flags,
        "Calculated_Triggers": calculated_triggers,
        "Planet_Significations": planet_significations,
        "House_Lordships": house_lordships,
        "Cross_Chart_Data": cross_chart,
        # Extra fields used internally
        "_natal": natal,
        "_kp_cusps_raw": kp_cusps_raw,
        "_kp_details_raw": kp_details_raw,
        "_lagna_sign_num": lagna_sign_num,
        "_moon_lon": moon_lon,
        "_birth_dt": birth_dt,
        "_tz": tz,
        "_lat": lat,
        "_lon": lon,
    }


def _compute_monthly_triggers(engine, tz, lat, lon, natal, kp_cusps_raw,
                               kp_details_raw, scan_start, scan_end,
                               moon_lon, birth_dt):
    """Compute per-cusp transit triggers for each month in the scan window.

    Returns dict keyed by "YYYY-MM" -> { cusp_key -> { "Macro_Hits": [...], ... } }
    """
    triggers = {}
    transit_planets = ["Jupiter", "Saturn", "Rahu"]
    target_cusps = cfg.TARGET_CUSPS  # [3, 4, 8, 10, 12]

    # Pre-compute natal cusp sub-lord degrees
    cusp_sublord_degrees = {}
    for c in range(1, 13):
        cdata = kp_cusps_raw.get(c, {})
        sublord = cdata.get("sub_lord", "")
        if sublord and sublord in natal:
            cusp_sublord_degrees[c] = natal[sublord].get("longitude", 0)

    # Iterate month by month
    cur = datetime.datetime(scan_start.year, scan_start.month, 15)
    end = datetime.datetime(scan_end.year, scan_end.month, 28)

    while cur <= end:
        month_key = cur.strftime("%Y-%m")
        month_triggers = {}

        try:
            dt_aware = tz.localize(cur)
            transit = engine.calculate_planetary_positions(dt_aware, lat, lon)
        except Exception:
            cur = _next_month(cur)
            continue

        # Also get current dasha
        try:
            dasha_info = engine.calculate_dasha(moon_lon, birth_dt, tz.localize(cur))
            md_lord = dasha_info.get("current_date_dasha", "")
            ad_lord = dasha_info.get("current_antardasha", "")
            pd_lord = dasha_info.get("current_pratyantardasha", "")
        except Exception:
            md_lord = ad_lord = pd_lord = ""

        for cusp_num in target_cusps:
            cusp_key = f"Cusp_{cusp_num}"
            sublord_deg = cusp_sublord_degrees.get(cusp_num)
            cusp_sublord = kp_cusps_raw.get(cusp_num, {}).get("sub_lord", "")

            macro_hits = []
            for tp in transit_planets:
                tdata = transit.get(tp, {})
                t_lon = tdata.get("longitude", 0)
                if sublord_deg is not None:
                    # Check all aspects
                    aspects = cfg.VEDIC_ASPECTS.get(tp, [0, 180])
                    for asp in aspects:
                        diff = abs(((t_lon + asp) - sublord_deg + 180) % 360 - 180)
                        if diff <= cfg.TRANSIT_ORB:
                            macro_hits.append({
                                "planet": tp,
                                "aspect": asp,
                                "orb": round(diff, 2),
                                "target_degree": round(sublord_deg, 2),
                            })

            # Check if cusp sub-lord IS the current dasha lord (Dasha Lock)
            dasha_lock = cusp_sublord in (md_lord, ad_lord, pd_lord) if cusp_sublord else False
            dasha_lock_level = ""
            if dasha_lock:
                if cusp_sublord == md_lord: dasha_lock_level = "MD"
                elif cusp_sublord == ad_lord: dasha_lock_level = "AD"
                elif cusp_sublord == pd_lord: dasha_lock_level = "PD"

            month_triggers[cusp_key] = {
                "Macro_Hits": macro_hits,
                "Dasha_Lock": dasha_lock,
                "Dasha_Lock_Level": dasha_lock_level,
                "Cusp_SubLord": cusp_sublord,
            }

        # Also store dasha state for this month
        month_triggers["_dasha"] = {
            "md": md_lord, "ad": ad_lord, "pd": pd_lord,
        }

        triggers[month_key] = month_triggers
        cur = _next_month(cur)

    return triggers


def _next_month(dt):
    if dt.month == 12:
        return datetime.datetime(dt.year + 1, 1, 15)
    return datetime.datetime(dt.year, dt.month + 1, 15)
