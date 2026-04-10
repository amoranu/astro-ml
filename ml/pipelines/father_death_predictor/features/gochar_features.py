"""Sign-based Gochar (transit) + Ashtakavarga filter features.

Parashari canonical transit discrimination: dasha indicates potential,
Gochar triggers the event. Sign-based (not degree-based) for crisp
binary features that vary across pratyantardasha candidates.

~35 features per candidate (18 original + ~17 new fast-planet + dignity + BAV).
"""

import swisseph as swe

from ..astro_engine.houses import get_sign, get_sign_lord, get_house_lord
from ..astro_engine.ashtakavarga import compute_bav, compute_sav, BAV_TABLES
from ..astro_engine.dignity import uchcha_bala, sign_dignity
from ..astro_engine.aspects import aspect_strength

# Mrityu Bhaga (death degrees) per sign — classical BPHS
# Index 0=Aries, 11=Pisces. Degree within sign (0-30).
_MRITYU_BHAGA = [1, 9, 22, 22, 25, 2, 4, 23, 18, 20, 24, 10]

# Kakshya lords — 8 segments of 3.75° each within a sign
# Order: Saturn, Jupiter, Mars, Sun, Venus, Mercury, Moon, Lagna-lord
_KAKSHYA_LORDS_BASE = ['Saturn', 'Jupiter', 'Mars', 'Sun',
                       'Venus', 'Mercury', 'Moon']  # 7 fixed + lagna_lord
_NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}

swe.set_sid_mode(swe.SIDM_LAHIRI)

# Planet IDs for swisseph
_SWE_IDS = {
    'Sun': swe.SUN, 'Moon': swe.MOON, 'Mars': swe.MARS,
    'Mercury': swe.MERCURY, 'Jupiter': swe.JUPITER, 'Venus': swe.VENUS,
    'Saturn': swe.SATURN, 'Rahu': swe.MEAN_NODE,
}


def _transit_sign(jd: float, planet_id: int) -> int:
    """Get sidereal sign (0-11) of a planet at a given Julian Day."""
    lon = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)[0][0]
    return int(lon / 30) % 12


def _transit_longitude(jd: float, planet_id: int) -> float:
    """Get sidereal longitude of a planet at a given Julian Day."""
    return swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)[0][0]


def _transit_speed(jd: float, planet_id: int) -> float:
    """Get daily motion speed (deg/day) of a planet at a given Julian Day."""
    return swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL | swe.FLG_SPEED)[0][3]


def _vedic_aspect_signs(planet: str, sign: int) -> set:
    """Return set of signs aspected by planet from given sign (Parashari).

    All planets have 7th aspect (opposition).
    Mars: additional 4th, 8th.
    Jupiter: additional 5th, 9th.
    Saturn: additional 3rd, 10th.
    Includes conjunction (own sign).
    """
    signs = {sign, (sign + 6) % 12}  # conjunction + 7th aspect

    if planet == 'Mars':
        signs.add((sign + 3) % 12)   # 4th aspect
        signs.add((sign + 7) % 12)   # 8th aspect
    elif planet == 'Jupiter':
        signs.add((sign + 4) % 12)   # 5th aspect
        signs.add((sign + 8) % 12)   # 9th aspect
    elif planet == 'Saturn':
        signs.add((sign + 2) % 12)   # 3rd aspect
        signs.add((sign + 9) % 12)   # 10th aspect

    return signs


def precompute_gochar_context(natal_asc: float, natal_chart: dict) -> dict:
    """Precompute chart-level data for gochar features (once per chart).

    Returns dict with father's key signs, BAV/SAV arrays.
    """
    # Father's lagna = 9th house sign
    asc_sign = get_sign(natal_asc)
    h9_sign = (asc_sign + 8) % 12  # 9th house sign = father's 1st

    # Father's maraka houses (2nd and 7th from father's lagna)
    father_maraka_signs = {
        (h9_sign + 1) % 12,   # father's 2nd = natal 10th
        (h9_sign + 6) % 12,   # father's 7th = natal 3rd
    }
    father_death_signs = {
        (h9_sign + 7) % 12,   # father's 8th = natal 4th
    }
    father_loss_signs = {
        (h9_sign + 11) % 12,  # father's 12th = natal 8th
    }
    danger_signs = father_maraka_signs | father_death_signs | father_loss_signs

    # BAV arrays for transit filtering
    sat_bav = compute_bav(BAV_TABLES['Saturn'], natal_chart, natal_asc)
    mars_bav = compute_bav(BAV_TABLES['Mars'], natal_chart, natal_asc)
    sun_bav = compute_bav(BAV_TABLES['Sun'], natal_chart, natal_asc)
    merc_bav = compute_bav(BAV_TABLES['Mercury'], natal_chart, natal_asc)
    venus_bav = compute_bav(BAV_TABLES['Venus'], natal_chart, natal_asc)

    # SAV (total bindus per sign)
    sav = compute_sav(natal_chart, natal_asc)

    # Minimum SAV in maraka signs
    maraka_savs = [sav[s] for s in father_maraka_signs]
    maraka_sav_min = min(maraka_savs)

    # Father's functional malefic lords (lords of 6, 8, 12 from 9th house)
    func_malefic_lords = {
        get_house_lord((9 + 5) % 12 or 12, natal_asc),  # 6th from 9th = 2nd house
        get_house_lord((9 + 7) % 12 or 12, natal_asc),  # 8th from 9th = 4th house
        get_house_lord((9 + 11) % 12 or 12, natal_asc), # 12th from 9th = 8th house
    }

    # Natal degree positions for degree-level transit features
    natal_sun_deg = natal_chart['Sun']['longitude']
    natal_moon_deg = natal_chart['Moon']['longitude']
    h9_cusp_deg = (natal_asc + 8 * 30) % 360  # 9th house cusp (equal house)

    # Lagna lord for Kakshya 8th segment
    lagna_lord = get_sign_lord(asc_sign)

    return {
        'h9_sign': h9_sign,
        'father_maraka_signs': father_maraka_signs,
        'father_death_signs': father_death_signs,
        'father_loss_signs': father_loss_signs,
        'danger_signs': danger_signs,
        'sat_bav': sat_bav,
        'mars_bav': mars_bav,
        'sun_bav': sun_bav,
        'merc_bav': merc_bav,
        'venus_bav': venus_bav,
        'sav': sav,
        'maraka_sav_min': maraka_sav_min,
        'maraka_sav_weak': 1.0 if maraka_sav_min < 25 else 0.0,
        'func_malefic_lords': func_malefic_lords,
        'natal_sun_deg': natal_sun_deg,
        'natal_moon_deg': natal_moon_deg,
        'h9_cusp_deg': h9_cusp_deg,
        'lagna_lord': lagna_lord,
    }


def extract_gochar_features(candidate: dict,
                            gochar_ctx: dict) -> dict:
    """Sign-based Gochar transit features at candidate midpoint.

    Args:
        candidate: pratyantardasha period dict
        gochar_ctx: precomputed from precompute_gochar_context()

    Returns:
        ~18 features dict.
    """
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2

    father_mk = gochar_ctx['father_maraka_signs']
    father_death = gochar_ctx['father_death_signs']
    danger = gochar_ctx['danger_signs']
    sat_bav = gochar_ctx['sat_bav']

    f = {}

    # --- Saturn transit ---
    sat_sign = _transit_sign(midpoint_jd, swe.SATURN)
    sat_asp = _vedic_aspect_signs('Saturn', sat_sign)

    f['gc_sat_on_maraka'] = 1.0 if sat_sign in father_mk else 0.0
    f['gc_sat_on_death'] = 1.0 if sat_sign in father_death else 0.0
    f['gc_sat_on_danger'] = 1.0 if sat_sign in danger else 0.0
    f['gc_sat_asp_maraka'] = 1.0 if bool(sat_asp & father_mk) else 0.0
    f['gc_sat_asp_danger'] = 1.0 if bool(sat_asp & danger) else 0.0

    # --- Jupiter transit ---
    jup_sign = _transit_sign(midpoint_jd, swe.JUPITER)
    jup_asp = _vedic_aspect_signs('Jupiter', jup_sign)

    f['gc_jup_asp_maraka'] = 1.0 if bool(jup_asp & father_mk) else 0.0
    f['gc_jup_asp_danger'] = 1.0 if bool(jup_asp & danger) else 0.0

    # --- Double Transit (THE key Parashari trigger) ---
    # Both Jupiter AND Saturn aspecting a father maraka house
    f['gc_double_transit_maraka'] = 1.0 if (
        bool(sat_asp & father_mk) and bool(jup_asp & father_mk)
    ) else 0.0
    f['gc_double_transit_danger'] = 1.0 if (
        bool(sat_asp & danger) and bool(jup_asp & danger)
    ) else 0.0

    # --- Mars transit (fast planet trigger) ---
    mars_sign = _transit_sign(midpoint_jd, swe.MARS)
    mars_asp = _vedic_aspect_signs('Mars', mars_sign)

    f['gc_mars_asp_maraka'] = 1.0 if bool(mars_asp & father_mk) else 0.0

    # --- Triple activation (Saturn + Jupiter + Mars all on danger) ---
    f['gc_triple_activation'] = 1.0 if (
        bool(sat_asp & danger) and
        bool(jup_asp & danger) and
        bool(mars_asp & danger)
    ) else 0.0

    # --- Mercury transit (fast planet, ~6 sign changes in 6mo) ---
    merc_sign = _transit_sign(midpoint_jd, swe.MERCURY)
    merc_asp = _vedic_aspect_signs('Mercury', merc_sign)  # 7th only
    f['gc_merc_on_maraka'] = 1.0 if merc_sign in father_mk else 0.0
    f['gc_merc_asp_maraka'] = 1.0 if bool(merc_asp & father_mk) else 0.0
    f['gc_merc_asp_danger'] = 1.0 if bool(merc_asp & danger) else 0.0

    # --- Venus transit (fast planet, ~3 sign changes in 6mo) ---
    venus_sign = _transit_sign(midpoint_jd, swe.VENUS)
    venus_asp = _vedic_aspect_signs('Venus', venus_sign)  # 7th only
    f['gc_venus_on_maraka'] = 1.0 if venus_sign in father_mk else 0.0
    f['gc_venus_asp_maraka'] = 1.0 if bool(venus_asp & father_mk) else 0.0
    f['gc_venus_asp_danger'] = 1.0 if bool(venus_asp & danger) else 0.0

    # --- Mars expansion ---
    f['gc_mars_on_maraka'] = 1.0 if mars_sign in father_mk else 0.0
    f['gc_mars_asp_danger'] = 1.0 if bool(mars_asp & danger) else 0.0

    # --- Fast malefic count (Mars + any functional malefic on danger) ---
    func_mal = gochar_ctx.get('func_malefic_lords', set())
    fast_mal_count = int(bool(mars_asp & danger))
    if 'Mercury' in func_mal:
        fast_mal_count += int(bool(merc_asp & danger))
    if 'Venus' in func_mal:
        fast_mal_count += int(bool(venus_asp & danger))
    f['gc_fast_malefic_count'] = fast_mal_count

    # --- Rahu transit (malefic amplifier) ---
    rahu_sign = _transit_sign(midpoint_jd, swe.MEAN_NODE)
    rahu_asp = {rahu_sign, (rahu_sign + 6) % 12}
    f['gc_rahu_on_danger'] = 1.0 if bool(rahu_asp & danger) else 0.0

    # --- Transit planet dignity (Phase 2) ---
    sat_lon = _transit_longitude(midpoint_jd, swe.SATURN)
    jup_lon = _transit_longitude(midpoint_jd, swe.JUPITER)
    mars_lon = _transit_longitude(midpoint_jd, swe.MARS)

    f['gc_sat_transit_uchcha'] = uchcha_bala('Saturn', sat_lon)
    f['gc_sat_transit_dignity'] = sign_dignity('Saturn', sat_lon)
    f['gc_jup_transit_uchcha'] = uchcha_bala('Jupiter', jup_lon)
    f['gc_jup_transit_dignity'] = sign_dignity('Jupiter', jup_lon)
    f['gc_mars_transit_uchcha'] = uchcha_bala('Mars', mars_lon)
    f['gc_mars_transit_dignity'] = sign_dignity('Mars', mars_lon)

    # Saturn strong (exalted/own) AND on maraka = peak destruction
    f['gc_sat_strong_on_maraka'] = 1.0 if (
        f['gc_sat_on_maraka'] and f['gc_sat_transit_dignity'] >= 0.75
    ) else 0.0
    # Jupiter weak (debilitated/enemy) = reduced protection
    f['gc_jup_weak'] = 1.0 if f['gc_jup_transit_dignity'] <= 0.25 else 0.0

    # --- Ashtakavarga filter ---
    # Saturn BAV in its transit sign (low = destructive)
    f['gc_sat_bav'] = sat_bav[sat_sign]
    f['gc_sat_bav_low'] = 1.0 if sat_bav[sat_sign] <= 2 else 0.0

    # Mars BAV in its transit sign
    mars_bav = gochar_ctx['mars_bav']
    f['gc_mars_bav'] = mars_bav[mars_sign]
    f['gc_mars_bav_low'] = 1.0 if mars_bav[mars_sign] <= 2 else 0.0

    # Sun BAV at Saturn's transit sign (Sun = father karaka)
    sun_bav = gochar_ctx['sun_bav']
    f['gc_sun_bav_at_sat'] = sun_bav[sat_sign]

    # SAV weakness of maraka signs (chart-level, constant per chart)
    f['gc_maraka_sav_min'] = gochar_ctx['maraka_sav_min']
    f['gc_maraka_sav_weak'] = gochar_ctx['maraka_sav_weak']

    # Interaction: Saturn on maraka AND that sign has low BAV
    f['gc_sat_on_weak_maraka'] = 1.0 if (
        f['gc_sat_on_maraka'] and f['gc_sat_bav_low']
    ) else 0.0

    # Mars on danger AND low BAV
    f['gc_mars_on_weak_danger'] = 1.0 if (
        f['gc_mars_asp_danger'] and f['gc_mars_bav_low']
    ) else 0.0

    # Double transit + low SAV interaction
    f['gc_double_transit_weak'] = 1.0 if (
        f['gc_double_transit_maraka'] and f['gc_maraka_sav_weak']
    ) else 0.0

    # --- Transit speed features (Phase 5) ---
    # Slow/stationary planet = prolonged transit contact = stronger effect
    mars_speed = abs(_transit_speed(midpoint_jd, swe.MARS))
    sat_speed = abs(_transit_speed(midpoint_jd, swe.SATURN))
    f['gc_mars_speed'] = mars_speed
    f['gc_sat_speed'] = sat_speed
    # Mars normally ~0.5 deg/day; near station < 0.1
    f['gc_mars_stationary'] = 1.0 if mars_speed < 0.1 else 0.0
    # Saturn normally ~0.03-0.07 deg/day; near station < 0.01
    f['gc_sat_stationary'] = 1.0 if sat_speed < 0.01 else 0.0

    # --- Degree-level transit features (Phase C) ---
    # Continuous angular distance to key natal points — varies smoothly
    # across adjacent SDs unlike sign-based binary features.
    natal_sun = gochar_ctx['natal_sun_deg']
    natal_moon = gochar_ctx['natal_moon_deg']
    h9_cusp = gochar_ctx['h9_cusp_deg']

    def _arc(a, b):
        """Shortest angular distance (0-180)."""
        d = abs(a - b) % 360
        return min(d, 360 - d)

    # Midpoint longitudes (sat_lon, jup_lon, mars_lon already computed above)
    merc_lon = _transit_longitude(midpoint_jd, swe.MERCURY)
    venus_lon = _transit_longitude(midpoint_jd, swe.VENUS)

    # Saturn distances to natal points
    f['gc_sat_dist_sun'] = _arc(sat_lon, natal_sun)
    f['gc_sat_dist_h9'] = _arc(sat_lon, h9_cusp)
    f['gc_sat_dist_moon'] = _arc(sat_lon, natal_moon)

    # Jupiter distances
    f['gc_jup_dist_sun'] = _arc(jup_lon, natal_sun)
    f['gc_jup_dist_h9'] = _arc(jup_lon, h9_cusp)
    f['gc_jup_dist_moon'] = _arc(jup_lon, natal_moon)

    # Mars distances
    f['gc_mars_dist_sun'] = _arc(mars_lon, natal_sun)
    f['gc_mars_dist_h9'] = _arc(mars_lon, h9_cusp)
    f['gc_mars_dist_moon'] = _arc(mars_lon, natal_moon)

    # Mercury distances
    f['gc_merc_dist_sun'] = _arc(merc_lon, natal_sun)
    f['gc_merc_dist_h9'] = _arc(merc_lon, h9_cusp)

    # Venus distances
    f['gc_venus_dist_sun'] = _arc(venus_lon, natal_sun)
    f['gc_venus_dist_h9'] = _arc(venus_lon, h9_cusp)

    # Tight conjunction features (within 5 degrees = strong contact)
    f['gc_sat_tight_sun'] = 1.0 if _arc(sat_lon, natal_sun) <= 5 else 0.0
    f['gc_sat_tight_h9'] = 1.0 if _arc(sat_lon, h9_cusp) <= 5 else 0.0
    f['gc_jup_tight_sun'] = 1.0 if _arc(jup_lon, natal_sun) <= 5 else 0.0
    f['gc_mars_tight_sun'] = 1.0 if _arc(mars_lon, natal_sun) <= 5 else 0.0

    # Double transit at degree level (both within 15 deg of same point)
    f['gc_double_deg_h9'] = 1.0 if (
        _arc(sat_lon, h9_cusp) <= 15 and _arc(jup_lon, h9_cusp) <= 15
    ) else 0.0
    f['gc_double_deg_sun'] = 1.0 if (
        _arc(sat_lon, natal_sun) <= 15 and _arc(jup_lon, natal_sun) <= 15
    ) else 0.0

    # --- Within-SD planet movement + ingress (Phase C) ---
    start_jd = candidate['start_jd']
    end_jd = candidate['end_jd']

    # Fast planet positions at SD start and end
    mars_start = _transit_longitude(start_jd, swe.MARS)
    mars_end = _transit_longitude(end_jd, swe.MARS)
    merc_start = _transit_longitude(start_jd, swe.MERCURY)
    merc_end = _transit_longitude(end_jd, swe.MERCURY)
    venus_start = _transit_longitude(start_jd, swe.VENUS)
    venus_end = _transit_longitude(end_jd, swe.VENUS)

    # Degrees moved during this SD (absolute)
    f['gc_mars_movement'] = _arc(mars_start, mars_end)
    f['gc_merc_movement'] = _arc(merc_start, merc_end)
    f['gc_venus_movement'] = _arc(venus_start, venus_end)

    # Ingress detection: planet changes sign during this SD
    f['gc_mars_ingress'] = 1.0 if int(mars_start / 30) % 12 != int(mars_end / 30) % 12 else 0.0
    f['gc_merc_ingress'] = 1.0 if int(merc_start / 30) % 12 != int(merc_end / 30) % 12 else 0.0
    f['gc_venus_ingress'] = 1.0 if int(venus_start / 30) % 12 != int(venus_end / 30) % 12 else 0.0
    f['gc_any_fast_ingress'] = f['gc_mars_ingress'] + f['gc_merc_ingress'] + f['gc_venus_ingress']

    # Does a fast planet cross natal Sun during this SD?
    # Check if natal Sun degree is between start and end longitudes
    def _crosses(start_lon, end_lon, target):
        """Check if target degree is swept during start→end movement."""
        d_start = _arc(start_lon, target)
        d_end = _arc(end_lon, target)
        movement = _arc(start_lon, end_lon)
        # If start-to-target + target-to-end ≈ start-to-end, target is in path
        return (d_start + d_end) <= movement + 1.0  # 1-degree tolerance

    f['gc_mars_crosses_sun'] = 1.0 if _crosses(mars_start, mars_end, natal_sun) else 0.0
    f['gc_mars_crosses_h9'] = 1.0 if _crosses(mars_start, mars_end, h9_cusp) else 0.0
    f['gc_merc_crosses_sun'] = 1.0 if _crosses(merc_start, merc_end, natal_sun) else 0.0

    # --- Moon transit features (Step 2) ---
    # Moon moves ~13 deg/day → unique position per PD midpoint → high variance
    moon_lon = _transit_longitude(midpoint_jd, swe.MOON)
    moon_sign = int(moon_lon / 30) % 12

    f['gc_moon_on_maraka'] = 1.0 if moon_sign in father_mk else 0.0
    f['gc_moon_on_danger'] = 1.0 if moon_sign in danger else 0.0
    f['gc_moon_dist_natal_moon'] = _arc(moon_lon, natal_moon)
    f['gc_moon_dist_sun'] = _arc(moon_lon, natal_sun)
    f['gc_moon_dist_h9'] = _arc(moon_lon, h9_cusp)
    f['gc_moon_near_natal_moon'] = 1.0 if _arc(moon_lon, natal_moon) <= 15 else 0.0

    # Moon-malefic conjunctions at PD midpoint
    f['gc_moon_conj_saturn'] = 1.0 if _arc(moon_lon, sat_lon) <= 15 else 0.0
    f['gc_moon_conj_mars'] = 1.0 if _arc(moon_lon, mars_lon) <= 15 else 0.0
    t_rahu_lon = _transit_longitude(midpoint_jd, swe.MEAN_NODE)
    f['gc_moon_conj_rahu'] = 1.0 if _arc(moon_lon, t_rahu_lon) <= 15 else 0.0

    # --- Speed anomaly features (Step 3) ---
    # Instantaneous speed at midpoint (not movement = duration proxy)
    merc_speed = _transit_speed(midpoint_jd, swe.MERCURY)
    venus_speed = _transit_speed(midpoint_jd, swe.VENUS)
    f['gc_merc_speed'] = abs(merc_speed)
    f['gc_venus_speed'] = abs(venus_speed)

    # Transit retrograde flags (NOT natal retrograde — that's in retrograde_features)
    f['gc_mars_retro'] = 1.0 if _transit_speed(midpoint_jd, swe.MARS) < 0 else 0.0
    f['gc_merc_retro'] = 1.0 if merc_speed < 0 else 0.0
    f['gc_venus_retro'] = 1.0 if venus_speed < 0 else 0.0

    # Speed anomalies (z-score from average daily motion)
    f['gc_mars_speed_anom'] = (abs(_transit_speed(midpoint_jd, swe.MARS)) - 0.52) / 0.30
    f['gc_merc_speed_anom'] = (abs(merc_speed) - 1.0) / 0.50
    f['gc_venus_speed_anom'] = (abs(venus_speed) - 1.0) / 0.30

    # --- Prong 1b: Continuous Parashari aspect strength (transit to natal) ---
    # Smooth 0-1 via Gaussian kernel — much better for LambdaRank than binary
    f['gc_sat_asp_str_sun'] = aspect_strength('Saturn', 'Sun', sat_lon, natal_sun)
    f['gc_sat_asp_str_h9'] = aspect_strength('Saturn', 'point', sat_lon, h9_cusp)
    f['gc_jup_asp_str_sun'] = aspect_strength('Jupiter', 'Sun', jup_lon, natal_sun)
    f['gc_jup_asp_str_h9'] = aspect_strength('Jupiter', 'point', jup_lon, h9_cusp)
    f['gc_mars_asp_str_sun'] = aspect_strength('Mars', 'Sun', mars_lon, natal_sun)
    f['gc_mars_asp_str_h9'] = aspect_strength('Mars', 'point', mars_lon, h9_cusp)

    # --- Prong 1c: Mercury/Venus BAV at transit sign ---
    merc_bav_arr = gochar_ctx['merc_bav']
    venus_bav_arr = gochar_ctx['venus_bav']
    f['gc_merc_bav'] = merc_bav_arr[merc_sign]
    f['gc_venus_bav'] = venus_bav_arr[venus_sign]
    f['gc_merc_bav_low'] = 1.0 if merc_bav_arr[merc_sign] <= 2 else 0.0

    # --- Prong 2a: Mrityu Bhaga (death degrees) ---
    sat_deg_in_sign = sat_lon % 30
    mars_deg_in_sign = mars_lon % 30
    f['gc_sat_mrityu_dist'] = abs(sat_deg_in_sign - _MRITYU_BHAGA[sat_sign])
    f['gc_sat_near_mrityu'] = 1.0 if f['gc_sat_mrityu_dist'] <= 3 else 0.0
    f['gc_mars_mrityu_dist'] = abs(mars_deg_in_sign - _MRITYU_BHAGA[mars_sign])
    f['gc_mars_near_mrityu'] = 1.0 if f['gc_mars_mrityu_dist'] <= 3 else 0.0

    # --- Prong 2b: Kakshya sub-divisions (3.75 deg segments) ---
    # 8 kakshyas per sign, ruled by: Sat, Jup, Mars, Sun, Ven, Merc, Moon, Lagna-lord
    lagna_lord = gochar_ctx['lagna_lord']
    kakshya_lords = _KAKSHYA_LORDS_BASE + [lagna_lord]

    sat_kakshya_idx = int(sat_deg_in_sign / 3.75) % 8
    sat_kakshya_lord = kakshya_lords[sat_kakshya_idx]
    f['gc_sat_kakshya_maraka'] = 1.0 if sat_kakshya_lord in father_mk else 0.0
    f['gc_sat_kakshya_self'] = 1.0 if sat_kakshya_lord == 'Saturn' else 0.0
    f['gc_sat_kakshya_malefic'] = 1.0 if sat_kakshya_lord in _NATURAL_MALEFICS else 0.0

    jup_deg_in_sign = jup_lon % 30
    jup_kakshya_idx = int(jup_deg_in_sign / 3.75) % 8
    jup_kakshya_lord = kakshya_lords[jup_kakshya_idx]
    f['gc_jup_kakshya_maraka'] = 1.0 if jup_kakshya_lord in father_mk else 0.0
    f['gc_jup_kakshya_self'] = 1.0 if jup_kakshya_lord == 'Jupiter' else 0.0

    # --- Prong 2c: Zero-bindu catastrophe ---
    f['gc_sat_zero_bindu'] = 1.0 if sat_bav[sat_sign] == 0 else 0.0
    f['gc_mars_zero_bindu'] = 1.0 if mars_bav[mars_sign] == 0 else 0.0
    zero_count = int(sat_bav[sat_sign] == 0) + int(mars_bav[mars_sign] == 0)
    if merc_bav_arr[merc_sign] == 0:
        zero_count += 1
    if venus_bav_arr[venus_sign] == 0:
        zero_count += 1
    f['gc_zero_bindu_count'] = zero_count

    return f
