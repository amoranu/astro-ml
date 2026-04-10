"""KP-native sub-sub level features for Sookshma (depth=4) candidates.

Pure KP techniques only — no Parashari maraka tier classification, no
multi-reference ascendants, no D12 Saptamsha.

KP techniques implemented:
  1. 4-level significator chain (MD/AD/PD/SD)
  2. Cusp Sub-Lord (CSL) of 9th house death analysis
  3. Ruling Planets at SD midpoint (Day Lord, Moon Star/Sub Lord, Lagna Sub Lord)
  4. Star Lord and Sub Lord of each dasha lord
  5. Badhaka lord activation across the chain
  6. Significator strength via 4-level KP chain depth

Features per candidate: ~50.
"""

import swisseph as swe

from ..astro_engine.houses import get_sign, get_sign_lord, get_house_number
from ..astro_engine.dasha import NAKSHATRA_LORDS, DASHA_SEQUENCE
from .kp_features import (
    FATHER_DEATH_HOUSES, FATHER_MARAKA_HOUSES,
    _get_nakshatra_lord, _SIGN_QUALITY, _BADHAKA, ALL_PLANETS,
)

swe.set_sid_mode(swe.SIDM_LAHIRI)

# Day-of-week → ruling planet (Vara). 0=Mon, 1=Tue, ... 6=Sun
_VARA_LORDS = ['Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Sun']

# Nakshatra sub-lord (Vimshottari sub-divisions of each nakshatra).
# Sub-division proportional to dasha years: Ke=7, Ve=20, Su=6, Mo=10, Ma=7,
# Ra=18, Ju=16, Sa=19, Me=17 (total = 120)
_SUB_LORD_BOUNDARIES = []  # list of (boundary_in_nakshatra_fraction, lord)

def _build_sub_lord_table():
    """Each nakshatra is subdivided into 9 sub-portions in dasha-year ratio.

    Sub-lord starts from the nakshatra lord itself.
    """
    NAK_SPAN = 360.0 / 27.0  # 13.3333°
    DASHA_YEARS = {
        'Ketu': 7, 'Venus': 20, 'Sun': 6, 'Moon': 10, 'Mars': 7,
        'Rahu': 18, 'Jupiter': 16, 'Saturn': 19, 'Mercury': 17
    }
    TOTAL = 120.0
    table = []
    for nak_idx in range(27):
        nak_lord = NAKSHATRA_LORDS[nak_idx]
        nak_start = nak_idx * NAK_SPAN
        # Sub-sequence starts from nak_lord
        start_pos = DASHA_SEQUENCE.index(nak_lord)
        seq = DASHA_SEQUENCE[start_pos:] + DASHA_SEQUENCE[:start_pos]
        cur_offset = 0.0
        for sub in seq:
            sub_span = NAK_SPAN * (DASHA_YEARS[sub] / TOTAL)
            sub_start = nak_start + cur_offset
            sub_end = sub_start + sub_span
            table.append((sub_start, sub_end, nak_lord, sub))
            cur_offset += sub_span
    return table

_SUB_LORD_TABLE = _build_sub_lord_table()


def _get_sub_lord(longitude):
    """Return (star_lord, sub_lord) for a sidereal longitude."""
    long_norm = longitude % 360.0
    for sub_start, sub_end, star, sub in _SUB_LORD_TABLE:
        if sub_start <= long_norm < sub_end:
            return star, sub
    # Fallback (shouldn't happen)
    return _get_nakshatra_lord(long_norm), _get_nakshatra_lord(long_norm)


def precompute_kp_native_context(natal_chart, natal_asc, birth_jd):
    """Precompute KP context: significator chains for all 12 houses,
    star/sub lord of each natal planet, CSL of all 12 cusps.
    """
    asc_sign = get_sign(natal_asc)

    # Planet positions in houses
    planet_in_house = {}
    for planet in ALL_PLANETS:
        if planet in natal_chart:
            h = get_house_number(natal_chart[planet]['longitude'], natal_asc)
            planet_in_house[planet] = h

    # 4-level significators per house
    house_significators = {}
    for house in range(1, 13):
        sigs = set()
        # Level 4: House lord
        house_sign = (asc_sign + house - 1) % 12
        house_lord = get_sign_lord(house_sign)
        sigs.add(house_lord)
        # Level 3: Star lord of house lord
        if house_lord in natal_chart:
            star_of_lord = _get_nakshatra_lord(
                natal_chart[house_lord]['longitude'])
            sigs.add(star_of_lord)
        # Level 2: planets in house
        for planet, h in planet_in_house.items():
            if h == house:
                sigs.add(planet)
                star_of_occ = _get_nakshatra_lord(
                    natal_chart[planet]['longitude'])
                sigs.add(star_of_occ)
        house_significators[house] = sigs

    # Father death significator sets
    father_death_sigs = set()
    for h in FATHER_DEATH_HOUSES:
        father_death_sigs |= house_significators.get(h, set())
    father_maraka_sigs = set()
    for h in FATHER_MARAKA_HOUSES:
        father_maraka_sigs |= house_significators.get(h, set())

    # CSL of each cusp (equal house system)
    cusp_sub_lords = {}
    cusp_star_lords = {}
    for h in range(1, 13):
        cusp_long = (natal_asc + (h - 1) * 30) % 360
        star, sub = _get_sub_lord(cusp_long)
        cusp_star_lords[h] = star
        cusp_sub_lords[h] = sub

    # Star and Sub lord of each natal planet
    planet_star = {}
    planet_sub = {}
    for planet in ALL_PLANETS:
        if planet in natal_chart:
            star, sub = _get_sub_lord(natal_chart[planet]['longitude'])
            planet_star[planet] = star
            planet_sub[planet] = sub

    # Badhaka for 9th house
    h9_sign = (asc_sign + 8) % 12
    h9_quality = _SIGN_QUALITY[h9_sign]
    badhaka_house_from_9 = _BADHAKA[h9_quality]
    badhaka_sign = (h9_sign + badhaka_house_from_9 - 1) % 12
    badhaka_lord = get_sign_lord(badhaka_sign)

    # Birth day-of-week (Vara)
    # JD of true noon = birth_jd. Day of week: (jd + 1.5) % 7
    # JD 0 = Monday noon. Convert to weekday index (0=Mon).
    weekday = int((birth_jd + 0.5) % 7)  # 0=Mon, 6=Sun
    natal_vara = _VARA_LORDS[weekday]

    return {
        'house_significators': house_significators,
        'father_death_sigs': father_death_sigs,
        'father_maraka_sigs': father_maraka_sigs,
        'cusp_sub_lords': cusp_sub_lords,
        'cusp_star_lords': cusp_star_lords,
        'planet_star': planet_star,
        'planet_sub': planet_sub,
        'badhaka_lord': badhaka_lord,
        'natal_vara': natal_vara,
        'natal_asc': natal_asc,
        'asc_sign': asc_sign,
        'planet_in_house': planet_in_house,
    }


def extract_kp_native_features(candidate, kp_ctx, natal_chart):
    """KP-native features for a Sookshma (depth=4) candidate.

    Returns ~50 features. NO Parashari maraka tier classification.
    """
    f_death = kp_ctx['father_death_sigs']
    f_maraka = kp_ctx['father_maraka_sigs']
    h_sigs = kp_ctx['house_significators']
    cusp_sub = kp_ctx['cusp_sub_lords']
    cusp_star = kp_ctx['cusp_star_lords']
    planet_star = kp_ctx['planet_star']
    planet_sub = kp_ctx['planet_sub']
    badhaka = kp_ctx['badhaka_lord']
    natal_vara = kp_ctx['natal_vara']

    lords = candidate['lords']
    md = lords[0]
    ad = lords[1] if len(lords) > 1 else md
    pd = lords[2] if len(lords) > 2 else ad
    sd = lords[-1]  # depth=4

    f = {}

    # ── 1. Significator status of each chain level ──────────────────
    for name, lord in [('md', md), ('ad', ad), ('pd', pd), ('sd', sd)]:
        f[f'kp_{name}_is_death'] = 1.0 if lord in f_death else 0.0
        f[f'kp_{name}_is_maraka'] = 1.0 if lord in f_maraka else 0.0
        # Specific death houses signified
        for h in [10, 3, 4, 8]:
            f[f'kp_{name}_sig_h{h}'] = (
                1.0 if lord in h_sigs.get(h, set()) else 0.0)
        f[f'kp_{name}_death_count'] = sum(
            lord in h_sigs.get(h, set()) for h in FATHER_DEATH_HOUSES)

    # ── 2. 4-level cascade ──────────────────────────────────────────
    chain_count = sum([
        md in f_death, ad in f_death, pd in f_death, sd in f_death])
    f['kp_4chain_count'] = chain_count
    f['kp_all_4_death'] = 1.0 if chain_count == 4 else 0.0
    f['kp_atleast_3_death'] = 1.0 if chain_count >= 3 else 0.0
    f['kp_md_ad_pd_death'] = 1.0 if (
        md in f_death and ad in f_death and pd in f_death) else 0.0
    f['kp_pd_sd_death'] = 1.0 if (
        pd in f_death and sd in f_death) else 0.0
    f['kp_ad_sd_death'] = 1.0 if (
        ad in f_death and sd in f_death) else 0.0

    # ── 3. CSL of 9th house (father's lagna) ────────────────────────
    h9_csl = cusp_sub.get(9)
    h9_csstar = cusp_star.get(9)
    f['kp_csl9_is_death'] = 1.0 if h9_csl in f_death else 0.0
    f['kp_csl9_is_maraka'] = 1.0 if h9_csl in f_maraka else 0.0
    f['kp_csstar9_is_death'] = 1.0 if h9_csstar in f_death else 0.0
    f['kp_csl9_in_chain'] = 1.0 if h9_csl in (md, ad, pd, sd) else 0.0
    f['kp_csl9_eq_sd'] = 1.0 if h9_csl == sd else 0.0
    f['kp_csl9_eq_pd'] = 1.0 if h9_csl == pd else 0.0
    f['kp_csstar9_in_chain'] = 1.0 if h9_csstar in (md, ad, pd, sd) else 0.0

    # ── 4. CSL of all death houses (10, 3, 4, 8) ─────────────────────
    death_house_csls = {cusp_sub.get(h) for h in FATHER_DEATH_HOUSES}
    f['kp_death_csl_count_in_chain'] = sum(
        1 for csl in death_house_csls if csl in (md, ad, pd, sd))
    death_house_csls_in_death_sigs = sum(
        1 for csl in death_house_csls if csl in f_death)
    f['kp_death_csl_self_referential'] = float(death_house_csls_in_death_sigs)

    # Per-house CSL analysis (each of the 4 death houses)
    for h in [10, 3, 4, 8]:
        h_csl = cusp_sub.get(h)
        h_csstar = cusp_star.get(h)
        f[f'kp_csl{h}_in_chain'] = (
            1.0 if h_csl in (md, ad, pd, sd) else 0.0)
        f[f'kp_csl{h}_eq_sd'] = 1.0 if h_csl == sd else 0.0
        f[f'kp_csl{h}_is_death'] = 1.0 if h_csl in f_death else 0.0
        f[f'kp_csstar{h}_in_chain'] = (
            1.0 if h_csstar in (md, ad, pd, sd) else 0.0)

    # ── 5. Star Lord / Sub Lord of dasha lords ──────────────────────
    for name, lord in [('pd', pd), ('sd', sd)]:
        star = planet_star.get(lord, '')
        sub = planet_sub.get(lord, '')
        f[f'kp_{name}_star_is_death'] = (
            1.0 if star and star in f_death else 0.0)
        f[f'kp_{name}_sub_is_death'] = (
            1.0 if sub and sub in f_death else 0.0)
        f[f'kp_{name}_star_eq_sd'] = 1.0 if star == sd else 0.0
        f[f'kp_{name}_sub_eq_sd'] = 1.0 if sub == sd else 0.0

    # ── 6. Badhaka lord activation ──────────────────────────────────
    f['kp_badhaka_in_chain'] = 1.0 if badhaka in (md, ad, pd, sd) else 0.0
    f['kp_badhaka_eq_sd'] = 1.0 if badhaka == sd else 0.0
    f['kp_badhaka_eq_pd'] = 1.0 if badhaka == pd else 0.0

    # ── 7. Ruling Planets at SD midpoint ────────────────────────────
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2.0
    weekday = int((midpoint_jd + 0.5) % 7)
    transit_vara = _VARA_LORDS[weekday]

    moon_pos = swe.calc_ut(midpoint_jd, swe.MOON, swe.FLG_SIDEREAL)[0][0]
    moon_star, moon_sub = _get_sub_lord(moon_pos)

    rps = {transit_vara, moon_star, moon_sub}
    f['kp_rp_count'] = len(rps)
    f['kp_rp_death_count'] = sum(1 for r in rps if r in f_death)
    f['kp_rp_chain_overlap'] = sum(
        1 for r in rps if r in (md, ad, pd, sd))
    f['kp_vara_is_death'] = 1.0 if transit_vara in f_death else 0.0
    f['kp_vara_in_chain'] = 1.0 if transit_vara in (md, ad, pd, sd) else 0.0
    f['kp_moon_star_is_death'] = (
        1.0 if moon_star in f_death else 0.0)
    f['kp_moon_sub_is_death'] = 1.0 if moon_sub in f_death else 0.0
    f['kp_moon_star_in_chain'] = (
        1.0 if moon_star in (md, ad, pd, sd) else 0.0)
    f['kp_moon_sub_in_chain'] = (
        1.0 if moon_sub in (md, ad, pd, sd) else 0.0)
    f['kp_natal_vara_match'] = 1.0 if natal_vara == transit_vara else 0.0
    f['kp_natal_vara_in_chain'] = (
        1.0 if natal_vara in (md, ad, pd, sd) else 0.0)

    # ── 8. CSL movement: where is transit Saturn currently? ────────
    sat_pos = swe.calc_ut(midpoint_jd, swe.SATURN, swe.FLG_SIDEREAL)[0][0]
    sat_star, sat_sub = _get_sub_lord(sat_pos)
    f['kp_tsat_sub_is_death'] = 1.0 if sat_sub in f_death else 0.0
    f['kp_tsat_star_is_death'] = 1.0 if sat_star in f_death else 0.0
    f['kp_tsat_sub_in_chain'] = 1.0 if sat_sub in (md, ad, pd, sd) else 0.0
    f['kp_tsat_sub_eq_h9csl'] = 1.0 if sat_sub == h9_csl else 0.0

    # ── 9. Transit Jupiter sub lord ──────────────────────────────────
    jup_pos = swe.calc_ut(midpoint_jd, swe.JUPITER, swe.FLG_SIDEREAL)[0][0]
    _, jup_sub = _get_sub_lord(jup_pos)
    f['kp_tjup_sub_is_death'] = 1.0 if jup_sub in f_death else 0.0
    f['kp_tjup_sub_in_chain'] = 1.0 if jup_sub in (md, ad, pd, sd) else 0.0

    # ── 10. Transit Rahu sub lord (eclipse axis activator) ──────────
    rahu_pos = swe.calc_ut(midpoint_jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)[0][0]
    _, rahu_sub = _get_sub_lord(rahu_pos)
    f['kp_trahu_sub_is_death'] = 1.0 if rahu_sub in f_death else 0.0
    f['kp_trahu_sub_in_chain'] = 1.0 if rahu_sub in (md, ad, pd, sd) else 0.0
    f['kp_trahu_sub_eq_h9csl'] = 1.0 if rahu_sub == h9_csl else 0.0

    # ── 11. Continuous KP distances (angular, in degrees) ──────────
    # Distance from natal PD lord position to natal 9th CSL position.
    # Uses min(d, 360-d) to fold the circle.
    def _angle_dist(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    natal_asc = kp_ctx['natal_asc']
    h9_cusp_long = (natal_asc + 8 * 30) % 360

    # Natal positions of dasha lords
    if pd in natal_chart:
        pd_long = natal_chart[pd]['longitude']
        f['kp_pd_to_h9cusp_dist'] = _angle_dist(pd_long, h9_cusp_long)
        f['kp_pd_to_natal_sun_dist'] = _angle_dist(
            pd_long, natal_chart['Sun']['longitude'])
    else:
        f['kp_pd_to_h9cusp_dist'] = -1.0
        f['kp_pd_to_natal_sun_dist'] = -1.0

    if sd in natal_chart:
        sd_long = natal_chart[sd]['longitude']
        f['kp_sd_to_h9cusp_dist'] = _angle_dist(sd_long, h9_cusp_long)
        f['kp_sd_to_natal_sun_dist'] = _angle_dist(
            sd_long, natal_chart['Sun']['longitude'])
        f['kp_sd_to_natal_moon_dist'] = _angle_dist(
            sd_long, natal_chart['Moon']['longitude'])
    else:
        f['kp_sd_to_h9cusp_dist'] = -1.0
        f['kp_sd_to_natal_sun_dist'] = -1.0
        f['kp_sd_to_natal_moon_dist'] = -1.0

    # Transit Saturn / Jupiter / Rahu distance to 9th cusp
    f['kp_tsat_to_h9cusp_dist'] = _angle_dist(sat_pos, h9_cusp_long)
    f['kp_tjup_to_h9cusp_dist'] = _angle_dist(jup_pos, h9_cusp_long)
    f['kp_trahu_to_h9cusp_dist'] = _angle_dist(rahu_pos, h9_cusp_long)

    # ── 12. SD lord transit at midpoint (KP timing technique) ──────
    # When the SD lord is itself transiting through a death-significator's
    # sub-lord position, the timing is exact.
    sd_swe_id = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mars': swe.MARS,
        'Mercury': swe.MERCURY, 'Jupiter': swe.JUPITER, 'Venus': swe.VENUS,
        'Saturn': swe.SATURN, 'Rahu': swe.MEAN_NODE,
    }
    if sd in sd_swe_id:
        sd_t_pos = swe.calc_ut(midpoint_jd, sd_swe_id[sd], swe.FLG_SIDEREAL)[0][0]
        sd_t_star, sd_t_sub = _get_sub_lord(sd_t_pos)
        f['kp_tsd_sub_is_death'] = 1.0 if sd_t_sub in f_death else 0.0
        f['kp_tsd_sub_in_chain'] = (
            1.0 if sd_t_sub in (md, ad, pd, sd) else 0.0)
        f['kp_tsd_sub_eq_h9csl'] = 1.0 if sd_t_sub == h9_csl else 0.0
        f['kp_tsd_to_h9cusp_dist'] = _angle_dist(sd_t_pos, h9_cusp_long)
    else:
        f['kp_tsd_sub_is_death'] = 0.0
        f['kp_tsd_sub_in_chain'] = 0.0
        f['kp_tsd_sub_eq_h9csl'] = 0.0
        f['kp_tsd_to_h9cusp_dist'] = -1.0

    # PD lord transit at midpoint (changes every PD ~62 days)
    if pd in sd_swe_id:
        pd_t_pos = swe.calc_ut(midpoint_jd, sd_swe_id[pd], swe.FLG_SIDEREAL)[0][0]
        _, pd_t_sub = _get_sub_lord(pd_t_pos)
        f['kp_tpd_sub_is_death'] = 1.0 if pd_t_sub in f_death else 0.0
        f['kp_tpd_to_h9cusp_dist'] = _angle_dist(pd_t_pos, h9_cusp_long)
    else:
        f['kp_tpd_sub_is_death'] = 0.0
        f['kp_tpd_to_h9cusp_dist'] = -1.0

    # ── 13. Cuspal Interlinks: 9th CSL → its sub lord → death? ─────
    # Multi-hop: take h9_csl's natal position, find ITS sub lord, check
    # if that's a death sig.
    if h9_csl in natal_chart:
        h9_csl_long = natal_chart[h9_csl]['longitude']
        h9csl_star, h9csl_sub = _get_sub_lord(h9_csl_long)
        f['kp_h9csl_sub_is_death'] = (
            1.0 if h9csl_sub in f_death else 0.0)
        f['kp_h9csl_sub_in_chain'] = (
            1.0 if h9csl_sub in (md, ad, pd, sd) else 0.0)
        f['kp_h9csl_star_is_death'] = (
            1.0 if h9csl_star in f_death else 0.0)
    else:
        f['kp_h9csl_sub_is_death'] = 0.0
        f['kp_h9csl_sub_in_chain'] = 0.0
        f['kp_h9csl_star_is_death'] = 0.0

    # ── 13b. SD lord natal house + dignity + nakshatra-owner features ──
    # These vary by SD lord identity, providing within-PD discrimination.
    if sd in natal_chart:
        sd_natal_long = natal_chart[sd]['longitude']
        sd_natal_house = (
            int((sd_natal_long - natal_asc) / 30) % 12 + 1)
        f['kp_sd_natal_house'] = float(sd_natal_house)
        f['kp_sd_in_house9'] = 1.0 if sd_natal_house == 9 else 0.0
        f['kp_sd_in_death_house'] = (
            1.0 if sd_natal_house in (10, 3, 4, 8) else 0.0)
        # SD lord natal nakshatra index (categorical 0-26)
        sd_nak_idx = int(sd_natal_long / (360.0 / 27.0)) % 27
        f['kp_sd_natal_nak_idx'] = float(sd_nak_idx)
        # Distance of SD lord natal position to natal Moon (continuous)
        f['kp_sd_to_natal_asc_dist'] = _angle_dist(
            sd_natal_long, natal_asc)
    else:
        f['kp_sd_natal_house'] = 0.0
        f['kp_sd_in_house9'] = 0.0
        f['kp_sd_in_death_house'] = 0.0
        f['kp_sd_natal_nak_idx'] = -1.0
        f['kp_sd_to_natal_asc_dist'] = -1.0

    # PD lord natal house too (varies more slowly but adds info)
    if pd in natal_chart:
        pd_natal_long = natal_chart[pd]['longitude']
        pd_natal_house = (
            int((pd_natal_long - natal_asc) / 30) % 12 + 1)
        f['kp_pd_natal_house'] = float(pd_natal_house)
        f['kp_pd_in_death_house'] = (
            1.0 if pd_natal_house in (10, 3, 4, 8) else 0.0)
    else:
        f['kp_pd_natal_house'] = 0.0
        f['kp_pd_in_death_house'] = 0.0

    # ── 14. Count of all transit planets whose sub lord = death sig ─
    death_sub_count = 0
    for pos in (sat_pos, jup_pos, rahu_pos):
        _, sub = _get_sub_lord(pos)
        if sub in f_death:
            death_sub_count += 1
    f['kp_t_outer_death_sub_count'] = float(death_sub_count)

    # Mercury and Venus (faster planets, finer sub-lord transitions)
    merc_pos = swe.calc_ut(midpoint_jd, swe.MERCURY, swe.FLG_SIDEREAL)[0][0]
    ven_pos = swe.calc_ut(midpoint_jd, swe.VENUS, swe.FLG_SIDEREAL)[0][0]
    sun_pos = swe.calc_ut(midpoint_jd, swe.SUN, swe.FLG_SIDEREAL)[0][0]
    _, merc_sub = _get_sub_lord(merc_pos)
    _, ven_sub = _get_sub_lord(ven_pos)
    _, sun_sub = _get_sub_lord(sun_pos)
    f['kp_tmerc_sub_is_death'] = 1.0 if merc_sub in f_death else 0.0
    f['kp_tven_sub_is_death'] = 1.0 if ven_sub in f_death else 0.0
    f['kp_tsun_sub_is_death'] = 1.0 if sun_sub in f_death else 0.0
    f['kp_t_inner_death_sub_count'] = float(
        sum(s in f_death for s in (merc_sub, ven_sub, sun_sub)))
    f['kp_t_all_death_sub_count'] = (
        f['kp_t_outer_death_sub_count'] + f['kp_t_inner_death_sub_count'])

    # Continuous: transit planet distance to 9th cusp
    f['kp_tmerc_to_h9cusp_dist'] = _angle_dist(merc_pos, h9_cusp_long)
    f['kp_tven_to_h9cusp_dist'] = _angle_dist(ven_pos, h9_cusp_long)
    f['kp_tsun_to_h9cusp_dist'] = _angle_dist(sun_pos, h9_cusp_long)
    f['kp_tmoon_to_h9cusp_dist'] = _angle_dist(moon_pos, h9_cusp_long)

    # ── 15. Composite KP danger score ───────────────────────────────
    danger = 0.0
    danger += f['kp_4chain_count'] * 1.0
    danger += f['kp_csl9_in_chain'] * 1.5
    danger += f['kp_csl9_eq_sd'] * 1.5
    danger += f['kp_badhaka_in_chain'] * 0.8
    danger += f['kp_rp_death_count'] * 0.6
    danger += f['kp_tsat_sub_in_chain'] * 0.5
    danger += f['kp_trahu_sub_in_chain'] * 0.5
    danger += f['kp_tsd_sub_is_death'] * 1.2
    danger += f['kp_tsd_sub_eq_h9csl'] * 1.5
    danger += f['kp_h9csl_sub_in_chain'] * 0.8
    danger += f['kp_t_all_death_sub_count'] * 0.4
    f['kp_composite_danger'] = danger

    return f
