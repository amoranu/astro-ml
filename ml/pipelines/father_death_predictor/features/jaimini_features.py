"""Jaimini-native features for Chara Dasha pipeline (depth=3).

Sign-based system: each dasha activates a SIGN (not a planet).
Features check what planets occupy or aspect the dasha sign,
especially Chara Karakas (AK, DK, GnK, etc.).

Depth=3 (PD) features: ~60 features per candidate.
"""

import swisseph as swe

from ..astro_engine.houses import get_sign, get_sign_lord, get_house_number
from ..astro_engine.dignity import uchcha_bala, sign_dignity
from ..astro_engine.vargas import navamsha_sign
from .friendship import get_friendship

ALL_PLANETS = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter',
               'Venus', 'Saturn']  # No Rahu/Ketu for Chara Karakas
ALL_PLANETS_FULL = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter',
                    'Venus', 'Saturn', 'Rahu', 'Ketu']
NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}

swe.set_sid_mode(swe.SIDM_LAHIRI)

# Sign quality: 0=moveable, 1=fixed, 2=dual
_SIGN_QUALITY = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
SIGN_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
              'Libra', 'Scorpio', 'Sagittarius', 'Capricorn',
              'Aquarius', 'Pisces']


def _jaimini_aspect_signs(sign):
    """Jaimini Rashi Drishti: which signs does this sign aspect?

    Moveable (0,3,6,9): aspects Fixed signs except the one next to it
    Fixed (1,4,7,10): aspects Moveable signs except the one before it
    Dual (2,5,8,11): aspects other Dual signs
    """
    quality = sign % 3  # 0=moveable, 1=fixed, 2=dual
    if quality == 0:  # Moveable
        all_fixed = {1, 4, 7, 10}
        adjacent = (sign + 1) % 12
        return all_fixed - {adjacent}
    elif quality == 1:  # Fixed
        all_moveable = {0, 3, 6, 9}
        behind = (sign - 1) % 12
        return all_moveable - {behind}
    else:  # Dual
        all_dual = {2, 5, 8, 11}
        return all_dual - {sign}


def _arudha_pada(house_num, asc_sign, natal_chart):
    """Compute Arudha Pada of a house.

    Arudha = lord of house placed as many signs from house as it is from house.
    If Arudha falls in same sign or 7th from house, use 10th from house instead.
    """
    house_sign = (asc_sign + house_num - 1) % 12
    lord = get_sign_lord(house_sign)
    if lord in natal_chart:
        lord_sign = get_sign(natal_chart[lord]['longitude'])
    else:
        lord_sign = house_sign
    # Distance from house sign to lord
    dist = (lord_sign - house_sign) % 12
    # Arudha = lord_sign + dist = house_sign + 2*dist
    arudha = (house_sign + 2 * dist) % 12
    # Exception: if arudha is same as house sign or 7th from it
    if arudha == house_sign or arudha == (house_sign + 6) % 12:
        arudha = (house_sign + 9) % 12  # 10th from house
    return arudha


def precompute_jaimini_context(natal_chart, natal_asc):
    """Precompute Jaimini-specific chart data."""
    asc_sign = get_sign(natal_asc)
    h9_sign = (asc_sign + 8) % 12  # Father's house

    # Father maraka signs (2nd and 7th from 9th = natal 10th and 3rd)
    father_maraka_signs = {
        (h9_sign + 1) % 12,   # 2nd from 9th
        (h9_sign + 6) % 12,   # 7th from 9th
    }
    father_danger_signs = father_maraka_signs | {
        (h9_sign + 7) % 12,   # 8th from 9th
        (h9_sign + 11) % 12,  # 12th from 9th
    }
    # Dusthana signs from 9th (6th, 8th, 12th from 9th)
    father_dusthana_signs = {
        (h9_sign + 5) % 12,   # 6th from 9th
        (h9_sign + 7) % 12,   # 8th from 9th
        (h9_sign + 11) % 12,  # 12th from 9th
    }

    # Chara Karakas — sorted by longitude (highest = AK)
    planet_degrees = []
    for p in ALL_PLANETS:
        if p in natal_chart:
            lon = natal_chart[p]['longitude']
            planet_degrees.append((p, lon % 30))  # degree within sign

    planet_degrees.sort(key=lambda x: x[1], reverse=True)
    karaka_order = [p for p, _ in planet_degrees]

    # Karaka names: AK, AmK, BK, MK, PuK, GnK, DK
    karaka_names = ['AK', 'AmK', 'BK', 'MK', 'PuK', 'GnK', 'DK']
    karakas = {}
    karaka_to_planet = {}
    for i, name in enumerate(karaka_names):
        if i < len(karaka_order):
            karakas[name] = karaka_order[i]
            karaka_to_planet[name] = karaka_order[i]

    # Planets in each sign
    sign_occupants = {s: [] for s in range(12)}
    for p in ALL_PLANETS_FULL:
        if p in natal_chart:
            s = get_sign(natal_chart[p]['longitude'])
            sign_occupants[s].append(p)

    # Karakamsha = sign of AK in D9 (Navamsha)
    if 'AK' in karakas and karakas['AK'] in natal_chart:
        ak_lon = natal_chart[karakas['AK']]['longitude']
        karakamsha = navamsha_sign(ak_lon)
    else:
        karakamsha = 0

    # Arudha Pada of 9th house (A9 = Pitru Pada)
    a9_sign = _arudha_pada(9, asc_sign, natal_chart)

    # Father maraka lords (planets, not signs)
    father_maraka_lords = {get_sign_lord(s) for s in father_maraka_signs}

    # 9th lord (father's significator lord)
    h9_lord = get_sign_lord(h9_sign)

    # Sun's natal sign (natural pitrukaraka)
    sun_sign = get_sign(natal_chart['Sun']['longitude']) if 'Sun' in natal_chart else 0

    return {
        'h9_sign': h9_sign,
        'father_maraka_signs': father_maraka_signs,
        'father_danger_signs': father_danger_signs,
        'father_dusthana_signs': father_dusthana_signs,
        'karakas': karakas,
        'karaka_to_planet': karaka_to_planet,
        'sign_occupants': sign_occupants,
        'karakamsha': karakamsha,
        'asc_sign': asc_sign,
        'a9_sign': a9_sign,
        'father_maraka_lords': father_maraka_lords,
        'h9_lord': h9_lord,
        'sun_sign': sun_sign,
        'natal_asc': natal_asc,
    }


def extract_jaimini_features(candidate, jai_ctx, natal_chart):
    """Jaimini-native features for a Chara Dasha candidate.

    Supports depth=2 (AD) and depth=3 (PD).

    Args:
        candidate: Chara Dasha period with 'signs' field
        jai_ctx: from precompute_jaimini_context()
        natal_chart: from compute_chart()

    Returns: ~60 features dict.
    """
    father_mk = jai_ctx['father_maraka_signs']
    father_danger = jai_ctx['father_danger_signs']
    father_dusthana = jai_ctx['father_dusthana_signs']
    karakas = jai_ctx['karakas']
    occupants = jai_ctx['sign_occupants']
    karakamsha = jai_ctx['karakamsha']
    h9_sign = jai_ctx['h9_sign']
    asc_sign = jai_ctx['asc_sign']
    a9_sign = jai_ctx['a9_sign']
    father_mk_lords = jai_ctx['father_maraka_lords']

    signs = candidate['signs']
    md_sign = signs[0]
    depth = candidate.get('depth', len(signs))

    # Determine deepest level sign
    if depth >= 3 and len(signs) >= 3:
        ad_sign = signs[1]
        pd_sign = signs[2]
    elif depth >= 2 and len(signs) >= 2:
        ad_sign = signs[1]
        pd_sign = signs[-1]
    else:
        ad_sign = signs[-1]
        pd_sign = signs[-1]

    f = {}

    # ── MD Sign Features ──────────────────────────────────────────────
    f['jai_md_is_maraka'] = 1.0 if md_sign in father_mk else 0.0
    f['jai_md_is_danger'] = 1.0 if md_sign in father_danger else 0.0
    f['jai_md_is_h9'] = 1.0 if md_sign == h9_sign else 0.0

    # MD sign quality (moveable/fixed/dual)
    f['jai_md_quality'] = _SIGN_QUALITY[md_sign]

    # ── AD Sign Features ──────────────────────────────────────────────
    f['jai_ad_is_maraka'] = 1.0 if ad_sign in father_mk else 0.0
    f['jai_ad_is_danger'] = 1.0 if ad_sign in father_danger else 0.0
    f['jai_ad_is_h9'] = 1.0 if ad_sign == h9_sign else 0.0

    # ── PD Sign Features (NEW for depth=3) ────────────────────────────
    f['jai_pd_is_maraka'] = 1.0 if pd_sign in father_mk else 0.0
    f['jai_pd_is_danger'] = 1.0 if pd_sign in father_danger else 0.0
    f['jai_pd_is_h9'] = 1.0 if pd_sign == h9_sign else 0.0
    f['jai_pd_is_dusthana'] = 1.0 if pd_sign in father_dusthana else 0.0

    # PD sign quality
    f['jai_pd_quality'] = _SIGN_QUALITY[pd_sign]

    # Distance from PD sign to 9th house
    f['jai_pd_dist_h9'] = (pd_sign - h9_sign) % 12

    # ── 3-Level Cascade ───────────────────────────────────────────────
    mk_count = sum([md_sign in father_mk, ad_sign in father_mk,
                    pd_sign in father_mk])
    f['jai_maraka_cascade_3'] = mk_count

    dg_count = sum([md_sign in father_danger, ad_sign in father_danger,
                    pd_sign in father_danger])
    f['jai_danger_cascade_3'] = dg_count

    f['jai_all_three_maraka'] = 1.0 if mk_count == 3 else 0.0
    f['jai_all_three_danger'] = 1.0 if dg_count == 3 else 0.0
    f['jai_both_md_pd_maraka'] = 1.0 if (
        md_sign in father_mk and pd_sign in father_mk) else 0.0
    f['jai_both_ad_pd_maraka'] = 1.0 if (
        ad_sign in father_mk and pd_sign in father_mk) else 0.0
    f['jai_both_md_pd_danger'] = 1.0 if (
        md_sign in father_danger and pd_sign in father_danger) else 0.0

    # ── Rashi Drishti (Jaimini Sign Aspects) ──────────────────────────
    pd_aspects = _jaimini_aspect_signs(pd_sign)
    ad_aspects = _jaimini_aspect_signs(ad_sign)
    md_aspects = _jaimini_aspect_signs(md_sign)

    # PD sign aspects
    f['jai_pd_aspects_h9'] = 1.0 if h9_sign in pd_aspects else 0.0
    f['jai_pd_aspects_maraka'] = 1.0 if bool(pd_aspects & father_mk) else 0.0
    f['jai_pd_aspects_danger'] = 1.0 if bool(pd_aspects & father_danger) else 0.0

    # AD sign aspects
    f['jai_ad_aspects_h9'] = 1.0 if h9_sign in ad_aspects else 0.0
    f['jai_ad_aspects_maraka'] = 1.0 if bool(ad_aspects & father_mk) else 0.0

    # Cross-level: does MD aspect PD sign? AD aspect PD sign?
    f['jai_md_aspects_pd'] = 1.0 if pd_sign in md_aspects else 0.0
    f['jai_ad_aspects_pd'] = 1.0 if pd_sign in ad_aspects else 0.0
    # PD aspects MD or AD
    f['jai_pd_aspects_md'] = 1.0 if md_sign in pd_aspects else 0.0
    f['jai_pd_aspects_ad'] = 1.0 if ad_sign in pd_aspects else 0.0

    # Combined drishti score: how many levels aspect father's houses
    drishti_score = sum([
        bool(md_aspects & father_mk),
        bool(ad_aspects & father_mk),
        bool(pd_aspects & father_mk),
    ])
    f['jai_drishti_maraka_count'] = drishti_score

    # ── PD Sign Occupants ─────────────────────────────────────────────
    pd_occ = occupants.get(pd_sign, [])
    f['jai_n_planets_in_pd'] = len(pd_occ)
    f['jai_malefic_in_pd'] = sum(1 for p in pd_occ if p in NATURAL_MALEFICS)
    f['jai_sun_in_pd'] = 1.0 if 'Sun' in pd_occ else 0.0  # Natural pitrukaraka

    # Chara Karakas in PD sign
    ak = karakas.get('AK', '')
    if ak and ak in natal_chart:
        ak_sign = get_sign(natal_chart[ak]['longitude'])
        f['jai_ak_in_pd'] = 1.0 if ak_sign == pd_sign else 0.0
    else:
        f['jai_ak_in_pd'] = 0.0

    # GnK (Gnati Karaka = 6th = disease/enemy) in PD sign
    gnk = karakas.get('GnK', '')
    if gnk and gnk in natal_chart:
        gnk_sign = get_sign(natal_chart[gnk]['longitude'])
        f['jai_gnk_in_pd'] = 1.0 if gnk_sign == pd_sign else 0.0
    else:
        f['jai_gnk_in_pd'] = 0.0

    # MK (Matru Karaka = 4th = mother, but also represents parent axis)
    mk_karaka = karakas.get('MK', '')
    if mk_karaka and mk_karaka in natal_chart:
        mk_sign = get_sign(natal_chart[mk_karaka]['longitude'])
        f['jai_mk_in_pd'] = 1.0 if mk_sign == pd_sign else 0.0
    else:
        f['jai_mk_in_pd'] = 0.0

    # Count how many karakas are in PD sign
    karakas_in_pd = 0
    for k_name, k_planet in karakas.items():
        if k_planet in natal_chart:
            if get_sign(natal_chart[k_planet]['longitude']) == pd_sign:
                karakas_in_pd += 1
    f['jai_n_karakas_in_pd'] = karakas_in_pd

    # ── PD Sign Lord Analysis ─────────────────────────────────────────
    pd_lord = get_sign_lord(pd_sign)
    ad_lord = get_sign_lord(ad_sign)
    md_lord = get_sign_lord(md_sign)

    f['jai_pd_lord_is_malefic'] = 1.0 if pd_lord in NATURAL_MALEFICS else 0.0

    # PD lord natal house from lagna
    if pd_lord in natal_chart:
        pd_lord_house = get_house_number(
            natal_chart[pd_lord]['longitude'], jai_ctx['natal_asc'])
        f['jai_pd_lord_in_dusthana'] = 1.0 if pd_lord_house in (6, 8, 12) else 0.0
        f['jai_pd_lord_in_kendra'] = 1.0 if pd_lord_house in (1, 4, 7, 10) else 0.0

        # PD lord dignity
        f['jai_pd_lord_dignity'] = sign_dignity(
            pd_lord, natal_chart[pd_lord]['longitude'])
        f['jai_pd_lord_uchcha'] = uchcha_bala(
            pd_lord, natal_chart[pd_lord]['longitude'])
    else:
        f['jai_pd_lord_in_dusthana'] = 0.0
        f['jai_pd_lord_in_kendra'] = 0.0
        f['jai_pd_lord_dignity'] = 0.5
        f['jai_pd_lord_uchcha'] = 0.5

    # PD lord is a father maraka lord?
    f['jai_pd_lord_is_maraka_lord'] = (
        1.0 if pd_lord in father_mk_lords else 0.0)

    # PD lord is Sun (natural pitrukaraka)?
    f['jai_pd_lord_is_sun'] = 1.0 if pd_lord == 'Sun' else 0.0

    # PD lord is 9th lord?
    f['jai_pd_lord_is_h9_lord'] = (
        1.0 if pd_lord == jai_ctx['h9_lord'] else 0.0)

    # ── Sign Lord Relationships ───────────────────────────────────────
    # Natural friendship between MD lord and PD lord
    f['jai_md_pd_lord_friendly'] = get_friendship(md_lord, pd_lord)
    f['jai_ad_pd_lord_friendly'] = get_friendship(ad_lord, pd_lord)

    # Are MD and PD lords the same planet?
    f['jai_md_pd_same_lord'] = 1.0 if md_lord == pd_lord else 0.0
    f['jai_ad_pd_same_lord'] = 1.0 if ad_lord == pd_lord else 0.0

    # ── Karakamsha Connection ─────────────────────────────────────────
    f['jai_pd_is_karakamsha'] = 1.0 if pd_sign == karakamsha else 0.0
    f['jai_pd_aspects_karakamsha'] = (
        1.0 if karakamsha in pd_aspects else 0.0)

    # 9th from Karakamsha (father indicator in Jaimini)
    h9_from_km = (karakamsha + 8) % 12
    f['jai_pd_is_h9_from_km'] = 1.0 if pd_sign == h9_from_km else 0.0
    f['jai_pd_aspects_h9_from_km'] = (
        1.0 if h9_from_km in pd_aspects else 0.0)

    # ── Arudha Pada A9 (Pitru Pada) ──────────────────────────────────
    f['jai_pd_is_a9'] = 1.0 if pd_sign == a9_sign else 0.0
    f['jai_pd_aspects_a9'] = 1.0 if a9_sign in pd_aspects else 0.0

    # ── Argala (Intervention) ─────────────────────────────────────────
    # Malefics in 2nd, 4th, 11th from PD sign (Shubha/Papa Argala)
    argala_signs = [(pd_sign + 1) % 12, (pd_sign + 3) % 12,
                    (pd_sign + 10) % 12]
    argala_malefic = 0
    argala_total = 0
    for s in argala_signs:
        occ = occupants.get(s, [])
        argala_total += len(occ)
        argala_malefic += sum(1 for p in occ if p in NATURAL_MALEFICS)
    f['jai_pd_argala_malefic'] = argala_malefic
    f['jai_pd_argala_total'] = argala_total

    # Virodha Argala (obstruction): 3rd, 10th from PD sign
    virodha_signs = [(pd_sign + 2) % 12, (pd_sign + 9) % 12]
    virodha_count = sum(len(occupants.get(s, [])) for s in virodha_signs)
    f['jai_pd_virodha_argala'] = virodha_count

    # Net Argala (positive if Argala > Virodha)
    f['jai_pd_net_argala'] = argala_total - virodha_count

    # ── 9th House from PD Sign (father's vitality from PD perspective) ─
    h9_from_pd = (pd_sign + 8) % 12
    h9_pd_occ = occupants.get(h9_from_pd, [])
    f['jai_h9_from_pd_malefic'] = sum(
        1 for p in h9_pd_occ if p in NATURAL_MALEFICS)
    f['jai_h9_from_pd_n_planets'] = len(h9_pd_occ)

    # Lord of 9th from PD sign — is it afflicted?
    h9_from_pd_lord = get_sign_lord(h9_from_pd)
    if h9_from_pd_lord in natal_chart:
        h9l_house = get_house_number(
            natal_chart[h9_from_pd_lord]['longitude'], jai_ctx['natal_asc'])
        f['jai_h9_from_pd_lord_dusthana'] = (
            1.0 if h9l_house in (6, 8, 12) else 0.0)
    else:
        f['jai_h9_from_pd_lord_dusthana'] = 0.0

    # ── PD Lord in D9 (Navamsha Confirmation) ─────────────────────────
    if pd_lord in natal_chart:
        pd_lord_lon = natal_chart[pd_lord]['longitude']
        pd_lord_d9 = navamsha_sign(pd_lord_lon)
        # D9 dispositor
        d9_disp = get_sign_lord(pd_lord_d9)
        f['jai_pd_lord_d9_maraka'] = (
            1.0 if pd_lord_d9 in father_mk else 0.0)
        f['jai_pd_lord_d9_danger'] = (
            1.0 if pd_lord_d9 in father_danger else 0.0)
        f['jai_pd_lord_vargottama'] = (
            1.0 if get_sign(pd_lord_lon) == pd_lord_d9 else 0.0)
        # D9 dispositor is maraka lord?
        f['jai_pd_d9_disp_is_maraka'] = (
            1.0 if d9_disp in father_mk_lords else 0.0)
    else:
        f['jai_pd_lord_d9_maraka'] = 0.0
        f['jai_pd_lord_d9_danger'] = 0.0
        f['jai_pd_lord_vargottama'] = 0.0
        f['jai_pd_d9_disp_is_maraka'] = 0.0

    return f


def extract_jaimini_transit_features(candidate, jai_ctx, natal_chart):
    """Transit-Jaimini interaction features at PD midpoint.

    Combines Jaimini sign-based analysis with real-time transit positions
    for compound danger conditions. ~25 features.
    """
    signs = candidate['signs']
    pd_sign = signs[-1]
    ad_sign = signs[1] if len(signs) >= 2 else signs[-1]
    md_sign = signs[0]

    father_mk = jai_ctx['father_maraka_signs']
    father_danger = jai_ctx['father_danger_signs']
    h9_sign = jai_ctx['h9_sign']
    occupants = jai_ctx['sign_occupants']

    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
    f = {}

    # Transit positions
    t_sat_lon = swe.calc_ut(midpoint_jd, swe.SATURN, swe.FLG_SIDEREAL)[0][0]
    t_jup_lon = swe.calc_ut(midpoint_jd, swe.JUPITER, swe.FLG_SIDEREAL)[0][0]
    t_mars_lon = swe.calc_ut(midpoint_jd, swe.MARS, swe.FLG_SIDEREAL)[0][0]
    t_rahu_lon = swe.calc_ut(midpoint_jd, swe.MEAN_NODE, swe.FLG_SIDEREAL)[0][0]

    sat_sign = int(t_sat_lon / 30) % 12
    jup_sign = int(t_jup_lon / 30) % 12
    mars_sign = int(t_mars_lon / 30) % 12
    rahu_sign = int(t_rahu_lon / 30) % 12

    # ── Transit on PD sign (Jaimini: transit planet IN dasha sign) ─────
    f['jt_sat_in_pd'] = 1.0 if sat_sign == pd_sign else 0.0
    f['jt_jup_in_pd'] = 1.0 if jup_sign == pd_sign else 0.0
    f['jt_mars_in_pd'] = 1.0 if mars_sign == pd_sign else 0.0
    f['jt_rahu_in_pd'] = 1.0 if rahu_sign == pd_sign else 0.0

    # Malefic count transiting PD sign
    f['jt_malefic_in_pd'] = sum([
        sat_sign == pd_sign, mars_sign == pd_sign, rahu_sign == pd_sign])

    # ── Transit Rashi Drishti to PD sign ──────────────────────────────
    # Jaimini aspects from transit positions
    sat_t_aspects = _jaimini_aspect_signs(sat_sign)
    jup_t_aspects = _jaimini_aspect_signs(jup_sign)
    mars_t_aspects = _jaimini_aspect_signs(mars_sign)

    f['jt_sat_aspects_pd'] = 1.0 if pd_sign in sat_t_aspects else 0.0
    f['jt_jup_aspects_pd'] = 1.0 if pd_sign in jup_t_aspects else 0.0
    f['jt_mars_aspects_pd'] = 1.0 if pd_sign in mars_t_aspects else 0.0

    # Transit Saturn aspects maraka signs (Jaimini Rashi Drishti)
    f['jt_sat_aspects_maraka'] = (
        1.0 if bool(sat_t_aspects & father_mk) else 0.0)
    f['jt_jup_aspects_maraka'] = (
        1.0 if bool(jup_t_aspects & father_mk) else 0.0)

    # Jaimini Double Transit: both Sat+Jup aspecting maraka via Rashi Drishti
    f['jt_double_rashi_maraka'] = 1.0 if (
        bool(sat_t_aspects & father_mk) and
        bool(jup_t_aspects & father_mk)) else 0.0

    # ── Compound Jaimini danger conditions ────────────────────────────
    pd_is_mk = pd_sign in father_mk
    pd_is_danger = pd_sign in father_danger

    # PD is maraka AND transit Saturn on PD sign
    f['jt_maraka_sat_on_pd'] = 1.0 if (
        pd_is_mk and sat_sign == pd_sign) else 0.0
    # PD is maraka AND transit Saturn aspects PD (Rashi Drishti)
    f['jt_maraka_sat_aspects'] = 1.0 if (
        pd_is_mk and pd_sign in sat_t_aspects) else 0.0
    # PD is danger AND Rahu on PD
    f['jt_danger_rahu_on_pd'] = 1.0 if (
        pd_is_danger and rahu_sign == pd_sign) else 0.0
    # All three: PD maraka + Saturn aspects + Jupiter aspects
    f['jt_triple_convergence'] = 1.0 if (
        pd_is_mk and
        pd_sign in sat_t_aspects and
        pd_sign in jup_t_aspects) else 0.0

    # ── PD sign lord natal degree distances (CONTINUOUS) ──────────────
    pd_lord = get_sign_lord(pd_sign)
    if pd_lord in natal_chart:
        pd_lord_lon = natal_chart[pd_lord]['longitude']
        sun_lon = natal_chart['Sun']['longitude']
        h9_cusp = (jai_ctx['natal_asc'] + 8 * 30) % 360

        def _arc(a, b):
            d = abs(a - b) % 360
            return min(d, 360 - d)

        # PD lord's natal distance from Sun (pitrukaraka)
        f['jt_pd_lord_dist_sun'] = _arc(pd_lord_lon, sun_lon)
        # PD lord's natal distance from 9th cusp
        f['jt_pd_lord_dist_h9'] = _arc(pd_lord_lon, h9_cusp)
        # Transit Saturn distance from PD lord natal degree
        f['jt_sat_dist_pd_lord'] = _arc(t_sat_lon, pd_lord_lon)
        # Transit Jupiter distance from PD lord
        f['jt_jup_dist_pd_lord'] = _arc(t_jup_lon, pd_lord_lon)
        # Transit Rahu distance from PD lord
        f['jt_rahu_dist_pd_lord'] = _arc(t_rahu_lon, pd_lord_lon)

        # Tight transit conjunctions with PD lord
        f['jt_sat_tight_pd_lord'] = (
            1.0 if _arc(t_sat_lon, pd_lord_lon) <= 5 else 0.0)
        f['jt_rahu_tight_pd_lord'] = (
            1.0 if _arc(t_rahu_lon, pd_lord_lon) <= 5 else 0.0)
    else:
        f['jt_pd_lord_dist_sun'] = 90.0
        f['jt_pd_lord_dist_h9'] = 90.0
        f['jt_sat_dist_pd_lord'] = 90.0
        f['jt_jup_dist_pd_lord'] = 90.0
        f['jt_rahu_dist_pd_lord'] = 90.0
        f['jt_sat_tight_pd_lord'] = 0.0
        f['jt_rahu_tight_pd_lord'] = 0.0

    # ── Composite Jaimini danger score (continuous) ───────────────────
    score = 0.0
    if pd_is_mk:
        score += 3.0
    if pd_is_danger:
        score += 1.5
    if ad_sign in father_mk:
        score += 2.0
    if md_sign in father_mk:
        score += 1.5
    pd_aspects = _jaimini_aspect_signs(pd_sign)
    if h9_sign in pd_aspects:
        score += 1.0
    if bool(pd_aspects & father_mk):
        score += 0.5
    # Transit augmentation
    if f.get('jt_sat_aspects_pd', 0):
        score += 1.0
    if f.get('jt_double_rashi_maraka', 0):
        score += 1.5
    f['jt_composite_danger'] = score

    return f


def extract_jaimini_subperiod_features(candidate, jai_ctx):
    """Sub-period density features for Chara Dasha PD.

    For a depth=3 PD, compute 12 depth=4 sub-period signs (using Chara
    Dasha sequence starting from PD sign) and count maraka signs.

    ~10 features.
    """
    signs = candidate['signs']
    pd_sign = signs[-1]
    father_mk = jai_ctx['father_maraka_signs']
    father_danger = jai_ctx['father_danger_signs']
    asc_sign = jai_ctx['asc_sign']

    # Determine direction (same as MD direction)
    _SAVYA_SIGNS = {0, 2, 4, 6, 8, 10}
    is_savya = asc_sign in _SAVYA_SIGNS

    # Sub-period signs: 12 signs starting from PD sign, same direction
    if is_savya:
        sub_signs = [(pd_sign + i) % 12 for i in range(12)]
    else:
        sub_signs = [(pd_sign - i) % 12 for i in range(12)]

    n_maraka = sum(1 for s in sub_signs if s in father_mk)
    n_danger = sum(1 for s in sub_signs if s in father_danger)
    n_malefic_lord = sum(
        1 for s in sub_signs if get_sign_lord(s) in NATURAL_MALEFICS)

    # Longest consecutive maraka streak
    longest_streak = 0
    current_streak = 0
    for s in sub_signs:
        if s in father_mk:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    # Midpoint sub-period (6th = middle of 12)
    mid_sign = sub_signs[5]
    first_sign = sub_signs[0]

    f = {}
    f['jsk_maraka_count'] = n_maraka
    f['jsk_maraka_frac'] = n_maraka / 12.0
    f['jsk_danger_frac'] = n_danger / 12.0
    f['jsk_malefic_lord_frac'] = n_malefic_lord / 12.0
    f['jsk_longest_streak'] = longest_streak
    f['jsk_mid_is_maraka'] = 1.0 if mid_sign in father_mk else 0.0
    f['jsk_mid_is_danger'] = 1.0 if mid_sign in father_danger else 0.0
    f['jsk_first_is_maraka'] = 1.0 if first_sign in father_mk else 0.0
    f['jsk_first_is_danger'] = 1.0 if first_sign in father_danger else 0.0
    f['jsk_last_is_maraka'] = (
        1.0 if sub_signs[-1] in father_mk else 0.0)

    return f
