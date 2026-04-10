"""KP (Krishnamurti Paddhati) significator chain features.

KP determines event timing by checking whether the dasha lord is a
significator of relevant houses through a 4-level chain:
  Level 1: Star lord of planet occupying the house (strongest)
  Level 2: Planet occupying the house
  Level 3: Star lord of house lord
  Level 4: House lord (weakest)

For father death: houses 2, 7 (maraka from 9th), 4 (8th from 9th), 8 (12th from 9th).

~20 features per candidate.
"""

from ..astro_engine.houses import get_sign, get_sign_lord, get_house_number
from ..astro_engine.dasha import NAKSHATRA_LORDS

ALL_PLANETS = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter',
               'Venus', 'Saturn', 'Rahu', 'Ketu']

# Father death house groups (from natal lagna)
# Father = 9th house. Maraka from 9th = 2nd/7th from 9th = natal 10, 3
# Death house from 9th = 8th from 9th = natal 4
# Loss house from 9th = 12th from 9th = natal 8
FATHER_DEATH_HOUSES = {10, 3, 4, 8}  # natal house numbers
FATHER_MARAKA_HOUSES = {10, 3}  # 2nd and 7th from 9th

# Badhaka house: 11th for moveable, 9th for fixed, 7th for dual
_BADHAKA = {
    'moveable': 11, 'fixed': 9, 'dual': 7,
}
_SIGN_QUALITY = ['moveable', 'fixed', 'dual', 'moveable',
                 'fixed', 'dual', 'moveable', 'fixed',
                 'dual', 'moveable', 'fixed', 'dual']


def _get_nakshatra_lord(longitude):
    """Get nakshatra lord (star lord) from sidereal longitude."""
    nak_idx = min(int(longitude / (360.0 / 27.0)), 26)
    return NAKSHATRA_LORDS[nak_idx]


def precompute_kp_context(natal_chart, natal_asc):
    """Precompute KP significator chains for all 12 houses."""
    asc_sign = get_sign(natal_asc)

    # For each house (1-12), compute 4-level significators
    house_significators = {}  # house_num -> set of significator planets

    # Determine which planets occupy which houses
    planet_in_house = {}
    for planet in ALL_PLANETS:
        if planet in natal_chart:
            h = get_house_number(natal_chart[planet]['longitude'], natal_asc)
            planet_in_house[planet] = h

    for house in range(1, 13):
        sigs = set()
        # Level 4: House lord
        house_sign = (asc_sign + house - 1) % 12
        house_lord = get_sign_lord(house_sign)
        sigs.add(house_lord)

        # Level 3: Star lord of house lord
        if house_lord in natal_chart:
            hl_lon = natal_chart[house_lord]['longitude']
            star_of_lord = _get_nakshatra_lord(hl_lon)
            sigs.add(star_of_lord)

        # Level 2: Planets occupying the house
        for planet, h in planet_in_house.items():
            if h == house:
                sigs.add(planet)
                # Level 1: Star lord of occupant (strongest)
                if planet in natal_chart:
                    occ_lon = natal_chart[planet]['longitude']
                    star_of_occ = _get_nakshatra_lord(occ_lon)
                    sigs.add(star_of_occ)

        house_significators[house] = sigs

    # KP father death significator set
    father_death_sigs = set()
    for h in FATHER_DEATH_HOUSES:
        father_death_sigs |= house_significators.get(h, set())

    father_maraka_sigs = set()
    for h in FATHER_MARAKA_HOUSES:
        father_maraka_sigs |= house_significators.get(h, set())

    # 9th cusp sub-lord (CSL of 9th house)
    h9_sign = (asc_sign + 8) % 12
    h9_lord = get_sign_lord(h9_sign)
    h9_csl = _get_nakshatra_lord(natal_asc + 8 * 30)  # Equal house 9th cusp

    # Badhaka lord for 9th house
    h9_quality = _SIGN_QUALITY[h9_sign]
    badhaka_house_from_9 = _BADHAKA[h9_quality]
    badhaka_sign = (h9_sign + badhaka_house_from_9 - 1) % 12
    badhaka_lord = get_sign_lord(badhaka_sign)

    return {
        'house_significators': house_significators,
        'father_death_sigs': father_death_sigs,
        'father_maraka_sigs': father_maraka_sigs,
        'h9_csl': h9_csl,
        'h9_lord': h9_lord,
        'badhaka_lord': badhaka_lord,
        'planet_in_house': planet_in_house,
    }


def extract_kp_features(candidate, kp_ctx, natal_chart):
    """KP significator features for a Vimshottari PD candidate.

    Args:
        candidate: period dict with 'lords' field
        kp_ctx: from precompute_kp_context()
        natal_chart: from compute_chart()

    Returns: ~20 features dict.
    """
    f_death = kp_ctx['father_death_sigs']
    f_maraka = kp_ctx['father_maraka_sigs']
    h_sigs = kp_ctx['house_significators']

    lords = candidate['lords']
    pd_lord = lords[-1]  # Deepest level lord
    ad_lord = lords[1] if len(lords) > 1 else lords[0]
    md_lord = lords[0]

    f = {}

    # --- PD lord significator status ---
    f['kp_pd_is_death_sig'] = 1.0 if pd_lord in f_death else 0.0
    f['kp_pd_is_maraka_sig'] = 1.0 if pd_lord in f_maraka else 0.0

    # Which specific death houses does PD lord signify?
    for h in [10, 3, 4, 8]:
        f[f'kp_pd_sig_h{h}'] = 1.0 if pd_lord in h_sigs.get(h, set()) else 0.0

    # Count of death houses PD lord signifies (0-4)
    f['kp_pd_death_count'] = sum(
        pd_lord in h_sigs.get(h, set()) for h in FATHER_DEATH_HOUSES)

    # --- AD and MD lord significator status ---
    f['kp_ad_is_death_sig'] = 1.0 if ad_lord in f_death else 0.0
    f['kp_md_is_death_sig'] = 1.0 if md_lord in f_death else 0.0

    # --- Cascade: how many dasha lords are death significators ---
    f['kp_chain_count'] = sum([
        md_lord in f_death,
        ad_lord in f_death,
        pd_lord in f_death,
    ])
    f['kp_all_death'] = 1.0 if f['kp_chain_count'] == 3 else 0.0
    f['kp_md_ad_death'] = 1.0 if (
        md_lord in f_death and ad_lord in f_death
    ) else 0.0

    # --- 9th cusp sub-lord (CSL) analysis ---
    h9_csl = kp_ctx['h9_csl']
    f['kp_csl9_is_death_sig'] = 1.0 if h9_csl in f_death else 0.0
    f['kp_csl9_is_malefic'] = 1.0 if h9_csl in (
        'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu') else 0.0

    # Is CSL retrograde? (weakens 9th house = father vulnerable)
    if h9_csl in natal_chart and h9_csl not in ('Rahu', 'Ketu'):
        f['kp_csl9_retrograde'] = 1.0 if natal_chart[h9_csl].get(
            'speed', 1) < 0 else 0.0
    else:
        f['kp_csl9_retrograde'] = 0.0

    # --- Badhaka lord active ---
    badhaka = kp_ctx['badhaka_lord']
    f['kp_badhaka_in_chain'] = 1.0 if badhaka in (md_lord, ad_lord, pd_lord) else 0.0

    # --- PD lord's star lord is also death significator ---
    if pd_lord in natal_chart:
        pd_star = _get_nakshatra_lord(natal_chart[pd_lord]['longitude'])
        f['kp_pd_star_is_death'] = 1.0 if pd_star in f_death else 0.0
    else:
        f['kp_pd_star_is_death'] = 0.0

    return f
