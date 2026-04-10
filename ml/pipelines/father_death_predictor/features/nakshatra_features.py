"""Nakshatra-level micro-timing features.

Each pratyantardasha lord sits in a specific nakshatra. The nakshatra
lord's relationship to maraka houses provides fine-grained timing.
Also: nakshatra-level classification (deva/manushya/rakshasa) and
the tara bala (birth star compatibility) cycle.

~13 features per candidate (7 natal + 6 transit nakshatra).
"""

import swisseph as swe

from ..astro_engine.dasha import NAKSHATRA_LORDS

swe.set_sid_mode(swe.SIDM_LAHIRI)

# Nakshatra names for reference
NAKSHATRA_NAMES = [
    'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra',
    'Punarvasu', 'Pushya', 'Ashlesha', 'Magha', 'P.Phalguni', 'U.Phalguni',
    'Hasta', 'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha',
    'Mula', 'P.Ashadha', 'U.Ashadha', 'Shravana', 'Dhanishtha', 'Shatabhisha',
    'P.Bhadrapada', 'U.Bhadrapada', 'Revati',
]

# Nakshatra temperament: 0=Deva(divine), 1=Manushya(human), 2=Rakshasa(demon)
# Rakshasa nakshatras = more destructive potential
NAKSHATRA_TEMPERAMENT = [
    0, 1, 2, 1, 0, 1,   # Ashwini-Ardra
    0, 0, 2, 2, 1, 1,   # Punarvasu-U.Phalguni
    0, 2, 0, 2, 0, 2,   # Hasta-Jyeshtha
    2, 1, 1, 0, 2, 2,   # Mula-Shatabhisha
    1, 1, 0,             # P.Bhadrapada-Revati
]

NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}


def _get_nakshatra_idx(longitude):
    """Get nakshatra index (0-26) from sidereal longitude."""
    return min(int(longitude / (360.0 / 27.0)), 26)


def extract_nakshatra_features(candidate, natal_chart, father_marakas):
    """Nakshatra-level features for the pratyantardasha lord.

    Args:
        candidate: pratyantardasha period dict
        natal_chart: from compute_chart()
        father_marakas: set of maraka planet names

    Returns:
        ~7 features dict.
    """
    pd_lord = candidate['lords'][-1]
    pd_lon = natal_chart[pd_lord]['longitude']

    # PD lord's nakshatra
    nak_idx = _get_nakshatra_idx(pd_lon)
    nak_lord = NAKSHATRA_LORDS[nak_idx]

    f = {}

    # Nakshatra lord is a maraka (dispositor-level maraka)
    f['nk_lord_is_maraka'] = 1.0 if nak_lord in father_marakas else 0.0

    # Nakshatra lord is a natural malefic
    f['nk_lord_is_malefic'] = 1.0 if nak_lord in NATURAL_MALEFICS else 0.0

    # PD lord and its nakshatra lord are BOTH maraka (double confirmation)
    pd_is_maraka = pd_lord in father_marakas
    f['nk_double_maraka'] = 1.0 if (
        pd_is_maraka and f['nk_lord_is_maraka']
    ) else 0.0

    # Nakshatra temperament (rakshasa = more destructive)
    temperament = NAKSHATRA_TEMPERAMENT[nak_idx]
    f['nk_is_rakshasa'] = 1.0 if temperament == 2 else 0.0
    f['nk_temperament'] = temperament  # 0=deva, 1=manushya, 2=rakshasa

    # Tara Bala: cycle of 9 from birth Moon's nakshatra
    # Taras: 1=Janma(birth), 2=Sampat, 3=Vipat(danger), 4=Kshema,
    #         5=Pratyari(obstacle), 6=Sadhaka, 7=Vadha(death),
    #         8=Mitra, 9=Parama Mitra
    # Death tara (7=Vadha) and danger tara (3=Vipat, 5=Pratyari) are worst
    moon_nak = _get_nakshatra_idx(natal_chart['Moon']['longitude'])
    tara_num = ((nak_idx - moon_nak) % 27) % 9 + 1  # 1-9

    f['nk_tara_vadha'] = 1.0 if tara_num == 7 else 0.0  # Death tara
    f['nk_tara_dangerous'] = 1.0 if tara_num in (3, 5, 7) else 0.0

    # --- Transit nakshatra features (Phase C) ---
    # 2.25x finer than sign-based features
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2

    # Saturn transit nakshatra
    sat_lon = swe.calc_ut(midpoint_jd, swe.SATURN, swe.FLG_SIDEREAL)[0][0]
    sat_nak = _get_nakshatra_idx(sat_lon)
    sat_nak_lord = NAKSHATRA_LORDS[sat_nak]
    f['nk_tr_sat_lord_maraka'] = 1.0 if sat_nak_lord in father_marakas else 0.0
    f['nk_tr_sat_rakshasa'] = 1.0 if NAKSHATRA_TEMPERAMENT[sat_nak] == 2 else 0.0

    # Mars transit nakshatra
    mars_lon = swe.calc_ut(midpoint_jd, swe.MARS, swe.FLG_SIDEREAL)[0][0]
    mars_nak = _get_nakshatra_idx(mars_lon)
    mars_nak_lord = NAKSHATRA_LORDS[mars_nak]
    f['nk_tr_mars_lord_maraka'] = 1.0 if mars_nak_lord in father_marakas else 0.0
    f['nk_tr_mars_rakshasa'] = 1.0 if NAKSHATRA_TEMPERAMENT[mars_nak] == 2 else 0.0

    # Saturn transit tara from Moon (death/danger tara at transit level)
    sat_tara = ((sat_nak - moon_nak) % 27) % 9 + 1
    f['nk_tr_sat_tara_danger'] = 1.0 if sat_tara in (3, 5, 7) else 0.0

    # Mars transit tara from Moon
    mars_tara = ((mars_nak - moon_nak) % 27) % 9 + 1
    f['nk_tr_mars_tara_danger'] = 1.0 if mars_tara in (3, 5, 7) else 0.0

    # --- Moon transit nakshatra (Step 2b) ---
    # Moon moves ~13 deg/day → changes nakshatra every ~1 day
    # High variance across PD candidates, zero duration correlation
    moon_lon = swe.calc_ut(midpoint_jd, swe.MOON, swe.FLG_SIDEREAL)[0][0]
    moon_tr_nak = _get_nakshatra_idx(moon_lon)
    moon_tr_nak_lord = NAKSHATRA_LORDS[moon_tr_nak]
    f['nk_tr_moon_lord_maraka'] = 1.0 if moon_tr_nak_lord in father_marakas else 0.0
    f['nk_tr_moon_rakshasa'] = 1.0 if NAKSHATRA_TEMPERAMENT[moon_tr_nak] == 2 else 0.0

    # Moon transit tara from natal Moon
    moon_tara = ((moon_tr_nak - moon_nak) % 27) % 9 + 1
    f['nk_tr_moon_tara_danger'] = 1.0 if moon_tara in (3, 5, 7) else 0.0

    return f
