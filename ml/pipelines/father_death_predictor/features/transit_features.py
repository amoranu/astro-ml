"""Tier 1: Transit features — Saturn, Jupiter, double transit, Rahu/Ketu, Mars.

~28 features per candidate month. These are the ONLY features that vary
across the 24 candidates and form the primary diagnostic.
"""

import numpy as np
from ..astro_engine.houses import (
    get_house_position, get_house_lord, get_sign
)
from ..astro_engine.aspects import aspect_strength


def fold_angle(long1: float, long2: float) -> float:
    """Angular separation folded to 0-180 degrees."""
    delta = abs(long1 - long2) % 360
    return min(delta, 360 - delta)


def extract_saturn_transit(natal_chart: dict, natal_asc: float,
                           transit_pos: dict,
                           natal_sat_bav: list, natal_sav: list) -> dict:
    """Saturn transit features (~11 features)."""
    t_sat = transit_pos['Saturn']
    n_sun = natal_chart['Sun']['longitude']
    h9_mid = (natal_asc + 8 * 30 + 15) % 360
    h4_mid = (natal_asc + 3 * 30 + 15) % 360

    h9_lord_name = get_house_lord(9, natal_asc)
    n_h9_lord = natal_chart[h9_lord_name]['longitude']

    tr_sat_sign = get_sign(t_sat)

    return {
        'tr_sat_house': get_house_position(t_sat, natal_asc),
        'tr_sat_asp_h9': aspect_strength('Saturn', 'point', t_sat, h9_mid),
        'tr_sat_asp_h4': aspect_strength('Saturn', 'point', t_sat, h4_mid),
        'tr_sat_asp_sun': aspect_strength('Saturn', 'Sun', t_sat, n_sun),
        'tr_sat_asp_h9lord': aspect_strength(
            'Saturn', h9_lord_name, t_sat, n_h9_lord),
        'tr_sat_bav': natal_sat_bav[tr_sat_sign],
        'tr_sat_sav': natal_sav[tr_sat_sign],
        'tr_sat_sun_angle': fold_angle(t_sat, n_sun),
        'tr_sat_in_h9_sign': (
            1.0 if tr_sat_sign == (get_sign(natal_asc) + 8) % 12 else 0.0),
        'tr_sat_in_h4_sign': (
            1.0 if tr_sat_sign == (get_sign(natal_asc) + 3) % 12 else 0.0),
        'tr_sat_retrograde': (
            1.0 if transit_pos.get('Saturn_speed', 0) < 0 else 0.0),
    }


def extract_jupiter_transit(natal_chart: dict, natal_asc: float,
                            transit_pos: dict,
                            natal_sun_bav: list) -> dict:
    """Jupiter transit features (~8 features)."""
    t_jup = transit_pos['Jupiter']
    n_sun = natal_chart['Sun']['longitude']
    h9_mid = (natal_asc + 8 * 30 + 15) % 360
    h4_mid = (natal_asc + 3 * 30 + 15) % 360

    tr_jup_sign = get_sign(t_jup)

    return {
        'tr_jup_house': get_house_position(t_jup, natal_asc),
        'tr_jup_asp_h9': aspect_strength('Jupiter', 'point', t_jup, h9_mid),
        'tr_jup_asp_h4': aspect_strength('Jupiter', 'point', t_jup, h4_mid),
        'tr_jup_asp_sun': aspect_strength('Jupiter', 'Sun', t_jup, n_sun),
        'tr_jup_sun_angle': fold_angle(t_jup, n_sun),
        'tr_jup_sun_bav': natal_sun_bav[tr_jup_sign],
        'tr_jup_in_h9_sign': (
            1.0 if tr_jup_sign == (get_sign(natal_asc) + 8) % 12 else 0.0),
        'tr_jup_in_h4_sign': (
            1.0 if tr_jup_sign == (get_sign(natal_asc) + 3) % 12 else 0.0),
    }


def extract_double_transit(saturn_feats: dict, jupiter_feats: dict) -> dict:
    """Double transit: both Jupiter and Saturn aspecting key points (~3 features)."""
    return {
        'double_transit_h9': min(
            saturn_feats['tr_sat_asp_h9'], jupiter_feats['tr_jup_asp_h9']),
        'double_transit_h4': min(
            saturn_feats['tr_sat_asp_h4'], jupiter_feats['tr_jup_asp_h4']),
        'double_transit_sun': min(
            saturn_feats['tr_sat_asp_sun'], jupiter_feats['tr_jup_asp_sun']),
    }


def extract_rahu_transit(natal_chart: dict, natal_asc: float,
                         transit_pos: dict) -> dict:
    """Rahu-Ketu transit features (~4 features)."""
    t_rahu = transit_pos['Rahu']
    t_ketu = transit_pos['Ketu']
    n_sun = natal_chart['Sun']['longitude']

    h9_lord_name = get_house_lord(9, natal_asc)
    n_h9_lord = natal_chart[h9_lord_name]['longitude']

    return {
        'tr_rahu_house': get_house_position(t_rahu, natal_asc),
        'tr_rahu_sun_angle': fold_angle(t_rahu, n_sun),
        'tr_ketu_sun_angle': fold_angle(t_ketu, n_sun),
        'tr_rahu_h9lord_angle': fold_angle(t_rahu, n_h9_lord),
    }


def extract_mars_transit(natal_chart: dict, natal_asc: float,
                         transit_pos: dict) -> dict:
    """Mars transit features (~3 features). Fast-moving trigger planet."""
    t_mars = transit_pos['Mars']
    h9_mid = (natal_asc + 8 * 30 + 15) % 360
    h4_mid = (natal_asc + 3 * 30 + 15) % 360
    n_sun = natal_chart['Sun']['longitude']

    return {
        'tr_mars_asp_h9': aspect_strength('Mars', 'point', t_mars, h9_mid),
        'tr_mars_asp_h4': aspect_strength('Mars', 'point', t_mars, h4_mid),
        'tr_mars_sun_angle': fold_angle(t_mars, n_sun),
    }


def extract_all_transit_features(natal_chart: dict, natal_asc: float,
                                 transit_pos: dict,
                                 natal_sat_bav: list, natal_sun_bav: list,
                                 natal_sav: list) -> dict:
    """Extract all Tier 1 transit features (~29 features)."""
    sat = extract_saturn_transit(
        natal_chart, natal_asc, transit_pos, natal_sat_bav, natal_sav)
    jup = extract_jupiter_transit(
        natal_chart, natal_asc, transit_pos, natal_sun_bav)
    dbl = extract_double_transit(sat, jup)
    rahu = extract_rahu_transit(natal_chart, natal_asc, transit_pos)
    mars = extract_mars_transit(natal_chart, natal_asc, transit_pos)

    return {**sat, **jup, **dbl, **rahu, **mars}
