"""Navamsha (D9) confirmation features.

D9 is Parashari's confirmation chart. A promise in D1 that's confirmed
in D9 manifests; one that isn't, doesn't.

~4 features per candidate.
"""

from ..astro_engine.houses import get_sign, get_sign_lord
from ..astro_engine.vargas import navamsha_sign
from ..astro_engine.dignity import EXALTATION_DEGREES


def _is_debilitated_in_sign(planet: str, sign: int) -> bool:
    """Check if planet is debilitated in a given sign."""
    if planet not in EXALTATION_DEGREES:
        return False
    exalt_sign = int(EXALTATION_DEGREES[planet] / 30) % 12
    debil_sign = (exalt_sign + 6) % 12
    return sign == debil_sign


def extract_navamsha_features(candidate: dict,
                              natal_chart: dict,
                              natal_asc: float,
                              father_marakas: set) -> dict:
    """D9 confirmation of the pratyantardasha lord's maraka status.

    Args:
        candidate: pratyantardasha period dict
        natal_chart: from compute_chart()
        natal_asc: ascendant longitude
        father_marakas: set of planet names that are father's maraka lords

    Returns:
        ~4 features dict.
    """
    pd_lord = candidate['lords'][-1]
    pd_lon = natal_chart[pd_lord]['longitude']

    f = {}

    # PD lord's D9 sign
    d9_sign = navamsha_sign(pd_lon)

    # Debilitation in D9 → weakened planet more likely to cause harm
    f['d9_pd_debilitated'] = 1.0 if _is_debilitated_in_sign(pd_lord, d9_sign) else 0.0

    # D9 dispositor (lord of D9 sign) is a maraka?
    d9_dispositor = get_sign_lord(d9_sign)
    f['d9_dispositor_maraka'] = 1.0 if d9_dispositor in father_marakas else 0.0

    # PD lord in a maraka house in D9?
    # D9 ascendant = navamsha of natal ascendant
    d9_asc_sign = navamsha_sign(natal_asc)
    pd_house_in_d9 = (d9_sign - d9_asc_sign) % 12 + 1  # 1-12
    f['d9_pd_in_maraka_house'] = 1.0 if pd_house_in_d9 in (2, 7) else 0.0

    # Vargottama: same sign in D1 and D9 → strengthens the planet
    d1_sign = int(pd_lon / 30) % 12
    f['d9_pd_vargottama'] = 1.0 if d1_sign == d9_sign else 0.0

    # --- Phase 5: Extended D9 features ---
    # SD lord (deepest lord) — may differ from PD lord at depth 4
    sd_lord = candidate['lords'][-1]
    sd_lon = natal_chart[sd_lord]['longitude']
    sd_d9_sign = navamsha_sign(sd_lon)

    # SD lord's D9 dispositor is maraka
    sd_d9_dispositor = get_sign_lord(sd_d9_sign)
    f['d9_sd_dispositor_maraka'] = 1.0 if sd_d9_dispositor in father_marakas else 0.0

    # SD lord debilitated in D9
    f['d9_sd_debilitated'] = 1.0 if _is_debilitated_in_sign(sd_lord, sd_d9_sign) else 0.0

    # SD lord in D9 maraka house (2nd or 7th from D9 asc)
    sd_house_d9 = (sd_d9_sign - d9_asc_sign) % 12 + 1
    f['d9_sd_in_maraka_house'] = 1.0 if sd_house_d9 in (2, 7) else 0.0

    # SD lord vargottama
    sd_d1_sign = int(sd_lon / 30) % 12
    f['d9_sd_vargottama'] = 1.0 if sd_d1_sign == sd_d9_sign else 0.0

    # D9 dispositor dignity (continuous)
    from ..astro_engine.dignity import sign_dignity
    disp_lon = natal_chart[d9_dispositor]['longitude']
    f['d9_dispositor_dignity'] = sign_dignity(d9_dispositor, disp_lon)

    return f
