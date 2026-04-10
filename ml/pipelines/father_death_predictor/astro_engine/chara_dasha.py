"""Jaimini Chara Dasha — sign-based dasha (1-12 years per sign).

Each rashi (sign) rules for a number of years determined by the
distance from the sign to its lord. The sequence direction (forward
or backward through the zodiac) depends on the 9th house sign.

Sub-periods: each MD sign has 12 AD sub-periods (one per sign),
each AD has 12 PD sub-sub-periods.
"""

from .houses import get_sign, get_sign_lord

# Sign lords (standard Parashari — Jaimini uses same lordship)
# Scorpio: Mars (co-lord Ketu — use Mars for distance counting)
# Aquarius: Saturn (co-lord Rahu — use Saturn for distance counting)

# Direction groups for Chara Dasha
# Savya (forward) signs: Odd signs (Aries, Gemini, Leo, Libra, Sagittarius, Aquarius)
# Apasavya (backward) signs: Even signs (Taurus, Cancer, Virgo, Scorpio, Capricorn, Pisces)
_SAVYA_SIGNS = {0, 2, 4, 6, 8, 10}  # Aries, Gemini, Leo, Libra, Sag, Aquarius
_APASAVYA_SIGNS = {1, 3, 5, 7, 9, 11}  # Taurus, Cancer, Virgo, Scorpio, Cap, Pisces


def _sign_distance(from_sign, to_sign):
    """Count houses from from_sign to to_sign (1-12, inclusive of from)."""
    return ((to_sign - from_sign) % 12) or 12  # 0 maps to 12


def _get_lord_sign(planet, natal_chart):
    """Get the sign a planet is placed in."""
    if planet not in natal_chart:
        return 0
    return get_sign(natal_chart[planet]['longitude'])


def compute_chara_dasha(natal_chart, natal_asc, birth_jd,
                        max_depth=2):
    """Compute Jaimini Chara Dasha periods.

    Args:
        natal_chart: dict of planet positions
        natal_asc: ascendant longitude
        birth_jd: Julian day of birth
        max_depth: 1=MD (sign-level), 2=AD (sub-period), 3=PD

    Returns:
        List of period dicts matching Vimshottari format:
        {'lords': [sign_names], 'start_jd', 'end_jd', 'duration_days',
         'depth', 'system': 'chara', 'signs': [sign_indices]}
    """
    asc_sign = get_sign(natal_asc)

    # Determine direction based on lagna sign (odd=Savya, even=Apasavya)
    is_savya = asc_sign in _SAVYA_SIGNS

    SIGN_NAMES = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                  'Libra', 'Scorpio', 'Sagittarius', 'Capricorn',
                  'Aquarius', 'Pisces']

    # Compute MD duration for each sign
    def _md_years(sign):
        lord = get_sign_lord(sign)
        lord_sign = _get_lord_sign(lord, natal_chart)
        if lord_sign == sign:
            return 12  # Own sign = 12 years
        dist = _sign_distance(sign, lord_sign)
        return dist - 1 if dist > 1 else 12

    # Build sequence starting from lagna
    if is_savya:
        md_sequence = [(asc_sign + i) % 12 for i in range(12)]
    else:
        md_sequence = [(asc_sign - i) % 12 for i in range(12)]

    # Build periods
    all_periods = []
    current_jd = birth_jd

    # Run 2 full cycles (240 years max) to cover any lifespan
    for cycle in range(2):
        for md_sign in md_sequence:
            md_years = _md_years(md_sign)
            md_days = md_years * 365.25
            md_name = SIGN_NAMES[md_sign]

            if max_depth == 1:
                all_periods.append({
                    'lords': [md_name],
                    'signs': [md_sign],
                    'start_jd': current_jd,
                    'end_jd': current_jd + md_days,
                    'duration_days': md_days,
                    'depth': 1,
                    'system': 'chara',
                })
            else:
                # AD: 12 sub-periods, one per sign, same direction
                ad_days_each = md_days / 12.0
                ad_jd = current_jd

                if is_savya:
                    ad_sequence = [(md_sign + i) % 12 for i in range(12)]
                else:
                    ad_sequence = [(md_sign - i) % 12 for i in range(12)]

                for ad_sign in ad_sequence:
                    ad_name = SIGN_NAMES[ad_sign]

                    if max_depth == 2:
                        all_periods.append({
                            'lords': [md_name, ad_name],
                            'signs': [md_sign, ad_sign],
                            'start_jd': ad_jd,
                            'end_jd': ad_jd + ad_days_each,
                            'duration_days': ad_days_each,
                            'depth': 2,
                            'system': 'chara',
                        })
                    elif max_depth >= 3:
                        # PD: further subdivide each AD into 12
                        pd_days_each = ad_days_each / 12.0
                        pd_jd = ad_jd

                        if is_savya:
                            pd_seq = [(ad_sign + i) % 12 for i in range(12)]
                        else:
                            pd_seq = [(ad_sign - i) % 12 for i in range(12)]

                        for pd_sign in pd_seq:
                            pd_name = SIGN_NAMES[pd_sign]
                            all_periods.append({
                                'lords': [md_name, ad_name, pd_name],
                                'signs': [md_sign, ad_sign, pd_sign],
                                'start_jd': pd_jd,
                                'end_jd': pd_jd + pd_days_each,
                                'duration_days': pd_days_each,
                                'depth': 3,
                                'system': 'chara',
                            })
                            pd_jd += pd_days_each

                    ad_jd += ad_days_each

            current_jd += md_days

            if (current_jd - birth_jd) / 365.25 > 120:
                return all_periods

    return all_periods
