"""Vimshottari dasha calculator."""

NAKSHATRA_LORDS = [
    'Ketu', 'Venus', 'Sun', 'Moon', 'Mars', 'Rahu',
    'Jupiter', 'Saturn', 'Mercury'
] * 3  # 27 nakshatras

DASHA_YEARS = {
    'Ketu': 7, 'Venus': 20, 'Sun': 6, 'Moon': 10, 'Mars': 7,
    'Rahu': 18, 'Jupiter': 16, 'Saturn': 19, 'Mercury': 17
}

DASHA_SEQUENCE = [
    'Ketu', 'Venus', 'Sun', 'Moon', 'Mars',
    'Rahu', 'Jupiter', 'Saturn', 'Mercury'
]

TOTAL_YEARS = 120.0


def compute_vimshottari(moon_longitude: float, birth_jd: float) -> list:
    """Compute Vimshottari Mahadasha-Antardasha periods.

    Args:
        moon_longitude: Moon's sidereal longitude in degrees
        birth_jd: Julian day of birth

    Returns:
        List of dicts with keys: maha_lord, antar_lord, start_jd, end_jd
    """
    nakshatra_span = 360.0 / 27.0  # 13.3333°
    nakshatra_index = int(moon_longitude / nakshatra_span)
    if nakshatra_index >= 27:
        nakshatra_index = 26
    nakshatra_lord = NAKSHATRA_LORDS[nakshatra_index]

    # Fraction of nakshatra already traversed
    elapsed_in_nak = (moon_longitude % nakshatra_span) / nakshatra_span
    remaining_fraction = 1.0 - elapsed_in_nak

    # Build ordered sequence starting from nakshatra lord
    start_idx = DASHA_SEQUENCE.index(nakshatra_lord)
    ordered = DASHA_SEQUENCE[start_idx:] + DASHA_SEQUENCE[:start_idx]

    periods = []
    current_jd = birth_jd

    for cycle in range(2):  # 2 cycles = 240 years
        for i, maha_lord in enumerate(ordered):
            if cycle == 0 and i == 0:
                maha_years = DASHA_YEARS[maha_lord] * remaining_fraction
            else:
                maha_years = DASHA_YEARS[maha_lord]

            # Antardashas within this Mahadasha
            antar_start_idx = DASHA_SEQUENCE.index(maha_lord)
            antar_ordered = (
                DASHA_SEQUENCE[antar_start_idx:]
                + DASHA_SEQUENCE[:antar_start_idx]
            )

            for antar_lord in antar_ordered:
                antar_years = maha_years * DASHA_YEARS[antar_lord] / TOTAL_YEARS
                antar_end_jd = current_jd + antar_years * 365.25
                periods.append({
                    'maha_lord': maha_lord,
                    'antar_lord': antar_lord,
                    'start_jd': current_jd,
                    'end_jd': antar_end_jd,
                })
                current_jd = antar_end_jd

            if (current_jd - birth_jd) / 365.25 > 120:
                return periods

    return periods


def get_dasha_at_age(periods: list, birth_jd: float, age_years: float):
    """Returns (maha_lord, antar_lord) running at given age."""
    target_jd = birth_jd + age_years * 365.25
    for p in periods:
        if p['start_jd'] <= target_jd < p['end_jd']:
            return p['maha_lord'], p['antar_lord']
    return None, None


def compute_full_dasha(moon_longitude: float, birth_jd: float,
                       max_depth: int = 3,
                       collect_all_depths: bool = False) -> list:
    """Compute dasha periods down to pratyantardasha (depth=3) or sookshma (depth=4).

    Depth levels:
      1 = Mahadasha only
      2 = Antardasha
      3 = Pratyantardasha
      4 = Sookshma

    Args:
        collect_all_depths: If True, return periods at ALL depths (1..max_depth),
            not just the deepest. Needed for sandhi features.

    Returns flat list of periods:
    [{
        'lords': ['Venus', 'Sun', 'Mars'],  # chain from MD -> deepest
        'start_jd': float,
        'end_jd': float,
        'duration_days': float,
        'depth': int,
    }, ...]
    """
    nakshatra_span = 360.0 / 27.0
    nakshatra_index = int(moon_longitude / nakshatra_span)
    if nakshatra_index >= 27:
        nakshatra_index = 26
    nakshatra_lord = NAKSHATRA_LORDS[nakshatra_index]

    elapsed_in_nak = (moon_longitude % nakshatra_span) / nakshatra_span
    remaining_fraction = 1.0 - elapsed_in_nak

    start_idx = DASHA_SEQUENCE.index(nakshatra_lord)
    maha_sequence = DASHA_SEQUENCE[start_idx:] + DASHA_SEQUENCE[:start_idx]

    def _sub_sequence(lord):
        idx = DASHA_SEQUENCE.index(lord)
        return DASHA_SEQUENCE[idx:] + DASHA_SEQUENCE[:idx]

    def _subdivide(parent_start_jd, parent_duration_days, parent_lord,
                   lords_so_far, current_depth):
        """Recursively subdivide a dasha period into 9 sub-periods."""
        if current_depth > max_depth:
            return [{
                'lords': lords_so_far,
                'start_jd': parent_start_jd,
                'end_jd': parent_start_jd + parent_duration_days,
                'duration_days': parent_duration_days,
                'depth': len(lords_so_far),
            }]

        sub_seq = _sub_sequence(parent_lord)
        periods = []
        cur_jd = parent_start_jd

        for sub_lord in sub_seq:
            sub_fraction = DASHA_YEARS[sub_lord] / TOTAL_YEARS
            sub_duration = parent_duration_days * sub_fraction
            sub_lords = lords_so_far + [sub_lord]

            # Collect intermediate depths if requested
            if collect_all_depths:
                periods.append({
                    'lords': sub_lords,
                    'start_jd': cur_jd,
                    'end_jd': cur_jd + sub_duration,
                    'duration_days': sub_duration,
                    'depth': len(sub_lords),
                })

            if current_depth == max_depth:
                if not collect_all_depths:
                    periods.append({
                        'lords': sub_lords,
                        'start_jd': cur_jd,
                        'end_jd': cur_jd + sub_duration,
                        'duration_days': sub_duration,
                        'depth': len(sub_lords),
                    })
            else:
                periods.extend(_subdivide(
                    cur_jd, sub_duration, sub_lord,
                    sub_lords, current_depth + 1))

            cur_jd += sub_duration

        return periods

    all_periods = []
    current_jd = birth_jd

    for cycle in range(2):
        for i, maha_lord in enumerate(maha_sequence):
            if cycle == 0 and i == 0:
                maha_years = DASHA_YEARS[maha_lord] * remaining_fraction
            else:
                maha_years = DASHA_YEARS[maha_lord]

            maha_days = maha_years * 365.25

            periods = _subdivide(
                current_jd, maha_days, maha_lord,
                [maha_lord], current_depth=2)
            all_periods.extend(periods)
            current_jd += maha_days

            if (current_jd - birth_jd) / 365.25 > 120:
                return all_periods

    return all_periods
