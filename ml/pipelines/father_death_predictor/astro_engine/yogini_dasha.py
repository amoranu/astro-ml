"""Yogini Dasha — 36-year cycle, 8 Yogini-planet lords (BPHS Ch. 46)."""

YOGINI_NAMES = [
    'Mangala', 'Pingala', 'Dhanya', 'Bhramari',
    'Bhadrika', 'Ulka', 'Siddha', 'Sankata',
]

YOGINI_PLANETS = {
    'Mangala': 'Moon', 'Pingala': 'Sun', 'Dhanya': 'Jupiter',
    'Bhramari': 'Mars', 'Bhadrika': 'Mercury', 'Ulka': 'Saturn',
    'Siddha': 'Venus', 'Sankata': 'Rahu',
}

YOGINI_YEARS = {
    'Mangala': 1, 'Pingala': 2, 'Dhanya': 3, 'Bhramari': 4,
    'Bhadrika': 5, 'Ulka': 6, 'Siddha': 7, 'Sankata': 8,
}

YOGINI_TOTAL = 36.0  # 1+2+3+4+5+6+7+8


def compute_yogini_dasha(moon_longitude: float, birth_jd: float,
                         max_depth: int = 3) -> list:
    """Compute Yogini dasha periods.

    Seed: (nakshatra_index + 3) mod 8 -> starting Yogini.
    Sub-periods: 8 sub-periods per level, proportional to years.

    Returns flat list at target depth with keys:
        lords (Yogini names), planets (mapped planets),
        start_jd, end_jd, duration_days, depth, system='yogini'
    """
    nakshatra_span = 360.0 / 27.0
    nakshatra_index = int(moon_longitude / nakshatra_span)
    if nakshatra_index >= 27:
        nakshatra_index = 26

    start_idx = (nakshatra_index + 3) % 8
    ordered = YOGINI_NAMES[start_idx:] + YOGINI_NAMES[:start_idx]

    elapsed_fraction = (moon_longitude % nakshatra_span) / nakshatra_span
    remaining_fraction = 1.0 - elapsed_fraction

    def _sub_seq(yogini):
        idx = YOGINI_NAMES.index(yogini)
        return YOGINI_NAMES[idx:] + YOGINI_NAMES[:idx]

    def _subdivide(parent_start, parent_days, parent_yogini,
                   lords_so_far, depth):
        if depth > max_depth:
            return [{
                'lords': lords_so_far,
                'planets': [YOGINI_PLANETS[y] for y in lords_so_far],
                'start_jd': parent_start,
                'end_jd': parent_start + parent_days,
                'duration_days': parent_days,
                'depth': len(lords_so_far),
                'system': 'yogini',
            }]

        sub_seq = _sub_seq(parent_yogini)
        periods = []
        cur_jd = parent_start

        for sub_yog in sub_seq:
            sub_frac = YOGINI_YEARS[sub_yog] / YOGINI_TOTAL
            sub_days = parent_days * sub_frac
            sub_lords = lords_so_far + [sub_yog]

            if depth == max_depth:
                periods.append({
                    'lords': sub_lords,
                    'planets': [YOGINI_PLANETS[y] for y in sub_lords],
                    'start_jd': cur_jd,
                    'end_jd': cur_jd + sub_days,
                    'duration_days': sub_days,
                    'depth': len(sub_lords),
                    'system': 'yogini',
                })
            else:
                periods.extend(_subdivide(
                    cur_jd, sub_days, sub_yog, sub_lords, depth + 1))
            cur_jd += sub_days

        return periods

    all_periods = []
    current_jd = birth_jd

    for cycle in range(5):  # 180 years
        for i, yogini in enumerate(ordered):
            if cycle == 0 and i == 0:
                dur_years = YOGINI_YEARS[yogini] * remaining_fraction
            else:
                dur_years = YOGINI_YEARS[yogini]

            dur_days = dur_years * 365.25
            periods = _subdivide(
                current_jd, dur_days, yogini, [yogini], depth=2)
            all_periods.extend(periods)
            current_jd += dur_days

            if (current_jd - birth_jd) / 365.25 > 120:
                return all_periods

    return all_periods
