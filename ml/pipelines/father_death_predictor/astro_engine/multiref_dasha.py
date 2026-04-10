"""Multi-reference Vimshottari: same system, 3 independent seed points.

Moon (standard), Sun (Pitru Karaka), 9th cusp (father's lagna).
All use identical Vimshottari mechanics — only the starting nakshatra differs.
"""

from .dasha import NAKSHATRA_LORDS, DASHA_YEARS, DASHA_SEQUENCE, TOTAL_YEARS


def _compute_from_longitude(seed_longitude: float, birth_jd: float,
                            max_depth: int = 3) -> list:
    """Generic Vimshottari from any sidereal longitude.

    Identical to compute_full_dasha but accepts arbitrary seed.
    """
    nakshatra_span = 360.0 / 27.0
    nak_idx = int(seed_longitude / nakshatra_span)
    if nak_idx >= 27:
        nak_idx = 26
    nak_lord = NAKSHATRA_LORDS[nak_idx]

    elapsed = (seed_longitude % nakshatra_span) / nakshatra_span
    remaining = 1.0 - elapsed

    start_idx = DASHA_SEQUENCE.index(nak_lord)
    maha_seq = DASHA_SEQUENCE[start_idx:] + DASHA_SEQUENCE[:start_idx]

    def _sub_seq(lord):
        idx = DASHA_SEQUENCE.index(lord)
        return DASHA_SEQUENCE[idx:] + DASHA_SEQUENCE[:idx]

    def _subdivide(p_start, p_days, p_lord, lords, depth):
        if depth > max_depth:
            return [{
                'lords': lords, 'start_jd': p_start,
                'end_jd': p_start + p_days,
                'duration_days': p_days, 'depth': len(lords),
            }]
        sub = _sub_seq(p_lord)
        periods = []
        cur = p_start
        for sl in sub:
            sd = p_days * DASHA_YEARS[sl] / TOTAL_YEARS
            sl_lords = lords + [sl]
            if depth == max_depth:
                periods.append({
                    'lords': sl_lords, 'start_jd': cur,
                    'end_jd': cur + sd, 'duration_days': sd,
                    'depth': len(sl_lords),
                })
            else:
                periods.extend(_subdivide(cur, sd, sl, sl_lords, depth + 1))
            cur += sd
        return periods

    all_periods = []
    cur_jd = birth_jd
    for cycle in range(2):
        for i, ml in enumerate(maha_seq):
            if cycle == 0 and i == 0:
                yrs = DASHA_YEARS[ml] * remaining
            else:
                yrs = DASHA_YEARS[ml]
            days = yrs * 365.25
            all_periods.extend(_subdivide(cur_jd, days, ml, [ml], 2))
            cur_jd += days
            if (cur_jd - birth_jd) / 365.25 > 120:
                return all_periods
    return all_periods


def compute_multi_reference(chart: dict, asc: float, birth_jd: float,
                            max_depth: int = 3) -> dict:
    """Compute Vimshottari from 3 independent seed points.

    Returns dict with keys 'moon', 'sun', 'h9_cusp', each a list of periods.
    """
    moon_long = chart['Moon']['longitude']
    sun_long = chart['Sun']['longitude']
    h9_cusp = (asc + 8 * 30) % 360  # 9th house cusp (equal house)

    return {
        'moon': _compute_from_longitude(moon_long, birth_jd, max_depth),
        'sun': _compute_from_longitude(sun_long, birth_jd, max_depth),
        'h9_cusp': _compute_from_longitude(h9_cusp, birth_jd, max_depth),
    }


def find_period_at_jd(periods: list, target_jd: float,
                      target_depth: int = None) -> dict:
    """Find the period containing a given Julian day.

    If target_depth specified, only match periods at that depth.
    """
    for p in periods:
        if target_depth is not None and p['depth'] != target_depth:
            continue
        if p['start_jd'] <= target_jd < p['end_jd']:
            return p
    return None
