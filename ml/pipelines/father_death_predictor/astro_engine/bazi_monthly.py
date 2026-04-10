"""BaZi Monthly Pillar computation via solar terms (Jie Qi).

Each Chinese month starts at a Jie Qi (major solar term) when the
Sun reaches a specific ecliptic longitude. Monthly stem computed
via Five Tigers Escape formula.

Uses Swiss Ephemeris for precise solar longitude computation.
"""

import swisseph as swe
import math

# Five Tigers formula: year_stem_index % 5 -> month_1_stem_index
_FIVE_TIGERS = {0: 2, 1: 4, 2: 6, 3: 8, 4: 0}

# 10 Heavenly Stems
STEMS = ['Jia', 'Yi', 'Bing', 'Ding', 'Wu', 'Ji', 'Geng', 'Xin', 'Ren', 'Gui']

# 12 Earthly Branches
BRANCHES = ['Zi', 'Chou', 'Yin', 'Mao', 'Chen', 'Si',
            'Wu', 'Wei', 'Shen', 'You', 'Xu', 'Hai']

# Month branches are fixed: month 1 (Lichun) = Yin (index 2)
# Month N branch = (N + 1) % 12
_MONTH_BRANCH_OFFSET = 2  # Yin

# Jie Qi solar longitudes (major terms that start months)
# Month 1: Lichun = Sun at 315 degrees
# Month 2: Jingzhe = 345
# etc. Each 30 degrees apart
_JIE_QI_LONGS = [315, 345, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285]
# Month numbers: 1 (Lichun/Feb), 2 (Jingzhe/Mar), ..., 12 (Xiaohan/Jan)

# Stem element and yin/yang
_STEM_ELEMENT = ['Wood', 'Wood', 'Fire', 'Fire', 'Earth', 'Earth',
                 'Metal', 'Metal', 'Water', 'Water']
_STEM_YANG = [True, False, True, False, True, False, True, False, True, False]

# Branch element
_BRANCH_ELEMENT = ['Water', 'Earth', 'Wood', 'Wood', 'Earth', 'Fire',
                   'Fire', 'Earth', 'Metal', 'Metal', 'Earth', 'Water']

# Hidden stems per branch (main, middle, residual)
_HIDDEN_STEMS = {
    0: [9],           # Zi: Gui
    1: [5, 9, 7],     # Chou: Ji, Gui, Xin
    2: [0, 2, 4],     # Yin: Jia, Bing, Wu
    3: [1],           # Mao: Yi
    4: [4, 1, 9],     # Chen: Wu, Yi, Gui
    5: [2, 4, 6],     # Si: Bing, Wu, Geng
    6: [3, 5],        # Wu: Ding, Ji
    7: [5, 3, 1],     # Wei: Ji, Ding, Yi
    8: [6, 4, 8],     # Shen: Geng, Wu, Ren
    9: [7],           # You: Xin
    10: [4, 7, 3],    # Xu: Wu, Xin, Ding
    11: [8, 0],       # Hai: Ren, Jia
}

# Stem clashes (Heavenly Stem clashes, Tian Gan Chong)
_STEM_CLASHES = {0: 6, 1: 7, 2: 8, 3: 9, 4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5}

# Branch clashes (Earthly Branch clashes, Di Zhi Chong)
_BRANCH_CLASHES = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11,
                   6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}


def _find_solar_longitude_jd(target_lon, start_jd, direction=1):
    """Find JD when tropical Sun reaches target_lon degrees.

    Uses binary search with Swiss Ephemeris.
    """
    # Rough estimate: Sun moves ~1 degree per day
    jd = start_jd
    for _ in range(50):
        lon = swe.calc_ut(jd, swe.SUN)[0][0]  # tropical longitude
        diff = (target_lon - lon) % 360
        if diff > 180:
            diff -= 360
        if abs(diff) < 0.001:
            return jd
        jd += diff  # ~1 degree per day
    return jd


def compute_year_stem(year):
    """Compute Heavenly Stem index for a year.

    (year - 4) % 10: 2024 = (2024-4)%10 = 0 = Jia
    """
    return (year - 4) % 10


def compute_monthly_pillars(year_stem, start_jd, end_jd):
    """Compute all monthly pillars between start_jd and end_jd.

    Args:
        year_stem: stem index (0-9) of the starting year
        start_jd: Julian day of window start
        end_jd: Julian day of window end

    Returns:
        List of period dicts with:
        {stem, branch, stem_idx, branch_idx, start_jd, end_jd,
         duration_days, depth, system, lords, hidden_stems}
    """
    periods = []

    # Find the first Jie Qi before start_jd
    # Start searching from ~35 days before start_jd
    search_jd = start_jd - 35

    # Compute all Jie Qi boundaries in range
    jieqi_dates = []
    jd = search_jd
    while jd < end_jd + 35:
        # Get current solar longitude
        lon = swe.calc_ut(jd, swe.SUN)[0][0]

        # Find the next Jie Qi after this JD
        for month_num in range(12):
            target = _JIE_QI_LONGS[month_num]
            jq_jd = _find_solar_longitude_jd(target, jd)
            if jq_jd > search_jd and jq_jd not in [j for j, _, _ in jieqi_dates]:
                jieqi_dates.append((jq_jd, month_num, target))

        jd += 30  # Jump ahead a month

    # Sort by JD
    jieqi_dates.sort()
    # Remove duplicates (within 1 day)
    cleaned = [jieqi_dates[0]] if jieqi_dates else []
    for jd, mn, tgt in jieqi_dates[1:]:
        if abs(jd - cleaned[-1][0]) > 1:
            cleaned.append((jd, mn, tgt))
    jieqi_dates = cleaned

    # Build monthly periods from consecutive Jie Qi boundaries
    for i in range(len(jieqi_dates) - 1):
        m_start = jieqi_dates[i][0]
        m_end = jieqi_dates[i + 1][0]
        month_num = jieqi_dates[i][1]  # 0-11

        if m_end <= start_jd or m_start >= end_jd:
            continue

        # Compute year for this month
        # Chinese year starts at Lichun (month_num=0 = 315 deg)
        # Rough: Jie Qi month 0 (Lichun) is ~Feb 4
        # Year stem changes at Lichun
        approx_year = int(2000 + (m_start - 2451545.0) / 365.25)
        if month_num == 0:
            curr_year_stem = compute_year_stem(approx_year)
        else:
            # If before Lichun of this calendar year, use previous year stem
            lichun_jd = _find_solar_longitude_jd(315, m_start - 180)
            if m_start < lichun_jd:
                curr_year_stem = compute_year_stem(approx_year - 1)
            else:
                curr_year_stem = compute_year_stem(approx_year)

        # Five Tigers: month 1 stem from year stem
        m1_stem = _FIVE_TIGERS[curr_year_stem % 5]
        stem_idx = (m1_stem + month_num) % 10
        branch_idx = (_MONTH_BRANCH_OFFSET + month_num) % 12

        periods.append({
            'lords': [f'{STEMS[stem_idx]}-{BRANCHES[branch_idx]}'],
            'stem_idx': stem_idx,
            'branch_idx': branch_idx,
            'stem': STEMS[stem_idx],
            'branch': BRANCHES[branch_idx],
            'start_jd': m_start,
            'end_jd': m_end,
            'duration_days': m_end - m_start,
            'depth': 1,
            'system': 'bazi_monthly',
            'hidden_stems': _HIDDEN_STEMS.get(branch_idx, []),
            'month_num': month_num,
        })

    return periods
