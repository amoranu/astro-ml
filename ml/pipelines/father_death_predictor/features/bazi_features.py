"""BaZi (Four Pillars) native features for monthly pillar pipeline.

Features based on stem/branch interactions between monthly pillar
and natal day pillar (Day Master). Father = Direct/Indirect Resource star.

~20 features per candidate.
"""

from ..astro_engine.bazi_monthly import (
    STEMS, BRANCHES, _STEM_ELEMENT, _STEM_YANG,
    _BRANCH_ELEMENT, _HIDDEN_STEMS, _STEM_CLASHES, _BRANCH_CLASHES
)

# Ten God derivation
_ELEMENT_ORDER = ['Wood', 'Fire', 'Earth', 'Metal', 'Water']
_GENERATES = {'Wood': 'Fire', 'Fire': 'Earth', 'Earth': 'Metal',
              'Metal': 'Water', 'Water': 'Wood'}
_CONTROLS = {'Wood': 'Earth', 'Fire': 'Metal', 'Earth': 'Water',
             'Metal': 'Wood', 'Water': 'Fire'}

# Father = Resource star (element that generates Day Master element)
# Direct Resource (Zheng Yin): same polarity, Indirect Resource (Pian Yin): opposite


def _ten_god(dm_elem, dm_yang, other_elem, other_yang):
    """Derive Ten God relationship."""
    if other_elem == dm_elem:
        return 'Rob' if other_yang == dm_yang else 'Friend'
    if _GENERATES.get(other_elem) == dm_elem:
        return 'DR' if other_yang != dm_yang else 'IR'  # Resource
    if _GENERATES.get(dm_elem) == other_elem:
        return 'EG' if other_yang == dm_yang else 'HO'  # Output
    if _CONTROLS.get(dm_elem) == other_elem:
        return 'DW' if other_yang != dm_yang else 'IW'  # Wealth
    if _CONTROLS.get(other_elem) == dm_elem:
        return '7K' if other_yang == dm_yang else 'DO'  # Power
    return 'Unknown'


def precompute_bazi_context(day_stem_idx, day_branch_idx,
                             year_stem_idx, year_branch_idx,
                             month_stem_idx, month_branch_idx):
    """Precompute BaZi natal context (from birth chart four pillars)."""
    dm_elem = _STEM_ELEMENT[day_stem_idx]
    dm_yang = _STEM_YANG[day_stem_idx]

    # Father star element = element that generates DM element
    # i.e., Resource element
    father_elem = [e for e, g in _GENERATES.items() if g == dm_elem][0]

    return {
        'dm_stem': day_stem_idx,
        'dm_branch': day_branch_idx,
        'dm_elem': dm_elem,
        'dm_yang': dm_yang,
        'year_stem': year_stem_idx,
        'year_branch': year_branch_idx,
        'natal_month_stem': month_stem_idx,
        'natal_month_branch': month_branch_idx,
        'father_elem': father_elem,
    }


def extract_bazi_features(candidate, bazi_ctx):
    """BaZi features for a monthly pillar candidate.

    Args:
        candidate: monthly pillar period dict
        bazi_ctx: from precompute_bazi_context()

    Returns: ~20 features dict.
    """
    ms = candidate['stem_idx']
    mb = candidate['branch_idx']
    m_elem = _STEM_ELEMENT[ms]
    m_yang = _STEM_YANG[ms]

    dm_stem = bazi_ctx['dm_stem']
    dm_branch = bazi_ctx['dm_branch']
    dm_elem = bazi_ctx['dm_elem']
    dm_yang = bazi_ctx['dm_yang']
    father_elem = bazi_ctx['father_elem']

    f = {}

    # --- Ten God of monthly stem ---
    tg = _ten_god(dm_elem, dm_yang, m_elem, m_yang)
    f['bz_is_resource'] = 1.0 if tg in ('DR', 'IR') else 0.0  # Father star
    f['bz_is_power'] = 1.0 if tg in ('7K', 'DO') else 0.0     # Pressure
    f['bz_is_wealth'] = 1.0 if tg in ('DW', 'IW') else 0.0
    f['bz_is_7killing'] = 1.0 if tg == '7K' else 0.0           # Violent

    # --- Father star in monthly pillar ---
    # Father element in month stem
    f['bz_father_in_stem'] = 1.0 if m_elem == father_elem else 0.0
    # Father element in hidden stems
    hidden = candidate.get('hidden_stems', [])
    f['bz_father_in_hidden'] = 1.0 if any(
        _STEM_ELEMENT[h] == father_elem for h in hidden) else 0.0

    # --- Stem clash with Day Master ---
    f['bz_stem_clash_dm'] = 1.0 if _STEM_CLASHES.get(ms) == dm_stem else 0.0

    # --- Branch clash with Day Branch (Spouse Palace) ---
    f['bz_branch_clash_dm'] = 1.0 if _BRANCH_CLASHES.get(mb) == dm_branch else 0.0

    # --- Branch clash with Year Branch ---
    f['bz_branch_clash_year'] = 1.0 if _BRANCH_CLASHES.get(mb) == bazi_ctx['year_branch'] else 0.0

    # --- Fu Yin (same pillar as natal month) ---
    f['bz_fu_yin_month'] = 1.0 if (
        ms == bazi_ctx['natal_month_stem'] and mb == bazi_ctx['natal_month_branch']
    ) else 0.0

    # --- Fan Yin (clash with natal month) ---
    f['bz_fan_yin_month'] = 1.0 if (
        _STEM_CLASHES.get(ms) == bazi_ctx['natal_month_stem']
        and _BRANCH_CLASHES.get(mb) == bazi_ctx['natal_month_branch']
    ) else 0.0

    # --- Stem combination (Six Combinations) ---
    # Jia-Ji, Yi-Geng, Bing-Xin, Ding-Ren, Wu-Gui
    combo_partner = (ms + 5) % 10
    f['bz_stem_combo_dm'] = 1.0 if combo_partner == dm_stem else 0.0

    # --- Branch element vs DM ---
    b_elem = _BRANCH_ELEMENT[mb]
    f['bz_branch_controls_dm'] = 1.0 if _CONTROLS.get(b_elem) == dm_elem else 0.0
    f['bz_branch_generates_dm'] = 1.0 if _GENERATES.get(b_elem) == dm_elem else 0.0

    # --- Hidden stems contain father/7K ---
    for h in hidden:
        h_tg = _ten_god(dm_elem, dm_yang, _STEM_ELEMENT[h], _STEM_YANG[h])
        if h_tg == '7K':
            f['bz_hidden_7k'] = 1.0
            break
    else:
        f['bz_hidden_7k'] = 0.0

    # --- Malefic count in hidden stems ---
    malefic_tg = {'7K', 'IR', 'HO'}  # 7 Killing, Indirect Resource (偏印), Hurting Officer
    f['bz_hidden_malefic_count'] = sum(
        1 for h in hidden
        if _ten_god(dm_elem, dm_yang, _STEM_ELEMENT[h], _STEM_YANG[h]) in malefic_tg
    )

    # --- Month number (seasonality) ---
    f['bz_month_num'] = candidate.get('month_num', 0)

    return f
