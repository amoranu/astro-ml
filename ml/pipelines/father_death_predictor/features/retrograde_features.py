"""Retrograde dasha lord features.

Laghu Parashari: retrograde planets deliver amplified results —
a retrograde maraka is MORE dangerous; a retrograde benefic
provides LESS protection. Natal speed sign tells us this.

~6 features per candidate.
"""

NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}


def extract_retrograde_features(candidate, natal_chart):
    """Retrograde flags for MD/AD/PD lords.

    Uses natal chart speed < 0 as retrograde indicator.
    Rahu/Ketu are always retrograde (mean node), skip them.
    """
    lords = candidate['lords']
    md_lord = lords[0]
    ad_lord = lords[1]
    pd_lord = lords[-1]

    f = {}

    def _is_retro(planet):
        if planet in ('Rahu', 'Ketu'):
            return False  # Always retrograde, not informative
        return natal_chart[planet]['speed'] < 0

    md_retro = _is_retro(md_lord)
    ad_retro = _is_retro(ad_lord)
    pd_retro = _is_retro(pd_lord)

    f['rt_pd_retrograde'] = 1.0 if pd_retro else 0.0
    f['rt_ad_retrograde'] = 1.0 if ad_retro else 0.0
    f['rt_md_retrograde'] = 1.0 if md_retro else 0.0

    # Count of retrograde lords in chain
    f['rt_retro_count'] = sum([md_retro, ad_retro, pd_retro])

    # Retrograde maraka = amplified danger
    pd_is_maraka = candidate.get('maraka_type', 'none') != 'none'
    f['rt_retro_maraka'] = 1.0 if (pd_retro and pd_is_maraka) else 0.0

    # Retrograde malefic in chain (amplified malefic effect)
    retro_malefic = any(
        _is_retro(lords[i]) and lords[i] in NATURAL_MALEFICS
        for i in range(len(lords))
    )
    f['rt_retro_malefic'] = 1.0 if retro_malefic else 0.0

    return f
