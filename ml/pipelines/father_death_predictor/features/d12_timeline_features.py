"""D12 (Dwadashamsha) as independent timeline predictor.

D12 is the parent chart. Instead of just confirming D1's dasha lord,
use D12's own Vimshottari as an independent witness:
- Does D12's concurrent period also have a maraka lord?
- Do D1 and D12 agree on maraka status?

~6 features per candidate.
"""

from ..astro_engine.houses import get_sign, get_sign_lord


def _find_d12_at_jd(d12_p3, jd):
    """Find D12 pratyantardasha containing a given JD."""
    for p in d12_p3:
        if p['start_jd'] <= jd < p['end_jd']:
            return p
    return None


def extract_d12_timeline_features(candidate, d12_p3,
                                  d1_marakas, d12_marakas):
    """D12 independent timeline features.

    Args:
        candidate: Vimshottari pratyantardasha (D1)
        d12_p3: D12 Vimshottari depth-3 periods
        d1_marakas: father's maraka set from D1
        d12_marakas: maraka set computed from D12 chart

    Returns:
        ~6 features dict.
    """
    midpoint_jd = (candidate['start_jd'] + candidate['end_jd']) / 2
    d12p = _find_d12_at_jd(d12_p3, midpoint_jd)

    vim_pd = candidate['lords'][-1]
    vim_pd_maraka = vim_pd in d1_marakas

    f = {}

    if d12p is None:
        f['d12t_pd_is_d1_maraka'] = 0.0
        f['d12t_pd_is_d12_maraka'] = 0.0
        f['d12t_both_maraka'] = 0.0
        f['d12t_md_is_maraka'] = 0.0
        f['d12t_maraka_count'] = 0
        f['d12t_cross_agree'] = 0.0
        return f

    d12_pd = d12p['lords'][-1]
    d12_md = d12p['lords'][0]
    d12_ad = d12p['lords'][1] if len(d12p['lords']) > 1 else d12_md

    # D12's PD lord maraka status (in both reference frames)
    f['d12t_pd_is_d1_maraka'] = 1.0 if d12_pd in d1_marakas else 0.0
    f['d12t_pd_is_d12_maraka'] = 1.0 if d12_pd in d12_marakas else 0.0

    # Both D1 and D12 PD lords are maraka (strongest agreement)
    f['d12t_both_maraka'] = 1.0 if (
        vim_pd_maraka and f['d12t_pd_is_d1_maraka']
    ) else 0.0

    # D12 MD lord is maraka (D12 mahadasha context)
    f['d12t_md_is_maraka'] = 1.0 if d12_md in d1_marakas else 0.0

    # Count of D12 lords that are D1 marakas
    f['d12t_maraka_count'] = sum([
        d12_md in d1_marakas,
        d12_ad in d1_marakas,
        d12_pd in d1_marakas,
    ])

    # Cross-system cascade: D1 has maraka cascade AND D12 confirms
    vim_md_maraka = candidate['lords'][0] in d1_marakas
    f['d12t_cross_agree'] = 1.0 if (
        vim_md_maraka and f['d12t_md_is_maraka']
    ) else 0.0

    return f
