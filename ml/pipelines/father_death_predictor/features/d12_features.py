"""D12 Vimshottari features — parent-specific dasha timeline.

D12 Moon longitude differs from D1 Moon, giving an independent Vimshottari
with different period boundaries. D12's own maraka lords provide cross-chart
validation.
"""

from ..astro_engine.houses import get_sign_lord
from ..astro_engine.dignity import uchcha_bala
from ..astro_engine.multiref_dasha import _compute_from_longitude, find_period_at_jd

NATURAL_MALEFICS = {'Sun', 'Mars', 'Saturn', 'Rahu', 'Ketu'}


def d12_sign(longitude: float) -> int:
    """Compute which sign a planet falls in within D12.

    Parashari D12: divide each sign into 12 parts of 2.5 degrees.
    D12 sign = (natal_sign + amsha_number) mod 12.
    """
    sign = int(longitude / 30) % 12
    amsha = int((longitude % 30) / 2.5)  # 0-11
    return (sign + amsha) % 12


def d12_longitude(longitude: float) -> float:
    """Compute D12 longitude from D1 longitude."""
    ds = d12_sign(longitude)
    position_in_amsha = ((longitude % 30) % 2.5) / 2.5
    return ds * 30 + position_in_amsha * 30


def d12_maraka_lords(natal_asc: float) -> set:
    """Maraka lords for father within D12 chart.

    D12 ascendant from D1 ascendant degree.
    Father = 9th house of D12.
    Marakas = 2nd from 9th (= 10th of D12) + 7th from 9th (= 3rd of D12).
    """
    d12_asc_sign = d12_sign(natal_asc)
    h10_of_d12 = (d12_asc_sign + 9) % 12
    h3_of_d12 = (d12_asc_sign + 2) % 12
    return {get_sign_lord(h10_of_d12), get_sign_lord(h3_of_d12)}


def compute_d12_periods(natal_chart: dict, birth_jd: float,
                        max_depth: int = 3) -> list:
    """Compute Vimshottari from D12 Moon longitude."""
    d12_moon = d12_longitude(natal_chart['Moon']['longitude'])
    return _compute_from_longitude(d12_moon, birth_jd, max_depth)


def extract_d12_features(candidate: dict, natal_chart: dict,
                         natal_asc: float, d12_periods: list,
                         d1_marakas: set) -> dict:
    """Extract 6 D12-based features for a candidate pratyantardasha."""
    midpoint = (candidate['start_jd'] + candidate['end_jd']) / 2
    d12_p = find_period_at_jd(d12_periods, midpoint, target_depth=3)

    if d12_p is None:
        return {
            'd12_lord_is_d1_maraka': 0.0,
            'd12_lord_is_d12_maraka': 0.0,
            'd1_d12_same_lord': 0.0,
            'd1_d12_both_maraka': 0.0,
            'd12_lord_uchcha': 0.5,
            'd12_lord_malefic': 0.0,
        }

    d12_pl = d12_p['lords'][-1]
    d1_pl = candidate['lords'][-1]
    d12_mk = d12_maraka_lords(natal_asc)

    is_d1_mk = candidate.get('maraka_type', 'none') != 'none'
    is_d12_mk = d12_pl in d12_mk

    return {
        'd12_lord_is_d1_maraka': 1.0 if d12_pl in d1_marakas else 0.0,
        'd12_lord_is_d12_maraka': 1.0 if is_d12_mk else 0.0,
        'd1_d12_same_lord': 1.0 if d1_pl == d12_pl else 0.0,
        'd1_d12_both_maraka': 1.0 if is_d1_mk and is_d12_mk else 0.0,
        'd12_lord_uchcha': uchcha_bala(d12_pl, natal_chart[d12_pl]['longitude']),
        'd12_lord_malefic': 1.0 if d12_pl in NATURAL_MALEFICS else 0.0,
    }
