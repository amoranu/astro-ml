"""Parashari aspect strength with Gaussian kernel for continuous orbs."""

import numpy as np


def aspect_strength(planet_from: str, planet_to: str,
                    long_from: float, long_to: float) -> float:
    """Compute aspect strength from planet_from onto planet_to.

    Returns float [0, 1] — 1.0 = exact full aspect, 0.0 = no aspect.

    Parashari aspects:
    - All planets: 7th (180°) = full strength
    - Mars special: 4th (90°), 8th (210°) = 3/4 strength
    - Jupiter special: 5th (120°), 9th (240°) = full (trine)
    - Saturn special: 3rd (60°), 10th (300°) = 3/4 strength
    - Rahu/Ketu: only standard 7th aspect (traditional Parashari)
    """
    delta = (long_to - long_from) % 360
    sigma = 12.0  # orb tolerance in degrees

    # All planets have 7th aspect (180°)
    aspects = [(180.0, 1.0)]

    if planet_from == 'Mars':
        aspects += [(90.0, 0.75), (210.0, 0.75)]
    elif planet_from == 'Jupiter':
        aspects += [(120.0, 0.5), (240.0, 0.5)]
    elif planet_from == 'Saturn':
        aspects += [(60.0, 0.25), (300.0, 0.25)]

    max_str = 0.0
    for angle, base_strength in aspects:
        deviation = min(abs(delta - angle), 360 - abs(delta - angle))
        kernel = np.exp(-(deviation ** 2) / (2 * sigma ** 2))
        max_str = max(max_str, base_strength * kernel)

    return float(max_str)
