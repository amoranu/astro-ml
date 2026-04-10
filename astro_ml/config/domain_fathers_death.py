"""Domain configuration for Father's Death prediction.

House rotation (Bhavat Bhavam): Father = 9th house.
- Primary houses (cause death): 3 (7th from 9th), 4 (8th from 9th), 10 (2nd from 9th)
- Secondary houses: 8 (12th from 9th), 12 (general loss)
- Negative/protective houses: 9, 5, 1, 11 (protect father's vitality)
"""

DOMAIN_NAME = "fathers_death"

# --- House groups ---
PRIMARY_HOUSES = [3, 4, 10]          # 7th, 8th, 2nd from 9th — cause father's death
SECONDARY_HOUSES = [8, 12]           # 12th from 9th, general loss
NEGATIVE_HOUSES = [9, 5, 1, 11]      # protect father's vitality
TARGET_CUSPS = [3, 4, 8, 10, 12]     # all cusps to monitor

# --- Karakas ---
NATURAL_KARAKAS = ["Sun", "Saturn"]   # Sun = Pitrukaraka, Saturn = death/separation
DEATH_KARAKAS = ["Saturn", "Ketu"]    # death and moksha karakas
ALL_KARAKAS = ["Sun", "Saturn", "Ketu"]
IS_NEGATIVE_EVENT = True              # death = negative event

# --- Dasha quality scoring ---
FUNC_NATURE_ENCODING = {
    "YOGAKARAKA": 3, "Yogakaraka": 3,
    "BENEFIC": 2, "Benefic": 2,
    "MIXED": 1, "Mixed": 1,
    "NEUTRAL": 0, "Neutral": 0,
    "MALEFIC": -1, "Malefic": -1,
    "SHADOW_PLANET": None,  # use sign lord's nature
}

DIGNITY_ENCODING = {
    "EXALTED": 5, "Exalted": 5,
    "OWN_SIGN": 4, "Own": 4, "Own Sign": 4,
    "FRIENDLY": 3, "Friend": 3, "Friendly": 3,
    "NEUTRAL": 2, "Neutral": 2,
    "ENEMY": 1, "Enemy": 1,
    "DEBILITATED": 0, "Debilitated": 0,
}

DIGNITY_FACTOR = {
    "EXALTED": 1.5, "Exalted": 1.5,
    "OWN_SIGN": 1.3, "Own": 1.3, "Own Sign": 1.3,
    "FRIENDLY": 1.1, "Friend": 1.1, "Friendly": 1.1,
    "NEUTRAL": 1.0, "Neutral": 1.0,
    "ENEMY": 0.7, "Enemy": 0.7,
    "DEBILITATED": 0.5, "Debilitated": 0.5,
}

NAV_DIGNITY_ENCODING = {
    "VARGOTTAMA+EXALTED": 7,
    "VARGOTTAMA+OWN_SIGN": 6,
    "EXALTED": 5,
    "VARGOTTAMA": 4,
    "OWN_SIGN": 3,
    "FRIENDLY": 2,
    "NEUTRAL": 1,
    "DEBILITATED": 0,
}

# --- Convergence tier definitions ---
# S-TIER: All 3 primary cusps active + dasha lock + transit + yogakaraka/exalted + quality >= 6.0
# A-TIER: Dasha lock on primary OR 2+ primary with yogakaraka OR double activation. Quality >= 3.0
# B-TIER: 2+ primary cusps active OR 1 primary + yogakaraka
# C-TIER: 1 primary cusp, no lock, no yogakaraka
# D-TIER: Only secondary cusps active
# F-TIER: No target cusps or only negative cusps
TIER_SCORES = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

# --- KP constants ---
SIGN_LORDS = {
    1: "Mars", 2: "Venus", 3: "Mercury", 4: "Moon",
    5: "Sun", 6: "Mercury", 7: "Venus", 8: "Mars",
    9: "Jupiter", 10: "Saturn", 11: "Saturn", 12: "Jupiter",
}

SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

SIGN_TO_NUM = {s: i + 1 for i, s in enumerate(SIGNS)}

PLANETS = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"]
SHADOW_PLANETS = {"Rahu", "Ketu"}
MALEFICS = {"Sun", "Mars", "Saturn", "Rahu", "Ketu"}
BENEFICS = {"Moon", "Mercury", "Jupiter", "Venus"}

BADHAKA_MAP = {"movable": 11, "fixed": 9, "dual": 7}
SIGN_MOBILITY = {
    1: "movable", 2: "fixed", 3: "dual", 4: "movable", 5: "fixed", 6: "dual",
    7: "movable", 8: "fixed", 9: "dual", 10: "movable", 11: "fixed", 12: "dual",
}

# Vedic aspects for transit planets (degrees)
VEDIC_ASPECTS = {
    "Jupiter": [0, 120, 180, 240],
    "Saturn": [0, 60, 90, 180, 270],
    "Rahu": [0, 120, 180, 240],
    "Ketu": [0, 120, 180, 240],
}
TRANSIT_ORB = 3.2  # degrees

# Main cusp for father's death analysis = Cusp 4 (8th from 9th)
MAIN_CUSP = 4
