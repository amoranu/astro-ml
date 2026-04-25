"""Classical Parashari helpers: sign lordship, aspects, maraka/badhaka.

All logic is independent of astro-prod so the feature extractor can be
unit-tested without ephemeris calls.
"""
from __future__ import annotations

from typing import Iterable, List

# Sign name → index (1..12)
_SIGN_INDEX = {
    "Aries": 1, "Taurus": 2, "Gemini": 3, "Cancer": 4,
    "Leo": 5, "Virgo": 6, "Libra": 7, "Scorpio": 8,
    "Sagittarius": 9, "Capricorn": 10, "Aquarius": 11, "Pisces": 12,
}

# Classical Parashari sign lordship (no co-lords; Ketu/Rahu not assigned).
SIGN_LORD = {
    "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury",
    "Cancer": "Moon", "Leo": "Sun", "Virgo": "Mercury",
    "Libra": "Venus", "Scorpio": "Mars", "Sagittarius": "Jupiter",
    "Capricorn": "Saturn", "Aquarius": "Saturn", "Pisces": "Jupiter",
}

# Sign nature for badhaka rule.
#   movable → 11H is badhaka; fixed → 9H; dual → 7H.
SIGN_NATURE = {
    "Aries": "movable", "Cancer": "movable",
    "Libra": "movable", "Capricorn": "movable",
    "Taurus": "fixed", "Leo": "fixed",
    "Scorpio": "fixed", "Aquarius": "fixed",
    "Gemini": "dual", "Virgo": "dual",
    "Sagittarius": "dual", "Pisces": "dual",
}


def sign_lord(sign: str) -> str:
    """Classical lord of a sign. Raises KeyError for unknown names."""
    return SIGN_LORD[sign]


# Sign-index → sign-name (reverse lookup).
_SIGN_NAMES = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
               "Libra", "Scorpio", "Sagittarius", "Capricorn",
               "Aquarius", "Pisces"]


def compute_functional_roles(lagna_sign: str) -> dict:
    """Laghu Parashari Ch.1-4: classify each planet's functional role
    based on the houses it lords from the native's lagna.

    Returns:
        {
          "yogakarakas":        set[planet] — simultaneously lord K + T
          "functional_benefics": set[planet] — lord 1/4/5/7/9/10 without
                                             mixing trishadaya strongly
          "functional_malefics": set[planet] — lord 3/6/11 (trishadaya)
                                             or 8 (dusthana) primarily
          "lagna_lord":          planet
        }

    Classical principles applied:
      - Kendra (1,4,7,10) + Trikona (1,5,9) joint lordship = YK
      - Pure trikona (non-lagna) lord = benefic
      - Pure trishadaya (3,6,11) lord = malefic
      - Lagna lord always benefic
      - Rahu/Ketu always functional malefics
      - Mixed: trikona wins over trishadaya; pure kendra non-lagna = neutral
    """
    if lagna_sign not in _SIGN_INDEX:
        return {
            "yogakarakas": set(),
            "functional_benefics": set(),
            "functional_malefics": {"Rahu", "Ketu"},
            "lagna_lord": "",
        }

    lagna_idx = _SIGN_INDEX[lagna_sign] - 1  # 0-based
    # For each house (1..12), which sign and which lord.
    house_lord_map: dict[int, str] = {}
    for h in range(1, 13):
        sign_idx = (lagna_idx + h - 1) % 12
        sign_name = _SIGN_NAMES[sign_idx]
        house_lord_map[h] = SIGN_LORD[sign_name]

    # Invert: which houses does each planet lord?
    planet_houses: dict[str, set[int]] = {}
    for h, lord in house_lord_map.items():
        planet_houses.setdefault(lord, set()).add(h)

    kendras = {1, 4, 7, 10}
    trikonas = {1, 5, 9}
    trishadaya = {3, 6, 11}
    dusthanas = {6, 8, 12}  # trishadaya-6 overlaps

    yogakarakas: set[str] = set()
    functional_benefics: set[str] = set()
    functional_malefics: set[str] = set()

    lagna_lord = house_lord_map[1]

    for planet, houses in planet_houses.items():
        # Yogakaraka: simultaneously lords a NON-LAGNA kendra and a
        # NON-LAGNA trikona.
        non_lagna_k = (houses & kendras) - {1}
        non_lagna_t = (houses & trikonas) - {1}
        if non_lagna_k and non_lagna_t:
            yogakarakas.add(planet)
            continue

        # Lagna lord is benefic by default.
        if 1 in houses:
            functional_benefics.add(planet)
            continue

        has_trishadaya = bool(houses & trishadaya)
        has_8h = 8 in houses
        has_trikona = bool(houses & trikonas)  # non-1 since we skipped 1
        has_kendra = bool(houses & kendras)  # non-1

        # Pure trishadaya/8H lordship = malefic.
        if (has_trishadaya or has_8h) and not has_trikona:
            functional_malefics.add(planet)
            continue

        # Pure trikona = benefic.
        if has_trikona and not (has_trishadaya or has_8h):
            functional_benefics.add(planet)
            continue

        # Mixed trikona + trishadaya: Laghu Parashari favors trikona
        # unless the trishadaya lordship is stronger. Keep benefic by
        # default (trikona lordship dominates).
        if has_trikona:
            functional_benefics.add(planet)
            continue

        # Pure kendra (non-lagna): Parashari treats kendra lordships
        # as neutral — natural benefics lose sheen, natural malefics
        # improve. For simplicity, treat as benefic-leaning here.
        if has_kendra:
            functional_benefics.add(planet)
            continue

        # Fallback — shouldn't happen given house-lordship exhausts
        # all 12 houses across 7 planets.
        functional_malefics.add(planet)

    # Rahu/Ketu: universally functional malefics (no sign lordship).
    functional_malefics.add("Rahu")
    functional_malefics.add("Ketu")

    return {
        "yogakarakas": yogakarakas,
        "functional_benefics": functional_benefics,
        "functional_malefics": functional_malefics,
        "lagna_lord": lagna_lord,
    }


def sign_index(sign: str) -> int:
    return _SIGN_INDEX[sign]


def house_from(start_house: int, offset_houses: int) -> int:
    """1-based house arithmetic. `house_from(9, 2) = 10` = 2nd from 9H."""
    return ((start_house - 1) + (offset_houses - 1)) % 12 + 1


def maraka_houses(target_house: int) -> List[int]:
    """Maraka houses relative to a target — 2nd and 7th from target.

    Spec §6.4: "Maraka set for target_house: lords of 2H and 7H *from target*".
    """
    return [house_from(target_house, 2), house_from(target_house, 7)]


def badhaka_house(target_sign: str) -> int:
    """Badhaka sthana relative to a sign. 1-based house offset from the
    sign's house."""
    nature = SIGN_NATURE.get(target_sign, "dual")
    if nature == "movable":
        return 11
    if nature == "fixed":
        return 9
    return 7


# ── Parashari graha drishti (aspects) ────────────────────────────────
# All planets aspect the 7th. Jupiter also 5/9. Saturn 3/10. Mars 4/8.
# Rahu/Ketu are omitted (disputed in classical texts).
_SPECIAL_ASPECTS = {
    "Jupiter": [5, 7, 9],
    "Saturn": [3, 7, 10],
    "Mars": [4, 7, 8],
}


def planet_aspects(planet: str) -> List[int]:
    """House-offsets (1-based from planet) that this planet aspects."""
    return _SPECIAL_ASPECTS.get(planet, [7])


def aspects_house(planet_house: int, target_house: int, planet: str) -> bool:
    """Does `planet` placed in `planet_house` aspect `target_house`?"""
    for offset in planet_aspects(planet):
        if house_from(planet_house, offset) == target_house:
            return True
    return False


def occupants_of_house(
    planet_positions: dict, house: int, exclude: Iterable[str] = ("Lagna",),
) -> List[str]:
    ex = set(exclude)
    return [p for p, pp in planet_positions.items()
            if p not in ex and pp.house == house]


def planets_aspecting_house(
    planet_positions: dict, target_house: int,
    exclude: Iterable[str] = ("Lagna",),
) -> List[str]:
    ex = set(exclude)
    out: List[str] = []
    for p, pp in planet_positions.items():
        if p in ex:
            continue
        if aspects_house(pp.house, target_house, p):
            out.append(p)
    return out


# ── Phase F.9: Mrityu Bhaga, Vipreet Raja Yoga, Neecha Bhanga ────────

# Simplified Mrityu Bhaga (BPHS Ch.40): each planet activates death
# significance when its degree-in-sign falls on its critical degree
# (single-degree convention; ±1° tolerance). The full BPHS table varies
# the critical degree per sign; we use the modern condensed form which
# is what most practitioners use today.
MRITYU_BHAGA_DEG = {
    "Sun": 20.0,
    "Moon": 9.0,
    "Mars": 19.0,
    "Mercury": 15.0,
    "Jupiter": 11.0,
    "Venus": 27.0,
    "Saturn": 23.0,
    "Rahu": 14.0,
    "Ketu": 14.0,
    "Lagna": 1.0,
}


def in_mrityu_bhaga(
    planet: str, longitude: float, tolerance: float = 1.0,
) -> bool:
    """True iff the planet's degree-in-sign is within `tolerance`°
    of its classical Mrityu Bhaga degree."""
    crit = MRITYU_BHAGA_DEG.get(planet)
    if crit is None:
        return False
    deg_in_sign = float(longitude) % 30.0
    return abs(deg_in_sign - crit) <= tolerance


# Debilitation signs per planet (Parashari).
DEBILITATION_SIGN = {
    "Sun": "Libra", "Moon": "Scorpio", "Mars": "Cancer",
    "Mercury": "Pisces", "Jupiter": "Capricorn", "Venus": "Virgo",
    "Saturn": "Aries", "Rahu": "Scorpio", "Ketu": "Taurus",
}

# Exaltation signs per planet (Parashari).
EXALTATION_SIGN = {
    "Sun": "Aries", "Moon": "Taurus", "Mars": "Capricorn",
    "Mercury": "Virgo", "Jupiter": "Cancer", "Venus": "Pisces",
    "Saturn": "Libra", "Rahu": "Taurus", "Ketu": "Scorpio",
}

# Kendra (1/4/7/10) and Trikona (1/5/9) — kendrakona = union.
KENDRAKONA_HOUSES = {1, 4, 5, 7, 9, 10}


def detect_vipreet_raja_yoga(
    planet_positions: dict, house_signs: dict,
    target_houses: Iterable[int] = (6, 8, 12),
) -> List[dict]:
    """Detect Vipreet Raja Yoga: lord of a dusthana (6/8/12) sits in
    another dusthana (BPHS Ch.36). Returns one entry per dusthana whose
    lord is in another dusthana — "bad becomes good" inversion.

    Each entry: {dusthana_house, dusthana_lord, lord_in_house, peer_dusthana}
    """
    target_set = set(target_houses)
    out: List[dict] = []
    for h in target_set:
        sign = house_signs.get(h)
        if not sign:
            continue
        lord = SIGN_LORD.get(sign)
        if not lord:
            continue
        pos = planet_positions.get(lord)
        if pos is None:
            continue
        peer = pos.house
        if peer in target_set and peer != h:
            out.append({
                "dusthana_house": h,
                "dusthana_lord": lord,
                "lord_in_house": peer,
                "peer_dusthana": peer,
            })
    return out


_MALEFIC_NATURAL = {"Saturn", "Mars", "Rahu", "Ketu"}


def compute_father_natal_context(
    planet_positions: dict, house_signs: dict,
) -> dict:
    """v38.7: derive Pitri-karaka + 9th-bhava natal affliction primitives.

    Classical basis (Phaladeepika Ch.14; BPHS Ch.24-25):
      - Pitri karaka is Sun; 9H is father bhava. Father's longevity is
        read from (a) Sun's dignity + afflictions, (b) 9L's dignity +
        afflictions, (c) malefics in/aspecting 9H, (d) papa-kartari on
        Sun or on 9H, (e) pitri-dosha (Sun conj Saturn or Rahu).
      - Sun itself is NEVER counted as a malefic-on-9H — it indicates
        the karaka is AT its bhava (neutral-to-benefic placement).

    Returns a flat dict of booleans, ints, and a summary score suitable
    for direct assignment into chart_applicability.
    """
    sun = planet_positions.get("Sun")
    sun_house = getattr(sun, "house", 0) if sun else 0
    sun_dignity = getattr(sun, "dignity", "") if sun else ""
    sun_sign = getattr(sun, "sign", "") if sun else ""

    nine_sign = house_signs.get(9, "")
    ninth_lord = SIGN_LORD.get(nine_sign, "") if nine_sign else ""
    nl_pp = planet_positions.get(ninth_lord) if ninth_lord else None
    nl_house = getattr(nl_pp, "house", 0) if nl_pp else 0
    nl_dignity = getattr(nl_pp, "dignity", "") if nl_pp else ""
    nl_combust = bool(getattr(nl_pp, "combust", False)) if nl_pp else False

    # Sun affliction: conjunction (same sign) or aspect by natural malefic.
    sun_conj_malefics: List[str] = []
    if sun:
        for p, pp in planet_positions.items():
            if p in ("Sun", "Lagna"):
                continue
            if p in _MALEFIC_NATURAL and pp.sign == sun_sign:
                sun_conj_malefics.append(p)
    sun_aspected_by_malefics: List[str] = []
    if sun_house:
        for p in ("Saturn", "Mars"):  # nodes have no classical graha-drishti
            pp = planet_positions.get(p)
            if pp is None:
                continue
            if aspects_house(pp.house, sun_house, p):
                sun_aspected_by_malefics.append(p)
    sun_conj_malefic = bool(sun_conj_malefics)
    sun_aspected_by_malefic = bool(sun_aspected_by_malefics)
    sun_afflicted_natally = sun_conj_malefic or sun_aspected_by_malefic
    # Per-malefic flags for mechanism-matched rule gating. "Attacker is
    # already wired to the pitri karaka natally" → classical basis for
    # the attacker's transit/dasha to fire death.
    saturn_attacks_sun_natally = (
        "Saturn" in sun_conj_malefics or "Saturn" in sun_aspected_by_malefics
    )
    mars_attacks_sun_natally = (
        "Mars" in sun_conj_malefics or "Mars" in sun_aspected_by_malefics
    )
    rahu_ketu_attacks_sun_natally = (
        "Rahu" in sun_conj_malefics or "Ketu" in sun_conj_malefics
    )

    # 9L affliction: conjunction or aspect by natural malefic (excl. Sun).
    nl_conj_malefics: List[str] = []
    nl_aspected_by_malefics: List[str] = []
    nl_sign = getattr(nl_pp, "sign", "") if nl_pp else ""
    if nl_pp and nl_sign:
        for p, pp in planet_positions.items():
            if p in ("Lagna", ninth_lord):
                continue
            if p in _MALEFIC_NATURAL and pp.sign == nl_sign:
                nl_conj_malefics.append(p)
    if nl_house:
        for p in ("Saturn", "Mars"):
            if p == ninth_lord:
                continue
            pp = planet_positions.get(p)
            if pp is None:
                continue
            if aspects_house(pp.house, nl_house, p):
                nl_aspected_by_malefics.append(p)
    ninth_lord_afflicted = (
        bool(nl_conj_malefics) or bool(nl_aspected_by_malefics) or nl_combust
    )
    # Per-malefic flags on the 9L (father bhava's lord) — same pattern
    # as for Sun above. Enables rules to demand that the rule's agent
    # (Saturn / Mars / nodes) is natally wired to the father's bhava.
    saturn_attacks_ninth_lord_natally = (
        "Saturn" in nl_conj_malefics or "Saturn" in nl_aspected_by_malefics
    )
    mars_attacks_ninth_lord_natally = (
        "Mars" in nl_conj_malefics or "Mars" in nl_aspected_by_malefics
    )
    rahu_ketu_attacks_ninth_lord_natally = (
        "Rahu" in nl_conj_malefics or "Ketu" in nl_conj_malefics
    )

    # 9H malefic occupants / aspects (exclude Sun — Sun in 9H is the
    # karaka at its bhava, NOT an affliction).
    nine_occupants = occupants_of_house(planet_positions, 9)
    nine_occ_malefic_list = [
        p for p in nine_occupants if p in _MALEFIC_NATURAL
    ]
    nine_house_has_malefic = bool(nine_occ_malefic_list)
    nine_aspected_by = planets_aspecting_house(planet_positions, 9)
    nine_aspected_by_malefic_list = [
        p for p in nine_aspected_by if p in _MALEFIC_NATURAL
    ]
    nine_house_aspected_by_malefic = bool(nine_aspected_by_malefic_list)

    # Papa-kartari on Sun: malefic occupants in 2nd-from-Sun AND 12th-from-Sun.
    papa_kartari_on_sun = False
    if sun_house:
        h2 = house_from(sun_house, 2)
        h12 = house_from(sun_house, 12)
        occ2 = [p for p in occupants_of_house(planet_positions, h2)
                if p in _MALEFIC_NATURAL]
        occ12 = [p for p in occupants_of_house(planet_positions, h12)
                 if p in _MALEFIC_NATURAL]
        papa_kartari_on_sun = bool(occ2 and occ12)

    # Papa-kartari on 9H: malefic occupants in 8H AND 10H.
    occ8 = [p for p in occupants_of_house(planet_positions, 8)
            if p in _MALEFIC_NATURAL]
    occ10 = [p for p in occupants_of_house(planet_positions, 10)
             if p in _MALEFIC_NATURAL]
    papa_kartari_on_ninth = bool(occ8 and occ10)

    # Pitri Dosha: Sun with Saturn (same sign) or Sun with Rahu/Ketu
    # (same sign) in D1.
    saturn_pp = planet_positions.get("Saturn")
    rahu_pp = planet_positions.get("Rahu")
    ketu_pp = planet_positions.get("Ketu")
    pitri_sun_saturn = bool(
        sun and saturn_pp and sun_sign and sun_sign == saturn_pp.sign
    )
    pitri_sun_nodes = bool(
        sun and (
            (rahu_pp and sun_sign and sun_sign == rahu_pp.sign)
            or (ketu_pp and sun_sign and sun_sign == ketu_pp.sign)
        )
    )

    # Composite father-affliction score (0-6). Each factor contributes
    # at most 1. Designed so 0 = pristine, 3+ = severe.
    sun_in_dusthana = sun_house in (6, 8, 12)
    nl_in_dusthana = nl_house in (6, 8, 12)
    factors = [
        sun_in_dusthana,
        nl_in_dusthana or nl_combust,
        sun_afflicted_natally,
        ninth_lord_afflicted,
        nine_house_has_malefic or nine_house_aspected_by_malefic,
        papa_kartari_on_sun or papa_kartari_on_ninth,
        pitri_sun_saturn or pitri_sun_nodes,
    ]
    father_affliction_score = sum(1 for f in factors if f)
    # Stratification: the score is noisy, so we collapse to 3 buckets
    # used for rule gating:
    #   unafflicted   score<=1 (few rules should fire)
    #   moderate      score 2-3 (most rules fire, current default)
    #   severe        score>=4 (aggressive rules fire full-strength)
    father_unafflicted = father_affliction_score <= 1
    father_severely_afflicted = father_affliction_score >= 4
    father_moderately_afflicted = 2 <= father_affliction_score <= 3

    # Any-agent flag: is ANY natural malefic (Saturn/Mars/nodes)
    # natally attacking either Sun or 9L? Used as a soft "father bhava
    # has a wired attacker" gate — a more classical alternative to the
    # numeric affliction score.
    any_malefic_attacks_father = (
        saturn_attacks_sun_natally or mars_attacks_sun_natally
        or rahu_ketu_attacks_sun_natally
        or saturn_attacks_ninth_lord_natally
        or mars_attacks_ninth_lord_natally
        or rahu_ketu_attacks_ninth_lord_natally
    )

    return {
        "sun_house": sun_house,
        "sun_dignity": sun_dignity,
        "sun_in_dusthana": sun_in_dusthana,
        "sun_conj_malefic_natally": sun_conj_malefic,
        "sun_aspected_by_malefic_natally": sun_aspected_by_malefic,
        "sun_afflicted_natally": sun_afflicted_natally,
        "saturn_attacks_sun_natally": saturn_attacks_sun_natally,
        "mars_attacks_sun_natally": mars_attacks_sun_natally,
        "rahu_ketu_attacks_sun_natally": rahu_ketu_attacks_sun_natally,
        "ninth_lord": ninth_lord,
        "ninth_lord_house": nl_house,
        "ninth_lord_dignity": nl_dignity,
        "ninth_lord_combust": nl_combust,
        "ninth_lord_in_dusthana": nl_in_dusthana,
        "ninth_lord_afflicted_natally": ninth_lord_afflicted,
        "saturn_attacks_ninth_lord_natally": saturn_attacks_ninth_lord_natally,
        "mars_attacks_ninth_lord_natally": mars_attacks_ninth_lord_natally,
        "rahu_ketu_attacks_ninth_lord_natally": rahu_ketu_attacks_ninth_lord_natally,
        "any_malefic_attacks_father": any_malefic_attacks_father,
        "ninth_house_has_malefic": nine_house_has_malefic,
        "ninth_house_aspected_by_malefic": nine_house_aspected_by_malefic,
        "papa_kartari_on_sun": papa_kartari_on_sun,
        "papa_kartari_on_ninth": papa_kartari_on_ninth,
        "pitri_dosha_sun_saturn": pitri_sun_saturn,
        "pitri_dosha_sun_nodes": pitri_sun_nodes,
        "father_affliction_score": father_affliction_score,
        "father_unafflicted": father_unafflicted,
        "father_moderately_afflicted": father_moderately_afflicted,
        "father_severely_afflicted": father_severely_afflicted,
    }


def detect_neecha_bhanga(
    planet_positions: dict, house_signs: dict,
) -> List[dict]:
    """Detect Neecha Bhanga (debility cancellation) for any debilitated
    planet. Simplified rule (Phaladeepika Ch.6 condensed):

    A debilitated planet's debility is cancelled if EITHER:
      1. The lord of the sign in which it is debilitated (its dispositor)
         is in a kendra or trikona from lagna; OR
      2. The lord of the planet's exaltation sign is in a kendra or
         trikona from lagna.

    Returns one entry per cancelled debility.
    """
    out: List[dict] = []
    for planet, pos in planet_positions.items():
        if planet in ("Lagna",):
            continue
        if pos.dignity != "debilitated":
            continue
        # Dispositor = sign-lord of where the planet sits.
        dispositor = SIGN_LORD.get(pos.sign)
        disp_pos = (
            planet_positions.get(dispositor) if dispositor else None
        )
        # Lord of exaltation sign = the planet that owns the exaltation sign.
        exalt_sign = EXALTATION_SIGN.get(planet)
        exalt_lord = SIGN_LORD.get(exalt_sign) if exalt_sign else None
        exalt_lord_pos = (
            planet_positions.get(exalt_lord) if exalt_lord else None
        )
        cancellers: List[str] = []
        if disp_pos is not None and disp_pos.house in KENDRAKONA_HOUSES:
            cancellers.append(f"dispositor {dispositor} in H{disp_pos.house}")
        if (exalt_lord_pos is not None
                and exalt_lord_pos.house in KENDRAKONA_HOUSES):
            cancellers.append(
                f"exalt-lord {exalt_lord} in H{exalt_lord_pos.house}"
            )
        if cancellers:
            out.append({
                "planet": planet,
                "debilitation_sign": pos.sign,
                "cancellers": cancellers,
            })
    return out
