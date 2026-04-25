"""Lookup tables for focus area resolution (spec §2.2–§2.4, §15.1–§15.3).

Pure data. Keyed by enum values (strings) so callers can use either the
enum or the raw string. Extensible by adding rows — no code change.
"""
from ..schemas.enums import LifeArea, Relationship

# ── RELATION_HOUSE (spec §2.2, §15.1) ────────────────────────────────
# Natal house associated with a relationship. Used as the rotation base.
RELATION_HOUSE: dict[str, int] = {
    Relationship.SELF.value: 1,
    Relationship.YOUNGER_SIBLING.value: 3,
    Relationship.MOTHER.value: 4,
    Relationship.CHILDREN.value: 5,
    Relationship.MATERNAL_UNCLE.value: 6,          # 3rd from 4th
    Relationship.ENEMY.value: 6,
    Relationship.SPOUSE.value: 7,
    Relationship.IN_LAWS.value: 1,                 # 7th from 7th
    Relationship.FATHER.value: 9,
    Relationship.PATERNAL_GRANDFATHER.value: 5,    # 9th from 9th
    Relationship.MATERNAL_GRANDFATHER.value: 12,   # 9th from 4th
    Relationship.ELDER_SIBLING.value: 11,
    Relationship.FRIEND.value: 11,
    Relationship.PATERNAL_UNCLE.value: 11,         # 3rd from 9th
}

# ── DOMAIN_HOUSE (spec §2.2, §15.2) ──────────────────────────────────
# Natal house associated with a life area. Rotation offset.
# Nature maps to house 1 as a sentinel — focus_resolver treats it as
# "no rotation" (target == relationship's own house).
DOMAIN_HOUSE: dict[str, int] = {
    LifeArea.NATURE.value: 1,
    LifeArea.FINANCE.value: 2,
    LifeArea.COURAGE.value: 3,
    LifeArea.HOME.value: 4,
    LifeArea.PROPERTY.value: 4,
    LifeArea.VEHICLES.value: 4,
    LifeArea.EDUCATION.value: 4,        # basic education; spirituality/higher goes elsewhere
    LifeArea.CHILDREN.value: 5,
    LifeArea.HEALTH.value: 6,
    LifeArea.LITIGATION.value: 6,
    LifeArea.MARRIAGE.value: 7,
    LifeArea.LONGEVITY.value: 8,
    LifeArea.SPIRITUALITY.value: 9,     # dharma; D20 Vimshamsa also relevant
    LifeArea.CAREER.value: 10,
    LifeArea.FAME.value: 10,
    LifeArea.FOREIGN_TRAVEL.value: 12,
}

# ── RELATION_KARAKA (spec §2.3) ──────────────────────────────────────
# Parashari natural significators for a relationship. For spouse, gender
# affects the karaka — resolver picks Venus (male native) or Jupiter (female).
# Represented as a list so multi-karaka relationships (e.g. spouse-both)
# can be returned together.
RELATION_KARAKA: dict[str, list[str]] = {
    Relationship.SELF.value: ["Lagna_Lord"],   # resolver substitutes actual lord
    Relationship.FATHER.value: ["Sun"],
    Relationship.MOTHER.value: ["Moon"],
    Relationship.SPOUSE.value: ["Venus", "Jupiter"],  # gender-filtered by resolver
    Relationship.CHILDREN.value: ["Jupiter"],
    Relationship.ELDER_SIBLING.value: ["Jupiter"],
    Relationship.YOUNGER_SIBLING.value: ["Mars"],
    Relationship.PATERNAL_UNCLE.value: ["Mars"],
    Relationship.MATERNAL_UNCLE.value: ["Mars"],
    Relationship.PATERNAL_GRANDFATHER.value: ["Sun"],
    Relationship.MATERNAL_GRANDFATHER.value: ["Moon"],
    Relationship.IN_LAWS.value: [],
    Relationship.FRIEND.value: [],
    Relationship.ENEMY.value: [],
}

# Jaimini Chara Karaka alias per relationship.
RELATION_JAIMINI_KARAKA: dict[str, list[str]] = {
    Relationship.SELF.value: ["AK"],
    Relationship.FATHER.value: ["PiK"],
    Relationship.MOTHER.value: ["MK"],
    Relationship.SPOUSE.value: ["DK"],
    Relationship.CHILDREN.value: ["PK"],
    Relationship.YOUNGER_SIBLING.value: ["BK"],
    Relationship.ELDER_SIBLING.value: [],
    Relationship.PATERNAL_UNCLE.value: [],
    Relationship.MATERNAL_UNCLE.value: [],
    Relationship.PATERNAL_GRANDFATHER.value: [],
    Relationship.MATERNAL_GRANDFATHER.value: [],
    Relationship.IN_LAWS.value: [],
    Relationship.FRIEND.value: [],
    Relationship.ENEMY.value: [],
}

# ── DOMAIN_KARAKA (spec §2.3) ────────────────────────────────────────
# Natural significators for a life area. "{N}L" is resolved at runtime
# to "lord of Nth house from target" by the feature extractor.
DOMAIN_KARAKA: dict[str, list[str]] = {
    LifeArea.LONGEVITY.value: ["Saturn", "8L"],
    LifeArea.HEALTH.value: ["Sun", "Mars", "Saturn", "Rahu"],
    LifeArea.FINANCE.value: ["Jupiter", "Venus", "2L", "11L"],
    LifeArea.MARRIAGE.value: ["Venus", "7L"],
    LifeArea.CAREER.value: ["Saturn", "Sun", "Mercury", "10L"],
    LifeArea.EDUCATION.value: ["Jupiter", "Mercury", "4L", "5L"],
    LifeArea.CHILDREN.value: ["Jupiter", "5L"],
    LifeArea.FOREIGN_TRAVEL.value: ["Rahu", "9L", "12L"],
    LifeArea.PROPERTY.value: ["Mars", "4L"],
    LifeArea.VEHICLES.value: ["Venus", "4L"],
    LifeArea.COURAGE.value: ["Mars", "3L"],
    LifeArea.HOME.value: ["Moon", "4L"],
    LifeArea.FAME.value: ["Sun", "10L"],
    LifeArea.SPIRITUALITY.value: ["Jupiter", "Ketu", "9L", "12L"],
    LifeArea.LITIGATION.value: ["Mars", "Saturn", "6L"],
    LifeArea.NATURE.value: ["Lagna_Lord"],
}

# ── DOMAIN_VARGA (spec §2.4) ─────────────────────────────────────────
# D1 implicitly always required; listed here for explicitness.
DOMAIN_VARGA: dict[str, list[str]] = {
    LifeArea.MARRIAGE.value: ["D1", "D9"],
    LifeArea.CAREER.value: ["D1", "D10"],
    LifeArea.CHILDREN.value: ["D1", "D7"],
    LifeArea.FINANCE.value: ["D1", "D2", "D11"],
    LifeArea.HEALTH.value: ["D1", "D6", "D30"],
    LifeArea.EDUCATION.value: ["D1", "D24"],
    LifeArea.PROPERTY.value: ["D1", "D4", "D16"],
    LifeArea.VEHICLES.value: ["D1", "D4", "D16"],
    LifeArea.LONGEVITY.value: ["D1", "D8", "D12", "D30"],
    LifeArea.SPIRITUALITY.value: ["D1", "D20"],
    LifeArea.FOREIGN_TRAVEL.value: ["D1", "D12"],
    LifeArea.COURAGE.value: ["D1", "D3"],
    LifeArea.HOME.value: ["D1", "D4"],
    LifeArea.FAME.value: ["D1", "D10"],
    LifeArea.NATURE.value: ["D1"],
    LifeArea.LITIGATION.value: ["D1", "D6"],
}

# ── Relationship → Varga overrides ───────────────────────────────────
# Relationship-specific vargas applied regardless of life area. Parents
# always want D12 (Dvadasamsa). Spouse always wants D9 (Navamsa).
# Children always want D7 (Saptamsa).
RELATION_VARGA: dict[str, list[str]] = {
    Relationship.FATHER.value: ["D12"],
    Relationship.MOTHER.value: ["D12"],
    Relationship.SPOUSE.value: ["D9"],
    Relationship.CHILDREN.value: ["D7"],
}
