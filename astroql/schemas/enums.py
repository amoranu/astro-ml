"""Enums for FocusArea ontology (spec §2, §5)."""
from enum import Enum


class Relationship(str, Enum):
    SELF = "self"
    FATHER = "father"
    MOTHER = "mother"
    SPOUSE = "spouse"
    CHILDREN = "children"
    ELDER_SIBLING = "elder_sibling"
    YOUNGER_SIBLING = "younger_sibling"
    PATERNAL_UNCLE = "paternal_uncle"
    MATERNAL_UNCLE = "maternal_uncle"
    PATERNAL_GRANDFATHER = "paternal_grandfather"
    MATERNAL_GRANDFATHER = "maternal_grandfather"
    IN_LAWS = "in_laws"
    FRIEND = "friend"
    ENEMY = "enemy"


class LifeArea(str, Enum):
    NATURE = "nature"
    HEALTH = "health"
    LONGEVITY = "longevity"
    FINANCE = "finance"
    CAREER = "career"
    MARRIAGE = "marriage"
    CHILDREN = "children"
    EDUCATION = "education"
    PROPERTY = "property"
    VEHICLES = "vehicles"
    FOREIGN_TRAVEL = "foreign_travel"
    FAME = "fame"
    SPIRITUALITY = "spirituality"
    LITIGATION = "litigation"
    COURAGE = "courage"
    HOME = "home"


class Effect(str, Enum):
    EVENT_POSITIVE = "event_positive"
    EVENT_NEGATIVE = "event_negative"
    NATURE = "nature"
    MAGNITUDE = "magnitude"
    EXISTENCE = "existence"


class Modifier(str, Enum):
    TIMING = "timing"
    PROBABILITY = "probability"
    DESCRIPTION = "description"
    SCALE = "scale"
    NULL = "null"


class School(str, Enum):
    PARASHARI = "parashari"
    JAIMINI = "jaimini"
    KP = "kp"


class QueryType(str, Enum):
    TIMING = "timing"
    PROBABILITY = "probability"
    DESCRIPTION = "description"
    MAGNITUDE = "magnitude"
    YES_NO = "yes_no"
