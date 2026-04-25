"""Natural-language query parser (spec §6.1).

Heuristic regex-based parser for the common NL queries in spec §11.
Returns a partial FocusQuery (relationship/life_area/effect/modifier)
that the caller must complete with birth details.

For ambiguous queries (e.g. "how is my father") raises ClarifyRequired.
A future LLM-based upgrade can replace this; the interface stays the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schemas.enums import Effect, LifeArea, Modifier, Relationship


class ClarifyRequired(ValueError):
    """Raised when NL is too ambiguous to parse without clarification."""

    def __init__(self, message: str, suggestions: list[str]):
        super().__init__(message)
        self.suggestions = suggestions


@dataclass
class ParsedNL:
    relationship: Relationship
    life_area: LifeArea
    effect: Effect
    modifier: Modifier


# Keyword maps. Order matters — earlier entries take precedence.
# Possessive-form rels (must contain "my X" or "X's") indicate the
# relationship is the subject. Otherwise "child/spouse" appearing as a
# bare noun is more likely a life-area mention.
_REL_KEYWORDS = [
    (r"\bmy\s+father\b|\b(?:dad|daddy)\b", Relationship.FATHER),
    (r"\bmy\s+mother\b|\b(?:mom|mum)\b", Relationship.MOTHER),
    (r"\bmy\s+(?:spouse|wife|husband|partner)\b|"
     r"\b(?:wife|husband)'s\b", Relationship.SPOUSE),
    (r"\bmy\s+(?:child|children|son|daughter|kid)\b",
     Relationship.CHILDREN),
    (r"\bmy\s+elder\s+(?:brother|sister|sibling)\b",
     Relationship.ELDER_SIBLING),
    (r"\bmy\s+younger\s+(?:brother|sister|sibling)\b",
     Relationship.YOUNGER_SIBLING),
    (r"\bmy\s+paternal\s+grandfather\b",
     Relationship.PATERNAL_GRANDFATHER),
    (r"\bmy\s+maternal\s+grandfather\b",
     Relationship.MATERNAL_GRANDFATHER),
    (r"\bmy\s+friend\b", Relationship.FRIEND),
    (r"\bmy\s+enem(?:y|ies)\b", Relationship.ENEMY),
    # 'I'/'me'/'my' bare → SELF default. Last so explicit relationships
    # take precedence.
    (r"\b(I|me|my|mine|myself)\b", Relationship.SELF),
]

_LIFE_KEYWORDS = [
    (r"\b(die|death|pass(?:ing)?|longev|lifespan|live)\b", LifeArea.LONGEVITY),
    (r"\b(marry|marri\w+|wedding|spous\w*|partner|relationship)\b",
     LifeArea.MARRIAGE),
    (r"\b(career|job|work|profess|business)\b", LifeArea.CAREER),
    (r"\b(child|children|progeny|son|daughter|kids|conceive|baby)\b",
     LifeArea.CHILDREN),
    (r"\b(money|wealth|finan|rich|income|salary|wealthy)\b", LifeArea.FINANCE),
    (r"\b(health|disease|ill|sick)\b", LifeArea.HEALTH),
    (r"\b(educat|study|school|college|degree)\b", LifeArea.EDUCATION),
    (r"\b(property|house|home|land|real estate)\b", LifeArea.PROPERTY),
    (r"\b(vehicle|car|conveyance)\b", LifeArea.VEHICLES),
    (r"\b(foreign|abroad|overseas|emigrat)\b", LifeArea.FOREIGN_TRAVEL),
    (r"\b(fame|reputation|famous|celebrity)\b", LifeArea.FAME),
    (r"\b(spirit|moksha|enlightenment)\b", LifeArea.SPIRITUALITY),
    (r"\b(litigation|lawsuit|court)\b", LifeArea.LITIGATION),
    (r"\b(courage|bravery)\b", LifeArea.COURAGE),
    (r"\bnature\b|\bkind of\b|\bwhat (?:kind|type|sort)\b", LifeArea.NATURE),
]

_EFFECT_KEYWORDS = [
    (r"\b(die|death|pass(?:ing)?|fall|loss|bankrupt|fail|illness)\b",
     Effect.EVENT_NEGATIVE),
    (r"\b(get|gain|achieve|promotion|win|born|earn|inherit|acquire)\b",
     Effect.EVENT_POSITIVE),
    (r"\bwhat (?:kind|type|sort)\b|\bnature\b|\bcharacter\b|\bdescribe\b",
     Effect.NATURE),
    (r"\bhow (?:much|good|wealthy|successful)\b|\bmagnitude\b",
     Effect.MAGNITUDE),
    (r"\bwill .* (?:ever|at all|happen)\b|"
     r"\bwill\s+I\s+have\s+(?:children|kids|a\s+spouse)\b|"
     r"\b(any|some)\s+(?:children|spouse)\b",
     Effect.EXISTENCE),
]

_MODIFIER_KEYWORDS = [
    (r"\bwhen\b|\bdate\b|\btiming\b", Modifier.TIMING),
    (r"\bin\s+\d{4}\b|\bnext\s+(?:year|month|\d+)\b|\bduring\s+\d{4}\b",
     Modifier.PROBABILITY),
    (r"\bhow\s+(?:much|good|wealthy)\b", Modifier.SCALE),
    (r"\bwhat (?:kind|type|sort)\b|\bnature\b|\bdescribe\b",
     Modifier.DESCRIPTION),
]


def _first_match(pattern_pairs, text):
    for pat, val in pattern_pairs:
        if re.search(pat, text, re.IGNORECASE):
            return val
    return None


def parse_nl(query_text: str) -> ParsedNL:
    """Parse a NL query into a ParsedNL tuple. Raises ClarifyRequired
    when ambiguous."""
    text = query_text.strip()
    if not text:
        raise ClarifyRequired("empty query", [])

    rel = _first_match(_REL_KEYWORDS, text)
    if rel is None:
        raise ClarifyRequired(
            f"No relationship found in {query_text!r}",
            ["father", "mother", "spouse", "self"],
        )

    life = _first_match(_LIFE_KEYWORDS, text)
    if life is None:
        raise ClarifyRequired(
            f"No life area found in {query_text!r}",
            ["longevity", "marriage", "career", "health", "finance"],
        )

    effect = _first_match(_EFFECT_KEYWORDS, text)
    if effect is None:
        # Heuristic defaults by life area.
        defaults = {
            LifeArea.LONGEVITY: Effect.EVENT_NEGATIVE,
            LifeArea.MARRIAGE: Effect.EVENT_POSITIVE,
            LifeArea.NATURE: Effect.NATURE,
        }
        effect = defaults.get(life, Effect.NATURE)

    modifier = _first_match(_MODIFIER_KEYWORDS, text)
    if modifier is None:
        # Default by effect: events → timing, nature → description, etc.
        modifier = {
            Effect.EVENT_NEGATIVE: Modifier.TIMING,
            Effect.EVENT_POSITIVE: Modifier.TIMING,
            Effect.NATURE: Modifier.DESCRIPTION,
            Effect.MAGNITUDE: Modifier.SCALE,
            Effect.EXISTENCE: Modifier.NULL,
        }.get(effect, Modifier.NULL)

    return ParsedNL(
        relationship=rel, life_area=life,
        effect=effect, modifier=modifier,
    )
