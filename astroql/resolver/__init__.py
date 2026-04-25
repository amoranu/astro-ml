"""Focus area resolution: FocusQuery → ResolvedFocus."""
from .focus_resolver import FocusResolver, ResolverError
from .tables import (
    RELATION_HOUSE, DOMAIN_HOUSE,
    RELATION_KARAKA, DOMAIN_KARAKA,
    DOMAIN_VARGA, RELATION_JAIMINI_KARAKA, RELATION_VARGA,
)

__all__ = [
    "FocusResolver", "ResolverError",
    "RELATION_HOUSE", "DOMAIN_HOUSE",
    "RELATION_KARAKA", "DOMAIN_KARAKA",
    "DOMAIN_VARGA", "RELATION_JAIMINI_KARAKA", "RELATION_VARGA",
]
