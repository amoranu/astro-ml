"""FeatureBundle + Passage (spec §5.4, §7)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import School
from .focus import ResolvedFocus


@dataclass
class FeatureBundle:
    school: School
    focus: ResolvedFocus

    primary_house_data: Dict[str, Any] = field(default_factory=dict)
    karaka_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    varga_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    dasha_candidates: Optional[List[Dict]] = None
    transit_events: Optional[List[Dict]] = None

    jaimini_features: Optional[Dict[str, Any]] = None
    kp_features: Optional[Dict[str, Any]] = None
    strength_data: Optional[Dict[str, Any]] = None
    descriptive_signals: Optional[Dict[str, Any]] = None

    # v38: chart-level applicability (evaluated once per chart, drives
    # rule.applicable_when and window-applicability multipliers).
    # See astroql.features.classical.compute_functional_roles +
    # parashari.py for population logic.
    chart_applicability: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Passage:
    """A RAG-retrieved text chunk with metadata."""
    passage_id: str
    text: str
    source: str
    score: float = 0.0
    rule_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
