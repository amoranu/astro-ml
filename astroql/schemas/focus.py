"""FocusQuery + ResolvedFocus (spec §5.1, §5.2)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .birth import BirthDetails, ChartConfig
from .enums import Effect, LifeArea, Modifier, QueryType, Relationship, School


@dataclass
class FocusQuery:
    relationship: Relationship
    life_area: LifeArea
    effect: Effect
    modifier: Modifier = Modifier.NULL
    window: Optional[Tuple] = None

    birth: Optional[BirthDetails] = None
    config: Optional[ChartConfig] = None
    schools: List[School] = field(
        default_factory=lambda: [
            School.PARASHARI, School.JAIMINI, School.KP
        ]
    )

    min_confidence: float = 0.55
    require_confluence: int = 1
    school_weights: Dict[School, float] = field(
        default_factory=lambda: {
            School.PARASHARI: 0.4,
            School.JAIMINI: 0.2,
            School.KP: 0.4,
        }
    )
    explain: bool = False
    gender: Optional[str] = None


@dataclass
class ResolvedFocus:
    query: FocusQuery
    target_house_rotated: int
    target_house_direct: int
    relevant_houses: List[int]
    relation_karakas: List[str]
    domain_karakas: List[str]
    jaimini_karakas: List[str]
    vargas_required: List[str]
    dashas_required: List[str]
    need_transits: bool
    query_type: QueryType
