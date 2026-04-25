"""BirthDetails + ChartConfig (spec §5.1)."""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class BirthDetails:
    date: date
    time: Optional[str]
    tz: str
    lat: float
    lon: float
    time_accuracy: str = "exact"


@dataclass
class ChartConfig:
    ayanamsa: str = "lahiri"
    house_system: str = "whole_sign"
    karaka_scheme: str = "7"
    dasha_systems: List[str] = field(
        default_factory=lambda: ["vimshottari"]
    )
    vargas: List[str] = field(default_factory=lambda: ["D1", "D9"])
