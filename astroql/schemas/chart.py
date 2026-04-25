"""Chart schemas (spec §5.3)."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .birth import BirthDetails, ChartConfig


@dataclass
class PlanetPosition:
    planet: str
    longitude: float
    sign: str
    house: int
    nakshatra: str
    nakshatra_lord: str
    nakshatra_pada: int
    sub_lord: Optional[str] = None
    sub_sub_lord: Optional[str] = None
    retrograde: bool = False
    combust: bool = False
    dignity: str = "neutral"
    speed: float = 0.0


@dataclass
class Varga:
    name: str
    planet_positions: Dict[str, PlanetPosition]
    house_cusps: List[float]
    house_signs: Dict[int, str]


@dataclass
class DashaNode:
    lord: str
    start: datetime
    end: datetime
    level: int
    children: List["DashaNode"] = field(default_factory=list)


@dataclass
class Chart:
    birth: BirthDetails
    config: ChartConfig
    vargas: Dict[str, Varga]
    vimshottari: Optional[DashaNode] = None
    chara: Optional[DashaNode] = None
    yogini: Optional[DashaNode] = None

    chara_karakas: Dict[str, str] = field(default_factory=dict)
    arudhas: Dict[int, int] = field(default_factory=dict)

    kp_cusps: Optional[List[float]] = None
    kp_cuspal_sublords: Optional[Dict[int, str]] = None
    kp_significators: Optional[Dict[str, List[int]]] = None
    # Richer KP fields (supplement to spec §5.3 for feature extraction).
    kp_cusp_details: Optional[Dict[int, Dict]] = None
    kp_planet_details: Optional[Dict[str, Dict]] = None
    kp_planet_houses: Optional[Dict[str, int]] = None

    shadbala: Optional[Dict[str, float]] = None
    ashtakavarga: Optional[Dict[int, int]] = None
