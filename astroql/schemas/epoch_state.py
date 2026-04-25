"""Epoch State schema (NEUROSYMBOLIC_ENGINE_DESIGN.md §2.1).

Pre-calculated astronomical snapshot for a discrete time window at
sookshma dasha granularity (MD → AD → PD → SD).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PlanetEpochState:
    transit_sign: str
    transit_house: int
    natal_house: int
    shadbala_coefficient: float
    is_retrograde: bool
    # Planets currently aspecting this planet's TRANSIT sign.
    aspects_receiving: List[str] = field(default_factory=list)
    # Planets currently aspecting this planet's NATAL sign — the
    # gochara form most classical Parashari rules actually mean when
    # they say "X afflicts/protects Y."
    aspects_on_natal: List[str] = field(default_factory=list)
    # Sidereal sign at birth (e.g., "Cancer"). Stable across all
    # epochs for a given chart; included on each epoch for self-
    # contained analysis.
    natal_sign: str = ""
    # Raw virupas retained for debugging / display; not used in CF math.
    shadbala_virupas: Optional[float] = None


@dataclass
class DashaStack:
    maha: str
    antar: str
    pratyantar: str
    sookshma: str


@dataclass
class EpochState:
    epoch_id: str
    start_time: datetime
    end_time: datetime
    dashas: DashaStack
    planets: Dict[str, PlanetEpochState]
    # Sidereal sign of the natal lagna (ascendant). Stable across all
    # epochs for a given chart; included on each epoch so rule
    # predicates can compute lagna-relative quantities (9th-lord,
    # maraka houses 2/7, lagna lord, etc.) without external lookup.
    natal_lagna_sign: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "epoch_id": self.epoch_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "dashas": asdict(self.dashas),
            "natal_lagna_sign": self.natal_lagna_sign,
            "planets": {
                name: asdict(ps) for name, ps in self.planets.items()
            },
        }
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpochState":
        planets = {
            name: PlanetEpochState(**ps)
            for name, ps in d["planets"].items()
        }
        return cls(
            epoch_id=d["epoch_id"],
            start_time=datetime.fromisoformat(d["start_time"]),
            end_time=datetime.fromisoformat(d["end_time"]),
            dashas=DashaStack(**d["dashas"]),
            natal_lagna_sign=d.get("natal_lagna_sign", ""),
            planets=planets,
        )
