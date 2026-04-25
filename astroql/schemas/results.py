"""QueryResult and its variants (spec §5.7).

Phase F adds per-tradition planner output types alongside the legacy
combined `QueryResult`:
    YogaFiring         — one rule firing recorded against a window/line
    ExceptionFiring    — a classical override that fired (line- or
                         yoga-scoped attenuation)
    LineEvidence       — per-reasoning-line breakdown for one window
    RankedWindow       — final per-tradition ranked window with full evidence
    TraditionResult    — one school's complete output (plan + windows +
                         explanation)
    MultiTraditionResult — top-level Phase F return type bundling per-
                          tradition results without combining them
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import QueryType, School
from .focus import FocusQuery, ResolvedFocus
from .rules import FiredRule


@dataclass
class CandidateWindow:
    start: datetime
    end: datetime
    confidence_per_school: Dict[School, float] = field(default_factory=dict)
    aggregate_confidence: float = 0.0
    contributing_rules: List[FiredRule] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DescriptiveAttribute:
    attribute: str
    value: str
    confidence: float
    contributing_rules: List[FiredRule] = field(default_factory=list)


@dataclass
class QueryResult:
    query: FocusQuery
    resolved: ResolvedFocus
    query_type: QueryType

    windows: Optional[List[CandidateWindow]] = None
    probability: Optional[float] = None
    attributes: Optional[List[DescriptiveAttribute]] = None
    magnitude: Optional[Dict[str, Any]] = None
    yes_no: Optional[str] = None

    explain: Optional[Dict[str, Any]] = None
    inconclusive_reason: Optional[str] = None


# ── Phase F: per-tradition planner outputs ───────────────────────────

@dataclass
class YogaFiring:
    """One rule firing recorded inside a reasoning line."""
    yoga_id: str
    strength: float
    polarity: str                 # 'negative' (supports event) | 'positive' (attacks)
    tier: int = 2
    tier_weighted_strength: float = 0.0
    n_factors_matched: int = 0
    n_factors_total: int = 0
    matched_keys: List[str] = field(default_factory=list)
    source: str = ""


@dataclass
class ExceptionFiring:
    """An exception (classical override) that fired against this window."""
    exception_id: str
    scope: str                    # 'line' | 'yoga'
    attenuation: float            # multiplier in [0, ~2]; <1 dampens, >1 amplifies
    applies_to_yogas: List[str] = field(default_factory=list)
    applies_to_lines: List[str] = field(default_factory=list)
    reason: str = ""
    source: str = ""


@dataclass
class LineEvidence:
    """Per-reasoning-line scoring breakdown for one window."""
    line_id: str
    description: str
    net_strength: float
    raw_support: float
    raw_attack: float
    rank_in_line: int = -1
    support: List[YogaFiring] = field(default_factory=list)
    attacks: List[YogaFiring] = field(default_factory=list)
    exceptions_fired: List[ExceptionFiring] = field(default_factory=list)


@dataclass
class RankedWindow:
    """One ranked candidate window for a tradition (post-RRF)."""
    start: datetime
    end: datetime
    rrf_score: float
    final_rank: int
    line_evidence: Dict[str, LineEvidence] = field(default_factory=dict)
    structured_argument: str = ""
    candidate_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraditionResult:
    """One school's complete planner output. Independent of other schools."""
    school: School
    plan_id: str
    windows: List[RankedWindow] = field(default_factory=list)
    explanation: str = ""
    n_lines_total: int = 0
    n_windows_evaluated: int = 0
    aggregation_method: str = "rrf"
    inconclusive_reason: Optional[str] = None


@dataclass
class MultiTraditionResult:
    """Top-level Phase F output. Per-tradition results are isolated —
    no cross-tradition fusion happens here. Caller compares schools
    side-by-side (or picks one)."""
    query: FocusQuery
    resolved: ResolvedFocus
    query_type: QueryType
    parashari: Optional[TraditionResult] = None
    jaimini: Optional[TraditionResult] = None
    kp: Optional[TraditionResult] = None

    def by_school(self, school: School) -> Optional[TraditionResult]:
        return {
            School.PARASHARI: self.parashari,
            School.JAIMINI: self.jaimini,
            School.KP: self.kp,
        }.get(school)

    def schools_with_results(self) -> List[School]:
        out: List[School] = []
        if self.parashari is not None:
            out.append(School.PARASHARI)
        if self.jaimini is not None:
            out.append(School.JAIMINI)
        if self.kp is not None:
            out.append(School.KP)
        return out
