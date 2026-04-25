"""Core data schemas for AstroQL (spec §5).

Split by concern:
    enums       — Relationship, LifeArea, Effect, Modifier, School, QueryType
    birth       — BirthDetails, ChartConfig
    focus       — FocusQuery, ResolvedFocus
    chart       — PlanetPosition, Varga, DashaNode, Chart
    features    — FeatureBundle, Passage
    rules       — Rule, FiredRule
    results     — CandidateWindow, DescriptiveAttribute, QueryResult
"""
from .enums import (
    Relationship, LifeArea, Effect, Modifier, School, QueryType,
)
from .birth import BirthDetails, ChartConfig
from .focus import FocusQuery, ResolvedFocus
from .chart import PlanetPosition, Varga, DashaNode, Chart
from .features import FeatureBundle, Passage
from .rules import Rule, FiredRule
from .results import CandidateWindow, DescriptiveAttribute, QueryResult

__all__ = [
    "Relationship", "LifeArea", "Effect", "Modifier", "School", "QueryType",
    "BirthDetails", "ChartConfig",
    "FocusQuery", "ResolvedFocus",
    "PlanetPosition", "Varga", "DashaNode", "Chart",
    "FeatureBundle", "Passage",
    "Rule", "FiredRule",
    "CandidateWindow", "DescriptiveAttribute", "QueryResult",
]
