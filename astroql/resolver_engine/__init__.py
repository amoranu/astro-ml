"""Aggregator + contradiction resolver (spec §6.8, §7 aggregation).

Phase 1: single-school aggregator only.
Phase 2: cross-school IoU clustering + contradiction detection.
"""
from .aggregator import Aggregator, AggregatorError
from .contradictions import annotate_all, detect_contradictions

__all__ = [
    "Aggregator", "AggregatorError",
    "annotate_all", "detect_contradictions",
]
