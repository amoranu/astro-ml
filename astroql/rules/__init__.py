"""Structured rule library — YAML-defined rules per (school, life_area)."""
from .loader import StructuredRuleLibrary, RuleLoadError

__all__ = ["StructuredRuleLibrary", "RuleLoadError"]
