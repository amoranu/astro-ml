"""Rule discovery via backtracing from known event dates.

For each chart with a known father-death date, walks the natal chart's
dasha tree to identify the (MD, AD, PAD) lords active on the death date,
tags each lord with its classical role relative to FATHER/LONGEVITY,
captures transit + conjunction context, then aggregates these patterns
across the train set to surface high-lift signatures.

Outputs YAML rule candidates that can be appended to the rule library.
"""
from .backtrace import DeathBacktrace, backtrace_chart, WindowFeatures
from .aggregate import (
    aggregate_lifts, FeatureLift, render_rule_candidates,
    discover_compounds, CompoundLift, render_compound_rules,
)

__all__ = [
    "DeathBacktrace", "backtrace_chart", "WindowFeatures",
    "aggregate_lifts", "FeatureLift", "render_rule_candidates",
    "discover_compounds", "CompoundLift", "render_compound_rules",
]
