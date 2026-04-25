"""Planner subsystem (spec §9).

  Phase 8: ExecutionTrace + Pipeline runner (legacy combined output).
  Phase F: per-tradition QueryPlan + reasoning lines + RRF fusion.
"""
from .trace import ExecutionTrace, TimedStage
from .pipeline import run_pipeline, PipelineResult
from .plan import (
    QueryPlan, ReasoningLine, ExceptionRule, ExceptionLibrary,
    PlanLoadError, load_plan, load_exception_library,
)
from .lines import evaluate_line, apply_exceptions
from .exceptions import evaluate_chart_static, evaluate_window_dynamic
from .fusion import fuse_rrf, fuse_weighted_sum, annotate_line_ranks
from .runner import run_tradition, run_multi_tradition

__all__ = [
    "ExecutionTrace", "TimedStage", "run_pipeline", "PipelineResult",
    "QueryPlan", "ReasoningLine", "ExceptionRule", "ExceptionLibrary",
    "PlanLoadError", "load_plan", "load_exception_library",
    "evaluate_line", "apply_exceptions",
    "evaluate_chart_static", "evaluate_window_dynamic",
    "fuse_rrf", "fuse_weighted_sum", "annotate_line_ranks",
    "run_tradition", "run_multi_tradition",
]
