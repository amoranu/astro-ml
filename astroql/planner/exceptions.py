"""Exception evaluator (Phase F.3).

Evaluates classical-override exception conditions against a chart and,
optionally, a candidate window. Two tiers of conditions:

  chart_static   : evaluated once per chart (e.g. "natal Saturn in own
                   sign aspected by Jupiter" softens Sade Sati)
  window_dynamic : evaluated per candidate window (e.g. condition uses
                   `candidate.*` paths)

The condition grammar is the same as rule antecedents (list of clauses
AND-combined, evaluated by the engine's `_evaluate_antecedent`). This
keeps the YAML authoring surface uniform: anything you can express in a
rule antecedent, you can express in an exception condition.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..engine.rule_engine import _evaluate_antecedent
from ..schemas.features import FeatureBundle
from .plan import ExceptionLibrary, ExceptionRule


def _bindings_to_reason(bindings: Dict[str, Any], cond_type: str) -> str:
    """Compress bindings into a short human-readable string."""
    if not bindings:
        return f"{cond_type} condition met"
    parts = [f"{k}={v}" for k, v in list(bindings.items())[:6]]
    return ", ".join(parts)


def evaluate_chart_static(
    library: ExceptionLibrary,
    bundle: FeatureBundle,
    exception_ids: List[str],
) -> List[Tuple[ExceptionRule, str]]:
    """Evaluate chart_static exceptions once per chart.

    Returns list of (exception, reason) for those that fired.
    """
    out: List[Tuple[ExceptionRule, str]] = []
    for eid in exception_ids:
        exc = library.by_id.get(eid)
        if exc is None or exc.condition_type != "chart_static":
            continue
        if not exc.condition:
            continue
        ok, bindings = _evaluate_antecedent(
            exc.condition, bundle, candidate=None,
        )
        if ok:
            out.append((exc, _bindings_to_reason(bindings, "chart_static")))
    return out


def evaluate_window_dynamic(
    library: ExceptionLibrary,
    bundle: FeatureBundle,
    candidate: Dict[str, Any],
    exception_ids: List[str],
) -> List[Tuple[ExceptionRule, str]]:
    """Evaluate window_dynamic exceptions for one candidate window."""
    out: List[Tuple[ExceptionRule, str]] = []
    for eid in exception_ids:
        exc = library.by_id.get(eid)
        if exc is None or exc.condition_type != "window_dynamic":
            continue
        if not exc.condition:
            continue
        ok, bindings = _evaluate_antecedent(
            exc.condition, bundle, candidate=candidate,
        )
        if ok:
            out.append((exc, _bindings_to_reason(bindings, "window_dynamic")))
    return out
