"""Declarative JSON-DSL evaluator for rule conditions.

Replaces the Python-lambda bottleneck so the LLM critic can emit
fully-executable rules as pure JSON. Conditions evaluate against an
`EpochState` (or any nested object/dict) via dot-path traversal +
operators + logical combinators.

Condition forms:

    {"path": "<dot.path>", "op": "<operator>", "value": <literal>}
    {"all": [<cond>, <cond>, ...]}     # logical AND
    {"any": [<cond>, <cond>, ...]}     # logical OR
    {"not": <cond>}                    # negation

Supported operators:

    ==, !=, <, <=, >, >=    Standard comparisons (numeric / string)
    in, not_in              Membership against a list literal
    contains                LHS list/string contains RHS literal
    is_in_set               LHS is a member of RHS set (alias for `in`)
    truthy, falsy           No `value`; LHS bool-coerced

Path traversal:

    "planets.Mars.transit_house"     → state.planets["Mars"].transit_house
    "dashas.maha"                    → state.dashas.maha
    "derived_lords.ninth_lord"       → state.derived_lords["ninth_lord"]

Path resolution accepts both attribute access (dataclass) and
mapping access (dict). A missing key/attr raises `DSLEvalError`.

This module is dependency-free (pure Python) so it can be unit-
tested without loading the ephemeris layer.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, Iterable, List, Mapping


class DSLEvalError(ValueError):
    """Raised when a condition is malformed or a path is unreachable."""


_BINARY_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "is_in_set": lambda a, b: a in b,
    "contains": lambda a, b: b in a,
}

_UNARY_OPS = {
    "truthy": lambda a: bool(a),
    "falsy":  lambda a: not bool(a),
}


def resolve_path(state: Any, path: str) -> Any:
    """Walk a dot-path against a nested object/mapping.

    Examples (`state` is an `EpochState`):
        resolve_path(state, "dashas.maha")          → "Saturn"
        resolve_path(state, "planets.Mars.is_retrograde") → True

    Raises DSLEvalError if any segment is missing.
    """
    if not isinstance(path, str) or not path:
        raise DSLEvalError(f"path must be a non-empty string, got {path!r}")
    cur: Any = state
    for i, seg in enumerate(path.split(".")):
        if isinstance(cur, Mapping):
            if seg not in cur:
                raise DSLEvalError(
                    f"path {path!r} segment[{i}]={seg!r} not in mapping "
                    f"with keys {sorted(list(cur.keys()))[:8]}"
                )
            cur = cur[seg]
        elif hasattr(cur, seg):
            cur = getattr(cur, seg)
        else:
            raise DSLEvalError(
                f"path {path!r} segment[{i}]={seg!r} unreachable on "
                f"{type(cur).__name__}"
            )
    return cur


def evaluate(condition: Any, state: Any) -> bool:
    """Evaluate a condition node against `state`.

    `condition` is typically a `Dict[str, Any]`. An empty dict `{}` is
    treated as a vacuous true (back-compat: existing rules use `{}` for
    Python-lambda-evaluated modifiers — they should not be re-evaluated
    via the DSL, the loader gates that out).
    """
    if condition is None:
        raise DSLEvalError("condition cannot be None")
    if not isinstance(condition, dict):
        raise DSLEvalError(
            f"condition must be a dict, got {type(condition).__name__}"
        )
    if not condition:
        # Vacuous true. Callers should treat `condition={}` as "use the
        # legacy Python predicate" rather than calling this evaluator.
        return True

    # Logical combinators (mutually exclusive, evaluated first)
    if "all" in condition:
        clauses = condition["all"]
        if not isinstance(clauses, list):
            raise DSLEvalError("'all' clauses must be a list")
        return all(evaluate(c, state) for c in clauses)
    if "any" in condition:
        clauses = condition["any"]
        if not isinstance(clauses, list):
            raise DSLEvalError("'any' clauses must be a list")
        return any(evaluate(c, state) for c in clauses)
    if "not" in condition:
        return not evaluate(condition["not"], state)

    # Leaf clause: must have path + op
    if "path" not in condition or "op" not in condition:
        raise DSLEvalError(
            f"leaf condition must have 'path' and 'op' keys, got "
            f"{sorted(condition.keys())}"
        )
    path = condition["path"]
    op = condition["op"]
    lhs = resolve_path(state, path)

    if op in _UNARY_OPS:
        if "value" in condition:
            raise DSLEvalError(
                f"unary op {op!r} must not carry a 'value' field"
            )
        return _UNARY_OPS[op](lhs)
    if op in _BINARY_OPS:
        if "value" not in condition:
            raise DSLEvalError(
                f"binary op {op!r} requires a 'value' field"
            )
        return _BINARY_OPS[op](lhs, condition["value"])
    raise DSLEvalError(
        f"unknown operator {op!r}. Valid: "
        f"{sorted(set(_BINARY_OPS) | set(_UNARY_OPS))}"
    )


def evaluate_modifier_indices(
    modifier_conditions: Iterable[Dict[str, Any]],
    state: Any,
) -> List[int]:
    """Evaluate a list of modifier conditions and return the indices
    of those that fire.

    A condition that's `{}` (vacuous) is treated as **not fired** here
    — the engine should fall back to legacy `modifier_predicates` for
    `{}` modifiers. This lets a single rule mix Python-lambda
    modifiers (legacy) and JSON-condition modifiers (LLM-emitted)
    without ambiguity.
    """
    fired: List[int] = []
    for idx, cond in enumerate(modifier_conditions):
        if not cond:
            continue  # vacuous → defer to legacy path
        try:
            if evaluate(cond, state):
                fired.append(idx)
        except DSLEvalError:
            # A malformed JSON condition must not crash the prediction
            # loop; skip it and let the loader / unit tests catch it.
            continue
    return fired
