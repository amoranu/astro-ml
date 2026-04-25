"""End-to-end CF predictor: chart + window → PredictedEvent.

Wraps Modules A and B for evaluation use. Rules are passed as
`(Rule, predicate)` pairs where `predicate(epoch_state) -> bool`. The
predictor emits sookshma epochs across [query_start, query_end], fires
the predicates per epoch, runs `cf_engine.infer_cf`, and returns the
single most-extreme epoch (most negative for negative-event aspects)
as the predicted event date.

This is the v1 testing path that lets us run real evaluation without
yet building a full feature-path clause evaluator. CF-native YAML
rules with formal antecedent clauses come in v2 once the EpochState
clause schema is defined.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

from ..schemas.birth import BirthDetails
from ..schemas.epoch_state import EpochState
from ..schemas.rules import FiredRule, Rule
from ..schemas.trace import ExecutionTrace
from . import cf_engine, dsl_evaluator, epoch_emitter
from . import shadbala as _sb


RulePredicate = Callable[[EpochState], bool]
JSONCondition = dict
FiresWhen = Union[RulePredicate, JSONCondition]


def _resolve_predicate(
    pred: FiresWhen, ep: EpochState,
) -> bool:
    """Evaluate a fires_when / modifier predicate against an EpochState.

    Accepts either:
      * Python callable `(EpochState) -> bool` (legacy, human-authored)
      * JSON-DSL dict (LLM-autonomous, see `dsl_evaluator`)

    A dict that is empty is treated as vacuous-true (matches the
    DSL evaluator's behavior).
    """
    if callable(pred):
        return bool(pred(ep))
    if isinstance(pred, dict):
        return dsl_evaluator.evaluate(pred, ep)
    raise TypeError(
        f"predicate must be callable or dict, got {type(pred).__name__}"
    )


@dataclass
class CFRuleSpec:
    """A rule plus the predicates needed to fire it and any modifiers.

    Predicates may be either Python callables (legacy) OR JSON-DSL
    dicts (LLM-autonomous). The two forms are interchangeable; the
    engine evaluates whichever shape it receives.

    `modifier_predicates` is parallel to `rule.modifiers`: index i
    decides whether modifier i fires. A predicate that returns False
    (or a JSON condition that evaluates to False) means the modifier
    is not applied for this epoch.

    For LLM-emitted rules: prefer attaching the JSON condition to
    `rule.modifiers[i].condition` and leaving `modifier_predicates`
    empty. The engine will pick those up via the DSL path inside
    `cf_engine.infer_cf` without any spec-level glue.
    """
    rule: Rule
    fires_when: FiresWhen
    modifier_predicates: List[FiresWhen] = field(default_factory=list)


@dataclass
class EpochScore:
    epoch_id: str
    epoch_start: _dt.datetime
    epoch_end: _dt.datetime
    midpoint: _dt.datetime
    cf: float
    fired_rule_ids: List[str]
    trace: ExecutionTrace


@dataclass
class PredictionResult:
    predicted_date: _dt.date
    cf: float
    epoch_scores: List[EpochScore]
    extreme_epoch: EpochScore  # the one we picked


def _natal_mu_from_first_epoch(
    epochs: List[EpochState],
) -> dict:
    """Natal μ is identical across all epochs — pull from epoch[0]."""
    if not epochs:
        return {}
    return {
        p: ps.shadbala_coefficient
        for p, ps in epochs[0].planets.items()
    }


RuleInput = Union[CFRuleSpec, Tuple[Rule, RulePredicate]]


def _normalize(rule_input: RuleInput) -> CFRuleSpec:
    """Accept either a (Rule, predicate) tuple (legacy v1) or a full
    CFRuleSpec. Tuples are wrapped with empty modifier_predicates.
    """
    if isinstance(rule_input, CFRuleSpec):
        return rule_input
    rule, pred = rule_input
    return CFRuleSpec(rule=rule, fires_when=pred)


def predict_extreme_epoch(
    birth: BirthDetails,
    query_start: _dt.datetime,
    query_end: _dt.datetime,
    rules: Sequence[RuleInput],
    polarity: str = "negative",
    max_window_years: float = 5.0,
) -> Optional[PredictionResult]:
    """Predict the date of an extreme-CF epoch within the window.

    polarity="negative": pick epoch with the MOST-negative CF
    polarity="positive": pick epoch with the MOST-positive CF
    polarity="absolute": pick epoch with the largest |CF|

    Returns None if no rule fires anywhere in the window or no epochs
    are emitted (window too narrow / chart invalid).
    """
    specs = [_normalize(r) for r in rules]
    epochs = epoch_emitter.emit_epochs(
        birth, query_start, query_end,
        max_window_years=max_window_years,
    )
    if not epochs:
        return None
    mu = _natal_mu_from_first_epoch(epochs)

    scores: List[EpochScore] = []
    for ep in epochs:
        fired: List[FiredRule] = []
        for spec in specs:
            try:
                if not _resolve_predicate(spec.fires_when, ep):
                    continue
            except Exception:
                # A predicate that errors on a particular epoch is a
                # rule-authoring bug, not a per-chart abort. Skip.
                continue
            mod_idxs: List[int] = []
            for idx, mp in enumerate(spec.modifier_predicates):
                try:
                    if _resolve_predicate(mp, ep):
                        mod_idxs.append(idx)
                except Exception:
                    continue
            fired.append(FiredRule(
                rule=spec.rule,
                fired_modifier_indices=mod_idxs,
            ))
        if not fired:
            continue
        cf, trace = cf_engine.infer_cf(
            fired, mu, target_aspect="longevity", query_id=ep.epoch_id,
            epoch_state=ep,
        )
        mid = ep.start_time + (ep.end_time - ep.start_time) / 2
        scores.append(EpochScore(
            epoch_id=ep.epoch_id,
            epoch_start=ep.start_time,
            epoch_end=ep.end_time,
            midpoint=mid,
            cf=cf,
            fired_rule_ids=[fr.rule.rule_id for fr in fired],
            trace=trace,
        ))

    if not scores:
        return None
    if polarity == "negative":
        extreme = min(scores, key=lambda s: s.cf)
    elif polarity == "positive":
        extreme = max(scores, key=lambda s: s.cf)
    else:
        extreme = max(scores, key=lambda s: abs(s.cf))
    return PredictionResult(
        predicted_date=extreme.midpoint.date(),
        cf=extreme.cf,
        epoch_scores=scores,
        extreme_epoch=extreme,
    )
