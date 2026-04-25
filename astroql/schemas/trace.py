"""Execution trace schema (NEUROSYMBOLIC_ENGINE_DESIGN.md §2.3).

The trace is what the LLM critic consumes when a prediction fails. It
captures every step of CF evaluation for one (chart, target_aspect,
epoch) triple: which rules fired, what their initial/modulated CFs
were, which were subsumed by yoga-bhanga, and the final aggregate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FiredRuleTrace:
    rule_id: str
    initial_cf: float          # base_cf (before μ modulation)
    strength_multiplier: float  # μ
    modifiers_applied: List[str] = field(default_factory=list)
    final_cf: float = 0.0      # base_cf * μ, plus any modifier effects


@dataclass
class ExecutionTrace:
    query_id: str
    target_aspect: str
    final_score: float
    rules_fired: List[FiredRuleTrace] = field(default_factory=list)
    rules_subsumed: List[str] = field(default_factory=list)
    veto_fired: Optional[str] = None  # rule_id of triggering veto

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "target_aspect": self.target_aspect,
            "final_score": self.final_score,
            "rules_fired": [asdict(r) for r in self.rules_fired],
            "rules_subsumed": list(self.rules_subsumed),
            "veto_fired": self.veto_fired,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionTrace":
        return cls(
            query_id=d["query_id"],
            target_aspect=d["target_aspect"],
            final_score=d["final_score"],
            rules_fired=[FiredRuleTrace(**r) for r in d["rules_fired"]],
            rules_subsumed=list(d.get("rules_subsumed", [])),
            veto_fired=d.get("veto_fired"),
        )
