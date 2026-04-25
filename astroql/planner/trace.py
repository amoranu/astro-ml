"""Execution trace (spec §9.4)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List


@dataclass
class TimedStage:
    name: str
    ms: float
    cached: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    stages: List[TimedStage] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, **extra) -> Iterator[Dict[str, Any]]:
        """Context manager. Pass extra kwargs that get attached to the stage.
        Mutate the yielded dict to add fields after the body runs.
        """
        t0 = time.perf_counter()
        bag: Dict[str, Any] = dict(extra)
        try:
            yield bag
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.stages.append(TimedStage(name=name, ms=ms, extra=bag))

    def to_list(self) -> List[Dict[str, Any]]:
        return [
            {"name": s.name, "ms": round(s.ms, 1),
             "cached": s.cached, **s.extra}
            for s in self.stages
        ]

    def total_ms(self) -> float:
        return sum(s.ms for s in self.stages)
