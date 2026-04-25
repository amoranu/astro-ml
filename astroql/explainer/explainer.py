"""Explainer (spec §6.9, §9.4).

Produces a structured EXPLAIN dict + human narrative from a QueryResult.
Phase 1 is single-school; `stages` list is minimal and will grow as
planner/executor gains per-stage timings in Phase 5+.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas.enums import QueryType, School
from ..schemas.results import CandidateWindow, QueryResult


class Explainer:
    def explain(self, result: QueryResult,
                extra_stages: Optional[List[Dict]] = None,
                trace: Optional[Any] = None) -> Dict[str, Any]:
        q = result.query
        r = result.resolved
        out: Dict[str, Any] = {
            "query": {
                "relationship": q.relationship.value,
                "life_area": q.life_area.value,
                "effect": q.effect.value,
                "modifier": q.modifier.value,
                "schools": [s.value for s in q.schools],
                "gender": q.gender,
            },
            "resolved": {
                "target_house_rotated": r.target_house_rotated,
                "target_house_direct": r.target_house_direct,
                "relation_karakas": r.relation_karakas,
                "domain_karakas": r.domain_karakas,
                "vargas_required": r.vargas_required,
                "dashas_required": r.dashas_required,
                "query_type": r.query_type.value,
            },
            "stages": list(extra_stages or []),
        }
        if trace is not None:
            try:
                out["stages"] = trace.to_list()
                out["total_ms"] = round(trace.total_ms(), 1)
            except AttributeError:
                pass

        if result.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            out["windows"] = [
                self._window_to_dict(w) for w in (result.windows or [])
            ]
        if result.query_type == QueryType.DESCRIPTION:
            out["attributes"] = [
                {"attribute": a.attribute, "value": a.value,
                 "confidence": round(a.confidence, 3),
                 "contributing_rule_ids": [
                     fr.rule.rule_id for fr in a.contributing_rules
                 ]}
                for a in (result.attributes or [])
            ]
        if result.query_type == QueryType.MAGNITUDE:
            out["magnitude"] = result.magnitude
        if result.query_type == QueryType.YES_NO:
            out["yes_no"] = result.yes_no
            out["evidence"] = result.magnitude
        if result.inconclusive_reason:
            out["inconclusive_reason"] = result.inconclusive_reason
        return out

    def narrate(self, result: QueryResult) -> str:
        lines: List[str] = []
        q = result.query
        lines.append(
            f"Query: {q.relationship.value} + {q.life_area.value} + "
            f"{q.effect.value} + {q.modifier.value}"
        )
        lines.append(
            f"Target house (rotated): {result.resolved.target_house_rotated}, "
            f"direct: {result.resolved.target_house_direct}. "
            f"Karakas: {result.resolved.relation_karakas + result.resolved.domain_karakas}."
        )
        if result.query_type == QueryType.DESCRIPTION:
            attrs = result.attributes or []
            if not attrs:
                reason = result.inconclusive_reason or "no attributes"
                lines.append(f"Inconclusive: {reason}")
                return "\n".join(lines)
            lines.append(f"Descriptive attributes ({len(attrs)}):")
            by_attr: Dict[str, List] = {}
            for a in attrs:
                by_attr.setdefault(a.attribute, []).append(a)
            for attr_name, entries in by_attr.items():
                lines.append(f"  {attr_name}:")
                for e in entries[:5]:
                    lines.append(
                        f"    - {e.value}  conf={e.confidence:.2f}  "
                        f"({len(e.contributing_rules)} rules)"
                    )
            return "\n".join(lines)
        if result.query_type == QueryType.MAGNITUDE:
            m = result.magnitude or {}
            lines.append(
                f"Magnitude: {m.get('ordinal','?')}  "
                f"(score={m.get('score',0):+.2f}, "
                f"pos={m.get('positive_sum',0):.2f}, "
                f"neg={m.get('negative_sum',0):.2f})"
            )
            return "\n".join(lines)
        if result.query_type == QueryType.YES_NO:
            m = result.magnitude or {}
            lines.append(
                f"Answer: "
                f"{result.yes_no.upper() if result.yes_no else '?'}  "
                f"(pos={m.get('positive_evidence',0):.2f}, "
                f"neg={m.get('negative_evidence',0):.2f})"
            )
            if result.yes_no == "inconclusive":
                lines.append(f"Reason: {result.inconclusive_reason or ''}")
            return "\n".join(lines)
        if result.query_type in (QueryType.TIMING, QueryType.PROBABILITY):
            windows = result.windows or []
            if not windows:
                reason = result.inconclusive_reason or "no candidate windows"
                lines.append(f"Inconclusive: {reason}")
                return "\n".join(lines)
            lines.append(f"Top {len(windows)} candidate windows:")
            for i, w in enumerate(windows, 1):
                duration = (w.end - w.start).days
                lines.append(
                    f"  {i}. {w.start.date()} -> {w.end.date()} "
                    f"({duration}d)  conf={w.aggregate_confidence:.2f}  "
                    f"rules={len(w.contributing_rules)}"
                )
                # Top 3 contributing rules for brevity
                # Show per-school confidence if multi-school.
                if len(w.confidence_per_school) > 1:
                    sch = "  ".join(
                        f"{s.value}={c:.2f}"
                        for s, c in sorted(
                            w.confidence_per_school.items(),
                            key=lambda kv: kv[0].value,
                        )
                    )
                    lines.append(f"       schools: {sch}")
                for fr in w.contributing_rules[:4]:
                    lines.append(
                        f"       - {fr.rule.rule_id}  "
                        f"[{fr.rule.rule_type}]  s={fr.strength:.2f}"
                    )
                if w.contradictions:
                    for c in w.contradictions:
                        lines.append(
                            f"       ! contradiction ({c['type']}): "
                            f"{c['detail']}"
                        )
            lines.append(
                "\nProbabilistic, not deterministic. "
                "Consult a professional for sensitive decisions."
            )
        return "\n".join(lines)

    def _window_to_dict(self, w: CandidateWindow) -> Dict[str, Any]:
        return {
            "start": w.start.isoformat(),
            "end": w.end.isoformat(),
            "duration_days": (w.end - w.start).days,
            "aggregate_confidence": round(w.aggregate_confidence, 3),
            "confidence_per_school": {
                s.value: round(v, 3)
                for s, v in w.confidence_per_school.items()
            },
            "contributing_rules": [
                {
                    "rule_id": fr.rule.rule_id,
                    "rule_type": fr.rule.rule_type,
                    "source": fr.rule.source,
                    "polarity": fr.polarity,
                    "strength": round(fr.strength, 3),
                    "bindings": fr.bindings,
                }
                for fr in w.contributing_rules
            ],
            "contradictions": w.contradictions,
        }
