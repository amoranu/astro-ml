"""Cross-school contradiction detection (spec §6.8, §12.5).

A contradiction is recorded when, for the same cluster window, schools
disagree — e.g. Parashari fires a negative-polarity rule while KP fires
a positive-polarity rule. Phase 2 records only basic polarity clashes;
richer doctrinal contradiction (e.g. "Parashari maraka dasha vs KP
auspicious DBAS") is a Phase 3+ enrichment.
"""
from __future__ import annotations

from typing import Dict, List

from ..schemas.enums import School
from ..schemas.results import CandidateWindow


def _polarity_summary(rules_by_school: Dict[School, List]) -> Dict:
    """{school -> {polarity -> max_strength}} for the rules seen."""
    out: Dict[School, Dict[str, float]] = {}
    for school, rules in rules_by_school.items():
        per_pol: Dict[str, float] = {}
        for fr in rules:
            p = fr.polarity
            per_pol[p] = max(per_pol.get(p, 0.0), float(fr.strength))
        out[school] = per_pol
    return out


def detect_contradictions(window: CandidateWindow) -> List[Dict]:
    """Mark contradictions on a single window. Returns a list of dicts
    describing each contradiction so the Explainer can render them.
    """
    contradictions: List[Dict] = []
    if len(window.confidence_per_school) < 2:
        return contradictions

    # Bucket rules by school + polarity.
    per_school: Dict[School, List] = {s: [] for s in window.confidence_per_school}
    for fr in window.contributing_rules:
        school = fr.rule.school
        if school in per_school:
            per_school[school].append(fr)
    summary = _polarity_summary(per_school)

    # For every pair of schools, find conflicts.
    schools = list(window.confidence_per_school)
    for i in range(len(schools)):
        for j in range(i + 1, len(schools)):
            a, b = schools[i], schools[j]
            sa = summary.get(a, {})
            sb = summary.get(b, {})
            # Classic polarity clash: one negative, one positive.
            if sa.get("negative", 0) >= 0.4 and sb.get("positive", 0) >= 0.4:
                contradictions.append({
                    "type": "polarity_clash",
                    "schools": [a.value, b.value],
                    "detail": (
                        f"{a.value} negative "
                        f"(max s={sa['negative']:.2f}) vs "
                        f"{b.value} positive (max s={sb['positive']:.2f})"
                    ),
                })
            if sb.get("negative", 0) >= 0.4 and sa.get("positive", 0) >= 0.4:
                contradictions.append({
                    "type": "polarity_clash",
                    "schools": [b.value, a.value],
                    "detail": (
                        f"{b.value} negative "
                        f"(max s={sb['negative']:.2f}) vs "
                        f"{a.value} positive (max s={sa['positive']:.2f})"
                    ),
                })

    # Confidence gap: schools agree on polarity but diverge sharply.
    confs = sorted(window.confidence_per_school.values())
    if confs and (confs[-1] - confs[0]) > 0.4:
        contradictions.append({
            "type": "confidence_gap",
            "schools": [s.value for s in window.confidence_per_school],
            "detail": (
                f"per-school conf spread {confs[0]:.2f}..{confs[-1]:.2f} "
                f"(> 0.4)"
            ),
        })
    return contradictions


def annotate_all(windows: List[CandidateWindow]) -> None:
    """Mutate each window's `.contradictions` in place."""
    for w in windows:
        w.contradictions = detect_contradictions(w)
