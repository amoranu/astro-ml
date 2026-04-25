"""Reciprocal Rank Fusion across reasoning lines (Phase F.4).

For each reasoning line we compute a per-window net strength
(`LineEvidence.net_strength`). RRF then converts those into ranks within
each line and fuses them across lines:

    rrf_score(window) = sum over lines L of:
        weight[L] / (k + rank[L][window])

where rank starts at 1 for the highest net_strength in that line.
Windows that did not fire on a line are treated as "absent" (no
contribution from that line) — we don't penalize them; RRF rewards
windows that show up high-ranked across many lines.

The classical RRF constant k=60 (Cormack-Clarke-Buettcher 2009)
suppresses tail noise — windows ranked >100 contribute negligibly.

We also expose a `weighted_sum` aggregator as an alternative, used when
plan.aggregation_method == "weighted_sum":

    score(window) = sum over lines L of: weight[L] * net_strength[L][window]

RRF is the default because it's robust to score-scale differences
between lines (Parashari yoga lines typically saturate higher than KP
significator lines, etc).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from ..schemas.results import LineEvidence


_WindowKey = Tuple[datetime, datetime]


def _rank_within_line(
    line_id: str,
    line_scores: Dict[_WindowKey, LineEvidence],
) -> Dict[_WindowKey, int]:
    """Return {window: rank} sorted by net_strength desc.

    Windows with net_strength <= 0 are NOT ranked (omitted). This is
    the standard RRF "non-firing means absent" semantics.

    **Tied-rank handling**: windows with identical net_strength share
    the same rank (the lowest rank in the tie group). This is critical
    when a reasoning line is composed entirely of static yogas
    (window=None firings) — every candidate ends up with the SAME
    net_strength, and without tied ranking, dict-iteration order would
    produce ranks 1..N inside the tie, polluting RRF with noise.
    With tied ranking, all N tied windows contribute the same RRF
    increment from that line, so the line becomes a constant offset
    that doesn't disturb cross-line ranking.
    """
    items = [
        (w, ev.net_strength) for w, ev in line_scores.items()
        if ev.net_strength > 0
    ]
    items.sort(key=lambda r: -r[1])
    out: Dict[_WindowKey, int] = {}
    prev_score: float = float("nan")
    prev_rank: int = 0
    for i, (w, score) in enumerate(items, start=1):
        if score == prev_score:
            out[w] = prev_rank
        else:
            out[w] = i
            prev_rank = i
            prev_score = score
    return out


def fuse_rrf(
    per_line_scores: Dict[str, Dict[_WindowKey, LineEvidence]],
    line_weights: Dict[str, float],
    k: int = 60,
) -> Dict[_WindowKey, float]:
    """Reciprocal Rank Fusion across reasoning lines.

    Args:
        per_line_scores : {line_id: {window: LineEvidence}}
        line_weights    : {line_id: weight}; missing lines default to 1.0
        k               : RRF constant (default 60)

    Returns: {window: rrf_score}
    """
    # Precompute ranks per line.
    per_line_ranks: Dict[str, Dict[_WindowKey, int]] = {}
    all_windows: set = set()
    for line_id, scores in per_line_scores.items():
        per_line_ranks[line_id] = _rank_within_line(line_id, scores)
        all_windows |= set(per_line_ranks[line_id].keys())

    out: Dict[_WindowKey, float] = {}
    for w in all_windows:
        s = 0.0
        for line_id, ranks in per_line_ranks.items():
            r = ranks.get(w)
            if r is None:
                continue
            weight = float(line_weights.get(line_id, 1.0))
            s += weight / (k + r)
        out[w] = s
    return out


def fuse_weighted_sum(
    per_line_scores: Dict[str, Dict[_WindowKey, LineEvidence]],
    line_weights: Dict[str, float],
) -> Dict[_WindowKey, float]:
    """Plain weighted sum of per-line net_strengths."""
    all_windows: set = set()
    for scores in per_line_scores.values():
        all_windows |= set(scores.keys())
    out: Dict[_WindowKey, float] = {}
    for w in all_windows:
        s = 0.0
        for line_id, scores in per_line_scores.items():
            ev = scores.get(w)
            if ev is None:
                continue
            weight = float(line_weights.get(line_id, 1.0))
            s += weight * ev.net_strength
        out[w] = s
    return out


def annotate_line_ranks(
    per_line_scores: Dict[str, Dict[_WindowKey, LineEvidence]],
) -> None:
    """Set LineEvidence.rank_in_line for every windowed evidence object.

    Lines with net_strength <= 0 get rank_in_line = -1 (sentinel for
    "not ranked / absent"). Mutates in place.
    """
    for line_id, scores in per_line_scores.items():
        ranks = _rank_within_line(line_id, scores)
        for w, ev in scores.items():
            ev.rank_in_line = ranks.get(w, -1)
