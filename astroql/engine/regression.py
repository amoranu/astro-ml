"""Held-out regression harness (Module E of NEUROSYMBOLIC_ENGINE_DESIGN.md).

Deterministic 50/50 train/holdout split with fixed seed, per-aspect
accuracy tracking, and a commit-gate that only admits new rules when
they improve holdout accuracy without regressing any aspect.

The harness is prediction-function agnostic: it takes
  `predict_fn(record) -> PredictedEvent | None`
so it can wrap the CF engine, a random baseline, or any future model.

Record shape is loose — a plain dict. Each record MUST declare:
  * `id` (stable identifier used for split membership)
  * the aspect-specific truth date field (e.g. `father_death_date`)
The record is otherwise opaque to the harness.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Types ──────────────────────────────────────────────────────────

@dataclass
class PredictedEvent:
    predicted_date: date
    cf: float


@dataclass
class SplitManifest:
    """Which IDs landed in train vs holdout, plus the seed used.

    Persisted alongside evaluation runs so the same partition can be
    re-applied. Also lets the critic loop be audited: "was this chart
    in the critic's training view or held out?"
    """
    seed: int
    train_ids: List[str] = field(default_factory=list)
    holdout_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "holdout_ids": list(self.holdout_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitManifest":
        return cls(
            seed=d["seed"],
            train_ids=list(d.get("train_ids", [])),
            holdout_ids=list(d.get("holdout_ids", [])),
        )


@dataclass
class Metrics:
    n: int
    hits: int
    misses: int
    no_prediction: int
    hit_rate: float  # hits / (hits + misses + no_prediction)
    # Per-aspect breakout when the eval spans multiple aspects.
    per_aspect: Dict[str, "Metrics"] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "n": self.n,
            "hits": self.hits,
            "misses": self.misses,
            "no_prediction": self.no_prediction,
            "hit_rate": self.hit_rate,
        }
        if self.per_aspect:
            out["per_aspect"] = {
                k: v.to_dict() for k, v in self.per_aspect.items()
            }
        return out


# ── Deterministic split ────────────────────────────────────────────

def _stable_hash(text: str, seed: int) -> int:
    """Seed-mixed SHA-256 of text; gives a deterministic uint64 shard
    key that is robust across Python runs (Python's hash() is
    randomized by default). We take the first 8 bytes as an integer.
    """
    h = hashlib.sha256(f"{seed}::{text}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def split_records(
    records: List[Dict[str, Any]],
    seed: int = 42,
    train_frac: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], SplitManifest]:
    """Partition records into (train, holdout) deterministically.

    Split is stable across runs and across record-order permutations —
    membership is a function of `(seed, record["id"])` only.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    threshold = int(train_frac * (2 ** 64))
    for rec in records:
        rid = rec.get("id")
        if rid is None:
            raise ValueError(
                f"record missing 'id' (required for split stability): {rec}"
            )
        if _stable_hash(str(rid), seed) < threshold:
            train.append(rec)
        else:
            holdout.append(rec)
    manifest = SplitManifest(
        seed=seed,
        train_ids=[str(r["id"]) for r in train],
        holdout_ids=[str(r["id"]) for r in holdout],
    )
    return train, holdout, manifest


# ── Evaluation ─────────────────────────────────────────────────────

def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        # Accept YYYY-MM-DD and YYYY-MM-DDTHH:MM:SS.
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return None
    return None


def evaluate(
    records: List[Dict[str, Any]],
    predict_fn: Callable[[Dict[str, Any]], Optional[PredictedEvent]],
    truth_field: str,
    aspect: str,
    window_months: int = 6,
) -> Metrics:
    """Run `predict_fn` over `records` and score against truth dates.

    A hit = |predicted - truth| ≤ window_months months. Missing
    predictions (predict_fn returns None) count as no_prediction.
    Missing truth dates skip the record (treated as unlabeled).
    """
    hits = 0
    misses = 0
    no_pred = 0
    labeled = 0
    window_days = window_months * 30.5  # approximate, matches existing harness
    for rec in records:
        truth = _parse_date(rec.get(truth_field))
        if truth is None:
            continue
        labeled += 1
        pred = predict_fn(rec)
        if pred is None:
            no_pred += 1
            continue
        delta_days = abs((pred.predicted_date - truth).days)
        if delta_days <= window_days:
            hits += 1
        else:
            misses += 1
    total = hits + misses + no_pred
    hit_rate = (hits / total) if total > 0 else 0.0
    m = Metrics(
        n=labeled, hits=hits, misses=misses, no_prediction=no_pred,
        hit_rate=hit_rate,
    )
    # Single-aspect evaluation stores the rollup in per_aspect too for
    # uniform downstream access.
    m.per_aspect[aspect] = Metrics(
        n=labeled, hits=hits, misses=misses, no_prediction=no_pred,
        hit_rate=hit_rate,
    )
    return m


# ── Commit gate ────────────────────────────────────────────────────

def commit_gate(
    new: Metrics,
    old: Metrics,
    target_aspect: str,
    per_aspect_tolerance: float = 0.0,
) -> Tuple[bool, str]:
    """Decide whether a proposed rule should commit (NEUROSYMBOLIC_ENGINE_DESIGN.md §3.E).

    Commit iff:
      1. holdout hit_rate for target_aspect strictly improves
      2. no per-aspect hit_rate regresses by more than
         per_aspect_tolerance (prevents aggregate masking)

    Returns (commit, reason). `reason` is human-readable — used both
    for logging and for the critic-loop stop condition.
    """
    new_target = new.per_aspect.get(target_aspect)
    old_target = old.per_aspect.get(target_aspect)
    if new_target is None or old_target is None:
        return False, (
            f"target_aspect={target_aspect!r} missing from metrics "
            f"(old={old_target is not None}, new={new_target is not None})"
        )
    if new_target.hit_rate <= old_target.hit_rate:
        return False, (
            f"target holdout hit_rate did not improve: "
            f"old={old_target.hit_rate:.4f} -> new={new_target.hit_rate:.4f}"
        )
    # Per-aspect no-regression check across all aspects the old metrics
    # knew about (new may expose strictly more aspects — those are
    # additive by definition, no regression possible).
    for aspect, old_m in old.per_aspect.items():
        new_m = new.per_aspect.get(aspect)
        if new_m is None:
            # Harness reporting dropped this aspect — treat as regression.
            return False, (
                f"aspect {aspect!r} missing from new metrics"
            )
        drop = old_m.hit_rate - new_m.hit_rate
        if drop > per_aspect_tolerance:
            return False, (
                f"aspect {aspect!r} regressed by {drop:.4f} "
                f"(old={old_m.hit_rate:.4f} -> new={new_m.hit_rate:.4f}), "
                f"tolerance={per_aspect_tolerance:.4f}"
            )
    return True, (
        f"commit OK: target {target_aspect!r} "
        f"{old_target.hit_rate:.4f} -> {new_target.hit_rate:.4f}; "
        f"no per-aspect regression beyond {per_aspect_tolerance:.4f}"
    )


# ── IO helpers ─────────────────────────────────────────────────────

def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load a GT file encoded as UTF-8 (handles non-ASCII names).

    Raises if the payload is not a list — the harness expects a flat
    list of chart records.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list, got {type(data).__name__}")
    return data


def save_manifest(manifest: SplitManifest, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
