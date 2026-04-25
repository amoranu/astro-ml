"""Regression harness tests (Module E)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from astroql.engine.regression import (
    Metrics,
    PredictedEvent,
    SplitManifest,
    commit_gate,
    evaluate,
    load_records,
    save_manifest,
    split_records,
)


# ── Fixtures ────────────────────────────────────────────────────────

def _make_records(n: int, prefix: str = "c") -> List[Dict]:
    # Truth date = 1980-01-01 + i days so each record has a distinct
    # truth. The exact date doesn't matter for split tests.
    return [
        {"id": f"{prefix}{i}",
         "father_death_date": (
             date(1980, 1, 1) + timedelta(days=i)
         ).isoformat(),
        }
        for i in range(n)
    ]


# ── Split determinism ───────────────────────────────────────────────

def test_split_deterministic_across_runs():
    recs = _make_records(500)
    a_train, a_holdout, _ = split_records(recs, seed=7)
    b_train, b_holdout, _ = split_records(recs, seed=7)
    assert [r["id"] for r in a_train] == [r["id"] for r in b_train]
    assert [r["id"] for r in a_holdout] == [r["id"] for r in b_holdout]


def test_split_stable_under_reordering():
    # Split membership must depend only on (seed, id), not record order.
    recs = _make_records(500)
    a_train, a_holdout, _ = split_records(recs, seed=9)
    reversed_recs = list(reversed(recs))
    b_train, b_holdout, _ = split_records(reversed_recs, seed=9)
    assert (
        {r["id"] for r in a_train} == {r["id"] for r in b_train}
    )
    assert (
        {r["id"] for r in a_holdout} == {r["id"] for r in b_holdout}
    )


def test_different_seed_gives_different_partition():
    recs = _make_records(500)
    _, hold_a, _ = split_records(recs, seed=1)
    _, hold_b, _ = split_records(recs, seed=2)
    # Very unlikely to be identical.
    assert {r["id"] for r in hold_a} != {r["id"] for r in hold_b}


def test_50_50_split_is_approximately_balanced():
    recs = _make_records(2000)
    train, holdout, _ = split_records(recs, seed=42, train_frac=0.5)
    ratio = len(train) / len(recs)
    # With 2000 records hash-based, expect within ~3% of 50/50.
    assert 0.47 < ratio < 0.53, f"ratio={ratio}"
    # Partition is disjoint and total.
    assert len(train) + len(holdout) == len(recs)
    assert not (
        {r["id"] for r in train} & {r["id"] for r in holdout}
    )


def test_split_manifest_roundtrip():
    recs = _make_records(30)
    _, _, manifest = split_records(recs, seed=13)
    restored = SplitManifest.from_dict(manifest.to_dict())
    assert restored.seed == manifest.seed
    assert restored.train_ids == manifest.train_ids
    assert restored.holdout_ids == manifest.holdout_ids


def test_split_manifest_file_io(tmp_path: Path):
    recs = _make_records(20)
    _, _, manifest = split_records(recs, seed=5)
    p = tmp_path / "manifest.json"
    save_manifest(manifest, p)
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["seed"] == 5
    assert len(data["train_ids"]) + len(data["holdout_ids"]) == 20


def test_split_rejects_record_without_id():
    with pytest.raises(ValueError, match="missing 'id'"):
        split_records([{"no_id_here": "oops"}], seed=0)


def test_split_rejects_bad_train_frac():
    with pytest.raises(ValueError):
        split_records([{"id": "x"}], seed=0, train_frac=0.0)
    with pytest.raises(ValueError):
        split_records([{"id": "x"}], seed=0, train_frac=1.0)


# ── Evaluation ──────────────────────────────────────────────────────

def test_evaluate_hits_and_misses():
    recs = [
        {"id": "a", "truth": "2020-06-15"},
        {"id": "b", "truth": "2021-01-01"},
        {"id": "c", "truth": "2022-03-10"},
    ]

    def predict(rec: Dict) -> PredictedEvent:
        # a: hit (within 6mo), b: hit (same day), c: miss (2yr off)
        if rec["id"] == "a":
            return PredictedEvent(date(2020, 7, 1), cf=-0.5)
        if rec["id"] == "b":
            return PredictedEvent(date(2021, 1, 1), cf=-0.7)
        return PredictedEvent(date(2020, 1, 1), cf=-0.3)

    m = evaluate(recs, predict, truth_field="truth", aspect="longevity")
    assert m.hits == 2
    assert m.misses == 1
    assert m.no_prediction == 0
    assert m.hit_rate == pytest.approx(2 / 3)
    assert "longevity" in m.per_aspect


def test_evaluate_no_prediction_counted():
    recs = [
        {"id": "a", "truth": "2020-06-15"},
        {"id": "b", "truth": "2021-01-01"},
    ]
    m = evaluate(
        recs, lambda rec: None, truth_field="truth", aspect="longevity",
    )
    assert m.hits == 0
    assert m.misses == 0
    assert m.no_prediction == 2
    assert m.hit_rate == 0.0


def test_evaluate_skips_unlabeled_records():
    recs = [
        {"id": "a", "truth": "2020-06-15"},
        {"id": "b"},  # no truth_field — skipped
    ]
    m = evaluate(
        recs, lambda r: PredictedEvent(date(2020, 7, 1), cf=-0.5),
        truth_field="truth", aspect="longevity",
    )
    assert m.n == 1
    assert m.hits == 1


def test_evaluate_window_boundary():
    recs = [{"id": "a", "truth": "2020-06-15"}]
    # Exactly 6 months later = 30.5 * 6 = 183 days. Prediction 182 days
    # later should hit; 184 days later should miss.
    hit_pred = PredictedEvent(date(2020, 12, 14), cf=-0.5)
    miss_pred = PredictedEvent(date(2020, 12, 16), cf=-0.5)
    m_hit = evaluate(
        recs, lambda r: hit_pred, truth_field="truth", aspect="x",
    )
    m_miss = evaluate(
        recs, lambda r: miss_pred, truth_field="truth", aspect="x",
    )
    assert m_hit.hits == 1
    assert m_miss.misses == 1


# ── Commit gate ────────────────────────────────────────────────────

def _m(hit_rate: float) -> Metrics:
    m = Metrics(
        n=100, hits=int(hit_rate * 100), misses=100 - int(hit_rate * 100),
        no_prediction=0, hit_rate=hit_rate,
    )
    return m


def test_commit_gate_strict_improvement_commits():
    old = Metrics(n=100, hits=30, misses=70, no_prediction=0, hit_rate=0.30)
    old.per_aspect["longevity"] = _m(0.30)
    new = Metrics(n=100, hits=40, misses=60, no_prediction=0, hit_rate=0.40)
    new.per_aspect["longevity"] = _m(0.40)
    ok, reason = commit_gate(new, old, "longevity")
    assert ok, reason


def test_commit_gate_no_improvement_rejects():
    old = Metrics(n=100, hits=30, misses=70, no_prediction=0, hit_rate=0.30)
    old.per_aspect["longevity"] = _m(0.30)
    new = Metrics(n=100, hits=30, misses=70, no_prediction=0, hit_rate=0.30)
    new.per_aspect["longevity"] = _m(0.30)
    ok, _ = commit_gate(new, old, "longevity")
    assert not ok


def test_commit_gate_regression_on_other_aspect_rejects():
    old = Metrics(n=100, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    old.per_aspect["longevity"] = _m(0.30)
    old.per_aspect["marriage"] = _m(0.50)
    new = Metrics(n=100, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    new.per_aspect["longevity"] = _m(0.40)
    new.per_aspect["marriage"] = _m(0.30)  # regressed 20pp
    ok, reason = commit_gate(new, old, "longevity")
    assert not ok
    assert "marriage" in reason


def test_commit_gate_small_regression_within_tolerance_commits():
    old = Metrics(n=100, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    old.per_aspect["longevity"] = _m(0.30)
    old.per_aspect["marriage"] = _m(0.50)
    new = Metrics(n=100, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    new.per_aspect["longevity"] = _m(0.40)
    new.per_aspect["marriage"] = _m(0.49)  # 1pp regression
    ok, reason = commit_gate(
        new, old, "longevity", per_aspect_tolerance=0.02,
    )
    assert ok, reason


def test_commit_gate_missing_target_aspect_rejects():
    old = Metrics(n=0, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    new = Metrics(n=0, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    ok, reason = commit_gate(new, old, "longevity")
    assert not ok
    assert "missing" in reason


def test_commit_gate_rejects_if_new_drops_an_aspect():
    old = Metrics(n=0, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    old.per_aspect["longevity"] = _m(0.30)
    old.per_aspect["marriage"] = _m(0.50)
    new = Metrics(n=0, hits=0, misses=0, no_prediction=0, hit_rate=0.0)
    new.per_aspect["longevity"] = _m(0.40)
    # "marriage" dropped — harness regression.
    ok, reason = commit_gate(new, old, "longevity")
    assert not ok
    assert "marriage" in reason


# ── Real GT file load smoke ─────────────────────────────────────────

def test_load_records_utf8_smoke():
    p = Path(__file__).resolve().parents[3] / "ml" / (
        "mother_passing_date_v2.json"
    )
    if not p.exists():
        pytest.skip("GT file not present in this checkout")
    recs = load_records(p)
    assert len(recs) > 100
    assert "id" in recs[0]
    assert "mother_death_date" in recs[0]
