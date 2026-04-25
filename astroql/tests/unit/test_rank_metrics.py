"""Tests for rank-based metrics (MRR, top-k recall) in regression.py."""
from __future__ import annotations

import pytest

from astroql.engine.regression import (
    RankMetrics, evaluate_ranks, rank_commit_gate,
)


def test_evaluate_ranks_aggregates_correctly():
    records = [{"id": str(i)} for i in range(5)]
    # Truth ranks: 1, 2, 3, 5, 100
    ranks_iter = iter([(1, 200), (2, 200), (3, 200),
                       (5, 200), (100, 200)])
    rank_fn = lambda rec: next(ranks_iter)
    m = evaluate_ranks(records, rank_fn)
    assert m.n_ranked == 5
    # MRR = mean of 1/1 + 1/2 + 1/3 + 1/5 + 1/100
    expected_mrr = (1 + 0.5 + 1/3 + 0.2 + 0.01) / 5
    assert m.mrr == pytest.approx(expected_mrr)
    assert m.median_rank == 3
    assert m.top_1_recall == pytest.approx(0.2)   # only rank=1 hits
    assert m.top_3_recall == pytest.approx(0.6)   # 3 of 5 in top-3
    assert m.top_10_recall == pytest.approx(0.8)  # 4 of 5 in top-10


def test_evaluate_ranks_skips_none():
    """Records where rank_fn returns None (out of window, no
    prediction, etc.) drop out of the denominator entirely.
    """
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    results = iter([(1, 50), None, (10, 50)])
    rank_fn = lambda rec: next(results)
    m = evaluate_ranks(records, rank_fn)
    assert m.n_ranked == 2  # 'b' dropped


def test_empty_records_returns_zeros():
    m = evaluate_ranks([], lambda r: None)
    assert m.n_ranked == 0
    assert m.mrr == 0.0
    assert m.top_1_recall == 0.0


# ── rank_commit_gate ─────────────────────────────────────────────

def _mk_rm(*, n=10, mrr=0.5, median=3, t1=0.3, t3=0.6, t10=0.9):
    return RankMetrics(
        n_ranked=n, mrr=mrr, median_rank=median,
        top_1_recall=t1, top_3_recall=t3, top_10_recall=t10,
    )


def test_commit_accepts_strict_mrr_gain_no_top3_regression():
    old = _mk_rm(mrr=0.40, t3=0.50)
    new = _mk_rm(mrr=0.55, t3=0.55)
    ok, reason = rank_commit_gate(new, old)
    assert ok is True
    assert "commit OK" in reason


def test_commit_rejects_no_mrr_gain():
    old = _mk_rm(mrr=0.50)
    new = _mk_rm(mrr=0.50)
    ok, reason = rank_commit_gate(new, old)
    assert ok is False
    assert "MRR did not improve" in reason


def test_commit_rejects_top3_regression_even_if_mrr_up():
    """A rule that boosts MRR but tanks top-3 recall is rejected."""
    old = _mk_rm(mrr=0.40, t3=0.60)
    new = _mk_rm(mrr=0.50, t3=0.40)  # MRR up but -0.20 on top-3
    ok, reason = rank_commit_gate(new, old, top3_tolerance=0.0)
    assert ok is False
    assert "top-3 recall regressed" in reason


def test_commit_top3_tolerance_allows_small_dip():
    """Caller can opt into a tolerance window for top-3 drops."""
    old = _mk_rm(mrr=0.40, t3=0.60)
    new = _mk_rm(mrr=0.50, t3=0.55)  # -0.05 on top-3
    ok, _ = rank_commit_gate(new, old, top3_tolerance=0.10)
    assert ok is True


def test_commit_allows_top1_drop_if_mrr_up():
    """A rule trading a few exact wins for higher MRR commits.
    This is the central design decision: rank metric > exact-match.
    """
    old = _mk_rm(mrr=0.40, t1=0.30, t3=0.60)
    new = _mk_rm(mrr=0.55, t1=0.20, t3=0.65)  # top-1 down, top-3 up
    ok, reason = rank_commit_gate(new, old)
    assert ok is True
