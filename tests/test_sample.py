from pathlib import Path

import pytest
from pydantic import ValidationError

from axial.sample import (
    Review,
    all_reviews,
    create_session,
    read_session,
    sample_batch,
    write_session,
)
from axial.store import Store
from axial.traces import Trace


def traces(n: int, **meta_by_index) -> list[Trace]:
    out = []
    for i in range(n):
        meta = {k: v[i] for k, v in meta_by_index.items()}
        out.append(Trace(trace_id=f"t{i}", input=f"in{i}", output="o" * (i + 1), metadata=meta))
    return out


def test_random_sampling_is_reproducible_for_a_seed():
    pool = traces(50)
    a = [t.trace_id for t in sample_batch(pool, 10, seed=3)]
    b = [t.trace_id for t in sample_batch(pool, 10, seed=3)]
    assert a == b
    assert len(set(a)) == 10


def test_sampling_more_than_the_pool_returns_the_pool():
    pool = traces(4)
    assert len(sample_batch(pool, 10)) == 4


def test_outlier_strategy_ranks_by_the_metadata_key():
    pool = traces(20, latency_s=[float(i) for i in range(20)])
    picked = sample_batch(pool, 10, strategy="outlier", random_fraction=0.0, seed=0)
    assert picked[0].trace_id == "t19"


def test_outlier_strategy_excludes_booleans_from_numeric_keys():
    """When both boolean and numeric metadata keys are present, rank by the numeric one."""
    pool = traces(10, flagged=[i % 2 == 0 for i in range(10)], latency_s=[float(i) for i in range(10)])
    picked = sample_batch(pool, 5, strategy="outlier", random_fraction=0.0, seed=0)
    # Should rank by latency_s (9.0, 8.0, 7.0, ...) not by flagged (alphabetically first)
    assert picked[0].trace_id == "t9"


def test_a_targeted_strategy_still_includes_a_random_slice():
    """Targeted sampling is biased toward problems you already know about."""
    pool = traces(100, latency_s=[float(i) for i in range(100)])
    picked = sample_batch(pool, 10, strategy="outlier", random_fraction=0.3, seed=1)
    # 10 picks at random_fraction 0.3 means 7 ranked picks (t93..t99) and 3 drawn at
    # random from everything else, so exactly 3 must fall outside the ranked head.
    ranked_head = {f"t{i}" for i in range(93, 100)}
    assert len([t for t in picked if t.trace_id not in ranked_head]) == 3


def test_random_fraction_must_be_a_fraction():
    with pytest.raises(ValueError):
        sample_batch(traces(10), 5, random_fraction=1.5)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        sample_batch(traces(10), 5, strategy="vibes")


def test_create_session_writes_pending_rows(tmp_path: Path):
    store = Store(tmp_path)
    create_session(store, traces(3), name="s1")
    reviews = read_session(store, "s1")
    assert len(reviews) == 3
    assert all(r.status == "pending" and r.open_code is None for r in reviews)


def test_session_round_trips_after_annotation(tmp_path: Path):
    store = Store(tmp_path)
    create_session(store, traces(2), name="s1")
    reviews = read_session(store, "s1")
    reviews[0].open_code = "merged two stories"
    reviews[0].verdict = "fail"
    reviews[0].annotator = "andres"
    reviews[0].status = "done"
    write_session(store, "s1", reviews)
    assert read_session(store, "s1")[0].open_code == "merged two stories"


def test_all_reviews_spans_every_session(tmp_path: Path):
    store = Store(tmp_path)
    create_session(store, traces(2), name="s1")
    create_session(store, traces(3), name="s2")
    assert len(all_reviews(store)) == 5


def test_review_status_must_be_valid():
    """Review.status must be one of the allowed literal values."""
    with pytest.raises(ValidationError):
        Review(trace_id="t1", status="invalid")
