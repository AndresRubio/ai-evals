"""The review session file: creating, filling in and reading back sessions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from axial.errors import AxialError
from axial.reviews import Review, all_reviews, create_session, read_session, write_session
from axial.store import Store
from axial.traces import Trace


def traces(n: int, **meta_by_index) -> list[Trace]:
    out = []
    for i in range(n):
        meta = {k: v[i] for k, v in meta_by_index.items()}
        out.append(Trace(trace_id=f"t{i}", input=f"in{i}", output="o" * (i + 1), metadata=meta))
    return out



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


# --- the perplexity strategy --------------------------------------------------


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", ""])
def test_a_session_name_that_is_not_one_plain_file_name_is_refused(tmp_path: Path, name: str):
    """Every session path is checked, so no reader or writer can leave .axial/reviews/."""
    store = Store(tmp_path)
    for call in (
        lambda: create_session(store, traces(1), name=name),
        lambda: read_session(store, name),
        lambda: write_session(store, name, []),
    ):
        with pytest.raises(AxialError, match="session name .* is not allowed"):
            call()
    assert not (tmp_path / ".axial").exists()
