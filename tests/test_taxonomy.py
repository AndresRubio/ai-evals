import json
from pathlib import Path

import pytest

from axial.errors import InvariantViolation
from axial.sample import Review
from axial.store import Store
from axial.taxonomy import (
    FailureMode,
    Taxonomy,
    count_open_codes,
    propose_taxonomy,
    saturation_curve,
)


def coded(n: int) -> list[Review]:
    return [
        Review(trace_id=f"t{i}", open_code=f"code {i % 4}", annotator="andres", status="done")
        for i in range(n)
    ]


def fake_complete(_prompt: str) -> str:
    return json.dumps(
        {
            "modes": [
                {
                    "id": "merged_stories",
                    "name": "Merged stories",
                    "definition": "Two distinct stories emitted as one.",
                    "examples": ["t0"],
                }
            ]
        }
    )


def test_count_open_codes_ignores_pending_and_blank(tmp_path: Path):
    reviews = coded(3) + [Review(trace_id="x"), Review(trace_id="y", open_code="  ")]
    assert count_open_codes(reviews) == 3


def test_propose_refuses_below_thirty_open_codes():
    """Husain's floor: open coding is where you find failures you didn't know existed."""
    with pytest.raises(InvariantViolation, match="30"):
        propose_taxonomy(coded(29), fake_complete)


def test_propose_succeeds_at_exactly_thirty():
    taxonomy = propose_taxonomy(coded(30), fake_complete)
    assert [m.id for m in taxonomy.modes] == ["merged_stories"]


def test_the_floor_is_configurable_for_a_team_that_sets_its_own():
    taxonomy = propose_taxonomy(coded(5), fake_complete, min_open_codes=5)
    assert len(taxonomy.modes) == 1


def test_taxonomy_round_trips_through_yaml(tmp_path: Path):
    store = Store(tmp_path)
    original = Taxonomy(
        modes=[FailureMode(id="a", name="A", definition="d", examples=["t1"], prevalence=2)]
    )
    original.save(store)
    assert Taxonomy.load(store) == original


def test_load_is_empty_when_no_taxonomy_exists(tmp_path: Path):
    assert Taxonomy.load(Store(tmp_path)).modes == []


def test_prevalence_counts_reviews_whose_open_code_mentions_the_mode():
    reviews = [
        Review(trace_id="t1", open_code="merged stories again", status="done"),
        Review(trace_id="t2", open_code="hallucinated a source", status="done"),
    ]
    taxonomy = Taxonomy(
        modes=[FailureMode(id="merged", name="merged stories", definition="d", examples=[])]
    )
    taxonomy.recount(reviews)
    assert taxonomy.modes[0].prevalence == 1


def test_prevalence_respects_word_boundaries_single_word():
    """A mode named 'merged' should not match 'unmerged' (antonym)."""
    reviews = [
        Review(trace_id="t1", open_code="unmerged stories", status="done"),
        Review(trace_id="t2", open_code="merged two stories", status="done"),
    ]
    taxonomy = Taxonomy(
        modes=[FailureMode(id="merged", name="merged", definition="d", examples=[])]
    )
    taxonomy.recount(reviews)
    # Should only count "merged two stories", not "unmerged stories"
    assert taxonomy.modes[0].prevalence == 1


def test_prevalence_is_case_insensitive():
    """Matching should be case-insensitive with word boundaries."""
    reviews = [
        Review(trace_id="t1", open_code="Merged two stories", status="done"),
        Review(trace_id="t2", open_code="MERGED again", status="done"),
    ]
    taxonomy = Taxonomy(
        modes=[FailureMode(id="merged", name="merged", definition="d", examples=[])]
    )
    taxonomy.recount(reviews)
    assert taxonomy.modes[0].prevalence == 2


def test_prevalence_handles_multiword_mode_names():
    """Multi-word mode names should match as complete phrases with word boundaries."""
    reviews = [
        Review(trace_id="t1", open_code="merged stories together", status="done"),
        Review(trace_id="t2", open_code="some merged stories were created", status="done"),
        Review(trace_id="t3", open_code="unmerged stories remain", status="done"),
    ]
    taxonomy = Taxonomy(
        modes=[FailureMode(id="merged_stories", name="merged stories", definition="d", examples=[])]
    )
    taxonomy.recount(reviews)
    # Should match "merged stories together" and "some merged stories were created"
    # but not "unmerged stories"
    assert taxonomy.modes[0].prevalence == 2


def test_saturation_curve_flattens_once_codes_repeat():
    reviews = [Review(trace_id=f"t{i}", open_code=f"c{i}", status="done") for i in range(3)]
    reviews += [Review(trace_id=f"u{i}", open_code="c0", status="done") for i in range(3)]
    curve = saturation_curve(reviews)
    assert curve == [1, 2, 3, 3, 3, 3]


def test_propose_rejects_unparseable_model_output():
    with pytest.raises(ValueError, match="could not parse"):
        propose_taxonomy(coded(30), lambda _p: "not json at all")
