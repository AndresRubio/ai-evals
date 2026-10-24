import json
from collections.abc import Callable
from pathlib import Path

import pytest

from axial.errors import InvariantViolation, MalformedFile
from axial.sample import Review
from axial.store import Store, validate_name
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


def test_count_open_codes_counts_distinct_traces_not_rows():
    """Two review sessions can draw the identical batch (`sample` defaults to
    --seed 0), so annotating both produces two rows per trace. The floor must
    count traces looked at, not rows written."""
    reviews = coded(20) + coded(20)  # same 20 trace ids, twice
    assert len(reviews) == 40
    assert count_open_codes(reviews) == 20


def test_propose_refuses_when_duplicate_sessions_inflate_the_row_count():
    """40 open-code rows over only 20 distinct traces must still refuse: this is
    exactly the scenario where two sessions sampled with the same seed re-code
    the same batch, doubling the row count without doubling the evidence."""
    reviews = coded(20) + coded(20)
    with pytest.raises(InvariantViolation, match="30"):
        propose_taxonomy(reviews, fake_complete)


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


def test_a_misspelt_key_under_a_mode_in_taxonomy_yaml_is_refused_naming_the_file_and_key(
    tmp_path: Path,
):
    """`examples` and `prevalence` have defaults, so a hand-edited `exmaples:` would
    otherwise read as no examples and be dropped on the next save."""
    store = Store(tmp_path)
    Taxonomy(modes=[FailureMode(id="a", name="A", definition="d")]).save(store)
    store.taxonomy_path.write_text(
        store.taxonomy_path.read_text(encoding="utf-8") + "  exmaples: [t1]\n", encoding="utf-8"
    )
    with pytest.raises(MalformedFile, match=r"taxonomy\.yaml: modes\.0\.exmaples: unknown key$"):
        Taxonomy.load(store)


def test_a_model_reply_with_a_key_no_mode_has_is_still_accepted():
    """Only the file is strict: a model that adds a field nobody asked for has not
    misspelt anything."""
    reply = {"modes": [{"id": "a", "name": "A", "definition": "d", "severity": "high"}]}
    taxonomy = propose_taxonomy(coded(30), lambda _prompt: json.dumps(reply))
    assert [m.id for m in taxonomy.modes] == ["a"]


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


def proposing(*ids: str, name: str = "") -> Callable[[str], str]:
    """A model that proposes one mode per id, each named `name` or after its position."""
    modes = [{"id": i, "name": name or f"mode {n}", "definition": "d"} for n, i in enumerate(ids)]
    return lambda _prompt: json.dumps({"modes": modes})


@pytest.mark.parametrize(
    ("proposed", "saved"),
    [
        ("merged_stories", "merged_stories"),
        ("Wrong Topic", "wrong_topic"),
        ("naïve_mode", "na_ve_mode"),
        ("../judges/x", "judges_x"),
        ("--hallucinated--", "hallucinated"),
        ("v1.2-tone", "v1.2-tone"),
        ("a" * 150, "a" * 100),
    ],
)
def test_a_proposed_id_is_saved_as_a_mode_name_every_command_accepts(proposed: str, saved: str):
    """Every mode command refuses a mode that is not one plain directory name, so an id
    like "Wrong Topic" could be saved but never used."""
    taxonomy = propose_taxonomy(coded(30), proposing(proposed))
    assert [m.id for m in taxonomy.modes] == [saved]
    assert validate_name(saved, "mode") == saved


def test_an_id_with_nothing_usable_falls_back_to_the_name_then_to_mode():
    assert [m.id for m in propose_taxonomy(coded(30), proposing("!!!", name="Wrong topic")).modes] == [
        "wrong_topic"
    ]
    assert [m.id for m in propose_taxonomy(coded(30), proposing("!!!", name="???")).modes] == ["mode"]


def test_ids_that_slugify_alike_are_numbered_and_stay_within_the_length_cap():
    taxonomy = propose_taxonomy(
        coded(30), proposing("Wrong Topic", "wrong topic", "wrong_topic", "a" * 150, "a" * 120)
    )
    ids = [m.id for m in taxonomy.modes]
    assert ids == ["wrong_topic", "wrong_topic_2", "wrong_topic_3", "a" * 100, "a" * 98 + "_2"]
    assert all(validate_name(i, "mode") == i for i in ids)
