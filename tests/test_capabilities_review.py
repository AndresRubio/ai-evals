"""What the review stage can and cannot do: sampling, perplexity, annotation, taxonomy.

Each test pins one capability or one limit, and its docstring says which. Read top to
bottom it is a description of the stage. Everything runs offline against local data
and `FakeModel`; nothing here touches a network or an API key.

Where the code falls short of what the README or the page itself promises, the test
is kept and marked `xfail(strict=True)` with the reason, so the gap is documented and
fixing it turns the test into a failure until the marker is removed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from axial.annotate.app import create_app
from axial.errors import InvariantViolation
from axial.model import FakeModel
from axial.perplexity import DEGENERATE_SPREAD, CharNgramLM, score_traces, spread
from axial.reviews import Review, create_session, read_session
from axial.sample import STRATEGIES, sample_batch
from axial.store import Store
from axial.taxonomy import (
    FailureMode,
    Taxonomy,
    count_open_codes,
    propose_taxonomy,
    saturation_curve,
)
from axial.traces import Trace, ingest_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- helpers ------------------------------------------------------------------


def numbered(n: int, **meta_by_index: list) -> list[Trace]:
    """`n` traces whose output length grows with the index: t0 is shortest, t{n-1} longest."""
    return [
        Trace(
            trace_id=f"t{i}",
            input=f"in{i}",
            output="o" * (i + 1),
            metadata={k: v[i] for k, v in meta_by_index.items()},
        )
        for i in range(n)
    ]


def ids(batch: list[Trace]) -> list[str]:
    return [t.trace_id for t in batch]


TYPICAL = [
    "what is the refund policy for annual plans",
    "what is the refund policy for monthly plans",
    "how do i change the card on my annual plan",
    "how do i cancel my monthly plan",
    "can i get a refund for my annual plan",
    "what is the refund policy for my plan",
]
ODD = "▓▓ qzxjkv wmbfgh ▓▓"


def worded(inputs: list[str], outputs: list[str] | None = None) -> list[Trace]:
    outs = outputs if outputs is not None else ["ok"] * len(inputs)
    return [Trace(trace_id=f"t{i}", input=a, output=b) for i, (a, b) in enumerate(zip(inputs, outs))]


def varied_pool(n: int = 40) -> list[Trace]:
    """A pool every strategy can rank: numeric metadata, distinct outputs, varied inputs."""
    words = ["refund", "invoice", "shipping", "password", "export", "receipt", "delay", "card"]
    return [
        Trace(
            trace_id=f"t{i}",
            input=f"question {i * 7 % 13} about the {words[i % len(words)]} of order {i * 31}",
            output="o" * (1 + i * 5 % 17),
            metadata={"latency_s": float(i * 3 % 11)},
        )
        for i in range(n)
    ]


# =============================================================================
# sample: the four strategies
# =============================================================================


def test_random_strategy_returns_the_batch_in_pool_order():
    """`random` draws without replacement and hands the batch back in pool order, not draw order."""
    pool = numbered(30)

    picked = ids(sample_batch(pool, 8, strategy="random", seed=4))

    assert picked == sorted(picked, key=lambda i: int(i[1:]))
    assert len(set(picked)) == 8


def test_random_strategy_ignores_random_fraction():
    """`--random-fraction` only mixes a random slice into targeted batches; `random` is all random."""
    pool = numbered(30)

    assert ids(sample_batch(pool, 8, strategy="random", random_fraction=0.0, seed=2)) == ids(
        sample_batch(pool, 8, strategy="random", random_fraction=1.0, seed=2)
    )


def test_longest_strategy_ranks_by_output_length_not_input_length():
    """`longest` means longest output: a huge input does not make a trace a candidate."""
    pool = [
        Trace(trace_id="long_in", input="x" * 500, output="short"),
        Trace(trace_id="long_out", input="x", output="y" * 50),
        Trace(trace_id="mid_out", input="x", output="y" * 20),
        Trace(trace_id="tiny", input="x", output="y"),
    ]

    picked = sample_batch(pool, 2, strategy="longest", random_fraction=0.0)

    assert ids(picked) == ["long_out", "mid_out"]


def test_longest_strategy_keeps_pool_order_among_equal_lengths():
    """Ties are not shuffled: equal-length outputs stay in the order the corpus had them."""
    pool = [Trace(trace_id=f"t{i}", input="x", output="same") for i in range(6)]

    assert ids(sample_batch(pool, 3, strategy="longest", random_fraction=0.0)) == ["t0", "t1", "t2"]


def test_outlier_strategy_only_reaches_for_the_high_tail():
    """`outlier` is highest-value-first, not most-deviant: a very low value is never chosen."""
    pool = numbered(6, latency_s=[-1000.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    picked = sample_batch(pool, 1, strategy="outlier", random_fraction=0.0)

    assert ids(picked) == ["t5"]


def test_outlier_strategy_silently_picks_the_alphabetically_first_numeric_key():
    """With several numeric metadata keys `outlier` ranks on the first alphabetically, and says nothing.

    The other key here points the opposite way; there is no option to choose between them.
    """
    n = 10
    pool = numbered(
        n,
        latency_s=[float(i) for i in range(n)],
        a_cost=[float(n - i) for i in range(n)],
    )

    picked = sample_batch(pool, 3, strategy="outlier", random_fraction=0.0)

    assert ids(picked) == ["t0", "t1", "t2"]  # ranked by a_cost, not latency_s


def test_outlier_strategy_treats_a_trace_missing_the_key_as_zero():
    """A trace without the ranking key sorts as 0, so it outranks every negative value."""
    pool = [
        Trace(trace_id="neg", input="i", output="o", metadata={"score": -2.0}),
        Trace(trace_id="absent", input="i", output="o"),
        Trace(trace_id="pos", input="i", output="o", metadata={"score": 5.0}),
    ]

    picked = sample_batch(pool, 2, strategy="outlier", random_fraction=0.0)

    assert ids(picked) == ["pos", "absent"]


def test_outlier_strategy_without_numeric_metadata_degrades_to_longest():
    """No numeric metadata (numeric-looking strings do not count) means there is nothing to
    find outliers in, so `outlier` quietly becomes `longest`."""
    pool = numbered(8, latency=[str(i) for i in range(8)])

    picked = sample_batch(pool, 3, strategy="outlier", random_fraction=0.0)

    assert ids(picked) == ["t7", "t6", "t5"]


# =============================================================================
# sample: the random slice, determinism, and batch boundaries
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_is_reproducible_for_a_seed(strategy: str):
    """The same corpus, strategy and seed always give the same batch, in the same order."""
    pool = varied_pool()

    first = ids(sample_batch(pool, 12, strategy=strategy, seed=7))

    assert first == ids(sample_batch(pool, 12, strategy=strategy, seed=7))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_a_different_seed_changes_the_draw(strategy: str):
    """The seed is real: two seeds do not produce the same batch."""
    pool = varied_pool()

    assert set(ids(sample_batch(pool, 12, strategy=strategy, seed=1))) != set(
        ids(sample_batch(pool, 12, strategy=strategy, seed=2))
    )


@pytest.mark.parametrize("strategy", ["outlier", "longest", "perplexity"])
def test_the_seed_only_moves_the_random_slice_of_a_targeted_batch(strategy: str):
    """The ranked head is a pure function of the corpus; only the random tail depends on the seed."""
    pool = varied_pool()
    head = 12 - round(12 * 0.3)

    a = ids(sample_batch(pool, 12, strategy=strategy, random_fraction=0.3, seed=1))
    b = ids(sample_batch(pool, 12, strategy=strategy, random_fraction=0.3, seed=2))

    assert a[:head] == b[:head]
    assert a[head:] != b[head:]


@pytest.mark.parametrize("strategy", ["outlier", "longest", "perplexity"])
def test_random_fraction_zero_makes_a_targeted_batch_independent_of_the_seed(strategy: str):
    """With no random slice there is nothing left for the seed to decide."""
    pool = varied_pool()

    assert ids(sample_batch(pool, 10, strategy=strategy, random_fraction=0.0, seed=1)) == ids(
        sample_batch(pool, 10, strategy=strategy, random_fraction=0.0, seed=99)
    )


def test_random_fraction_one_fills_a_targeted_batch_entirely_at_random():
    """At `random_fraction=1` no ranked pick survives: the batch is exactly a `random` draw."""
    pool = numbered(40, latency_s=[float(i) for i in range(40)])

    targeted = sample_batch(pool, 6, strategy="outlier", random_fraction=1.0, seed=3)
    plain = sample_batch(pool, 6, strategy="random", seed=3)

    assert set(ids(targeted)) == set(ids(plain))


@pytest.mark.parametrize(
    ("n", "fraction", "n_random"),
    [(10, 0.3, 3), (10, 0.25, 2), (5, 0.5, 2), (1, 0.3, 0), (4, 1.0, 4), (4, 0.0, 0)],
)
def test_random_slice_size_is_round_n_times_fraction(n: int, fraction: float, n_random: int):
    """The random slice is `round(n * fraction)` picks, so a small batch can lose it entirely.

    Python's rounding is banker's: 2.5 rounds to 2, not 3, and n=1 at 0.3 has no random
    slice at all. The ranked head is the first `n - n_random` picks, in rank order.
    """
    pool = numbered(40, latency_s=[float(i) for i in range(40)])
    ranked = [f"t{i}" for i in range(39, -1, -1)]

    picked = ids(sample_batch(pool, n, strategy="outlier", random_fraction=fraction, seed=5))

    head = n - n_random
    assert len(picked) == n
    assert picked[:head] == ranked[:head]
    assert not set(picked[head:]) & set(ranked[:head])


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.7, 1.0])
def test_a_batch_never_repeats_a_trace(strategy: str, fraction: float):
    """The random slice is drawn from what the ranked head left behind, so nothing is picked twice."""
    pool = varied_pool()

    picked = ids(sample_batch(pool, 25, strategy=strategy, random_fraction=fraction, seed=0))

    assert len(picked) == 25
    assert len(set(picked)) == 25
    assert set(picked) <= set(ids(pool))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_asking_for_more_than_the_pool_returns_the_pool_unranked(strategy: str):
    """`n >= len(pool)` returns every trace in original order: no ranking, no random draw."""
    pool = varied_pool(10)

    assert sample_batch(pool, 10, strategy=strategy) == pool
    assert sample_batch(pool, 500, strategy=strategy) == pool


def test_a_batch_of_zero_is_empty_for_every_strategy():
    """`n=0` is a legal, empty batch rather than an error."""
    pool = varied_pool(10)

    assert all(sample_batch(pool, 0, strategy=s) == [] for s in STRATEGIES)


def test_negative_batch_size_is_refused():
    """A negative `n` fails loudly (with numpy's message, not axial's) instead of slicing from the end."""
    with pytest.raises(ValueError):
        sample_batch(numbered(10), -1)


def test_random_fraction_boundaries_are_inclusive():
    """0 and 1 are valid fractions; anything outside [0, 1] is refused."""
    pool = numbered(10)

    sample_batch(pool, 4, strategy="longest", random_fraction=0.0)
    sample_batch(pool, 4, strategy="longest", random_fraction=1.0)
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="between 0 and 1"):
            sample_batch(pool, 4, strategy="longest", random_fraction=bad)


def test_arguments_are_validated_even_when_the_pool_is_smaller_than_the_batch():
    """A typo'd strategy is caught up front, not hidden by the 'return everything' shortcut."""
    with pytest.raises(ValueError, match="unknown strategy"):
        sample_batch(numbered(3), 10, strategy="vibes")
    with pytest.raises(ValueError, match="between 0 and 1"):
        sample_batch(numbered(3), 10, random_fraction=2.0)


def test_a_bad_perplexity_field_is_rejected_even_by_other_strategies():
    """`perplexity_field` is validated globally, so a typo fails even when `random` never reads it."""
    with pytest.raises(ValueError, match="unknown field"):
        sample_batch(numbered(10), 3, strategy="random", perplexity_field="nonsense")


# =============================================================================
# sample: the perplexity strategy
# =============================================================================


def test_perplexity_batch_is_ordered_most_surprising_first():
    """The ranked head comes back in descending perplexity, so a reviewer opens the strangest first."""
    pool = worded([*TYPICAL, ODD, "select * from users where id = 1"])
    scores = score_traces(pool)

    picked = ids(sample_batch(pool, 4, strategy="perplexity", random_fraction=0.0))

    assert picked == sorted(picked, key=lambda i: scores[i], reverse=True)
    assert set(picked[:2]) == {"t6", "t7"}


def test_perplexity_ranks_the_input_side_by_default_and_is_blind_to_odd_outputs():
    """The default ranks on `input`, so a trace with a strange *output* is not surfaced by it."""
    inputs = [*TYPICAL[:5], TYPICAL[0]]
    outputs = ["Refunds take five days.", "Cancel from the billing page.", "Yes, within thirty days.",
               "Update it in settings.", "Invoices are under billing.", ODD]
    pool = worded(inputs, outputs)

    by_input = sample_batch(pool, 1, strategy="perplexity", random_fraction=0.0)
    by_output = sample_batch(
        pool, 1, strategy="perplexity", random_fraction=0.0, perplexity_field="output"
    )

    assert by_input[0].trace_id != "t5"
    assert by_output[0].trace_id == "t5"


def test_unscorable_traces_fill_a_perplexity_batch_last_when_nothing_else_is_left():
    """Empty inputs rank behind every scorable trace but can still be used to fill the batch."""
    pool = worded(["", *TYPICAL[:3], ""])

    picked = sample_batch(pool, 4, strategy="perplexity", random_fraction=0.0)

    assert "" not in [t.input for t in picked[:3]]
    assert picked[3].input == ""


# =============================================================================
# perplexity: what the score means, and what it does not
# =============================================================================


def test_a_single_trace_scores_finite_and_has_no_spread():
    """One trace is scored against a model that knows nothing (uniform), and cannot be compared."""
    scores = score_traces(worded(["only trace"]))

    assert math.isfinite(scores["t0"]) and scores["t0"] > 1.0
    assert math.isnan(spread(scores))


def test_scoring_no_traces_gives_no_scores():
    """An empty trace list scores to an empty mapping rather than raising."""
    assert score_traces([]) == {}


def test_an_exact_duplicate_protects_a_trace_from_looking_surprising():
    """Leave-one-out removes one copy only, so a text that appears twice is scored as known text."""
    once = score_traces(worded([*TYPICAL, ODD]))[f"t{len(TYPICAL)}"]
    twice = score_traces(worded([*TYPICAL, ODD, ODD]))[f"t{len(TYPICAL)}"]

    assert twice < once / 10


def test_identical_stub_outputs_sort_to_the_bottom_when_ranking_outputs():
    """Ranking outputs buries a repeated truncated stub; degenerate output is *unsurprising*.

    That is why `--perplexity-field output` is the wrong tool for finding truncations.
    """
    outputs = [
        "The refund window is thirty days from purchase.",
        "The refund window is thirty days.",
        "Annual plans can be cancelled from the billing page.",
        "[truncated]",
        "[truncated]",
        "[truncated]",
    ]

    scores = score_traces(worded(["q"] * 6, outputs), field="output")

    assert max(scores[f"t{i}"] for i in (3, 4, 5)) < min(scores[f"t{i}"] for i in (0, 1, 2))


def test_perplexity_is_not_a_quality_metric():
    """A fluent, confident, fabricated answer scores like every other answer; a terse *correct* one
    is the outlier.

    Perplexity says how expected a text was, never whether it was true. The corpus is
    consistently phrased, so a wrong answer in the house style ("ninety days") is ordinary
    and a right answer in a different register ("30d") is flagged.
    """
    outputs = [
        "The refund window is thirty days from purchase.",
        "The refund window is thirty days from the date of purchase.",
        "The refund window is fourteen days from purchase.",
        "The refund window is thirty days from purchase, no exceptions.",
        "The refund window is thirty days from delivery.",
        "The refund window is ninety days from purchase.",  # fluent and wrong
        "30d",  # terse and right
    ]

    scores = score_traces(worded(["q"] * 7, outputs), field="output")

    assert scores["t5"] < 5 * min(scores[f"t{i}"] for i in range(5))
    assert scores["t6"] > 100 * scores["t5"]


def test_scores_are_relative_to_the_corpus_and_do_not_travel():
    """The same text gets a very different score under a different reference corpus."""
    support = CharNgramLM.fit(TYPICAL)
    sql = CharNgramLM.fit(["select * from users where id = 1", "drop table users; --",
                           "insert into logs values (1)"])
    text = "what is the refund policy for monthly plans"

    assert sql.perplexity(text) > 10 * support.perplexity(text)


def test_the_output_field_reads_outputs_and_ignores_inputs():
    """Changing only the inputs cannot change scores computed on the output field."""
    outputs = ["a normal answer", "a normal answer here", "▓qzxvv"]

    a = score_traces(worded(["one", "two", "three"], outputs), field="output")
    b = score_traces(worded(["▓▓▓", "▓▓▓▓", "▓"], outputs), field="output")

    assert a == b


# =============================================================================
# perplexity: saying when the ranking is noise
# =============================================================================


def test_the_degenerate_threshold_is_the_documented_two():
    """README: `sample` warns below a spread of 2.0."""
    assert DEGENERATE_SPREAD == 2.0


def test_the_committed_newsroom_corpus_is_flagged_as_noise():
    """README claim: `examples/newsroom` (25 templates, 400 traces) reads about 1.05, below 2.0."""
    traces = ingest_jsonl(REPO_ROOT / "examples" / "newsroom" / "traces.jsonl")

    ratio = spread(score_traces(traces))

    assert len(traces) == 400
    assert ratio < DEGENERATE_SPREAD


def test_spread_of_identical_scores_is_exactly_one():
    """No trace stands out at all: max over median is 1."""
    assert spread({"a": 3.0, "b": 3.0, "c": 3.0}) == 1.0


def test_spread_is_unchanged_by_the_scale_of_the_scores():
    """It is a ratio, so multiplying every score by a constant leaves it alone."""
    assert spread({"a": 3.0, "b": 6.0, "c": 9.0}) == pytest.approx(
        spread({"a": 30.0, "b": 60.0, "c": 90.0})
    )


def test_spread_catches_three_outliers_in_a_large_corpus():
    """It uses the maximum, not a percentile: 3 strange traces in 123 must still register."""
    scores = {f"t{i}": 1.0 for i in range(120)} | {f"odd{i}": 800.0 for i in range(3)}

    assert spread(scores) == pytest.approx(800.0)
    assert spread(scores) > DEGENERATE_SPREAD


def test_spread_is_undefined_when_no_score_is_usable():
    """All-NaN scores, or a non-positive median, give NaN, and the CLI stays silent for NaN."""
    assert math.isnan(spread({"a": math.nan, "b": math.nan}))
    assert math.isnan(spread({"a": 0.0, "b": 0.0}))


# =============================================================================
# annotate: the app
# =============================================================================


def build_app(
    tmp_path: Path, *, annotator: str = "andres", metadata: dict | None = None
) -> tuple[Store, TestClient, list[Trace]]:
    store = Store(tmp_path)
    traces = [
        Trace(trace_id=f"t{i}", input=f"in{i}", output=f"out{i}", metadata=metadata or {})
        for i in range(4)
    ]
    store.write_traces(traces)
    create_session(store, traces, name="s1")
    return store, TestClient(create_app(store, "s1", annotator)), traces


def post(client: TestClient, trace_id: str, **body: object):
    return client.post(f"/api/review/{trace_id}", json={"verdict": "fail", **body})


def test_a_session_resumes_where_it_left_off_in_a_fresh_app(tmp_path: Path):
    """Annotations live in the session file, so a restarted app shows earlier work and progress."""
    store, client, _ = build_app(tmp_path)
    post(client, "t0", open_code="merged stories", critique="two in one")
    post(client, "t2", open_code="fine", verdict="pass")

    resumed = TestClient(create_app(store, "s1", "andres")).get("/api/session").json()

    by_id = {i["trace_id"]: i for i in resumed["items"]}
    assert resumed["done"] == 2 and resumed["total"] == 4
    assert by_id["t0"]["open_code"] == "merged stories"
    assert by_id["t0"]["critique"] == "two in one"
    assert by_id["t2"]["verdict"] == "pass"
    assert by_id["t1"]["status"] == "pending"


def test_items_come_back_in_session_order_with_trace_metadata(tmp_path: Path):
    """The app walks the batch in the order `sample` produced it and carries metadata to the page."""
    _, client, _ = build_app(tmp_path, metadata={"latency_s": 1.5})

    items = client.get("/api/session").json()["items"]

    assert [i["trace_id"] for i in items] == ["t0", "t1", "t2", "t3"]
    assert items[0]["metadata"] == {"latency_s": 1.5}


def test_a_trace_can_be_re_annotated_and_the_last_review_wins(tmp_path: Path):
    """Re-posting overwrites the earlier review in place; progress does not double-count it."""
    store, client, _ = build_app(tmp_path)
    post(client, "t0", open_code="first", critique="c1")

    second = post(client, "t0", open_code="second", verdict="pass", critique="c2")

    saved = read_session(store, "s1")[0]
    assert second.json() == {"ok": True, "done": 1}
    assert (saved.open_code, saved.verdict, saved.critique) == ("second", "pass", "c2")
    assert len(read_session(store, "s1")) == 4


def test_a_verdict_alone_is_a_complete_review_but_is_not_an_open_code(tmp_path: Path):
    """Open code and critique are optional: the review is 'done', yet earns no credit toward the
    30-open-code floor."""
    store, client, _ = build_app(tmp_path)

    response = client.post("/api/review/t0", json={"verdict": "pass"})

    saved = read_session(store, "s1")[0]
    assert response.status_code == 200
    assert saved.status == "done" and saved.open_code == ""
    assert count_open_codes(read_session(store, "s1")) == 0


def test_a_whitespace_open_code_is_stored_verbatim_but_never_counted(tmp_path: Path):
    """The app does not trim what you type; blank codes are filtered later, where they are counted."""
    store, client, _ = build_app(tmp_path)

    post(client, "t0", open_code="   ")

    reviews = read_session(store, "s1")
    assert reviews[0].open_code == "   "
    assert count_open_codes(reviews) == 0


def test_a_second_annotator_overwrites_the_first_with_no_history(tmp_path: Path):
    """Limit: a session file holds one review per trace, so two people on one session clobber each other."""
    store, first, _ = build_app(tmp_path, annotator="andres")
    post(first, "t0", open_code="mine")

    post(TestClient(create_app(store, "s1", "bob")), "t0", open_code="bob's")

    saved = read_session(store, "s1")[0]
    assert (saved.annotator, saved.open_code) == ("bob", "bob's")


def test_a_trace_missing_from_the_store_is_flagged_and_keeps_its_prior_review(tmp_path: Path):
    """A review survives its trace being dropped from the store; the page is told it is missing."""
    store, client, traces = build_app(tmp_path)
    post(client, "t1", open_code="kept")
    store.write_traces([t for t in traces if t.trace_id != "t1"])

    item = {i["trace_id"]: i for i in client.get("/api/session").json()["items"]}["t1"]

    assert item["missing"] is True
    assert item["open_code"] == "kept"


@pytest.mark.xfail(
    strict=True,
    reason="app.py post_review accepts a verdict for a trace missing from the store, although "
    "index.html says such a trace 'cannot be open-coded until the trace is restored'",
)
def test_a_review_of_a_trace_the_reviewer_could_not_see_is_refused(tmp_path: Path):
    """The page promises a missing trace cannot be coded; the API should refuse it too."""
    store, client, traces = build_app(tmp_path)
    store.write_traces([t for t in traces if t.trace_id != "t1"])

    response = post(client, "t1", open_code="guess")

    assert response.status_code >= 400
    assert read_session(store, "s1")[1].status == "pending"


@pytest.mark.xfail(
    strict=True,
    reason="app.py get_session returns 200 with zero items for a session that does not exist, so a "
    "typo in `axial annotate --name` opens an empty page instead of an error",
)
def test_opening_a_session_that_does_not_exist_is_an_error(tmp_path: Path):
    """A misspelt session name should fail, not present an empty, finished-looking review."""
    store, _, _ = build_app(tmp_path)

    response = TestClient(create_app(store, "typo", "andres")).get("/api/session")

    assert response.status_code >= 400


def test_a_trace_id_containing_a_slash_can_be_annotated(tmp_path: Path):
    """Ids from adapters are often paths or URLs (`run/123`); they should be reviewable."""
    store = Store(tmp_path)
    traces = [Trace(trace_id="run/123", input="i", output="o")]
    store.write_traces(traces)
    create_session(store, traces, name="s1")
    client = TestClient(create_app(store, "s1", "andres"))

    response = client.post(f"/api/review/{quote('run/123', safe='')}", json={"verdict": "fail"})

    assert response.status_code == 200
    assert read_session(store, "s1")[0].status == "done"


def test_trace_ids_with_spaces_and_punctuation_can_be_annotated(tmp_path: Path):
    """The page URL-encodes the id, and ordinary punctuation survives the round trip."""
    store = Store(tmp_path)
    weird = ["a b", "a%b", "é?", "x&y=1"]
    traces = [Trace(trace_id=i, input="i", output="o") for i in weird]
    store.write_traces(traces)
    create_session(store, traces, name="s1")
    client = TestClient(create_app(store, "s1", "andres"))

    for trace_id in weird:
        assert client.post(f"/api/review/{quote(trace_id, safe='')}", json={"verdict": "pass"}).json()[
            "ok"
        ]

    assert all(r.status == "done" for r in read_session(store, "s1"))


# =============================================================================
# annotate: the page and its keyboard
# =============================================================================


@pytest.fixture
def html(tmp_path: Path) -> str:
    _, client, _ = build_app(tmp_path)
    return client.get("/").text


def test_page_maps_p_and_f_to_pass_and_fail_and_arrows_to_navigation(html: str):
    """Outside a text field: `p` passes, `f` fails, and the arrow keys move between traces."""
    assert 'e.key === "p") save("pass")' in html
    assert 'e.key === "f") save("fail")' in html
    assert 'e.key === "ArrowLeft") move(-1)' in html
    assert 'e.key === "ArrowRight") move(1)' in html


def test_page_ignores_letter_shortcuts_while_typing_in_a_field(html: str):
    """Typing `p` or `f` into the open-code box must not save a verdict: fields return before the
    letter shortcuts."""
    in_field = html.index('e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA"')
    letters = html.index('e.key === "p") save("pass")')

    assert in_field < html.index("return;", in_field) < letters


def test_page_checks_the_shift_chord_before_the_plain_one(html: str):
    """Cmd/Ctrl+Shift+Enter must be tested first, or it would be swallowed as plain Cmd/Ctrl+Enter."""
    shift = html.index("(e.metaKey || e.ctrlKey) && e.shiftKey")
    plain = html.index('(e.metaKey || e.ctrlKey)) { save("fail")')

    assert shift < plain
    assert 'e.shiftKey) { save("pass")' in html


def test_page_never_saves_on_a_bare_enter(html: str):
    """Enter without Cmd/Ctrl is just a newline; every Enter-triggered save requires a modifier."""
    enter_lines = [line for line in html.splitlines() if 'e.key === "Enter"' in line]

    assert enter_lines
    assert all("e.metaKey || e.ctrlKey" in line for line in enter_lines)


def test_page_opens_on_the_first_unannotated_trace(html: str):
    """Resuming lands on the first trace whose status is not done; a finished session wraps to 0."""
    assert 'items.findIndex(x => x.status !== "done")' in html
    assert "firstPending === -1 ? 0 : firstPending" in html


def test_page_renders_trace_text_as_text_not_html(html: str):
    """A trace containing markup is displayed literally: the page assigns `textContent`, never HTML."""
    assert "textContent = item.input" in html
    assert "textContent = item.output" in html
    assert "innerHTML" not in html


# =============================================================================
# taxonomy: the 30-open-code floor
# =============================================================================


def coded(n: int, code: str = "merged stories") -> list[Review]:
    return [
        Review(trace_id=f"t{i}", open_code=f"{code} {i}", annotator="andres", status="done")
        for i in range(n)
    ]


def proposal(*modes: dict[str, object]) -> FakeModel:
    return FakeModel(lambda _prompt: json.dumps({"modes": list(modes)}))


def ask(model: FakeModel, reviews: list[Review], **kwargs: int) -> Taxonomy:
    return propose_taxonomy(reviews, lambda p: model.complete(p).text, **kwargs)


MERGED = {"id": "merged", "name": "merged stories", "definition": "Two stories emitted as one."}


def test_a_refusal_never_calls_the_model():
    """Below the floor the model is not consulted at all, so a refused run costs nothing."""
    model = proposal(MERGED)

    with pytest.raises(InvariantViolation):
        ask(model, coded(29))

    assert model.calls == []


def test_the_refusal_says_exactly_how_many_more_traces_to_annotate():
    """The error names the shortfall, so the next step is a number, not a guess."""
    with pytest.raises(InvariantViolation, match=r"25 traces carry an open code.*Annotate 5 more"):
        ask(proposal(MERGED), coded(25))


def test_blank_open_codes_do_not_count_toward_the_floor():
    """30 rows, 10 of them blank or whitespace, is 20 open codes: still refused."""
    reviews = coded(20) + [Review(trace_id=f"b{i}", open_code=" ", status="done") for i in range(5)]
    reviews += [Review(trace_id=f"p{i}") for i in range(5)]
    assert len(reviews) == 30

    with pytest.raises(InvariantViolation, match="20 traces"):
        ask(proposal(MERGED), reviews)


# =============================================================================
# taxonomy: what the model sees and what it gets to decide
# =============================================================================


def test_the_prompt_lists_only_real_open_codes_each_tagged_with_its_trace():
    """The model is shown stripped open codes keyed by trace id; pending and blank reviews are left out."""
    reviews = coded(30)
    reviews[0].open_code = "  padded code  "
    reviews += [Review(trace_id="pending"), Review(trace_id="blank", open_code="   ")]
    model = proposal(MERGED)

    ask(model, reviews)

    (prompt,) = model.calls
    assert "- [t0] padded code\n" in prompt
    assert "[pending]" not in prompt and "[blank]" not in prompt
    assert prompt.count("\n- [") == 30


def test_the_model_is_called_once_per_proposal():
    """One proposal is one model call: there is no hidden retry or refinement loop."""
    model = proposal(MERGED)

    ask(model, coded(30))

    assert len(model.calls) == 1


def test_prevalence_is_computed_locally_and_a_model_supplied_number_is_discarded():
    """A model cannot report its own prevalence: whatever it claims is overwritten by a recount."""
    model = proposal({**MERGED, "prevalence": 999})

    taxonomy = ask(model, coded(30))

    assert taxonomy.modes[0].prevalence == 30


def test_json_wrapped_in_prose_or_a_code_fence_is_still_parsed():
    """Models often chat around their JSON; the first `{` to the last `}` is taken as the payload."""
    payload = json.dumps({"modes": [MERGED]})
    model = FakeModel(lambda _p: f"Sure! Here you go:\n```json\n{payload}\n```\nHope that helps.")

    assert [m.id for m in ask(model, coded(30)).modes] == ["merged"]


def test_an_empty_proposal_is_returned_not_refused():
    """Limit: `{"modes": []}` yields an empty taxonomy silently; only the CLI guards decision models."""
    assert ask(proposal(), coded(30)).modes == []


def test_a_proposal_missing_required_fields_is_rejected():
    """A mode without a definition fails validation instead of entering the taxonomy half-formed."""
    with pytest.raises(ValueError):
        ask(proposal({"id": "merged", "name": "merged stories"}), coded(30))


def test_example_trace_ids_from_the_model_are_kept_unchecked():
    """Limit: `examples` are taken as given, so an id the model invented is not caught."""
    taxonomy = ask(proposal({**MERGED, "examples": ["no-such-trace"]}), coded(30))

    assert taxonomy.modes[0].examples == ["no-such-trace"]


# =============================================================================
# taxonomy: prevalence
# =============================================================================


def mode(name: str) -> FailureMode:
    return FailureMode(id=name.lower().replace(" ", "_"), name=name, definition="d")


def reviewed(*codes: str | None) -> list[Review]:
    return [
        Review(trace_id=f"t{i}", open_code=c, status="done" if c else "pending")
        for i, c in enumerate(codes)
    ]


def test_a_code_counts_once_for_a_mode_however_often_it_repeats_the_name():
    """Prevalence counts reviews, not mentions."""
    taxonomy = Taxonomy(modes=[mode("merged")])

    taxonomy.recount(reviewed("merged merged merged"))

    assert taxonomy.modes[0].prevalence == 1


def test_one_code_can_count_toward_several_modes():
    """Modes are not exclusive: a code naming two of them raises both prevalences."""
    taxonomy = Taxonomy(modes=[mode("merged stories"), mode("hallucinated")])

    taxonomy.recount(reviewed("hallucinated and merged stories"))

    assert [m.prevalence for m in taxonomy.modes] == [1, 1]


def test_prevalence_only_matches_the_mode_name_not_its_id_or_a_paraphrase():
    """Limit: a code that says `merged_stories` or 'joined two stories' is not counted for the
    mode named 'Merged stories'; prevalence undercounts anything not phrased like the name."""
    taxonomy = Taxonomy(modes=[mode("Merged stories")])

    taxonomy.recount(reviewed("merged_stories", "joined two stories together", "merged stories"))

    assert taxonomy.modes[0].prevalence == 1


def test_recount_resets_prevalence_and_skips_pending_reviews():
    """Recounting starts from zero each time, and a review without an open code counts for nothing."""
    taxonomy = Taxonomy(modes=[FailureMode(id="m", name="merged", definition="d", prevalence=9)])

    taxonomy.recount(reviewed(None, "merged"))
    assert taxonomy.modes[0].prevalence == 1

    taxonomy.recount([])
    assert taxonomy.modes[0].prevalence == 0


@pytest.mark.xfail(
    strict=True,
    reason="taxonomy.py Taxonomy.recount wraps the name in \\b...\\b, and \\b cannot match after a "
    "trailing non-word character, so a mode named 'Wrong topic!' or 'Off-topic (other)' never counts",
)
def test_a_mode_name_ending_in_punctuation_is_still_counted():
    """An LLM-proposed name like 'Wrong topic!' should still match the codes that say it."""
    taxonomy = Taxonomy(modes=[mode("Wrong topic!")])

    taxonomy.recount(reviewed("Wrong topic! again", "wrong topic!"))

    assert taxonomy.modes[0].prevalence == 2


@pytest.mark.xfail(
    strict=True,
    reason="Design question, not a confirmed defect: taxonomy.py Taxonomy.recount counts review rows, while count_open_codes deliberately "
    "counts distinct traces; two sessions coding one trace inflate its mode's prevalence to 2",
)
def test_prevalence_counts_traces_not_duplicate_review_rows():
    """`count_open_codes` treats two sessions over one trace as one piece of evidence; prevalence
    should too."""
    taxonomy = Taxonomy(modes=[mode("merged")])
    twice = [
        Review(trace_id="t1", open_code="merged", status="done"),
        Review(trace_id="t1", open_code="merged", status="done"),
    ]

    taxonomy.recount(twice)

    assert taxonomy.modes[0].prevalence == 1


# =============================================================================
# taxonomy: saturation and persistence
# =============================================================================


def test_saturation_curve_ignores_case_and_whitespace_and_skips_uncoded_reviews():
    """One point per *coded* review; 'A ' and 'a' are the same code, blanks and pending add no point."""
    reviews = [
        Review(trace_id="1", open_code="A ", status="done"),
        Review(trace_id="2", open_code="a", status="done"),
        Review(trace_id="3", open_code="   ", status="done"),
        Review(trace_id="4"),
        Review(trace_id="5", open_code="b", status="done"),
    ]

    assert saturation_curve(reviews) == [1, 1, 2]


def test_taxonomy_save_creates_the_directory_and_keeps_mode_order_and_prevalence(tmp_path: Path):
    """A taxonomy saved into a fresh repo loads back identical, modes in the order they were proposed."""
    store = Store(tmp_path / "fresh")
    original = Taxonomy(
        modes=[
            FailureMode(id="z", name="Zed", definition="last alphabetically", prevalence=3),
            FailureMode(id="a", name="Aye", definition="first alphabetically", examples=["t1"]),
        ]
    )

    original.save(store)

    assert Taxonomy.load(store) == original
    assert [m.id for m in Taxonomy.load(store).modes] == ["z", "a"]


def test_an_empty_taxonomy_file_loads_as_no_modes(tmp_path: Path):
    """A blank `taxonomy.yaml` (say, freshly touched) is an empty taxonomy, not a crash."""
    store = Store(tmp_path)
    store.taxonomy_path.parent.mkdir(parents=True)
    store.taxonomy_path.write_text("")

    assert Taxonomy.load(store).modes == []
