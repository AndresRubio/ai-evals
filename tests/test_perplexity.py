"""Perplexity: how surprising a text is under a model fitted to your own corpus.

Nothing here touches the network or a model API. Perplexity in axial is a
sampling signal computed locally, not a call to anybody's endpoint.
"""

from __future__ import annotations

import math

import pytest

from axial.perplexity import DEGENERATE_SPREAD, CharNgramLM, score_traces, spread
from axial.traces import Trace

CORPUS = [
    "what is the refund policy for annual plans",
    "what is the refund policy for monthly plans",
    "how do i change the card on my annual plan",
    "how do i cancel my monthly plan",
    "can i get a refund for my annual plan",
]


def fitted(order: int = 5) -> CharNgramLM:
    return CharNgramLM.fit(CORPUS, order=order)


# --- the metric itself --------------------------------------------------------


def test_in_distribution_text_is_less_surprising_than_gibberish():
    lm = fitted()

    typical = lm.perplexity("what is the refund policy for monthly plans")
    odd = lm.perplexity("zqxjkv wmbfgh zzzqqq")

    assert typical < odd


def test_perplexity_is_never_below_one():
    """PPL = exp(mean -log p) and p <= 1, so PPL < 1 means a probability above 1."""
    lm = fitted()

    for text in [*CORPUS, "totally unseen", "!!!", "a"]:
        assert lm.perplexity(text) >= 1.0


def test_an_unseen_character_stays_finite():
    """A zero-probability character would make perplexity infinite and the ranking useless."""
    lm = fitted()

    assert math.isfinite(lm.perplexity("你好 \U0001f600 \x00"))


def test_empty_text_has_no_perplexity():
    """Nothing was predicted, so the mean log-likelihood is undefined — not 1.0."""
    assert math.isnan(fitted().perplexity(""))


def test_seeing_a_text_more_often_makes_it_less_surprising():
    once = CharNgramLM.fit(["the cat sat on the mat", "unrelated filler text here"])
    often = CharNgramLM.fit(["the cat sat on the mat"] * 20 + ["unrelated filler text here"])

    assert often.perplexity("the cat sat on the mat") < once.perplexity("the cat sat on the mat")


def test_scoring_is_deterministic():
    assert fitted().perplexity("what is the refund policy") == fitted().perplexity(
        "what is the refund policy"
    )


def test_a_longer_context_uses_more_of_the_corpus():
    """Order is the n in n-gram: how many preceding characters condition a prediction."""
    text = "what is the refund policy for annual plans"

    assert CharNgramLM.fit(CORPUS, order=6).perplexity(text) < CharNgramLM.fit(
        CORPUS, order=1
    ).perplexity(text)


def test_order_must_be_at_least_one():
    with pytest.raises(ValueError, match="order"):
        CharNgramLM.fit(CORPUS, order=0)


def test_an_empty_corpus_is_refused():
    """A model fitted on nothing ranks nothing; silently returning a flat score hides that."""
    with pytest.raises(ValueError, match="empty"):
        CharNgramLM.fit([])


# --- scoring a corpus of traces ----------------------------------------------


def _traces(texts: list[str], *, outputs: list[str] | None = None) -> list[Trace]:
    outs = outputs if outputs is not None else ["ok"] * len(texts)
    return [Trace(trace_id=f"t{i}", input=t, output=o) for i, (t, o) in enumerate(zip(texts, outs))]


def test_every_trace_gets_a_score_keyed_by_id():
    scores = score_traces(_traces(CORPUS))

    assert set(scores) == {f"t{i}" for i in range(len(CORPUS))}
    assert all(math.isfinite(v) for v in scores.values())


def test_the_odd_trace_out_scores_highest():
    traces = _traces([*CORPUS, "▓▓▓ qzx vvv ▓▓▓"])

    scores = score_traces(traces)

    assert max(scores, key=lambda k: scores[k]) == f"t{len(CORPUS)}"


def test_scoring_the_output_field_reads_outputs():
    traces = _traces(["same input"] * 3, outputs=["a normal answer", "a normal answer", "▓qzxvv"])

    scores = score_traces(traces, field="output")

    assert max(scores, key=lambda k: scores[k]) == "t2"


def test_an_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="field"):
        score_traces(_traces(CORPUS), field="metadata")


# --- leave-one-out ------------------------------------------------------------


def test_forgetting_a_text_and_restoring_it_is_exact():
    lm = fitted()
    before = lm.perplexity("what is the refund policy for annual plans")

    with lm.without(CORPUS[0]):
        pass

    assert lm.perplexity("what is the refund policy for annual plans") == before


def test_forgetting_a_text_makes_it_more_surprising():
    lm = fitted()
    seen = lm.perplexity(CORPUS[0])

    with lm.without(CORPUS[0]):
        unseen = lm.perplexity(CORPUS[0])

    assert unseen > seen


def test_a_traces_own_score_excludes_its_own_counts():
    """score_traces must fit on the others, or every trace part-grades itself."""
    traces = _traces(CORPUS)

    scored = score_traces(traces)["t0"]
    held_out = CharNgramLM.fit(CORPUS[1:]).perplexity(CORPUS[0])

    assert scored == pytest.approx(held_out)


def test_a_lone_oddity_in_a_small_corpus_still_ranks_first():
    """The regression this exists for.

    Self-inclusion is not a rounding error at small n. With one strange trace in
    seven, its own counts taught the model its own rare sequences: it scored 199
    held out and 1.21 left in, landing second behind ordinary text and inverting
    the ranking the perplexity strategy exists to produce.
    """
    traces = _traces([*CORPUS, "▓▓ qzxjkv wmbfgh ▓▓"])

    scores = score_traces(traces)

    assert max(scores, key=lambda k: scores[k]) == f"t{len(CORPUS)}"
    assert scores[f"t{len(CORPUS)}"] > 10 * max(scores[f"t{i}"] for i in range(len(CORPUS)))


# --- knowing when the ranking carries nothing ---------------------------------


def test_spread_is_large_when_one_trace_really_is_an_outlier():
    scores = score_traces(_traces([*CORPUS, "▓▓ qzxjkv wmbfgh ▓▓"]))

    assert spread(scores) > DEGENERATE_SPREAD


def test_spread_is_flat_for_a_templated_corpus():
    """400 traces from one template differ only in a number. There is nothing to rank."""
    templated = _traces([f"THE WIRE - issue {i}\n\nNorthwind Labs released a model." for i in range(60)])

    assert spread(templated and score_traces(templated)) < DEGENERATE_SPREAD


def test_spread_needs_at_least_two_scores():
    assert math.isnan(spread({}))
    assert math.isnan(spread({"t0": 4.0}))


def test_spread_ignores_traces_with_no_score():
    """An empty trace scores NaN, which would poison a percentile."""
    assert math.isfinite(spread({"t0": 1.0, "t1": 2.0, "t2": math.nan}))
