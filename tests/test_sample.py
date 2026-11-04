
import pytest

from axial.sample import sample_batch
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



def _worded(texts: list[str]) -> list[Trace]:
    return [Trace(trace_id=f"t{i}", input=t, output="ok") for i, t in enumerate(texts)]


TYPICAL = [
    "what is the refund policy for annual plans",
    "what is the refund policy for monthly plans",
    "how do i change the card on my annual plan",
    "how do i cancel my monthly plan",
    "can i get a refund for my annual plan",
    "what is the refund policy for my plan",
]


def test_perplexity_strategy_surfaces_the_unusual_trace_first():
    pool = _worded([*TYPICAL, "▓▓ qzxjkv wmbfgh ▓▓"])

    picked = sample_batch(pool, 3, strategy="perplexity", random_fraction=0.0, seed=0)

    assert picked[0].trace_id == f"t{len(TYPICAL)}"


def test_perplexity_strategy_can_rank_on_the_output_side():
    pool = [
        Trace(trace_id=f"t{i}", input="same question", output=o)
        for i, o in enumerate(["a normal answer here", "a normal answer there", "▓qzxjkv wmbf"])
    ]

    picked = sample_batch(
        pool, 1, strategy="perplexity", random_fraction=0.0, perplexity_field="output", seed=0
    )

    assert picked[0].trace_id == "t2"


def test_perplexity_strategy_still_includes_a_random_slice():
    """Invariant 10 applies to every targeted strategy, this one included."""
    pool = _worded([*TYPICAL] * 20)

    picked = sample_batch(pool, 10, strategy="perplexity", random_fraction=0.3, seed=1)

    assert len(picked) == 10
    assert len({t.trace_id for t in picked}) == 10


def test_perplexity_strategy_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="field"):
        sample_batch(_worded(TYPICAL), 3, strategy="perplexity", perplexity_field="nonsense")


def test_an_unscorable_trace_never_takes_the_top_review_slot():
    """An empty input has no perplexity, and NaN sorts wherever it likes.

    Left alone it lands at the front of the ranking, so the first thing a reviewer
    opens is a blank trace instead of the outlier the strategy was run to find.
    Absent is not surprising; it sorts last.
    """
    pool = _worded(["", *TYPICAL, "▓▓ qzxjkv wmbfgh ▓▓", ""])

    picked = sample_batch(pool, 3, strategy="perplexity", random_fraction=0.0, seed=0)

    assert picked[0].input == "▓▓ qzxjkv wmbfgh ▓▓"
    assert "" not in [t.input for t in picked]
