"""Perplexity: how surprising a trace is, relative to your own corpus.

Perplexity is the exponentiated average negative log-likelihood a language model
assigns to a text::

    PPL(x) = exp( -1/N * sum_i log p(x_i | x_<i) )

A model that found the text entirely predictable scores 1. A model that was as
surprised as a uniform guess over V symbols scores V.

**What it is for here: choosing which traces a human reads.** `axial sample
--strategy perplexity` ranks traces by how far they sit from the rest of the
corpus, because the failures worth finding cluster in the unusual inputs, not in
the hundredth paraphrase of the most common question. That is a selection
signal, and selection is the one job perplexity is genuinely good at.

**What it is not for: judging quality.** Perplexity measures how expected a text
was, not whether it is correct, grounded or useful. A fluent, confident,
completely fabricated answer is *low* perplexity — the failure mode axial exists
to catch is the one perplexity scores best. It is never a verdict, never a label
and never an input to alignment; invariant 2 means a machine score cannot become
a label in dev or test, and this is a machine score.

**The model is fitted on your corpus, not pretrained.** The question being asked
is "unusual compared to my production traffic", so the reference distribution is
that traffic. It also keeps `sample` free of an API key and a download, and the
default judge model — Anthropic's — exposes no token logprobs at all, so a
pretrained reference could not be reached through the path axial already uses.

Scores are comparable *within* one corpus and meaningless *across* corpora or
against published numbers. That is true of perplexity generally — it is always
relative to a tokenisation and a reference model — and doubly so here, where the
reference model is your corpus.

Scoring is leave-one-out, and that is not a refinement. A trace included in the
model that scores it teaches that model its own rare character sequences, and on
a small corpus it teaches them loudly: in a 7-trace corpus, one deliberately
strange trace scored 199 when held out and 1.21 when left in — second place,
indistinguishable from ordinary text. Self-inclusion does not merely flatten the
score, it inverts the ranking the strategy exists to produce. `score_traces`
therefore removes each trace's counts before scoring it and puts them back
after.

Characters, not words, are the unit: no vocabulary to build, no unknown-word
case to fudge, and it degrades gracefully on code, identifiers and languages the
corpus barely contains.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np

from axial.core.traces import Trace

DEFAULT_ORDER = 5
FIELDS = ("input", "output", "both")

# Below this ratio between the most surprising trace and the typical one, a
# perplexity ranking is not carrying information. The number is a heuristic, not
# a calibrated boundary: it sits in a very wide empty gap between the two corpora
# available to measure -- `examples/newsroom`, 400 traces that are 25 templates
# with an issue number substituted in, scores 1.05, and a varied support corpus
# with three planted oddities scores in the thousands.
DEGENERATE_SPREAD = 2.0

# Sentinel padding the start of a text, so the first real character is predicted
# from a context rather than skipped. Chosen to be absent from real traces.
_BOS = "\x02"


class CharNgramLM:
    """A character n-gram language model with Witten-Bell interpolated smoothing.

    Smoothing is not a detail here: an unsmoothed n-gram assigns probability zero
    to any character it never saw in that context, which sends perplexity to
    infinity and collapses the ranking into "everything unseen is equally weird".
    Witten-Bell backs a sparse context off to a shorter one in proportion to how
    many distinct continuations that context has actually shown, and bottoms out
    in a uniform distribution, so no probability is ever zero.
    """

    def __init__(self, order: int = DEFAULT_ORDER) -> None:
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        self.order = order
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._totals: dict[str, int] = defaultdict(int)
        self._vocab: set[str] = set()

    @classmethod
    def fit(cls, texts: Iterable[str], *, order: int = DEFAULT_ORDER) -> CharNgramLM:
        lm = cls(order=order)
        seen = False
        for text in texts:
            seen = True
            lm._observe(text)
        if not seen:
            raise ValueError("cannot fit a language model on an empty corpus")
        return lm

    def _observe(self, text: str) -> None:
        self._vocab.update(text)
        self._update(text, 1)

    def _update(self, text: str, delta: int) -> None:
        padded = _BOS * (self.order - 1) + text
        for i in range(self.order - 1, len(padded)):
            char = padded[i]
            context = padded[i - (self.order - 1) : i]
            # Every suffix of the context is a backoff level, including the empty
            # one, which is the unigram distribution.
            for k in range(len(context) + 1):
                key = context[len(context) - k :]
                counts = self._counts[key]
                counts[char] += delta
                self._totals[key] += delta
                # A count that reaches zero must go, not linger: `_probability`
                # reads an empty Counter as "this context was never seen" and
                # backs off, and a zero entry would make it interpolate instead.
                if counts[char] <= 0:
                    del counts[char]
                if not counts:
                    del self._counts[key]
                    del self._totals[key]

    @contextmanager
    def without(self, text: str) -> Iterator[None]:
        """Temporarily forget `text`, so it can be scored as if never seen.

        The vocabulary is deliberately left intact. It only sets the uniform
        floor every backoff bottoms out in, and holding it fixed keeps scores on
        one scale across traces — which is the whole point of a ranking.
        """
        self._update(text, -1)
        try:
            yield
        finally:
            self._update(text, 1)

    @property
    def _vocab_size(self) -> int:
        # +1 reserves mass for a character the corpus never contained.
        return len(self._vocab) + 1

    def _probability(self, context: str, char: str) -> float:
        probability = 1.0 / self._vocab_size
        for k in range(len(context) + 1):
            counts = self._counts.get(context[len(context) - k :])
            if not counts:
                # Nothing was ever seen in this context; keep the shorter estimate.
                continue
            distinct = len(counts)
            total = self._totals[context[len(context) - k :]]
            probability = (counts[char] + distinct * probability) / (total + distinct)
        return probability

    def log_likelihood(self, text: str) -> tuple[float, int]:
        """Total log probability of `text`, and how many characters it covers."""
        padded = _BOS * (self.order - 1) + text
        total = 0.0
        predicted = 0
        for i in range(self.order - 1, len(padded)):
            context = padded[i - (self.order - 1) : i]
            total += math.log(self._probability(context, padded[i]))
            predicted += 1
        return total, predicted

    def perplexity(self, text: str) -> float:
        """Per-character perplexity, or NaN when there is nothing to predict.

        An empty text gets NaN rather than 1.0 because no prediction was made at
        all — the same reason `checks.verbosity_bias` reports NaN instead of a
        correlation it cannot compute.
        """
        total, predicted = self.log_likelihood(text)
        if predicted == 0:
            return math.nan
        return math.exp(-total / predicted)


def trace_text(trace: Trace, field: str) -> str:
    if field == "input":
        return trace.input
    if field == "output":
        return trace.output
    if field == "both":
        return f"{trace.input}\n{trace.output}"
    raise ValueError(f"unknown field {field!r}; expected one of {FIELDS}")


def score_traces(
    traces: Sequence[Trace],
    *,
    field: str = "input",
    order: int = DEFAULT_ORDER,
) -> dict[str, float]:
    """Perplexity of every trace under a model fitted to all the *others*.

    Defaults to the input side. An unusual *input* is an edge case the system was
    asked to handle and may not have; an unusual *output* is often just a long
    one. Note that the most degenerate outputs — a truncated stub, the same
    sentence repeated — are the least surprising text there is, so ranking
    outputs by perplexity sorts those to the bottom rather than the top.
    """
    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}; expected one of {FIELDS}")
    if not traces:
        return {}
    texts = [trace_text(t, field) for t in traces]
    lm = CharNgramLM.fit(texts, order=order)
    scores: dict[str, float] = {}
    for trace, text in zip(traces, texts, strict=True):
        with lm.without(text):
            scores[trace.trace_id] = lm.perplexity(text)
    return scores


def spread(scores: Mapping[str, float]) -> float:
    """How much more surprising the most surprising trace is than the typical one.

    A perplexity ranking is only worth a reviewer's time if the corpus actually
    contains something unusual. When it does not, the ranking silently
    degenerates into whichever incidental token varies -- on `examples/newsroom`,
    the issue number in the header, which is noise dressed as a signal.

    The maximum is deliberate, and a percentile is not. Outliers are rare by
    definition: three strange traces in 123 leave the 90th percentile sitting
    inside the ordinary population, which reported a spread of 1.14 for a corpus
    where the ranking was in fact discriminating by a factor of eight hundred.
    The statistic has to be sensitive to a handful of traces, because a handful
    of traces is exactly what the strategy exists to surface.

    Reporting this alongside the batch is the same discipline as `cost_usd`
    returning None instead of $0.00: the tool says when its own output does not
    mean what it looks like it means. NaN when fewer than two traces carry a
    score, since one point is not a spread.
    """
    finite = [v for v in scores.values() if math.isfinite(v)]
    if len(finite) < 2:
        return math.nan
    median = float(np.median(finite))
    if median <= 0:
        return math.nan
    return max(finite) / median
