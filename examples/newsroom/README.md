# The newsroom example

A toy story-extraction app: given newsletter text, emit discrete stories with a topic
label and sources. `traces.jsonl` holds 400 traces with four failure modes planted in
a repeating cycle, recorded in each trace's `planted_mode` metadata so the tutorial can
check its own work.

The corpus is entirely invented. It derives from no real newsletter and contains no
personal data. Regenerate it with:

    uv run python examples/newsroom/build_corpus.py

Walk the pipeline:

    uv run axial init
    uv run axial ingest examples/newsroom/traces.jsonl
    uv run axial sample --n 40 --name first-pass

Uncomment and set `annotator = "your-name"` near the top of `.axial/config.toml`
(every label records who made it), then:

    uv run axial annotate --name first-pass     # annotate 30+, then stop
    uv run axial taxonomy

`axial taxonomy` is the first command here that calls your configured model, so
from this point on you need your own model credentials for it — `axial annotate`
itself calls no model at all; it just serves a local review UI. The full pipeline
beyond taxonomy — human labels, a frozen split, a judge, alignment, and promotion —
runs entirely offline against this corpus in `tests/test_end_to_end.py`, using
`FakeModel` in place of a real model. Read that test alongside this file for the
rest of the walkthrough: it is the part of the pipeline validated on every CI run.
