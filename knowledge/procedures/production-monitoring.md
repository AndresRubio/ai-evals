---
name: production-monitoring
description: Sample live traces, measure them with a promoted judge, alert on the interval's lower bound, and feed flagged traces back into review
shelf: procedures
updated: 2026-10-20
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://docs.evidentlyai.com/docs/platform/monitoring_scheduled_evals.md
    read: 2026-10-07
  - url: https://docs.evidentlyai.com/docs/platform/monitoring_local_batch.md
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/06-common-automations
    read: 2026-10-07
related: [methodology-spine, evals-in-ci, evidently, warp-factories, factory-lifecycle]
---
# Production monitoring

Steps twelve and thirteen of the [methodology spine](../methodology/methodology-spine.md)
as a procedure. Production has no reference outputs, so it leans on a reference-free
judge run asynchronously over sampled live traces, and the loop closes when what the
judge flags is read by a person and promoted into the CI set.

## Steps

1. **Collect traces.**
   Trigger: live traffic.
   Artifact: a trace store with the complete record of each request: messages, tool
   calls, retrievals. Evidently's Tracely and any OpenTelemetry pipeline produce this;
   axial ingests JSONL or CSV through an adapter.
   Gate: none.
2. **Sample a batch.**
   Trigger: a schedule, daily or weekly, matched to traffic.
   Artifact: a batch mixing a random slice with targeted slices, because every targeted
   method has a bias and the random slice is where unknown failures appear.
   Gate: none.
3. **Measure with a promoted judge.**
   Trigger: the batch exists.
   Artifact: a corrected failure rate with a confidence interval. A judge with known
   true-positive and true-negative rates on its test split gives a raw flag rate that is
   neither the failure rate nor an interval. axial's `measure` adds both: it corrects
   the batch's flag rate with the judge's test-split error rates and reports a 95%
   interval that includes the judge's own uncertainty, plus the interval the batch
   alone would give if the judge's rates were exact. Evidently's scheduled evals and
   batch monitoring jobs run the equivalent for its own descriptors.
   Gate: none.
4. **Alert on the lower bound.**
   Trigger: the interval's lower bound crosses the threshold.
   Artifact: an alert, and from it a ticket. The common-automations example of
   [Warp Factories](../landscape/warp-factories.md) shows the ticket form: an issue with
   the failing step, the error and the commit range, which re-enters intake. A soft
   warning is not monitoring.
   Gate: none.
5. **Promote flagged traces into review.**
   Trigger: the measurement exists.
   Artifact: a review batch drawn from the measured batch, mixing flagged and unflagged
   traces, that a person annotates. New failure modes found here go through open
   coding and into the taxonomy; fixed ones become CI cases.
   Gate: a human reads them. This is the irreducibly human step.
6. **Re-check the judge on what it measured.**
   Trigger: the labels from step five exist.
   Artifact: the judge's true-positive and true-negative rates on this batch beside the
   rates from its test split. The correction in step three assumed they still hold;
   production drifts, so this is the check on the checker.
   Gate: a human decides whether the judge needs a new version, which starts the
   alignment loop again.

## Drift

A batch that looks unlike the data the judge was aligned on is a reason to distrust
step three before any alert fires. Evidently detects it with embedding and
distribution drift over text. axial has no drift check today; its perplexity model
ranks traces for review within one corpus and is not comparable across corpora.
