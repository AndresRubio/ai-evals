---
name: data-contracts-at-ingest
description: Expectation, checkpoint and data docs from Great Expectations mapped onto contract, gate and alignment report, with checks for axial ingest
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://greatexpectations.io/
    read: 2026-10-07
related: [great-expectations, methodology-spine]
---
# Data contracts at ingest

The data-quality loop as [Great Expectations](../landscape/great-expectations.md)
defines it, and what it says about the traces an eval pipeline consumes. The point is
that a contract is checked where data enters, and a failed check stops the run rather
than warning.

## The mapping

| Great Expectations | What it is | In axial |
|---|---|---|
| Expectation | one testable assertion about data | one labelled failure mode |
| Expectation Suite | the assertions that define acceptable data | the labelled split for a mode |
| Validation | running a suite against a batch | `judge` on a split, `measure` on a batch |
| Checkpoint | the orchestration point that runs validations and acts | `gate`, and its exit code |
| Data Docs | generated human-readable record of each run | `alignment.md` and `alignment.json` |

## Steps

1. **Write the expectations before the data arrives.**
   Trigger: a new trace source is wired up.
   Artifact: the contract for a trace: required fields present, input and output
   non-empty, trace ids unique, timestamps inside the export's range, metadata keys
   known. The [methodology spine](../methodology/methodology-spine.md) calls the trace
   the unit everything else depends on.
   Gate: a human writes the contract.
2. **Validate on entry.**
   Trigger: `axial ingest` reads a file.
   Artifact: a pass, or a list of every violation with its row. Today ingest fails on
   the first missing required field; the checks above would run before any trace is
   written and report all of them.
   Gate: a failed contract stops the ingest. Nothing is written.
3. **Validate on exit.**
   Trigger: a review batch, a split or a measurement is written.
   Artifact: the same contract applied to the artifact: a split with every trace id
   resolvable, a batch with no duplicates, a measurement whose traces all parsed.
   Gate: a failed check stops the command.
4. **Keep the record human-readable.**
   Trigger: any validation.
   Artifact: a Markdown report of what was checked and what failed, committed beside
   the data, as Data Docs are generated beside a pipeline.
   Gate: none.
5. **Alert and ticket.**
   Trigger: a failed validation in a scheduled job.
   Artifact: a notification that becomes a work item, as in
   [production-monitoring](production-monitoring.md).
   Gate: none.
