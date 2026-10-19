---
name: evals-in-ci
description: Gate a pull request on a promoted judge and keep the gate honest with deterministic checks first, intervals not points, and a saturation watch
shelf: procedures
updated: 2026-10-20
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://anthropic.com/engineering/demystifying-evals-for-ai-agents
    read: 2026-09-17
  - url: https://docs.evidentlyai.com/examples/GitHub_actions.md
    read: 2026-10-07
related: [methodology-spine, numbers-and-rules-of-thumb, evidently, production-monitoring]
---
# Evals in CI

Step eleven of the [methodology spine](../methodology/methodology-spine.md) as a
procedure. The contract is a frozen regression set and a threshold; the check runs on
every change; the gate blocks the merge. The numbers quoted are collected in
[numbers-and-rules-of-thumb](../methodology/numbers-and-rules-of-thumb.md).

## Steps

1. **Freeze the regression set.**
   Trigger: a judge is promoted, or a bug is fixed and its trace becomes a case.
   Artifact: a small purpose-built dataset, a hundred or more examples, covering core
   features, fixed bugs and known edge cases. In axial this is the labelled split and
   the promoted judge directory under `.axial/judges/`, committed with the code.
   Gate: a human decides what goes in. The test split is write-once.
2. **Prefer deterministic assertions.**
   Trigger: writing a check.
   Artifact: a code assertion wherever one can express the failure, because CI runs
   constantly and judge calls cost money. A judge only for failure modes code cannot
   express.
   Gate: none.
3. **Run on every change and every model upgrade.**
   Trigger: a pull request, and any change to the model behind the system.
   Artifact: a verdict per case. For non-deterministic systems, run each case several
   times and report pass@k when one working answer is enough or pass^k when every run
   must succeed.
   Gate: none.
4. **Gate on an interval, not a point.**
   Trigger: the verdicts exist.
   Artifact: a pass or fail exit code. Two runs of the same system differ by a few
   points routinely, so a gate set inside that noise fails builds at random until
   someone disables it. axial's `measure` grades the CI inputs with a promoted judge
   and corrects the flag rate for the judge's test-split errors; `gate --batch` then
   fails on a corrected rate whose interval reaches above a ceiling (`--max-rate`), or
   on a rise against a baseline batch (`--baseline-batch`). Running the same inputs
   through both versions lets the comparison pair them, which catches more: in axial's
   simulation, 1,000 shared inputs caught a 4-point rise 81% of the time paired and 58%
   unpaired.
   Gate: the exit code blocks the merge.
5. **Watch for saturation.**
   Trigger: a suite stops failing anything.
   Artifact: a warning that the eval has stopped carrying signal, which axial's gate
   prints. Retire or replace the suite.
   Gate: a human decides what replaces it.

## The Evidently form

Evidently documents the same loop as a GitHub Actions job: load the dataset, run Tests
with pass or fail conditions, upload the result to the platform, fail the job on a
failed test. See [evidently](../landscape/evidently.md). The difference is upstream:
Evidently's judge is a prompt you wrote; axial's is one whose agreement with your
labels was measured before it was allowed into this job.
