---
name: factory-lifecycle
description: Issue to merged PR with agents doing every stage and humans at three gates, as the Warp factory examples define it
shelf: procedures
updated: 2026-10-07
sources:
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/02-sdlc-issue-to-pr
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/05-ui-verification
    read: 2026-10-07
  - url: https://github.com/warpdotdev/warp-factory-examples/tree/main/examples/06-common-automations
    read: 2026-10-07
related: [warp-factories, production-monitoring, evals-in-ci]
---
# The factory lifecycle

A standing engineering loop in which agents do every stage and a person decides only
what gets built and what ships. Taken from the full-lifecycle, UI-verification and
common-automations examples of [Warp Factories](../landscape/warp-factories.md). Every
stage starts from an event and ends by emitting an artifact that is itself the next
trigger; nothing runs on a schedule except standing audits.

## Steps

1. **Intake.**
   Trigger: a Slack mention or DM, a Linear agent session, or a GitHub mention or
   assignment on an issue carrying the factory's routing label. Filters combine with
   AND, so the label is the volume control.
   Artifact: a work item the foreman owns. The foreman first checks the event is
   addressed to this factory and stops silently when it is not.
   Gate: none.
2. **Triage.**
   Trigger: the foreman dispatches it whenever cause or scope is unknown, the work needs
   a tracked issue, or it may duplicate existing work. When in doubt, triage.
   Artifact: a Linear issue that is the durable record: the request in the requester's
   words, findings, code locations, reproduction steps for bugs. Plus a complexity
   estimate and the ambiguities a spec would need to resolve. Runs on a cheap model.
   Gate: none.
3. **Spec.**
   Trigger: triage reported the work as ambiguous or large.
   Artifact: the smallest Markdown spec that resolves the ambiguity, with desired
   behaviour, non-goals and validation criteria, committed in a draft PR on a branch
   named for the work item.
   Gate: a human approves the spec. The agent never approves on a human's behalf.
4. **Implement.**
   Trigger: the foreman dispatches with the work item and, when one exists, the spec PR.
   Artifact: the change, tests for the changed behaviour, and test and linter output
   pasted into the PR description, in the spec's own PR and branch. Runs on a larger
   runner with the toolchain image. Scope growth is reported, not absorbed.
   Gate: none.
5. **Review.**
   Trigger: the implementation PR exists.
   Artifact: a findings report to the foreman, each finding with file and line,
   severity, problem, impact and correction, classified as unambiguous or a judgment
   call. The reviewer runs on a different model from the implementer so the two do not
   share blind spots, reviews against the issue and spec rather than style, and checks
   that the tests exercise the changed behaviour. The foreman relays unambiguous
   findings back to the implementer until the PR is clean.
   Gate: judgment calls wait for a human.
6. **Verify.**
   Trigger: a PR labelled for a UI check, or an issue labelled for reproduction.
   Artifact: a comment with screenshots or a recording showing the behaviour, made by
   checking out the branch, running the app and driving it in a browser. The verifier
   never pushes fixes.
   Gate: none.
7. **Ship.**
   Trigger: review is clean and verification posted.
   Artifact: the merged PR.
   Gate: a human merges. The foreman never merges.
8. **Close the loop.**
   Trigger: a factory-labelled PR is merged or closed.
   Artifact: merged, the linked issues move to done; closed without merging, a report
   on the issue instead.
   Gate: none.
9. **Monitor.**
   Trigger: a CI workflow fails on the default branch.
   Artifact: a fix PR when the cause is small and clear, otherwise an issue with the
   failing step, the error and the commit range. That issue is step one's input.
   Gate: none.
10. **Grade the agents.**
    Trigger: a sampled fraction of implementer and reviewer runs.
    Artifact: a scorer verdict per run: did the PR deliver exactly what the issue asked,
    judged from the issue text; did each review finding carry a location, impact and
    correction. The implementer scorer feeds its verdicts back into the agent.
    Gate: none, which is the problem the next section names.

## What to keep and what to add

Keep the explicit "you do not do X" clause on every agent, so handoffs are the only way
work moves, and the routing label, so a person can inject work anywhere by applying it.

Add a measurement of the scorers before self-improvement is turned on. The scorers are
LLM judges with no recorded agreement against human labels; see
[warp-factories](../landscape/warp-factories.md). Validating one is the
[methodology spine](../methodology/methodology-spine.md) from step seven onward, and
feeding production back into it is [production-monitoring](production-monitoring.md).
