---
name: guardrails
description: Block a clear-cut, high-impact failure in the request path with a fast check, and treat every false block as a production bug
shelf: procedures
updated: 2026-11-24
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/whats-the-difference-between-guardrails-evaluators.html
    read: 2026-11-24
  - url: https://hamel.dev/notes/llm/evals/flashcards/
    read: 2026-11-24
related: [evals-in-ci, production-monitoring, methodology-spine, mechanizable-vs-human]
---
# Guardrails

A guardrail is a check that sits in the request and response path and stops a failure
before a user sees it. Husain and Shankar's FAQ answer on guardrails and evaluators
(2025-06-29) separates the two by when they run and what they may cost: a guardrail
runs inline and has to be fast, and an evaluator runs after the response and can
afford an LLM judge. Two of Husain's flashcards (2025-11-30), *Types of Automated
Evals* and *Deploy Evals*, place guardrails beside the two deployments this shelf
already covers, [evals-in-ci](evals-in-ci.md) and
[production-monitoring](production-monitoring.md).

## Three ways to deploy a check

The *Deploy Evals* flashcard compares them on five questions.

| | CI/CD | Online monitoring | Guardrails |
|---|---|---|---|
| Goal | Prevent regressions | Discover new failures and track performance | Enforce safety and block high-impact errors |
| When | Pre-merge, on the pull request | Asynchronous, after the response | Synchronous, before the response |
| How | Unit tests, LLM judge | Unit tests, LLM judge, A/B testing | Unit tests, small classifiers |
| Data | Curated test cases | Sampled production traffic | 100% of live traffic |
| On failure | Block the merge | Trigger an alert | Block the response, retry, or fall back to an alternative |

## Steps

1. Pick the failures that earn a guardrail.
   Trigger: error analysis or an incident shows a failure that is clear-cut and
   high-impact. The FAQ's examples are PII leaks, profanity, disallowed instructions,
   SQL injection, malformed JSON and invalid code syntax.
   Artifact: a short list of failures to block. A subjective or nuanced criterion goes
   to an evaluator instead.
   Gate: none in the sources. This entry's view is that a person decides what is worth
   blocking.
2. Write the check as code or a small classifier.
   Trigger: a failure on the list.
   Artifact: a regex, a keyword block-list, a schema or type validator, or a
   lightweight classifier. The FAQ describes guardrails as fast and deterministic,
   with typically a few milliseconds of latency budget, and as simple and explainable.
   An LLM judge inline is possible only when the latency budget and reliability
   targets allow it; the FAQ suggests a slow judge might be feasible in a cascade that
   runs on the minority of borderline cases. It warns against using an LLM guardrail
   off the shelf without reading its prompt.
   Gate: none.
3. Decide what happens when it fires.
   Trigger: the check exists.
   Artifact: the response to a trigger. The FAQ lists redact, refuse or regenerate;
   the flashcard lists block the response, retry, or fall back to an alternative.
   Gate: none in the sources, which name the options and not who chooses.
4. Run it on every request, before the response.
   Trigger: live traffic. A guardrail sees all of it, where monitoring sees a sample.
   Artifact: a response that went through, or one that was blocked, redacted or
   regenerated, and a log line for every trigger.
   Gate: none. This is the automated step.
5. Treat a false block as a production bug.
   Trigger: a guardrail fires on a valid response. The user sees it happen, which is
   why the FAQ counts a false positive as a bug.
   Artifact: versioned guardrail rules, a log of every trigger, and a monitored
   trigger rate, all three kept so the rules stay conservative. The flashcard says
   guardrails tend to be fast and have a low false positive rate, to avoid blocking
   valid responses.
   Gate: the sources do not say who reads the triggers or how often.

## With evaluators

The FAQ's summary is that the two are layers. Guardrails give immediate protection
against objective failures that need intervention. Evaluators measure what simple
rules cannot, such as factual correctness and completeness, and their verdicts feed
dashboards, regression tests and improvement loops without blocking the original
answer.

## In axial

`axial` does not run guardrails. Its checks run before a merge (`axial gate`) and on
sampled batches after the response (`axial measure`), and nothing in it sits in the
request path. The design note in
[mechanizable-vs-human](../methodology/mechanizable-vs-human.md) lists guardrails as
mechanizable and architecturally separate from evaluators.
