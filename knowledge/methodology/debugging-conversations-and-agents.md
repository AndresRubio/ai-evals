---
name: debugging-conversations-and-agents
description: Find the first failure in a conversation or agent run, cut it to the smallest failing test, and count failures by step in a transition matrix
shelf: methodology
updated: 2026-11-24
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/how-do-i-debug-multi-turn-conversation-traces.html
    read: 2026-11-24
  - url: https://hamel.dev/blog/posts/evals-faq/how-do-i-evaluate-agentic-workflows.html
    read: 2026-11-24
  - url: https://hamel.dev/notes/llm/evals/flashcards/
    read: 2026-11-24
related: [methodology-spine, points-of-disagreement, mechanizable-vs-human]
---

# Debugging conversations and agents

The [methodology spine](methodology-spine.md) reads one trace and notes its first
failure. A conversation of many turns, or an agent that plans, calls tools and hands
off, makes that harder: the trace is long and the failure sits somewhere inside it.
Husain and Shankar cover this in two FAQ answers, on multi-turn conversation traces
(2025-05-29) and on agentic workflows (2025-06-29), and in two of Husain's flashcards
(2025-11-30), *Traces for Evals* and *Transition Matrix*.

## Reading a conversation

1. Judge the whole conversation first, pass or fail, on whether it met the user's
   goal.
2. Read the user-visible parts before the tool calls and intermediate steps.
3. Find the first upstream failure and annotate only that one at first. Later failures
   often cascade from it, so fixing it can resolve them. With experience, a reviewer
   can annotate independent failure modes in the same trace to go faster.

For a flow with several agents, the multi-turn answer says to give each user request a
session or trace ID and to log every message with its source (which agent or tool),
that ID and its position in the sequence, so the path from query to result can be
rebuilt.

## Cutting the failure down

Reproduce the failure with the simplest test that still fails. The FAQ's example is a
shopping bot that gives the wrong return policy on turn 4. Ask it the single-turn
question "What is the return window for product X1000?". If that fails too, the error
is not about conversation context, and is likely a retrieval or knowledge problem that
is easier to debug. The flashcard puts it as reproducing the failure in as few turns
and with as little complexity as possible. The agentic answer gives the same rule for
agents and adds that a multi-turn test is for a failure that depends on conversation
context.

## Turning failures into test cases

The FAQ names two ways to get multi-turn test cases.

N-1 testing gives the system the first N-1 turns of a real conversation and tests what
it does next. The FAQ says this often works better, because the prefix is a real
conversation, and that it is less flexible. The flashcard builds the dataset from the
failures: collect the minimally reproduced error traces and use the N-1 turns before
the error as test cases. It adds that this assumes the system is not changing rapidly.

Simulating the user with a second LLM produces whole synthetic conversations. The
flashcard lists it as advanced and says doing it well can be challenging. It lists one
more advanced step for coverage: modify the N-1 traces in valid ways with an LLM, such
as rephrasing the user's question.

## Agents: the outcome first, then the steps

The agentic answer evaluates in two phases.

End-to-end task success comes first. Treat the agent as a black box, define a precise
success rule per task, and measure it with human review or a validated LLM judge.
Record the first upstream failure during error analysis.

Step-level diagnostics follow, once error analysis shows which workflows fail most
often. The answer lists six things to score: tool choice, parameter extraction, error
handling (empty results, API failures), context retention (earlier constraints kept),
efficiency (steps, seconds and tokens) and goal checkpoints in long workflows.

For tool calls it says to test the tool name, the arguments, the result and the
resulting state as separate checks, with code assertions where the expected behaviour
is objective. Its example verifies that the agent chose `cancel_order`, passed the
right order ID, got a successful response and changed the order's status before it
told the user the cancellation succeeded. It also says to test authorization and
preconditions, since a valid tool call is still wrong when the user did not approve
the action or a required check was skipped.

This ordering bears on item h of [points-of-disagreement](points-of-disagreement.md),
trajectory against outcome: the outcome decides whether the task succeeded, and the
steps are then scored, each pass or fail, to find out why it did not.

## The transition failure matrix

When an agent has many steps, a transition failure matrix shows which step fails most.
The flashcard gives three steps.

1. List the states the agent can be in, such as Plan, Search, Code and Finalize.
2. Draw a grid with the "from" state on the rows and the "to" state on the columns.
3. For each failure, find the last successful transition before the error and add one
   to that cell.

The FAQ words the axes as the last successful state on the rows and the state where
the first failure occurred on the columns. In its text-to-SQL example the transition
from generating SQL to executing it holds 12 failures and the one from deciding on a
tool to planning a calendar step holds 2, which says where to look first. It also
shows a matrix from Bryan Bischof's talk *Failure is a Funnel* (Data Council, 2025)
and notes that how the matrix is organised depends on the application: Bischof's
agent has an inherently sequential workflow, which he uses for further analysis.

## In axial

None of this is implemented. A `Trace` in `axial` is one `input`, one `output` and a
`metadata` dictionary (`src/axial/core/traces.py`), with no turns, steps or states. A
review holds an open code, a verdict and a critique for the whole trace and has no
field for the turn or step that failed. A whole conversation can be ingested as text
and the first failure named in the open code, which covers the three reading steps
above and not the per-message logging for several agents.
Cutting an N-1 prefix and counting transitions both need a trace that knows its own
turns and steps, and that is a change to the trace model.
