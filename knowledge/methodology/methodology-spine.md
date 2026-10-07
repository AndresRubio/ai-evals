---
name: methodology-spine
description: The thirteen-step error-analysis-first loop from instrumentation to production monitoring, with each step's artifact
shelf: methodology
updated: 2026-09-17
sources:
  - url: https://hamel.dev/blog/posts/evals-faq/
    read: 2026-09-17
  - url: https://hamel.dev/blog/posts/llm-judge/
    read: 2026-09-17
  - url: https://hamel.dev/blog/posts/field-guide/
    read: 2026-09-17
  - url: https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill
    read: 2026-09-17
  - url: https://www.davidokpare.com/blog/a-primer-to-evals
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/product-evals/
    read: 2026-09-17
  - url: https://anthropic.com/engineering/demystifying-evals-for-ai-agents
    read: 2026-09-17
  - url: https://platform.claude.com/docs/en/docs/test-and-evaluate/develop-tests
    read: 2026-09-17
  - url: https://developers.openai.com/api/docs/guides/evaluation-best-practices
    read: 2026-09-17
related: [numbers-and-rules-of-thumb, mechanizable-vs-human]
---

# The methodology spine

All the practitioner sources converge on the same loop. Husain and Shankar name it **Analyze → Measure → Improve**, wrapped around their **Three Gulfs** diagnosis (Comprehension: you don't know what your data does; Specification: you haven't said what "good" is; Generalization: it works on A and B but not C). Eugene Yan compresses it to three steps: label data → align an evaluator → run a harness on every change. OpenAI's cookbook lifecycle and Anthropic's agent-eval guide describe the same shape with different entry points.

## Step 0 — Instrumentation
**Input:** a running system. **Artifact:** a **trace store**. A trace is the complete record from one user query to final response — messages, tool calls, retrievals, intermediate spans (Husain's definition). Without this there is nothing to analyze. Husain: store prompts in Git as software artifacts, versioned and deployed atomically with code; prompt-management SaaS creates indirection and can't execute your application code.

## Step 1 — Build a custom data viewer
**Artifact:** a **bespoke annotation interface**. Husain calls this the single highest-leverage investment in the whole workflow, and claims custom tools iterate ~10x faster than off-the-shelf. Required properties: render traces domain-appropriately (emails look like emails, code syntax-highlighted); progressive disclosure with less-relevant detail collapsed; progress counter ("45 of 100"); keyboard hotkeys for next/label; filter, sort, cluster, semantic search; one-click open-ended note capture; all context from all systems on one screen so non-technical reviewers judge outcomes, not implementation. Off-the-shelf is justified only for dozens of distributed annotators needing enterprise access control. Stack he suggests for Python devs: FastHTML + MonsterUI; a spreadsheet is an acceptable v0.

## Step 2 — Assemble a review sample
**Artifact:** a **working pool of diverse traces**. Sampling methods, ordered exploratory → targeted (Husain): random; embedding clustering; data analysis on extreme values (latency, turn count, retries); classifier flagging; user-feedback-negative; targeted search for known signals. Each method has a bias — classification favours known problems, feedback misses unreported ones — so keep a random slice in every batch for serendipitous discovery. If you have no users yet, bootstrap with synthetic inputs (Step 2b).

## Step 2b — Synthetic bootstrap (when real data is thin)
**Artifact:** a **dimension table + tuple set + generated queries + real traces**. The recipe (Field Guide, LLM-judge post, FAQ): define dimensions — **features × scenarios × personas** — then write ~20 tuples by hand, then scale by LLM in two stages: (1) generate structured tuples, (2) convert each tuple to a natural-language query. Cross-product-then-filter guarantees coverage; direct LLM generation is more realistic but misses rare scenarios. Generate **user inputs only**, never outputs — run the synthetic inputs through the real system to capture full traces. Real-estate example dimensions: features {property search, market analysis, scheduling, follow-up} × scenarios {exact match, multiple matches, no match, invalid criteria} × personas {first-time buyer, investor, luxury client, relocating family}. Verify the generated scenario actually triggers the intended case. Unreliable for: complex domain content (legal/medical), low-resource languages, unverifiable domains, high-stakes settings, underrepresented user groups.

## Step 3 — Open coding
**Artifact:** a **spreadsheet of free-text observations, one row per trace**. A human reads traces and writes open-ended notes on what went wrong. Two rules: (a) focus on the **first upstream failure** — errors compound, so early failures are the real ones; (b) do **not** write a rubric first. Husain is explicit that a pre-written rubric obstructs discovery and causes reviewers to miss problems outside it. Stop at **theoretical saturation** — when new traces stop producing new failure modes or changing existing ones.

## Step 4 — Axial coding
**Artifact:** a **failure taxonomy with counts**. Group the open codes into categories, then count frequency per category (pivot table). This is where prioritization comes from. NurtureBoss example: three categories — conversation flow, handoff failures, rescheduling — accounted for **over 60%** of all problems; fixing date handling moved success from **33% → 95%**. LLMs may propose the first-pass grouping (typically 5–6 categories, per the Lenny's interview) but a human must validate it.

## Step 5 — Fix the obvious things first
**Artifact:** an updated prompt/pipeline. Husain: many "failures" are just preferences you never specified (length, format, step-by-step reasoning). Do not build an expensive evaluator for something a prompt line fixes. Only persistent failures — ones you iterate on repeatedly — earn an automated evaluator.

## Step 6 — Choose evaluator type per failure mode
**Artifact:** an **evaluator registry** mapping failure mode → evaluator. The cost hierarchy is universal:
1. **Code assertions** — regex, schema/JSON validation, execution tests, outcome/state checks. Cheap, deterministic, reproducible. Anthropic: prefer these wherever possible.
2. **LLM-as-judge** — only for subjective qualities. Expensive: ~100+ hand labels to build, ongoing maintenance.
3. **Human review** — gold standard, doesn't scale.
Anthropic's agent guide adds the crucial agent-specific rule: grade the **outcome/final state**, not the trajectory. An agent can say "your flight is booked" without a reservation existing; and rigid path-checking punishes valid approaches the designer didn't anticipate. Grade transcripts only as a secondary signal.

## Step 7 — Label a ground-truth set (binary + critique)
**Artifact:** a **labeled dataset**: `{trace, binary verdict, written critique}`. The single domain expert ("**benevolent dictator**") labels. Format is **pass/fail + a detailed critique** explaining the reasoning — terse critiques are the common mistake; the critique should be detailed enough that a new employee understands the judgment. Critiques are load-bearing twice over: they become few-shot examples for the judge, and they surface criteria drift.

## Step 8 — Split
**Artifact:** **train / dev / test manifests**. Husain's split: **train 10–20%** (examples that may appear in the judge prompt), **dev 40–45%** (refining), **test 40–45%** (frozen, final). Target **30–50 Pass and 30–50 Fail examples in each of dev and test** where possible.

## Step 9 — Build and align the judge
**Artifact:** an **aligned judge** + an **alignment report**. Iterate: seed the prompt with 3–5 diverse expert examples (with the same external context the judge will have at runtime), run against dev labels, inspect disagreements, revise, repeat. Measure **TPR and TNR separately** — never raw agreement. Husain's worked argument: at a 5% error rate, a judge that predicts "Pass" always scores 95% agreement and catches 0% of errors. Context discipline: give the judge only what that failure mode needs; full traces by default cause "context rot" and make the judge worse. Validate context choices by ablation — remove one element at a time, see if alignment moves.

## Step 10 — Error analysis on the judge's output
**Artifact:** **error rates sliced by dimension** (feature × scenario × persona) and a **root-cause classification**. Husain reports this is often ~15 minutes of spreadsheet work over ~20 failed interactions across 4–5 root-cause buckets. This tells you where to spend engineering effort — and whether a specialized second judge or a code assertion is warranted.

## Step 11 — CI gate
**Artifact:** a **frozen regression set + threshold config**. Husain's CI profile: small purpose-built datasets (100+ examples) covering core features, regression tests for fixed bugs, and known edge cases; **favour deterministic assertions over LLM judges in CI** for cost reasons since it runs constantly. Anthropic: run on every agent change and model upgrade; handle non-determinism with repeated runs and report **pass@k** (at least one of k succeeds) or **pass^k** (all k succeed) depending on whether the product needs one working answer or consistency. Monitor **saturation** — 100% pass rate means the eval has stopped giving signal and should be retired or de-prioritized.

## Step 12 — Production monitoring
**Artifact:** **sampled online scores with confidence intervals**. Production has no reference outputs, so it leans on reference-free evaluators (LLM judges) run asynchronously on sampled live traces. Husain: track confidence intervals; if the **lower bound crosses your threshold, investigate**. The loop closes here — new patterns found in production get error-analyzed and promoted into the CI dataset.

## Step 13 — Re-run error analysis (the loop is non-terminal)
Cadence guidance below. Gold sets go stale as the product and usage change; evals cost more to maintain than unit tests, so weigh maintenance against signal and retire evals that always pass.
