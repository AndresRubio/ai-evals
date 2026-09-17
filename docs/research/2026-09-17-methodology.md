# LLM Evaluation Methodology as Practiced by Leading Practitioners

Research compiled 2026-09-17. Primary source: Hamel Husain's AI Evals FAQ (pub. 2025-05-28, last modified 2026-09-13).

---

## 1. The methodology spine

All the practitioner sources converge on the same loop. Husain and Shankar name it **Analyze → Measure → Improve**, wrapped around their **Three Gulfs** diagnosis (Comprehension: you don't know what your data does; Specification: you haven't said what "good" is; Generalization: it works on A and B but not C). Eugene Yan compresses it to three steps: label data → align an evaluator → run a harness on every change. OpenAI's cookbook lifecycle and Anthropic's agent-eval guide describe the same shape with different entry points.

### Step 0 — Instrumentation
**Input:** a running system. **Artifact:** a **trace store**. A trace is the complete record from one user query to final response — messages, tool calls, retrievals, intermediate spans (Husain's definition). Without this there is nothing to analyze. Husain: store prompts in Git as software artifacts, versioned and deployed atomically with code; prompt-management SaaS creates indirection and can't execute your application code.

### Step 1 — Build a custom data viewer
**Artifact:** a **bespoke annotation interface**. Husain calls this the single highest-leverage investment in the whole workflow, and claims custom tools iterate ~10x faster than off-the-shelf. Required properties: render traces domain-appropriately (emails look like emails, code syntax-highlighted); progressive disclosure with less-relevant detail collapsed; progress counter ("45 of 100"); keyboard hotkeys for next/label; filter, sort, cluster, semantic search; one-click open-ended note capture; all context from all systems on one screen so non-technical reviewers judge outcomes, not implementation. Off-the-shelf is justified only for dozens of distributed annotators needing enterprise access control. Stack he suggests for Python devs: FastHTML + MonsterUI; a spreadsheet is an acceptable v0.

### Step 2 — Assemble a review sample
**Artifact:** a **working pool of diverse traces**. Sampling methods, ordered exploratory → targeted (Husain): random; embedding clustering; data analysis on extreme values (latency, turn count, retries); classifier flagging; user-feedback-negative; targeted search for known signals. Each method has a bias — classification favours known problems, feedback misses unreported ones — so keep a random slice in every batch for serendipitous discovery. If you have no users yet, bootstrap with synthetic inputs (Step 2b).

### Step 2b — Synthetic bootstrap (when real data is thin)
**Artifact:** a **dimension table + tuple set + generated queries + real traces**. The recipe (Field Guide, LLM-judge post, FAQ): define dimensions — **features × scenarios × personas** — then write ~20 tuples by hand, then scale by LLM in two stages: (1) generate structured tuples, (2) convert each tuple to a natural-language query. Cross-product-then-filter guarantees coverage; direct LLM generation is more realistic but misses rare scenarios. Generate **user inputs only**, never outputs — run the synthetic inputs through the real system to capture full traces. Real-estate example dimensions: features {property search, market analysis, scheduling, follow-up} × scenarios {exact match, multiple matches, no match, invalid criteria} × personas {first-time buyer, investor, luxury client, relocating family}. Verify the generated scenario actually triggers the intended case. Unreliable for: complex domain content (legal/medical), low-resource languages, unverifiable domains, high-stakes settings, underrepresented user groups.

### Step 3 — Open coding
**Artifact:** a **spreadsheet of free-text observations, one row per trace**. A human reads traces and writes open-ended notes on what went wrong. Two rules: (a) focus on the **first upstream failure** — errors compound, so early failures are the real ones; (b) do **not** write a rubric first. Husain is explicit that a pre-written rubric obstructs discovery and causes reviewers to miss problems outside it. Stop at **theoretical saturation** — when new traces stop producing new failure modes or changing existing ones.

### Step 4 — Axial coding
**Artifact:** a **failure taxonomy with counts**. Group the open codes into categories, then count frequency per category (pivot table). This is where prioritization comes from. NurtureBoss example: three categories — conversation flow, handoff failures, rescheduling — accounted for **over 60%** of all problems; fixing date handling moved success from **33% → 95%**. LLMs may propose the first-pass grouping (typically 5–6 categories, per the Lenny's interview) but a human must validate it.

### Step 5 — Fix the obvious things first
**Artifact:** an updated prompt/pipeline. Husain: many "failures" are just preferences you never specified (length, format, step-by-step reasoning). Do not build an expensive evaluator for something a prompt line fixes. Only persistent failures — ones you iterate on repeatedly — earn an automated evaluator.

### Step 6 — Choose evaluator type per failure mode
**Artifact:** an **evaluator registry** mapping failure mode → evaluator. The cost hierarchy is universal:
1. **Code assertions** — regex, schema/JSON validation, execution tests, outcome/state checks. Cheap, deterministic, reproducible. Anthropic: prefer these wherever possible.
2. **LLM-as-judge** — only for subjective qualities. Expensive: ~100+ hand labels to build, ongoing maintenance.
3. **Human review** — gold standard, doesn't scale.
Anthropic's agent guide adds the crucial agent-specific rule: grade the **outcome/final state**, not the trajectory. An agent can say "your flight is booked" without a reservation existing; and rigid path-checking punishes valid approaches the designer didn't anticipate. Grade transcripts only as a secondary signal.

### Step 7 — Label a ground-truth set (binary + critique)
**Artifact:** a **labeled dataset**: `{trace, binary verdict, written critique}`. The single domain expert ("**benevolent dictator**") labels. Format is **pass/fail + a detailed critique** explaining the reasoning — terse critiques are the common mistake; the critique should be detailed enough that a new employee understands the judgment. Critiques are load-bearing twice over: they become few-shot examples for the judge, and they surface criteria drift.

### Step 8 — Split
**Artifact:** **train / dev / test manifests**. Husain's split: **train 10–20%** (examples that may appear in the judge prompt), **dev 40–45%** (refining), **test 40–45%** (frozen, final). Target **30–50 Pass and 30–50 Fail examples in each of dev and test** where possible.

### Step 9 — Build and align the judge
**Artifact:** an **aligned judge** + an **alignment report**. Iterate: seed the prompt with 3–5 diverse expert examples (with the same external context the judge will have at runtime), run against dev labels, inspect disagreements, revise, repeat. Measure **TPR and TNR separately** — never raw agreement. Husain's worked argument: at a 5% error rate, a judge that predicts "Pass" always scores 95% agreement and catches 0% of errors. Context discipline: give the judge only what that failure mode needs; full traces by default cause "context rot" and make the judge worse. Validate context choices by ablation — remove one element at a time, see if alignment moves.

### Step 10 — Error analysis on the judge's output
**Artifact:** **error rates sliced by dimension** (feature × scenario × persona) and a **root-cause classification**. Husain reports this is often ~15 minutes of spreadsheet work over ~20 failed interactions across 4–5 root-cause buckets. This tells you where to spend engineering effort — and whether a specialized second judge or a code assertion is warranted.

### Step 11 — CI gate
**Artifact:** a **frozen regression set + threshold config**. Husain's CI profile: small purpose-built datasets (100+ examples) covering core features, regression tests for fixed bugs, and known edge cases; **favour deterministic assertions over LLM judges in CI** for cost reasons since it runs constantly. Anthropic: run on every agent change and model upgrade; handle non-determinism with repeated runs and report **pass@k** (at least one of k succeeds) or **pass^k** (all k succeed) depending on whether the product needs one working answer or consistency. Monitor **saturation** — 100% pass rate means the eval has stopped giving signal and should be retired or de-prioritized.

### Step 12 — Production monitoring
**Artifact:** **sampled online scores with confidence intervals**. Production has no reference outputs, so it leans on reference-free evaluators (LLM judges) run asynchronously on sampled live traces. Husain: track confidence intervals; if the **lower bound crosses your threshold, investigate**. The loop closes here — new patterns found in production get error-analyzed and promoted into the CI dataset.

### Step 13 — Re-run error analysis (the loop is non-terminal)
Cadence guidance below. Gold sets go stale as the product and usage change; evals cost more to maintain than unit tests, so weigh maintenance against signal and retire evals that always pass.

---

## 2. Concrete numbers and rules of thumb

### Effort and cadence
| Number | Claim | Source |
|---|---|---|
| **60–80%** | of development time on error analysis and evaluation | Husain, FAQ |
| **30 min** | minimum viable setup: manually review 20–50 outputs after significant changes | Husain, FAQ |
| **~30 min/week** | steady-state cost after initial setup | Husain & Shankar, Lenny's, 2025-09-25 |
| **2–4 weeks** | typical full error-analysis cycle | Husain, FAQ |
| **10–20 traces/week** | between major cycles, focused on outliers | Husain, FAQ |
| **100+ fresh traces** | per review cycle | Husain, FAQ |
| weekly → monthly | new systems need weekly analysis until patterns stabilize; mature systems monthly | Husain, FAQ |
| **~15 min** | to classify ~20 failed interactions into 4–5 root causes | Husain, LLM-judge post |
| 2 wk / 1 mo / 6 wk | data feasibility / technical feasibility / prototype + A/B | Eugene Yan, via Field Guide |

### Sample sizes
| Number | Claim | Source |
|---|---|---|
| **≥100 diverse traces** | for error discovery; **annotate the first ≥30 yourself** | Husain, FAQ |
| **theoretical saturation** | the real stopping rule, not a fixed count | Husain/Shankar |
| **100–200 examples per failure mode** | to build and validate an LLM judge | Husain, FAQ |
| **~100 per failure mode; ≥60 minimum** | below 60, confidence intervals are too wide | Husain, LLM-judge post |
| **30–50 Pass + 30–50 Fail** | in each of dev and test | Husain, FAQ + LLM-judge post |
| **~50/50 pass–fail** | aim for balanced classes | Husain, LLM-judge post |
| **100+ examples** | size of the repeatable/CI eval set | Husain, FAQ |
| **~20 tuples by hand** | before scaling synthetic generation | Husain, FAQ |
| **3–5 few-shot examples** | minimum in the judge prompt | Husain, LLM-judge post |
| **20–50 tasks** | "a great start" for agent evals, drawn from real failures | Anthropic, *Demystifying evals for AI agents*, 2026-01-09 |
| **200+ samples, ≥50–100 failures** | product eval dataset | Eugene Yan, *Product Evals in Three Simple Steps*, 2025-11 |
| **75 / 25** | alignment / held-out test split | Eugene Yan, product-evals |
| **20 labels → eval mode; 50+ → optimize; 25 per dev and test** | AlignEval's gates | Eugene Yan, *AlignEval*, 2024-10 |
| **1,000+ / 500+ / 200+ / 100+ / 50+** | exact-match / privacy / summarization / tone & context-utilization / consistency test cases | Anthropic docs, *Define success criteria and build evaluations* |
| **±2.4% at n=200; ±1.7% at n=400** | confidence interval widths | Eugene Yan, product-evals |
| **~1,000 samples** | enough to fine-tune an NLI factual-consistency model | Eugene Yan, *Task-Specific LLM Evals*, 2024-03 |

### Agreement thresholds and judge quality
- **>90% agreement in 3 iterations** — what the Honeycomb judge took to converge (Husain, LLM-judge post; also cited in Field Guide).
- **Critiques raise agreement 15–20%** when used as few-shot examples (Field Guide, 2025-03-24).
- **Report TPR and TNR separately**; raw agreement is misleading under class imbalance (Husain, repeatedly).
- **Cohen's κ** for inter-annotator agreement — it corrects for chance. Eugene Yan's product-evals post targets **0.4–0.6 as substantial, >0.7 as excellent**, and notes **human–human κ is often only 0.2–0.3**, so that's the realistic ceiling. (Note: his LLM-evaluators survey uses the conventional scale where 0.41–0.60 is only "moderate" — the two posts are not perfectly consistent.)
- **F1 0.571 → 0.722 on dev, 0.727 on test** — AlignEval's demonstrated optimization gain (Eugene Yan, 2024-10).
- Judge biases with numbers (Eugene Yan, *Evaluating the Effectiveness of LLM-Evaluators*, 2024-08): position bias 50% (gpt-3.5) to 70% (claude-v1); **verbosity bias >90%** preference for longer answers in both; self-enhancement +25% (claude-v1) and +10% (gpt-4); gpt-4 hit **85% agreement with human experts on MT-Bench**; gpt-3.5-turbo managed only **30–60% recall** on factual inconsistency; gpt-4 κ=0.84 and llama-3-70b κ=0.79 on reference-based QA vs **human–human κ=0.97**.
- Task-specific numbers (Eugene Yan, 2024-03): NLI factual-consistency ROC-AUC **0.56 → 0.85** after fine-tuning; realistic inconsistency/irrelevance rates **5–10% even with RAG**, and **below 2% is prohibitively hard**; ChatGPT/Claude distinguish hallucinations at only **53.8–58.5%**; grammatical errors from a decent LLM ~**1 in 10k**; toxicity threshold Perspective API **≥0.5**, with adversarial prompts eliciting toxic output in **at least 10%** of cases; Voiceflow's harness caught a **10% drop** on a GPT version upgrade.
- **pass@k vs pass^k** (Anthropic, 2026-01): 75% per-trial success gives **pass^3 ≈ 42%**; the two metrics are identical at k=1 and diverge hard by k=10. A frontier model at **0% pass@100** means the task is broken, not the agent.
- **SPADE** (Shankar et al., arXiv 2401.03038, 2024-01): auto-generated assertions from prompt version history cut assertion count **14%** and false failures **21%** vs baselines; deployed across **2,000+ pipelines** via LangSmith; evaluated on 9 real pipelines.
- Husain's contrarian metric: a **70% pass rate may be a more meaningful eval than 100%**. Optimize for catching real issues, not for green dashboards.
- Anthropic docs' multi-criteria example: F1 ≥ 0.85, 99.5% non-toxic, 95% of responses <200ms, <0.1% of 10,000 trials flagged for toxicity.

---

## 3. Points of disagreement

**a) Eval-driven development: yes or no.**
Husain says **generally no** — LLM failure surface is effectively infinite, you can't anticipate what breaks, so write evaluators for *discovered* errors, not imagined ones. His only exception is a hard constraint with an unambiguous success criterion (e.g. "never mention competitors"). OpenAI's best-practices guide says the opposite outright, telling teams to evaluate early and often as a discipline, and its *Eval Driven System Design* cookbook is explicitly eval-centric rather than data-first: build a V0 system, generate draft outputs, have experts correct small batches, iterate on eval signal. Anthropic's agent guide sits nearer OpenAI — start before production, because waiting means reverse-engineering success criteria from a live system. **The reconciliation:** OpenAI and Anthropic mean "instrument and measure early"; Husain means "don't author rubrics before you've read outputs". They genuinely differ on whether the rubric precedes the data.

**b) Binary vs Likert.**
Husain, Shankar, and Yan are emphatic on **binary pass/fail**: the 3-vs-4 boundary is subjective and inconsistent across annotators, annotators centre-bias, and detecting differences on a scale needs bigger samples. Yan adds that binary outputs give you recall/precision, which map to production behavior, whereas correlation metrics do not. Anthropic's own docs, however, present a **1–5 Likert grader as a first-class pattern**, and its agent guide uses rubric scoring with **partial credit** across multi-component tasks. Husain's concession is to decompose instead of scaling: "4 of 5 expected facts present" as a set of binary checks.

**c) Sample size philosophy.**
Anthropic's docs say **prioritize volume over quality** — more questions with slightly noisier automated grading beats fewer hand-graded ones — and list 1,000+ test cases for exact-match evals. Husain says the opposite: **100 well-chosen traces beat thousands labeled superficially**, and careful investigation of a few failures beats rushing many. Anthropic's *agent* guide actually agrees with Husain (20–50 well-designed tasks; a good task beats hundreds of ambiguous ones), so Anthropic is internally split between its docs and its engineering blog.

**d) Same model as judge.**
Anthropic docs: generally best practice to use a **different** model for evaluation than generation. Husain: usually fine — the judge is doing a different task, and what actually matters is measured alignment (TPR/TNR) on a held-out labeled set; if alignment is poor, swap models then. Yan documents the self-enhancement bias (+10 to +25% self-preference) that motivates Anthropic's rule.

**e) Off-the-shelf / generic metrics.**
Husain: generic evaluations waste time and manufacture false confidence — useful only as a *sampling* signal to surface traces for review. Anthropic's docs include ROUGE-L and cosine-similarity graders. Yan splits the difference by task: he rejects ROUGE/METEOR/BERTScore/G-Eval for summarization, but endorses chrF, BLEURT, COMET and COMETKiwi for translation (BLEU is "bottom of the leaderboard" at WMT22/23) and fine-tuned NLI models for factual consistency. So "generic metrics are useless" is really "generic metrics are useless for *product* quality in open-ended domains".

**f) Who annotates, and how many.**
Husain: one **benevolent dictator** domain expert for most small/medium companies; multiple annotators only for genuinely multi-domain orgs, and then measure Cohen's κ and run alignment sessions. OpenAI recommends multiple review rounds with **consensus voting** aggregation and lists purchased/crowdsourced eval data as legitimate; Anthropic lists crowdsourcing as a human grader method. Husain calls outsourcing annotation "usually a big mistake" because it breaks the observe→understand→fix loop, allowing it only for mechanical tasks after an internal rubric exists.

**g) Automated prompt/assertion generation.**
Husain is skeptical of automated prompt optimization — writing prompts is how you externalize requirements, and optimizers hill-climb a fixed metric so they structurally cannot discover new failure modes; he permits it only for the "final mile" once good evals exist. Shankar's own **SPADE** does exactly the thing he's wary of — synthesizing assertions automatically from prompt version history — and reports real gains. Her **EvalGen** is a middle position: generate candidate Python functions and grader prompts, then use human grades on a subset to *select* among them.

**h) Trajectory vs outcome.**
Anthropic: grade what the agent produced, not the path, to avoid punishing valid unanticipated approaches. Husain's error-analysis method is fundamentally trajectory-based — find the **first upstream failure**. Not strictly contradictory (diagnosis vs scoring) but they pull tooling in different directions: one wants full-trace rendering, the other wants final-state assertions.

**i) Is rigorous eval for everyone?**
Shankar (*In Defense of AI Evals, for Everyone*, 2025-09-05) concedes ground to the anti-eval camp: teams in well-represented domains (coding agents) can lean on upstream model providers, and experienced teams with deep domain intuition can substitute continuous dogfooding plus error analysis for formal metrics. Her line is that newcomers and teams without data backgrounds cannot. Husain's framing in *"It's Hard to Eval" Is a Product Smell* (2026-06-29) goes further sideways: if verification is hard, that's a **product design defect** — surface intermediate artifacts, anchor outputs to vetted references with diffs, make outputs reviewable in small units — fix the product rather than building a heroic eval.

---

## 4. Mechanizable in Python vs irreducibly human

### Mechanizable — a tool should own these outright

**Trace plumbing.** Capture, normalize, version and index traces; link every eval run to a prompt version, dataset version and code commit. Git-backed prompt storage, not a SaaS indirection layer.

**Sampling.** Random, stratified, embedding-clustered (k-means/HDBSCAN over trace embeddings), outlier-ranked on computable signals (latency, token count, turn count, retry count, tool-error count), classifier-flagged, feedback-filtered, and semantic search. The tool should *enforce* that every review batch contains a random slice, since every targeted method is biased toward known problems.

**Annotation UI generation.** Progressive disclosure, domain-specific renderers, hotkeys, progress counters, filters, next/prev navigation, one-click note capture, resumable sessions. This is the highest-ROI thing to build and it is pure engineering — it holds no judgment.

**Saturation instrumentation.** Plot new-unique-codes per trace reviewed. Theoretical saturation is the stated stopping rule in every source but nobody gives a number; a tool can turn it into a visible curve and a suggested stop.

**Taxonomy bookkeeping.** Counts, frequency × severity pivot tables, ranked failure modes, coverage of taxonomy over the trace pool, drift in category proportions over time.

**Synthetic generation scaffolding.** Dimension registry (features × scenarios × personas), cross-product enumeration, sampling with coverage constraints, tuple → natural-language conversion, near-duplicate filtering, execution of generated inputs through the real system to capture traces, and a check that each generated case actually triggered its intended scenario.

**Dataset management.** Stratified train/dev/test splitting with class-balance constraints (10–20 / 40–45 / 40–45; ≥30 Pass and ≥30 Fail per split), frozen test manifests with hashes, leakage detection between splits, and a hard refusal to let a test row be touched by the optimization loop.

**Judge harness.** Prompt templating, few-shot injection drawn from labeled critiques in the train split only, concurrency, caching, retry, cost accounting.

**All measurement.** Confusion matrix, TPR, TNR, precision, recall, F1, Cohen's κ, bootstrap confidence intervals, per-slice breakdown by feature/scenario/persona, dev→test generalization gap. The tool should refuse to print raw agreement without TPR/TNR beside it.

**Bias and robustness checks.** Position-swap tests for pairwise judges, correlation of verdict with output length (verbosity bias), self-preference check by swapping the judge model, prompt-paraphrase sensitivity, and **context ablation** — drop one context element at a time and recompute alignment.

**Bounded judge-prompt optimization.** Search over judge prompt variants on dev only, with a trial budget, always reporting the dev→test gap. AlignEval proves this is tractable and also shows the honest failure mode (dev 0.722 vs test 0.727 is fine; a large gap means overfitting).

**Non-determinism handling.** k runs per task, pass@k and pass^k, variance reporting, environment isolation so flakiness isn't mistaken for regression.

**CI gating.** Fixed-set runner, per-metric thresholds, paired bootstrap CI on delta versus baseline (gate on the CI, not the point estimate), regression report naming which cases flipped, and a **saturation warning** when a suite hits 100%.

**Production monitoring.** Async sampled judging, rate estimates with confidence intervals and lower-bound alerting, input-distribution drift detection (embedding shift, new cluster emergence), auto-promotion of newly-flagged production traces into the review queue.

**Guardrails.** Inline deterministic validators — regex, schema, blocklist, PII, JSON/SQL well-formedness — with trigger logging and false-positive tracking. Architecturally separate from evaluators: guardrails block, evaluators measure.

### Irreducibly human — a tool must not automate these

**Defining what "good" means.** Criteria drift is the load-bearing finding: Shankar et al. establish that users need criteria to grade outputs, but grading outputs is how users define criteria. Criteria cannot be fully specified before observing outputs, so no tool can pre-populate them.

**Initial open coding.** Husain is explicit: do not outsource this to an LLM. It is where you discover failures you didn't know existed, build intuition about your data, and feel user pain. He sets a floor of **30 traces annotated by you personally** before any agent suggestions enter the loop.

**Validating the failure taxonomy.** LLMs mis-group. The proposal can be machine-generated; the acceptance cannot.

**Ground-truth labels.** Every label in the judge's dev and test sets must be hand-validated. A judge is aligned *to* something; if that something is machine-generated the whole construct is circular.

**Writing critiques.** The critique is the transmission medium for tacit domain knowledge and the raw material for few-shot examples. Terse or auto-generated critiques defeat the purpose.

**Adjudicating disagreement.** When two annotators differ, the resolution is a rubric edit — identifying which clause was ambiguous and adding a definition, rule or example, then relabeling. The tool computes κ and surfaces the disagreeing rows; a human writes the rule.

**Root-cause analysis and choosing the fix.** Only human review catches edge-case patterns and decides whether the answer is a prompt line, a retrieval change, decomposition, or a product redesign.

**Cost-benefit on each evaluator.** Deciding that a failure mode does *not* deserve a judge — the discipline Husain says most teams skip and where the ROI actually lives.

**Choosing the dimensions.** The features × scenarios × personas table is a domain artifact. An LLM can expand it; it cannot originate the right axes.

**Retirement decisions.** Knowing when a gold set is stale, when an always-passing eval should be dropped, and when the product itself should change so outputs become verifiable.

### The grey zone — LLM-assisted, human sign-off required
First-pass axial coding over 30–50 open-coded traces; mapping new annotations onto an existing taxonomy; drafting prompt improvements from open-coding notes; clustering and surfacing candidate traces; proposing judge prompt revisions; suggesting taxonomy merges. Husain's framing is the right design principle: **use LLMs to scale what you learned, not to avoid looking at data.**

### Design implications for a Python tool
1. Be a **state store plus a measurement engine plus an interface generator** — not a judgment engine.
2. Make the human steps *fast*, not optional. The bottleneck is reviewer throughput, so invest in rendering and hotkeys, not in auto-labeling.
3. **Refuse unsafe operations by construction**: no auto-labeling into dev/test, no test-set access during optimization, no raw-agreement-only reporting, no judge promoted without a test-set alignment report.
4. Emit **artifacts, not dashboards**: trace store → open codes → taxonomy → labeled set → splits → judge → alignment report → CI config. Each step's output is the next step's input, versioned.
5. Gate on **confidence intervals**, not point estimates, everywhere a decision is made.
6. Surface **saturation** and **staleness** as first-class signals, so the tool tells you when to stop reviewing and when an eval has stopped earning its maintenance cost.

---

## 5. Sources

**Hamel Husain**
- *AI Evals: Everything You Need to Know* (FAQ) — https://hamel.dev/blog/posts/evals-faq/ — pub. 2025-05-28, modified 2026-09-13 *(primary source)*
- *Creating a LLM-as-a-Judge That Drives Business Results* — https://hamel.dev/blog/posts/llm-judge/ — 2024-10-29, modified 2026-09-01
- *A Field Guide to Rapidly Improving AI Products* — https://hamel.dev/blog/posts/field-guide/ — 2025-03-24
- *Your AI Product Needs Evals* — https://hamelhusain.substack.com/p/evals — 2024-03-29
- *"It's Hard to Eval" Is a Product Smell* — https://hamelhusain.substack.com/p/its-hard-to-eval-is-a-product-smell — 2026-06-29
- AI Evals index — https://hamel.dev/notes/llm/evals/
- with Gergely Orosz, *A pragmatic guide to LLM evals for devs* — https://newsletter.pragmaticengineer.com/p/evals — 2025-12-02 (partly paywalled)

**Shreya Shankar**
- Shankar, Zamfirescu-Pereira, Hartmann, Parameswaran, Arawjo, *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences* (EvalGen) — https://arxiv.org/abs/2404.12272 — 2024-04-18
- Shankar et al., *SPADE: Synthesizing Data Quality Assertions for Large Language Model Pipelines* — https://arxiv.org/abs/2401.03038 — 2024-01-05, rev. 2024-03-31
- *In Defense of AI Evals, for Everyone* — https://www.sh-reya.com/blog/in-defense-ai-evals/ — 2025-09-05
- *Data Flywheels for LLM Applications* — https://www.sh-reya.com/blog/ai-engineering-flywheel/ — 2024-07-01
- *Short Musings on AI Engineering and "Failed AI Projects"* — https://www.sh-reya.com/blog/ai-engineering-short/ — 2024-06-24

**Husain & Shankar jointly**
- Lenny's Newsletter interview, *Why AI evals are the hottest new skill for product builders* — https://www.lennysnewsletter.com/p/why-ai-evals-are-the-hottest-new-skill — 2025-09-25
- Three Gulfs framework summary (third-party write-up of their course) — https://www.davidokpare.com/blog/a-primer-to-evals — 2025-07-27

**Eugene Yan**
- *Task-Specific LLM Evals that Do & Don't Work* — https://eugeneyan.com/writing/evals/ — 2024-03
- *Evaluating the Effectiveness of LLM-Evaluators (aka LLM-as-Judge)* — https://eugeneyan.com/writing/llm-evaluators/ — 2024-08
- *AlignEval: Building an App to Make Evals Easy, Fun, and Automated* — https://eugeneyan.com/writing/aligneval/ — 2024-10 (code: https://github.com/eugeneyan/align-app)
- *Product Evals in Three Simple Steps* — https://eugeneyan.com/writing/product-evals/ — 2025-11

**Anthropic**
- *Demystifying evals for AI agents* — https://anthropic.com/engineering/demystifying-evals-for-ai-agents — 2026-01-09
- *Define success criteria and build evaluations* — https://platform.claude.com/docs/en/docs/test-and-evaluate/develop-tests
- *Building Effective AI Agents* — https://www.anthropic.com/engineering/building-effective-agents

**OpenAI**
- *Evaluation best practices* — https://developers.openai.com/api/docs/guides/evaluation-best-practices
- *Eval Driven System Design — From Prototype to Production* (receipt inspection cookbook) — https://developers.openai.com/cookbook/examples/partners/eval_driven_system_design/receipt_inspection
- *Building resilient prompts using an evaluation flywheel* — https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel
- Note: OpenAI's hosted Evals platform goes read-only 2026-10-31 and shuts down 2026-11-30.
