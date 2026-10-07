---
name: numbers-and-rules-of-thumb
description: Effort, cadence, sample sizes and agreement thresholds the practitioners quote, and who quotes which
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
  - url: https://arxiv.org/abs/2401.03038
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/evals/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/llm-evaluators/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/aligneval/
    read: 2026-09-17
  - url: https://eugeneyan.com/writing/product-evals/
    read: 2026-09-17
  - url: https://anthropic.com/engineering/demystifying-evals-for-ai-agents
    read: 2026-09-17
  - url: https://platform.claude.com/docs/en/docs/test-and-evaluate/develop-tests
    read: 2026-09-17
related: [methodology-spine]
---

# Concrete numbers and rules of thumb

## Effort and cadence
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

## Sample sizes
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

## Agreement thresholds and judge quality
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
