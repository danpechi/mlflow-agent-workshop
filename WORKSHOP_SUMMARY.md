# Workshop Summary — PEMEX Knowledge Assistant

A recap of what you built, the patterns you applied, and the next experiments worth running.

## What you built

A Databricks Knowledge Assistant that answers PEMEX employee questions from five internal
procedure documents — safety, environmental compliance, emergency response, contractor
management, and operational standards. You took deliberately minimal V1 instructions and
used MLflow's evaluation and automated optimization stack to produce measurably better
answers — without writing the improved instructions by hand.

## What you covered

| Stage | Notebook | What you took away |
|-------|----------|--------------------|
| Configure | `00_config` | One source of truth — every name (catalog, schema, volume, KA, instructions, experiment) parameterized and user-scoped via `_short_name`. No hardcoded names anywhere else. |
| Seed data | `01_setup_data` | Uploaded 5 synthetic PEMEX procedure docs to a UC Volume; generated 15 hand-crafted and 15 LLM-generated Q&A eval pairs. |
| KA + V1 setup | `02a_setup_and_agent` | Created Knowledge Assistant via Databricks UI, registered V1 instructions in the MLflow Prompt Registry, verified the KA endpoint. |
| Tracing & observability | `02b_tracing_deep_dive` | Three objectives — KA auto-traces explored in the UI, `search_traces` + span drill-down for latency/error diagnosis, and **traces as governed Delta tables** with SQL latency analytics and JOIN against the eval dataset. |
| Baseline eval | `02c_evaluate_v1` | Quantified V1 failure modes — missing citations, incomplete answers, vague responses — with `mlflow.genai.evaluate`. |
| Auto-optimize | `02d_optimize_prompt` | Used GEPA (`mlflow.genai.optimize_prompts`) to generate improved instructions from V1 failure patterns. |
| Compare | `02e_evaluate_and_compare` | Re-ran the same scorers against the optimized instructions and confirmed lift across all four metrics. |
| Apply optimized | `02f_redeploy_app` | Updated the live KA instructions via SDK and spot-checked improved response quality. |

## Patterns to take with you

- **Parameterize first, hardcode never.** Single config notebook + user-scoped names lets the same workshop run for any PEMEX user in any workspace without collision.
- **Zero-code tracing for Knowledge Assistants.** KA traces are generated automatically — you spend your time *reading* them, not writing instrumentation code.
- **Treat traces as data, not telemetry.** Once traces land in Delta, every SQL pattern you use for analytics applies: JOIN traces with eval datasets, Q&A tables, or user feedback. Same Lakehouse, same governance.
- **Version instructions by alias, not by code.** The KA reads `v1` or `optimized` from the Prompt Registry. Promotion is a one-line registry update.
- **Optimize on real failure data.** GEPA uses actual V1 outputs as its signal. The better your scorers and eval dataset, the better the optimization.

## Try yourself — recommended next experiments

### 1. Collect human feedback to close the quality loop

**Why it matters for PEMEX:** The workshop scored answers automatically. In production, PEMEX subject-matter experts know best whether an answer is accurate for their domain. Capturing thumbs-up / thumbs-down signals from real users gives you a live ground-truth stream that beats any static eval set.

**What you'll learn:** `mlflow.log_feedback()` for end-user signals, in-UI annotations for expert notes, and **labeling sessions** with custom schemas for structured domain-expert review. Feed the resulting labels back into `mlflow.genai.evaluate` with a `Correctness` scorer.

**Link:** [Collect Human Feedback — Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/human-feedback)

### 2. Build custom judges, then align them with domain experts

**Why it matters for PEMEX:** Build an LLM-as-judge that scores answer quality the way a PEMEX safety engineer would — not by guessing the rubric, but by **aligning** the judge against the feedback collected in #1. Aligned judges agree with domain experts 30–50% better than baseline judges, and they scale expert evaluation standards to thousands of questions.

**What you'll learn:** `make_judge()` with a custom instruction template, then `judge.align(SIMBAAlignmentOptimizer(...), traces_with_feedback)` to refine the judge from expert corrections.

**Link:** [Align judges with humans — Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/align-judges)

### 3. Store OpenTelemetry traces in Unity Catalog for agents running outside Databricks

**Why it matters for PEMEX:** If PEMEX runs other AI systems outside Databricks (on-premises or another cloud), those agents can still ship OTEL traces directly to UC tables via a managed serverless endpoint — same SQL, same governance, same eval flow as your KA traces.

**What you'll learn:** `mlflow.set_experiment(trace_location=UnityCatalog(...))` and the OTEL exporter env-var recipe to redirect any third-party OTEL client at your workspace.

**Link:** [Store OpenTelemetry traces in Unity Catalog — Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)

### 4. Expand your Knowledge Assistant with Genie for structured data queries

**Why it matters for PEMEX:** Your current KA answers questions from procedure documents. Add a **Genie Space** over structured PEMEX data (maintenance records, incident logs, inspection schedules) and build a Supervisor Agent (MAS) that routes questions to either the KA or Genie depending on whether the answer lives in documents or tables.

**What you'll learn:** Creating a Genie Space via the Databricks UI, using `manage_mas` to compose a multi-agent supervisor, and routing logic between document and structured data sources.

**Link:** [Agent Bricks — Databricks docs](https://docs.databricks.com/aws/en/generative-ai/agent-bricks/)

## A natural progression

```
1 (human feedback) → 2 (align judges) → 3 (OTEL for other systems) → 4 (add Genie)
   ground truth          scale evaluation      full observability         richer KA
```

Steps 1 and 2 improve quality measurement. Step 3 extends observability to the rest of PEMEX's AI stack. Step 4 expands the KA to answer questions that currently require a database query. Together they take the workshop prototype to a production-grade knowledge system.
