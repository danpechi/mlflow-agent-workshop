# Workshop Summary — Zscaler Security Alert Triage Agent

A recap of what you built, the patterns you applied, and the next experiments worth running on your own.

## What you built

A LangGraph-based security alert triage agent that classifies Zscaler ZIA/ZPA alerts as **benign**, **suspicious**, or **malicious**, deployed as a Databricks App and observed end-to-end in MLflow. You took a deliberately weak baseline prompt and used MLflow's evaluation + automated optimization stack to ship a measurably better version — without writing the new prompt by hand.

## What you covered

| Stage | Notebook | What you took away |
|-------|----------|--------------------|
| Configure | `00_config` | One source of truth — every name (catalog, schema, volume, prompt, app, experiment) parameterized and user-scoped via `_short_name`. No hardcoded names anywhere else. |
| Seed data | `01_setup_data` | Generated tool fixtures, hand-crafted alerts, and an LLM-generated eval dataset on a UC volume. |
| Agent + V1 deploy | `02a_setup_and_agent` | LangGraph ReAct agent with four fixture-backed tools, custom scorers, V1 prompt registered in the MLflow Prompt Registry, V1 app deployed. |
| Tracing & observability | `02b_tracing_deep_dive` | Four objectives — manual `start_span`, automatic `autolog()`/`@mlflow.trace`, `search_traces` + span analysis, and **traces as governed Delta tables** with SQL latency forensics, wrong-verdict JOINs, and long-pole span pinpointing. |
| Baseline eval | `02c_evaluate_v1` | Quantified V1 failure modes — structure, PII handling, injection resistance — with `mlflow.genai.evaluate`. |
| Auto-optimize | `02d_optimize_prompt` | Used GEPA (`mlflow.genai.optimize_prompts`) to generate an improved prompt from V1 trace data. |
| Compare | `02e_evaluate_and_compare` | Re-ran the same scorers against the optimized prompt and confirmed lift. |
| Ship optimized | `02f_redeploy_app` | Redeployed the same Databricks App with `AGENT_PROMPT_VERSION=optimized` — zero agent code change, alias swap only. |

## Patterns to take with you

- **Parameterize first, hardcode never.** Single config notebook + bundle variables + user-scoped names lets the same workshop run for any user in any workspace without collision.
- **Trace before you optimize.** You can't fix what you can't see. `start_span` works for any agent (not just LangGraph); `autolog()` is a bonus.
- **Treat traces as data, not telemetry.** Once they land in Delta, every SQL pattern you use for SIEM/security analytics applies to your agent. JOIN traces with alerts, eval datasets, customer outcomes — same Lakehouse, same governance.
- **Version prompts by alias, not by code.** The deployed app reads `v1` or `optimized` at startup. Promotion is a one-line registry update, not a redeploy.
- **Optimize on real failure data.** GEPA used the actual V1 trace outputs as its signal. The better your traces, the better the optimization.

## Try yourself — recommended next experiments

The workshop covered the foundation. Each of these picks up where the workshop left off and pushes one concept further. None require infrastructure beyond what you already have.

### 1. Redact sensitive data from traces — client-side PII masking

**Why this matters here:** Your agent sees real Zscaler alert content — IPs, hostnames, usernames, raw evidence strings. Today those land in spans verbatim. For production with real customer data you want PII redacted *before* the span leaves the agent process — not after it lands in storage.

**What you'll learn:** Register a span processor (`mlflow.tracing.configure(span_processors=[...])`) that mutates spans in-place using regex, type-aware logic, or Microsoft Presidio. Combine with the UC governance you already have (column masking, row filtering) for defense in depth.

**Link:** [Redacting Sensitive Data from Traces — MLflow docs](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/masking/)

### 2. Store OpenTelemetry traces directly in Unity Catalog — for agents anywhere

**Why this matters here:** Objective 4 of `02b_tracing_deep_dive` showed traces as Delta tables via `search_traces` → snapshot. That's the *pull-based* path. The *push-based* counterpart lets agents running **outside Databricks** (TrueFoundry, k8s, local) ship OTEL traces directly to UC tables in real time, via a managed serverless endpoint. Same SQL, same governance, same eval flow — no agent code changes, just env vars on the deployment.

**What you'll learn:** `mlflow.set_experiment(trace_location=UnityCatalog(...))` to bind an experiment to UC trace tables, and the OTEL exporter env-var recipe (`OTEL_EXPORTER_OTLP_ENDPOINT`, headers, `X-Databricks-UC-Table-Name`) to redirect any third-party OTEL client at your workspace.

**Link:** [Store OpenTelemetry traces in Unity Catalog — Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) *(Public Preview — enable on the Previews page first.)*

### 3. Collect human feedback on traces — close the SOC analyst loop

**Why this matters here:** The workshop scored verdicts with synthetic checks (`verdict_accuracy`, `structured_output`, etc.). In production, T1 SOC analysts know best whether a triage was right. Capturing that signal — thumbs up/down on bad triages, structured expert review for ambiguous cases — gives you a live ground-truth stream that beats any static eval set.

**What you'll learn:** `mlflow.log_feedback()` from your app for end-user signals, in-UI annotations for developer notes, and **labeling sessions** with custom schemas (`InputCategorical`, `InputText`) for structured expert review. Plus how to feed the resulting `expected_response` labels straight back into `mlflow.genai.evaluate` with a `Correctness` scorer.

**Link:** [Collect Human Feedback — Databricks 10-min demo](https://docs.databricks.com/aws/en/mlflow3/genai/getting-started/human-feedback)

### 4. Build custom judges, then align them with humans

**Why this matters here:** Combine the previous step with this one and you get a flywheel. Build an LLM-as-judge that scores triage quality the way a senior security analyst would — but instead of guessing the rubric, **align** the judge against the feedback you collected in #3. Aligned judges agree with humans 30–50% better than baseline judges, and they scale senior-analyst evaluation standards across millions of alerts where actual human review is impossible.

**What you'll learn:** `make_judge()` with a template-based instruction, then `judge.align(SIMBAAlignmentOptimizer(...), traces_with_feedback)` to iteratively refine the judge from human corrections. Register the aligned judge and use it as a scorer alongside the ones you built in `02c`/`02e`.

**Link:** [Align judges with humans — Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/align-judges)

## A natural progression

Run them in order and they compound:

```
2 (OTEL → UC) → 1 (mask PII)  →  3 (collect human feedback)  →  4 (align judges)
   route traces         protect data         get ground truth         scale evaluation
```

The first two enable trustworthy production tracing for agents anywhere. The last two convert that trace stream into a continuously-improving eval stack. That's the production-grade closed loop the workshop pointed at — these four are how you actually build it.
