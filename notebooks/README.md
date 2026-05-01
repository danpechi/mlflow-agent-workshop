# Zscaler Triage Agent — Tracing & Evaluation Workshop

A 3-hour hands-on workshop demonstrating how to build, trace, evaluate, and optimize
an LLM-powered security alert triage agent on Databricks.

## Workshop Overview

Participants build a triage agent that classifies Zscaler ZIA/ZPA security alerts as
**benign**, **suspicious**, or **malicious**. The workshop walks through the full
iteration loop: instrument → evaluate → optimize → compare → deploy.

**Target audience:** Security engineering and ML teams currently using custom agent
architectures (e.g., TrueFoundry) who want to adopt Databricks MLflow for
observability and evaluation.

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to a Foundation Model endpoint (default: `databricks-claude-sonnet-4-5`)
- Catalog with CREATE SCHEMA permissions (default: `zscaler-demo`)
- ~3 hours

## Notebook Guide

### Setup

| Notebook | Purpose | Run Time |
|----------|---------|----------|
| `00_config` | Single source of truth for all workshop parameters (catalog, schema, volume, endpoints). Edit widget defaults here — never hardcode names in other notebooks. | < 1 min |
| `01_setup_data` | Generates seed data: 20 hand-crafted alerts, tool fixtures (threat intel, user history, asset criticality, log search), and 30 LLM-generated evaluation examples. Writes to Unity Catalog tables and Volume. | ~3 min |

### Core Workshop

| Notebook | Purpose | Run Time |
|----------|---------|----------|
| `02a_setup_and_agent` | Shared foundation for all workshop notebooks. Installs dependencies, loads config, builds the LangGraph triage agent, defines custom scorers (`verdict_accuracy`, `structured_output`, `pii_handling`, `injection_resistance`, `safety`), and registers the V1 baseline prompt. All other `02*` notebooks `%run` this. | ~2 min |
| `02b_tracing_deep_dive` | **MLflow Tracing for Agent Observability.** Three focused objectives: (1) Instrument custom agents with `start_span()`, typed spans, and trace-level tags (2) Automatic tracing with `autolog()` and `@mlflow.trace` decorator (3) Search, analyze, and diagnose agent failures from traces. | ~35 min |
| `02c_evaluate_v1` | Run the full evaluation suite (27 alerts × 5 scorers) against the V1 baseline prompt. Establishes the baseline metrics showing V1's weaknesses: no structured output, inconsistent verdicts, PII leakage, injection vulnerability. | ~10 min |
| `02d_optimize_prompt` | Use MLflow GEPA (`optimize_prompts()`) to automatically generate an improved prompt from V1 evaluation results. Registers the optimized prompt in the Prompt Registry with an `optimized` alias. | ~20 min |
| `02e_evaluate_and_compare` | Re-run the same evaluation suite with the GEPA-optimized prompt. Prints a side-by-side V1 vs Optimized comparison table showing metric improvements. | ~15 min |
| `02f_redeploy_app` | Deploy the optimized prompt to the Databricks App by setting `AGENT_PROMPT_VERSION=optimized`. Includes workshop summary. | ~5 min |

## Key Concepts Demonstrated

### Tracing & Observability
- **Manual tracing** for custom agents: `start_span()`, `set_inputs/outputs()`, typed spans (AGENT, TASK, TOOL, LLM)
- **Automatic instrumentation**: `mlflow.langchain.autolog()` for LangGraph/LangChain
- **Function-level tracing**: `@mlflow.trace` decorator for custom business logic
- **Trace-level tags**: `update_current_trace()` for searchable metadata
- **Programmatic search**: `search_traces()` with filters, `MlflowClient().get_trace()` for span analysis

### Evaluation
- **Custom scorers**: `verdict_accuracy` (requires JSON structure), `structured_output`
- **LLM-as-judge scorers**: `Safety()`, `Guidelines()` for PII handling and injection resistance
- **Batch evaluation**: `mlflow.genai.evaluate()` across full alert datasets

### Prompt Optimization
- **GEPA**: `mlflow.genai.optimize_prompts()` for automatic prompt improvement
- **Prompt Registry**: Version control with `register_prompt()` and alias management (`v1`, `optimized`)
- **A/B comparison**: Side-by-side metric tables from MLflow experiment runs

## Configuration

All configurable values live in `00_config`. Edit widget defaults or set environment
variables for bundle deploys:

| Widget | Env Var | Default |
|--------|---------|--------|
| `catalog` | `WORKSHOP_CATALOG` | `zscaler-demo` |
| `schema` | `WORKSHOP_SCHEMA` | `zscaler_workshop` |
| `volume` | `WORKSHOP_VOLUME` | `workshop_data` |
| `llm_endpoint` | `LLM_ENDPOINT_NAME` | `databricks-claude-sonnet-4-5` |
| `app_name` | `WORKSHOP_APP_NAME` | `zscaler-triage-agent` |

## Workshop Flow

```
00_config          →  Set parameters
01_setup_data      →  Generate alerts, fixtures, eval dataset
02a_setup_and_agent →  Build agent, define scorers, register V1
     ↓
02b_tracing        →  Instrument, trace, search, diagnose
02c_evaluate_v1    →  Baseline metrics (V1 fails on structure, PII, injection)
02d_optimize       →  GEPA auto-generates improved prompt
02e_compare        →  Confirm improvement with same eval suite
02f_redeploy       →  Push optimized prompt to production app
```

## Timing Guide (3 hours)

| Block | Duration | Notebooks |
|-------|----------|----------|
| Setup & intro | 15 min | `00_config`, `01_setup_data` |
| Agent foundation | 15 min | `02a_setup_and_agent` |
| Tracing deep dive | 45 min | `02b_tracing_deep_dive` + UI walkthrough |
| Break | 10 min | — |
| Evaluation baseline | 20 min | `02c_evaluate_v1` |
| Prompt optimization | 30 min | `02d_optimize_prompt` |
| Compare & deploy | 25 min | `02e_evaluate_and_compare`, `02f_redeploy_app` |
| Q&A / wrap-up | 20 min | — |
