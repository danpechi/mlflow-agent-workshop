# PEMEX Knowledge Assistant — Tracing & Evaluation Workshop

A 3-hour hands-on workshop demonstrating how to build, trace, evaluate, and optimize
a Databricks Knowledge Assistant for PEMEX operational documentation.

## Workshop Overview

Participants build a Knowledge Assistant (KA) that answers employee questions from
PEMEX internal documents: safety procedures, environmental compliance guidelines,
emergency response protocols, contractor requirements, and operational standards.

The workshop walks through the full iteration loop:
**create KA → trace → evaluate → optimize instructions → compare → redeploy**

**Target audience:** Teams evaluating Databricks Agent Bricks (Knowledge Assistants)
who want to use MLflow for systematic quality evaluation and instruction optimization.

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to a Foundation Model endpoint (default: `databricks-claude-sonnet-4-5`)
- Catalog with CREATE SCHEMA permissions (default: `pemex_lab`)
- ~3 hours

## Notebook Guide

### Setup

| Notebook | Purpose | Run Time |
|----------|---------|----------|
| `00_config` | Single source of truth for all workshop parameters (catalog, schema, volume, endpoints). Edit widget defaults here — never hardcode names in other notebooks. | < 1 min |
| `01_setup_data` | Generates seed data: 5 synthetic PEMEX documents uploaded to UC Volume, 15 hand-crafted Q&A pairs, and 15 LLM-generated evaluation examples. Writes to Unity Catalog tables. | ~3 min |

### Core Workshop

| Notebook | Purpose | Run Time |
|----------|---------|----------|
| `02a_setup_and_agent` | Create the Knowledge Assistant via the Databricks UI (guided walkthrough), register V1 instructions in the MLflow Prompt Registry, and verify the KA endpoint is live. | ~10 min |
| `02b_tracing_deep_dive` | **MLflow Tracing for KA Observability.** Three objectives: (1) Query the KA and explore auto-generated traces in the MLflow UI (2) Programmatic trace search with `search_traces()` — find slow or poor-quality responses (3) Production observability: snapshot traces to a Delta table, run SQL analytics | ~35 min |
| `02c_evaluate_v1` | Run the full evaluation suite (30 Q&A pairs × 4 scorers) against the V1 baseline instructions. Establishes the baseline showing V1's weaknesses: vague answers, missing citations, incomplete coverage. | ~10 min |
| `02d_optimize_prompt` | Use MLflow GEPA (`optimize_prompts()`) to automatically generate improved KA instructions from V1 evaluation results. Registers the optimized instructions in the Prompt Registry with an `optimized` alias. | ~20 min |
| `02e_evaluate_and_compare` | Re-run the same evaluation suite with the GEPA-optimized instructions. Prints a side-by-side V1 vs Optimized comparison table showing metric improvements. | ~15 min |
| `02f_redeploy_app` | Apply the optimized instructions to the deployed Knowledge Assistant endpoint. | ~5 min |

## Key Concepts Demonstrated

### Tracing & Observability
- **Auto-generated KA traces**: every query → retrieval → LLM response is captured automatically
- **Programmatic search**: `search_traces()` with filters to find degraded responses at scale
- **Span analysis**: `MlflowClient().get_trace()` to inspect retrieval quality and LLM reasoning
- **Traces as Delta**: snapshot to Unity Catalog, run SQL latency/error queries

### Evaluation
- **Answer quality scorer**: custom `@mlflow.genai.scorer` checking factual correctness
- **LLM-as-judge scorers**: `Safety()`, `Guidelines(groundedness)`, `Guidelines(completeness)`
- **Batch evaluation**: `mlflow.genai.evaluate()` across the full Q&A dataset

### Instruction Optimization
- **GEPA**: `mlflow.genai.optimize_prompts()` for automatic instruction improvement
- **Prompt Registry**: version KA instructions with `register_prompt()` and aliases (`v1`, `optimized`)
- **A/B comparison**: side-by-side metric tables from MLflow experiment runs

## Configuration

All configurable values live in `00_config`. Edit widget defaults or set environment
variables for bundle deploys:

| Widget | Env Var | Default |
|--------|---------|--------|
| `catalog` | `WORKSHOP_CATALOG` | `pemex_lab` |
| `schema` | `WORKSHOP_SCHEMA` | `default` |
| `volume` | `WORKSHOP_VOLUME` | `{user}_pemex_docs` |
| `llm_endpoint` | `LLM_ENDPOINT_NAME` | `databricks-claude-sonnet-4-5` |
| `ka_name` | `WORKSHOP_KA_NAME` | `{user}-pemex-ka` |
| `ka_endpoint` | `WORKSHOP_KA_ENDPOINT` | `{user}-pemex-ka` |

## Workshop Flow

```
00_config          →  Set parameters
01_setup_data      →  Upload PEMEX docs, generate Q&A, create eval dataset
02a_setup_ka       →  Create KA (UI), register V1 instructions, verify endpoint
     ↓
02b_tracing        →  Query KA, view traces, search/diagnose, Delta observability
02c_evaluate_v1    →  Baseline metrics (V1 instructions underperform)
02d_optimize       →  GEPA auto-generates improved instructions
02e_compare        →  Confirm improvement with same eval suite
02f_update_ka      →  Apply optimized instructions to the live KA
```

## Timing Guide (3 hours)

| Block | Duration | Notebooks |
|-------|----------|----------|
| Setup & intro | 15 min | `00_config`, `01_setup_data` |
| KA foundation | 20 min | `02a_setup_and_agent` |
| Tracing deep dive | 40 min | `02b_tracing_deep_dive` + UI walkthrough |
| Break | 10 min | — |
| Evaluation baseline | 20 min | `02c_evaluate_v1` |
| Instruction optimization | 25 min | `02d_optimize_prompt` |
| Compare & update | 20 min | `02e_evaluate_and_compare`, `02f_redeploy_app` |
| Q&A / wrap-up | 10 min | — |
