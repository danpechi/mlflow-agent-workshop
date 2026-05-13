# PEMEX Knowledge Assistant

A workshop demonstrating how to build, evaluate, and optimize a Databricks Knowledge Assistant
for PEMEX operational documentation using MLflow.

## Architecture

```
MLflow Prompt Registry          Databricks Knowledge Assistant
┌──────────────────────┐       ┌──────────────────────────────┐
│ <instructions_name>  │       │ <ka_name>                    │
│   @v1 (baseline)     │──────▶│                              │
│   @optimized (GEPA)  │       │ Auto-retrieval from UC docs  │
└──────────────────────┘       │   + Claude Sonnet 4.5        │
                               │   + 5 PEMEX procedure docs   │
MLflow Experiment              │                              │
┌──────────────────────┐       │ /invocations (API)           │
│ Eval runs + traces   │◀──────│ Chat UI (Agents UI)          │
│ V1 vs Optimized      │       └──────────────────────────────┘
└──────────────────────┘
```

The KA answers employee questions from PEMEX internal documents — safety procedures,
environmental compliance guidelines, emergency response protocols, contractor requirements,
and operational standards.

## Multi-User Scoping

All collision-prone resource names are automatically prefixed with the deploying user's
`short_name` (the part of their email before `@`, dots replaced by underscores). Multiple
participants can deploy to the **same workspace** without name collisions.

| Resource | Naming Pattern | Example (`jane_doe`) |
|---|---|---|
| KA name | `<short_name>-pemex-ka` | `jane_doe-pemex-ka` |
| KA endpoint | `<short_name>-pemex-ka` | `jane_doe-pemex-ka` |
| Experiment | `<short_name>-pemex-ka-eval` | `jane_doe-pemex-ka-eval` |
| Volume | `<short_name>_pemex_docs` | `jane_doe_pemex_docs` |
| Tables | `<short_name>_sample_qa`, `<short_name>_eval_dataset` | `jane_doe_sample_qa` |
| Instructions | `<short_name>_ka_instructions` | `jane_doe_ka_instructions` |

**Shared resources** (no prefix): catalog `pemex_lab`, schema `default`.

## Project Structure

```
notebooks/
  00_config.py                  # Single source of truth: catalog, schema, volume, KA name, experiment
  01_setup_data.py              # Upload PEMEX documents, generate Q&A and eval dataset
  02a_setup_and_agent.py        # Create KA (UI), register V1 instructions, verify endpoint
  02b_tracing_deep_dive.py      # KA tracing: auto-traces, programmatic search, Delta observability
  02c_evaluate_v1.py            # Evaluate V1 baseline instructions
  02d_optimize_prompt.py        # GEPA instruction optimization
  02e_evaluate_and_compare.py   # Compare V1 vs optimized
  02f_redeploy_app.py           # Apply optimized instructions to live KA
```

## Workshop Flow

```
00_config  →  01_setup_data  →  02a_setup_ka  →  02b_tracing  →  02c_eval_v1
                                                                       ↓
                                              02f_update_ka  ←  02e_compare  ←  02d_optimize
```

## Configuration — Single Source of Truth

All names are defined in **one** place: `notebooks/00_config.py`. It computes `_short_name`
from the current user's email and uses it to build user-scoped defaults.

| Variable | Default | Description |
|---|---|---|
| `catalog` | `pemex_lab` | Unity Catalog catalog |
| `schema` | `default` | Unity Catalog schema |
| `volume` | `<short_name>_pemex_docs` | UC volume for documents and eval data |
| `llm_endpoint` | `databricks-claude-sonnet-4-5` | LLM serving endpoint |
| `ka_name` | `<short_name>-pemex-ka` | Knowledge Assistant name |
| `ka_endpoint` | `<short_name>-pemex-ka` | KA serving endpoint name |
| `instructions_name` | `<short_name>_ka_instructions` | MLflow Prompt Registry entry |
| `experiment_name` | `<short_name>-pemex-ka-eval` | MLflow experiment subdir |
| `qa_table` | `<short_name>_sample_qa` | UC table for hand-crafted Q&A |
| `eval_table` | `<short_name>_eval_dataset` | UC table for eval dataset |

## Quick Start

1. Run `01_setup_data` to upload PEMEX documents and generate the eval dataset.
2. Run `02a_setup_and_agent` to create the KA (UI walkthrough) and register V1 instructions.
3. Run `02b` through `02f` sequentially to trace, evaluate, optimize, and redeploy.

## PEMEX Documents (Knowledge Sources)

Five synthetic procedure documents uploaded to the UC Volume:

| Document | Content |
|---|---|
| `hsse_procedures.md` | PPE requirements, work permits, incident reporting, safety induction |
| `emergency_response.md` | Spill response, fire procedures, medical emergency, evacuation |
| `environmental_compliance.md` | Waste management, air quality monitoring, water discharge standards |
| `contractor_management.md` | Pre-qualification requirements, HSSE standards, performance KPIs |
| `operational_standards.md` | LOTO, confined space entry, Management of Change, PTW system |

## Scorers

| Scorer | What it measures |
|---|---|
| `answer_quality` | Key facts from expected answer are present in the response |
| `Safety` | MLflow built-in safety check |
| `groundedness` | Response cites or references a PEMEX source document |
| `completeness` | All parts of a multi-part question are addressed |
