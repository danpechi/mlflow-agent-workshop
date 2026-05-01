# Zscaler Security Alert Triage Agent

A workshop demonstrating how to build, evaluate, and optimize a security alert triage agent using Databricks Apps, LangGraph, and MLflow.

## Architecture

```
MLflow Prompt Registry          Databricks App
┌──────────────────────┐       ┌──────────────────────────────┐
│ <prompt_name>        │       │ <app_name>                   │
│   @v1 (baseline)     │──────▶│                              │
│   @optimized (GEPA)  │       │ LangGraph Agent              │
└──────────────────────┘       │   + Claude Sonnet 4.5        │
                               │   + 4 triage tools           │
MLflow Experiment              │                              │
┌──────────────────────┐       │ /invocations (API)           │
│ Eval runs + traces   │◀──────│ / (Chat UI)                  │
│ V1 vs Optimized      │       └──────────────────────────────┘
└──────────────────────┘
```

The agent triages security alerts by calling four tools (threat intelligence, user history, asset criticality, log search) and returning a structured verdict with confidence and reasoning.

## Multi-User Scoping

All collision-prone resource names are automatically prefixed with the deploying user's `short_name` (the part of their email before `@`, with dots replaced by underscores). This lets multiple people deploy to the **same workspace** without name collisions.

| Resource | Naming Pattern | Example (`jane_doe`) |
|---|---|---|
| Job | `<short_name>-zscaler-workshop` | `jane_doe-zscaler-workshop` |
| App | `<short_name>-zscaler-triage-agent` | `jane_doe-zscaler-triage-agent` |
| Experiment | `<short_name>-zscaler-triage-agent-eval` | `jane_doe-zscaler-triage-agent-eval` |
| Volume | `<short_name>_workshop_data` | `jane_doe_workshop_data` |
| Tables | `<short_name>_sample_alerts`, `<short_name>_eval_dataset` | `jane_doe_sample_alerts` |
| Prompt | `<short_name>_triage_agent_prompt` | `jane_doe_triage_agent_prompt` |

**Shared resources** (no prefix): catalog `bricks_lab`, schema `default`.

## Project Structure

```
agent_server/
  agent.py           # LangGraph agent with @invoke/@stream handlers
  prompts.py         # Prompt loading from MLflow Prompt Registry (fallback to inline V1)
  tools.py           # Triage tools reading from bundled fixtures
  start_server.py    # FastAPI server entry point + chat UI
  utils.py           # Streaming utilities
  fixtures/
    tool_fixtures.json  # Bundled threat intel, user history, asset, log data

notebooks/
  00_config.py                  # Single source of truth: catalog, schema, volume, endpoint, app, prompt, experiment
  01_setup_data.py              # Generate alerts, fixtures, eval dataset on UC volume
  02a_setup_and_agent.py        # Agent foundation, scorers, V1 prompt registration, app deployment
  02b_tracing_deep_dive.py      # Tracing deep dive
  02c_evaluate_v1.py            # Evaluate V1 baseline
  02d_optimize_prompt.py        # GEPA prompt optimization
  02e_evaluate_and_compare.py   # Compare V1 vs optimized
  02f_redeploy_app.py           # Redeploy with optimized prompt

app.yaml              # Databricks App config (overwritten dynamically at deploy time)
databricks.yml        # Asset bundle: single workflow with 3 dependent tasks
requirements.txt      # Python dependencies
```

## Workflow

The bundle defines a single job (`workshop_pipeline`) with three sequential tasks:

```
run_config  →  setup_data  →  setup_and_deploy_agent
```

1. **run_config** — Runs `00_config.py` to establish all widget values and derived paths.
2. **setup_data** — Runs `01_setup_data.py` to create catalog/schema/volume, generate fixtures, sample alerts, and eval dataset.
3. **setup_and_deploy_agent** — Runs `02a_setup_and_agent.py` to set up the agent, register the V1 prompt, and deploy the Databricks App.

## Configuration — Single Source of Truth

All names are defined in **one** place: `notebooks/00_config.py`. It computes `_short_name` from the current user's email and uses it to build user-scoped defaults. Both other notebooks start with `%run ./00_config` and inherit those variables. The defaults match the bundle-variable defaults in `databricks.yml`.

To override values, use any of these layers:

| Layer | What to edit | When it takes effect |
|---|---|---|
| Notebook widgets | Top of `notebooks/00_config.py` | Interactive notebook runs |
| Bundle variables | `variables:` block in `databricks.yml`, or `--var key=value` on the CLI | `databricks bundle deploy / run` |
| App env vars | `env:` list in `app.yaml` (rendered dynamically by 02a/02f) | Deployed Databricks App |

| Variable | Default | Where it's used |
|---|---|---|
| `catalog` | `bricks_lab` | All notebooks, bundle, volume path |
| `schema` | `default` | All notebooks, bundle, volume path |
| `volume` | `<short_name>_workshop_data` | UC volume holding fixtures, alerts, eval dataset |
| `llm_endpoint` | `databricks-claude-sonnet-4-5` | Agent + eval-data generation + GEPA |
| `app_name` | `<short_name>-zscaler-triage-agent` | Databricks App name |
| `prompt_name` | `<short_name>_triage_agent_prompt` | Prompt registry entry |
| `experiment_name` | `<short_name>-zscaler-triage-agent-eval` | MLflow experiment subdir |
| `alerts_table` | `<short_name>_sample_alerts` | UC table for hand-crafted alerts |
| `eval_table` | `<short_name>_eval_dataset` | UC table for eval dataset |

## Quick Start

1. Deploy the bundle from your Databricks workspace:
   ```bash
   databricks bundle deploy
   databricks bundle run workshop_pipeline
   ```
2. The workflow runs all 3 tasks sequentially and deploys the app with the V1 prompt.
3. Open the app URL printed at the end of the workflow to interact with the triage agent.

## How It Works

### The Agent

The agent uses a LangGraph `create_react_agent` with four tools backed by fixture data:

| Tool | Purpose |
|---|---|
| `lookup_threat_intel` | IP/hash reputation lookup |
| `get_user_history` | User behavioral profile and anomaly score |
| `get_asset_criticality` | Host criticality and data classification |
| `search_logs` | Recent security log entries for a host |

### The Workshop Flow

1. **V1 Baseline** — Agent runs with a deliberately weak prompt and gets evaluated. Scores are low on structured output and verdict accuracy.
2. **GEPA Optimization** — `mlflow.genai.optimize_prompts()` with `GepaPromptOptimizer` automatically generates an improved prompt that produces structured JSON output with verdicts, confidence scores, and proper PII handling.
3. **Evaluation** — Same scorers re-run against the optimized prompt. Scores improve significantly.
4. **Registry** — Both prompts are versioned in MLflow Prompt Registry. The app loads its prompt by alias (`v1` or `optimized`) at startup.

### Scorers

| Scorer | What it measures |
|---|---|
| `verdict_accuracy` | Correct verdict in structured JSON (not just keywords) |
| `structured_output` | All 5 required JSON fields present |
| `Safety` | MLflow built-in safety check |
| `pii_handling` | No PII echoed in response |
| `injection_resistance` | Injection attempts explicitly flagged |

## Runtime Configuration (deployed app)

The deployed app reads everything from env vars in `app.yaml` (rendered dynamically by notebooks `02a` and `02f`):

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking (`databricks`) |
| `MLFLOW_EXPERIMENT_NAME` | Full experiment path |
| `LLM_ENDPOINT_NAME` | Databricks LLM serving endpoint |
| `PROMPT_REGISTRY_NAME` | Full prompt registry path `<catalog>.<schema>.<prompt_name>` |
| `AGENT_PROMPT_VERSION` | Prompt alias to load (`v1` or `optimized`) |
| `FIXTURES_PATH` | Path to tool fixtures JSON on UC volume |
