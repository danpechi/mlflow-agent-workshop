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
  02_evaluate_and_optimize.py   # Eval V1 → GEPA optimize → Eval optimized → Redeploy

app.yaml              # Databricks App config (env vars consumed by agent_server/)
databricks.yml        # Asset bundle: variables + jobs for both notebooks
requirements.txt      # Python dependencies
LAB_GUIDE.md          # Step-by-step workshop instructions
```

## Configuration — single source of truth

All catalog / schema / volume / endpoint / app / prompt names are defined in **one** place: `notebooks/00_config.py`. Both other notebooks start with `%run ./00_config` and inherit those variables. The defaults in this notebook match the bundle-variable defaults in `databricks.yml` and the env-var defaults in `app.yaml`.

To retarget the workshop to your workspace, change the values in **one** of the following layers:

| Layer | What to edit | When it takes effect |
|---|---|---|
| Notebook widgets | Top of `notebooks/00_config.py` | Interactive notebook runs |
| Bundle variables | `variables:` block in `databricks.yml`, or `--var key=value` on the CLI | `databricks bundle deploy / run` |
| App env vars | `env:` list in `app.yaml` | Deployed Databricks App |
| Local `.env` | `WORKSHOP_*`, `LLM_ENDPOINT_NAME`, `PROMPT_REGISTRY_NAME`, `FIXTURES_PATH` | Local `start-app` and notebooks running outside the bundle |

| Variable | Default | Where it's used |
|---|---|---|
| `catalog` / `WORKSHOP_CATALOG` | `main` | All notebooks, bundle, fixtures volume path |
| `schema` / `WORKSHOP_SCHEMA` | `zscaler_workshop` | All notebooks, bundle, fixtures volume path |
| `volume` / `WORKSHOP_VOLUME` | `workshop_data` | UC volume holding fixtures, alerts, eval dataset |
| `llm_endpoint` / `LLM_ENDPOINT_NAME` | `databricks-claude-sonnet-4-5` | Agent + eval-data generation + GEPA reflection |
| `app_name` / `WORKSHOP_APP_NAME` | `zscaler-triage-agent` | Databricks App name |
| `prompt_name` / `WORKSHOP_PROMPT_NAME` | `triage_agent_prompt` | Last component of `<catalog>.<schema>.<prompt_name>` |
| `experiment_name` / `WORKSHOP_EXPERIMENT_NAME` | `zscaler-triage-agent-eval` | MLflow experiment subdir under `/Users/<current_user>/` |

## Quick Start

See [LAB_GUIDE.md](LAB_GUIDE.md) for full workshop instructions.

**TL;DR:**

1. Open `notebooks/00_config` and set the widget values for your workspace (or leave defaults).
2. Run `notebooks/01_setup_data` to generate evaluation data on the UC volume.
3. Run `notebooks/02_evaluate_and_optimize` to evaluate V1, run GEPA, and measure improvement.

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

The deployed app reads everything from env vars in `app.yaml`:

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking | `databricks` |
| `LLM_ENDPOINT_NAME` | Databricks LLM serving endpoint | `databricks-claude-sonnet-4-5` |
| `AGENT_PROMPT_VERSION` | Prompt alias to load from registry | `v1` |
| `PROMPT_REGISTRY_NAME` | Full prompt registry path `<catalog>.<schema>.<prompt_name>` | _(unset → inline fallback)_ |
| `FIXTURES_PATH` | Path to tool fixtures JSON | _(unset → bundled fixtures)_ |
