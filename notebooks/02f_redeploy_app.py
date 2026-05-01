# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Redeploy App with Optimized Prompt
# MAGIC
# MAGIC Now that we've confirmed the optimized prompt scores better, we redeploy the
# MAGIC Databricks App so it loads the new prompt from the Prompt Registry alias
# MAGIC `@optimized` set in `02d_optimize_prompt`.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02a_setup_and_agent` (loaded via `%run`) — provides `APP_NAME`, `LLM_ENDPOINT`,
# MAGIC   `PROMPT_REGISTRY_FQN`, `TOOL_FIXTURES_PATH`, etc.
# MAGIC - `02d_optimize_prompt` must have been run (alias `optimized` exists).
# MAGIC - `02e_evaluate_and_compare` should have confirmed improvement.

# COMMAND ----------

# DBTITLE 1,Run shared setup
# MAGIC %run ./02a_setup_and_agent

# COMMAND ----------

# DBTITLE 1,Source layout
# MAGIC %md
# MAGIC ## How the app is deployed
# MAGIC
# MAGIC This notebook deploys the FastAPI app **end-to-end** — no separate `databricks apps`
# MAGIC CLI invocation is needed. The flow is:
# MAGIC
# MAGIC 1. Resolve the project root in `/Workspace/...` (the directory that contains
# MAGIC    `app.yaml` + `agent_server/` + `requirements.txt`).
# MAGIC 2. Re-write `app.yaml` in the workspace so that `AGENT_PROMPT_VERSION=optimized`
# MAGIC    and the other runtime env vars (`LLM_ENDPOINT_NAME`, `PROMPT_REGISTRY_NAME`,
# MAGIC    `FIXTURES_PATH`, `MLFLOW_TRACKING_URI`) point at this workshop's resources.
# MAGIC 3. Create the Databricks App (if it doesn't exist).
# MAGIC 4. Call `WorkspaceClient.apps.deploy_and_wait(...)` with the project root as the
# MAGIC    `source_code_path` and `mode=SNAPSHOT`. Databricks Apps installs
# MAGIC    `requirements.txt`, runs the `command` from `app.yaml`
# MAGIC    (`uvicorn agent_server.start_server:app`), and injects the `env` block as
# MAGIC    process environment variables.
# MAGIC 5. Poll until the app reports `RUNNING`.
# MAGIC
# MAGIC **What controls each setting at runtime**
# MAGIC
# MAGIC | Env var                | Read by                       | Purpose                                                    |
# MAGIC |------------------------|-------------------------------|------------------------------------------------------------|
# MAGIC | `LLM_ENDPOINT_NAME`    | `agent_server/agent.py`       | Databricks model serving endpoint to call.                 |
# MAGIC | `PROMPT_REGISTRY_NAME` | `agent_server/prompts.py`     | UC FQN of the registered prompt.                           |
# MAGIC | `AGENT_PROMPT_VERSION` | `agent_server/prompts.py`     | Registry alias to load (`v1` or `optimized`).              |
# MAGIC | `FIXTURES_PATH`        | `agent_server/tools.py`       | Tool fixtures JSON in the UC volume.                       |
# MAGIC | `MLFLOW_TRACKING_URI`  | `mlflow` (server-side)        | Sends traces back to this workspace.                       |

# COMMAND ----------

# DBTITLE 1,Resolve source code path
import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# This notebook lives at <project_root>/notebooks/02f_redeploy_app.
# The app source is the project root (where app.yaml + agent_server/ live).
_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_project_root = os.path.dirname(os.path.dirname(_nb_path))
SOURCE_CODE_PATH = f"/Workspace{_project_root}"
APP_YAML_WS_PATH = f"{_project_root}/app.yaml"  # workspace path (no /Workspace prefix for workspace API)

print(f"Project root (workspace path): {_project_root}")
print(f"App source (FUSE path)       : {SOURCE_CODE_PATH}")
print(f"app.yaml (workspace path)    : {APP_YAML_WS_PATH}")

# COMMAND ----------

# DBTITLE 1,Render app.yaml with optimized env
# Re-write app.yaml in the workspace so the deploy picks up the optimized alias and
# the resolved catalog/schema/volume names from 00_config. We render a minimal YAML
# by hand to avoid a PyYAML dependency on the cluster.
APP_YAML_CONTENTS = f"""command:
  - "python"
  - "-m"
  - "uvicorn"
  - "agent_server.start_server:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"

env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: MLFLOW_EXPERIMENT_NAME
    value: "{EXPERIMENT_PATH}"
  - name: LLM_ENDPOINT_NAME
    value: "{LLM_ENDPOINT}"
  - name: PROMPT_REGISTRY_NAME
    value: "{PROMPT_REGISTRY_FQN}"
  - name: AGENT_PROMPT_VERSION
    value: "optimized"
  - name: FIXTURES_PATH
    value: "{TOOL_FIXTURES_PATH}"
"""

# Try direct file IO first (works for plain workspace folders); fall back to the
# Workspace API for read-only FUSE paths (e.g. when the project is in a Git folder).
import io
from databricks.sdk.service.workspace import ImportFormat

_local_path = f"{SOURCE_CODE_PATH}/app.yaml"
try:
    with open(_local_path, "w") as f:
        f.write(APP_YAML_CONTENTS)
    print(f"Wrote app.yaml via FUSE: {_local_path}")
except (PermissionError, OSError) as exc:
    print(f"FUSE write failed ({exc}); falling back to Workspace API…")
    w.workspace.upload(
        path=APP_YAML_WS_PATH,
        content=io.BytesIO(APP_YAML_CONTENTS.encode("utf-8")),
        format=ImportFormat.AUTO,
        overwrite=True,
    )
    print(f"Wrote app.yaml via Workspace API: {APP_YAML_WS_PATH}")

print("Contents:")
print(APP_YAML_CONTENTS)

# COMMAND ----------

# DBTITLE 1,Create app if it does not exist
def _state_str(obj):
    """Pull a human-readable state out of a possibly-None status object."""
    if obj is None or getattr(obj, "state", None) is None:
        return "unknown"
    s = obj.state
    return s.value if hasattr(s, "value") else str(s)


from databricks.sdk.service.apps import App

try:
    app = w.apps.get(APP_NAME)
    print(f"App '{APP_NAME}' exists. compute={_state_str(app.compute_status)} app={_state_str(app.app_status)}")
except Exception as e:
    msg = str(e).lower()
    if any(tok in msg for tok in ("does not exist", "not found", "resource_does_not_exist", "not_found")):
        print(f"Creating app '{APP_NAME}'...")
        app = w.apps.create_and_wait(
            app=App(
                name=APP_NAME,
                description="Zscaler Security Alert Triage Agent — Workshop Demo",
            )
        )
        print(f"App '{APP_NAME}' created.")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Grant App SP write access to the workshop experiment
# The deployed App runs as a service principal. To write traces to the workshop
# experiment (instead of an auto-created default), the SP needs CAN_EDIT on it.
try:
    from databricks.sdk.service.ml import (
        ExperimentAccessControlRequest,
        ExperimentPermissionLevel,
    )

    sp_application_id = app.service_principal_client_id  # OAuth client_id == SP application_id
    if sp_application_id:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
        if exp is None:
            mlflow.set_experiment(EXPERIMENT_PATH)
            exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
        w.experiments.update_permissions(
            experiment_id=exp.experiment_id,
            access_control_list=[
                ExperimentAccessControlRequest(
                    service_principal_name=sp_application_id,
                    permission_level=ExperimentPermissionLevel.CAN_EDIT,
                )
            ],
        )
        print(f"Granted CAN_EDIT on experiment {EXPERIMENT_PATH} to App SP {sp_application_id}.")
    else:
        print("App has no service_principal_client_id yet; skipping ACL grant.")
except Exception as e:
    print(f"Could not grant experiment ACL (non-fatal — traces may land in app default): {e}")

# COMMAND ----------

# DBTITLE 1,Deploy
from databricks.sdk.service.apps import AppDeployment, AppDeploymentMode

deployment = w.apps.deploy_and_wait(
    app_name=APP_NAME,
    app_deployment=AppDeployment(
        source_code_path=SOURCE_CODE_PATH,
        mode=AppDeploymentMode.SNAPSHOT,
    ),
)

state = deployment.status.state.value if hasattr(deployment.status.state, "value") else str(deployment.status.state)
print(f"\nDeployment finished: {deployment.deployment_id}  state={state}")
print(f"  AGENT_PROMPT_VERSION = optimized")
print(f"  Source: {SOURCE_CODE_PATH}")

# COMMAND ----------

# DBTITLE 1,Wait for app to become RUNNING and print URL
import time

print("Waiting for app to reach RUNNING state...")
for i in range(30):
    app = w.apps.get(APP_NAME)
    compute_state = _state_str(app.compute_status)
    app_state = _state_str(app.app_status)
    print(f"  [{i + 1:>2}/30] compute={compute_state}  app={app_state}")
    if compute_state.upper() in ("ACTIVE", "RUNNING"):
        url = getattr(app, "url", None) or f"https://{w.config.host}/apps/{APP_NAME}"
        print(f"\nApp is running with the optimized prompt.")
        print(f"  Open: {url}")
        break
    time.sleep(10)
else:
    print("\nApp did not reach RUNNING within 5 minutes. Inspect the Databricks Apps UI for logs.")

# COMMAND ----------

# DBTITLE 1,Workshop summary
# MAGIC %md
# MAGIC ## Workshop Summary
# MAGIC
# MAGIC In this workshop you learned:
# MAGIC
# MAGIC | Notebook | Topic | Key Takeaways |
# MAGIC |----------|-------|---------------|
# MAGIC | `02a` | **Setup & Agent** | Build LangGraph agent, define tools and scorers |
# MAGIC | `02b` | **Tracing** | Instrument agents with `start_span()`, autolog, and `@mlflow.trace`; search and diagnose failures |
# MAGIC | `02c` | **Evaluate V1** | Run baseline eval, identify systematic failures |
# MAGIC | `02d` | **Optimize Prompt** | Use GEPA to auto-generate an improved prompt and alias it as `optimized` |
# MAGIC | `02e` | **Evaluate & Compare** | Verify improvements, compare V1 vs optimized |
# MAGIC | `02f` | **Redeploy** | Push the `optimized` alias into the FastAPI app via `apps.deploy_and_wait` |
# MAGIC
# MAGIC **Key capabilities demonstrated:**
# MAGIC
# MAGIC 1. **Tracing & Observability** — manual `start_span`, `autolog`, `@mlflow.trace`, trace tags, programmatic search.
# MAGIC 2. **Evaluation & Iteration** — custom scorers (`verdict_accuracy`, `structured_output`) + LLM-as-judge (`Safety`, `Guidelines`).
# MAGIC 3. **Prompt Optimization** — GEPA, Prompt Registry versioning, A/B switching via aliases.
# MAGIC
# MAGIC **Next steps:**
# MAGIC - Open the MLflow Experiment UI to explore traces and per-example results.
# MAGIC - Try pasting adversarial alerts into the App chat UI to verify injection resistance.
# MAGIC - Add custom scorers for your specific security use cases.