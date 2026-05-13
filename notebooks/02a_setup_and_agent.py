# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Setup: PEMEX Knowledge Assistant
# MAGIC
# MAGIC This notebook creates the Databricks Knowledge Assistant for PEMEX, registers
# MAGIC V1 instructions in the MLflow Prompt Registry, and verifies the endpoint is live.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Install dependencies and load config
# MAGIC 2. Walk through creating the KA in the Databricks UI
# MAGIC 3. Register V1 (baseline) KA instructions in MLflow Prompt Registry
# MAGIC 4. Grant the KA endpoint access to the MLflow experiment
# MAGIC 5. Verify the KA endpoint responds correctly

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install mlflow[databricks] databricks-sdk "gepa>=0.0.26" --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import json
import mlflow
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# DBTITLE 1,Load config
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Section 1 — Create Knowledge Assistant
# MAGIC %md
# MAGIC ## Section 1: Create the Knowledge Assistant
# MAGIC
# MAGIC Knowledge Assistants are created through the Databricks UI or the Agent Bricks API.
# MAGIC Follow the steps below, then paste your KA endpoint name into the `ka_endpoint`
# MAGIC widget in `00_config` before proceeding.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Step-by-Step: Create the KA in the UI
# MAGIC
# MAGIC 1. In the left sidebar, navigate to **Machine Learning** → **Agents**.
# MAGIC 2. Click **+ Create agent** → select **Knowledge Assistant**.
# MAGIC 3. Fill in the form:
# MAGIC    - **Name**: Use the value from your `ka_name` widget (printed in the cell below)
# MAGIC    - **Description**: "PEMEX operational Q&A assistant for safety, environmental, and operational procedures"
# MAGIC    - **Instructions**: Leave blank for now — we will register instructions via the Prompt Registry and apply the optimized version after evaluation.
# MAGIC 4. **Add knowledge sources** — click **+ Add source** → **Files in UC Volume**:
# MAGIC    - Browse to the volume path printed below and add all 5 `.md` files from the `docs/` subdirectory.
# MAGIC 5. Click **Create Agent**.
# MAGIC
# MAGIC > Creation takes up to a few hours while Databricks indexes your documents.
# MAGIC > You will receive a notification when the KA is ready.
# MAGIC
# MAGIC 6. Once created, click **Endpoint** to get the serving endpoint name.
# MAGIC    Paste it into the `ka_endpoint` widget in `00_config` (default: same as `ka_name`).

# COMMAND ----------

# DBTITLE 1,Print KA setup values
print("=" * 60)
print("  KA SETUP VALUES")
print("=" * 60)
print(f"  KA name          : {KA_NAME}")
print(f"  KA endpoint      : {KA_ENDPOINT}")
print(f"  Documents path   : {DOCS_PATH}")
print()
print("  Document files to add as knowledge sources:")
import os
for fname in sorted(os.listdir(DOCS_PATH)):
    if fname.endswith(".md"):
        fpath = os.path.join(DOCS_PATH, fname)
        size = os.path.getsize(fpath)
        print(f"    - {DOCS_PATH}/{fname}  ({size:,} bytes)")
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Section 2 — Register V1 Instructions
# MAGIC %md
# MAGIC ## Section 2: Register V1 Instructions in MLflow Prompt Registry
# MAGIC
# MAGIC The **V1 instructions** are deliberately minimal — just enough to activate the KA
# MAGIC but without proper citation guidance, scope constraints, or answer formatting rules.
# MAGIC
# MAGIC This is our baseline. Later, GEPA will generate an improved version.

# COMMAND ----------

# DBTITLE 1,Register V1 instructions
# V1: deliberately minimal — no citation guidance, no scope constraint, no structured format
V1_INSTRUCTIONS = (
    "You are a helpful assistant for PEMEX employees. "
    "Answer questions about PEMEX procedures and policies."
)

INSTRUCTIONS_REGISTRY_NAME = INSTRUCTIONS_REGISTRY_FQN

v1_version = mlflow.genai.register_prompt(
    name=INSTRUCTIONS_REGISTRY_NAME,
    template=V1_INSTRUCTIONS,
    commit_message="V1: minimal baseline instructions — no citation, scope, or format guidance",
)
mlflow.genai.set_prompt_alias(
    name=INSTRUCTIONS_REGISTRY_NAME,
    alias="v1",
    version=v1_version.version,
)

print(f"Registered V1 instructions as '{INSTRUCTIONS_REGISTRY_FQN}@v1' (version {v1_version.version})")
print()
print("V1 Instructions:")
print("-" * 40)
print(V1_INSTRUCTIONS)
print("-" * 40)
print()
print("V1 weaknesses (expected failures in 02c):")
print("  - No citation requirement: may give answers without citing source documents")
print("  - No scope constraint: may answer questions outside PEMEX documentation")
print("  - No format guidance: inconsistent response structure")
print("  - No completeness requirement: may give partial answers to multi-part questions")

# COMMAND ----------

# DBTITLE 1,Section 3 — Verify KA Endpoint
# MAGIC %md
# MAGIC ## Section 3: Verify the KA Endpoint
# MAGIC
# MAGIC Once the KA finishes indexing (check the Agents UI for status), run this cell
# MAGIC to verify it responds to a test question.
# MAGIC
# MAGIC **Prerequisite:** The `ka_endpoint` widget in `00_config` must be set to the
# MAGIC KA's serving endpoint name.

# COMMAND ----------

# DBTITLE 1,Test KA endpoint
from mlflow.deployments import get_deploy_client

def test_ka_endpoint(endpoint_name: str, question: str) -> str:
    """Query the KA endpoint and return the response text."""
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint_name,
        inputs={
            "messages": [{"role": "user", "content": question}]
        },
    )
    # Handle both OpenAI-style and custom response formats
    if "choices" in response:
        return response["choices"][0]["message"]["content"]
    elif "content" in response:
        return response["content"]
    return str(response)


TEST_QUESTION = "What PPE is required for workers entering a refinery process unit?"

try:
    print(f"Testing KA endpoint: {KA_ENDPOINT}")
    print(f"Question: {TEST_QUESTION}")
    print()
    answer = test_ka_endpoint(KA_ENDPOINT, TEST_QUESTION)
    print("KA Response:")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    print()
    print("KA endpoint is responding. Proceed to 02b_tracing_deep_dive.")
except Exception as e:
    print(f"Could not reach KA endpoint '{KA_ENDPOINT}': {e}")
    print()
    print("Possible reasons:")
    print("  1. The KA is still indexing documents (may take up to a few hours).")
    print("  2. The 'ka_endpoint' widget value doesn't match the actual endpoint name.")
    print("     Check the Agents UI → your KA → Endpoint tab for the correct name.")
    print()
    print("You can continue with 02b_tracing_deep_dive using the local LLM fallback.")

# COMMAND ----------

# DBTITLE 1,Grant experiment access to KA endpoint
# MAGIC %md
# MAGIC ## Section 4: Grant Experiment Access (Optional)
# MAGIC
# MAGIC If you want the KA's traces to land in your workshop MLflow experiment
# MAGIC (instead of an auto-created default), grant the KA's service principal
# MAGIC `CAN_EDIT` on the experiment.

# COMMAND ----------

# DBTITLE 1,Grant experiment permissions
w = WorkspaceClient()

try:
    from databricks.sdk.service.ml import (
        ExperimentAccessControlRequest,
        ExperimentPermissionLevel,
    )
    from databricks.sdk.service.serving import EndpointStateReady

    # Look up the serving endpoint to find its service principal
    endpoint = w.serving_endpoints.get(KA_ENDPOINT)
    sp_id = getattr(endpoint, "creator", None)

    exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
    if exp is None:
        mlflow.set_experiment(EXPERIMENT_PATH)
        exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

    if sp_id:
        w.experiments.update_permissions(
            experiment_id=exp.experiment_id,
            access_control_list=[
                ExperimentAccessControlRequest(
                    user_name=sp_id,
                    permission_level=ExperimentPermissionLevel.CAN_EDIT,
                )
            ],
        )
        print(f"Granted CAN_EDIT on experiment '{EXPERIMENT_PATH}' to '{sp_id}'.")
    else:
        print("Could not determine KA service principal — skipping ACL grant.")
        print("Traces will still be logged to the KA's default experiment.")

except Exception as e:
    print(f"Permission grant skipped (non-fatal): {e}")
    print("KA traces will land in the KA's auto-created experiment.")
    print("You can still search them from any notebook using:")
    print("  mlflow.search_traces(experiment_ids=[...], ...)")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Status |
# MAGIC |------|--------|
# MAGIC | Documents uploaded to UC Volume | Done in `01_setup_data` |
# MAGIC | Knowledge Assistant created (UI) | Manual — check Agents UI for status |
# MAGIC | V1 instructions registered in Prompt Registry | Done |
# MAGIC | KA endpoint verified | Done (or pending KA indexing) |
# MAGIC | Experiment permissions granted | Done |
# MAGIC
# MAGIC **Next:** Run `02b_tracing_deep_dive` to explore KA traces and learn how to
# MAGIC search and diagnose responses at scale.
# MAGIC
# MAGIC **Note on V1 instructions:**
# MAGIC The V1 instructions are minimal by design. When you evaluate in `02c`, you'll see
# MAGIC the KA give answers without citations, go off-topic, or give incomplete answers.
# MAGIC That's expected — it's the baseline we'll improve with GEPA in `02d`.
