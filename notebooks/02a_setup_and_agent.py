# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Setup: PEMEX Knowledge Assistant
# MAGIC
# MAGIC This notebook creates the Databricks Knowledge Assistant for PEMEX via the API,
# MAGIC registers V1 instructions in the MLflow Prompt Registry, and verifies the endpoint.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Install dependencies and load config
# MAGIC 2. Create the KA via `POST /api/2.1/knowledge-assistants`
# MAGIC 3. Attach the documents volume via `POST /api/2.1/knowledge-assistants/{id}/knowledge-sources`
# MAGIC 4. Wait for `state == ACTIVE`
# MAGIC 5. Register V1 baseline instructions in MLflow Prompt Registry
# MAGIC 6. Grant the KA experiment access

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install mlflow[databricks] databricks-sdk "gepa>=0.0.26" --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import time
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
# MAGIC Uses `POST /api/2.1/knowledge-assistants`. Knowledge sources are attached separately.

# COMMAND ----------

# DBTITLE 1,Create KA
w = WorkspaceClient()

# display_name must match ^[\w.-]+$ and be 4-63 chars
ka_display_name = KA_NAME[:63]

response = w.api_client.do(
    "POST",
    "/api/2.1/knowledge-assistants",
    body={
        "display_name": ka_display_name,
        "description": "PEMEX operational Q&A assistant for safety, environmental, and operational procedures",
        "instructions": (
            "You are a helpful assistant for PEMEX employees. "
            "Answer questions about PEMEX procedures and policies."
        ),
    },
)

KA_ID = response.get("id")          # deprecated field but still returned
KA_NAME_RESOURCE = response.get("name")   # format: knowledge-assistants/{id}
KA_ENDPOINT = response.get("endpoint_name", KA_ENDPOINT)
KA_STATE = response.get("state", "CREATING")

print(f"KA created!")
print(f"  id            : {KA_ID}")
print(f"  name          : {KA_NAME_RESOURCE}")
print(f"  endpoint_name : {KA_ENDPOINT}")
print(f"  state         : {KA_STATE}")

# COMMAND ----------

# DBTITLE 1,Attach knowledge source (documents volume)
# MAGIC %md
# MAGIC ## Attach Knowledge Source
# MAGIC
# MAGIC Uses `POST /api/2.1/knowledge-assistants/{id}/knowledge-sources`

# COMMAND ----------

# DBTITLE 1,Create knowledge source
ks_response = w.api_client.do(
    "POST",
    f"/api/2.1/knowledge-assistants/{KA_ID}/knowledge-sources",
    body={
        "display_name": "PEMEX Procedure Documents",
        "description": "PEMEX operational procedure markdown files (HSSE, emergency response, environmental compliance, contractor management, operational standards)",
        "source_type": "files",
        "files": {"path": DOCS_PATH},
    },
)

KS_ID = ks_response.get("id")
print(f"Knowledge source created!")
print(f"  id    : {KS_ID}")
print(f"  state : {ks_response.get('state')}")
print(f"  path  : {DOCS_PATH}")

# COMMAND ----------

# DBTITLE 1,Section 2 — Wait for KA to be ACTIVE
# MAGIC %md
# MAGIC ## Section 2: Wait for KA Provisioning
# MAGIC
# MAGIC The KA indexes documents and provisions a serving endpoint. Typically 2–10 minutes.
# MAGIC State transitions: `CREATING` → `ACTIVE` (or `FAILED`).

# COMMAND ----------

# DBTITLE 1,Poll until ACTIVE
MAX_WAIT_SECONDS = 1800  # 30 min
POLL_INTERVAL = 30

print(f"Polling KA state (id={KA_ID})...")
start = time.time()
state = "CREATING"

while time.time() - start < MAX_WAIT_SECONDS:
    try:
        ka = w.api_client.do("GET", f"/api/2.1/knowledge-assistants/{KA_ID}")
        state = ka.get("state", "CREATING")
        endpoint_name = ka.get("endpoint_name", "")
        if endpoint_name and endpoint_name != KA_ENDPOINT:
            print(f"  NOTE: endpoint_name updated to '{endpoint_name}'")
            KA_ENDPOINT = endpoint_name
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s] state={state}  endpoint={KA_ENDPOINT or '(pending)'}")
        if state == "ACTIVE":
            print(f"\nKA is ACTIVE and ready!")
            break
        elif state == "FAILED":
            error = ka.get("error_info", "unknown error")
            raise RuntimeError(f"KA provisioning FAILED: {error}")
    except RuntimeError:
        raise
    except Exception as e:
        print(f"  Poll error (will retry): {e}")
    time.sleep(POLL_INTERVAL)
else:
    print(f"\nWARNING: KA did not reach ACTIVE within {MAX_WAIT_SECONDS // 60} min.")
    print("Re-run the test cell once the KA shows ACTIVE.")

# COMMAND ----------

# DBTITLE 1,Section 3 — Register V1 Instructions
# MAGIC %md
# MAGIC ## Section 3: Register V1 Instructions in MLflow Prompt Registry

# COMMAND ----------

# DBTITLE 1,Register V1 instructions
V1_INSTRUCTIONS = (
    "You are a helpful assistant for PEMEX employees. "
    "Answer questions about PEMEX procedures and policies."
)

v1_version = mlflow.genai.register_prompt(
    name=INSTRUCTIONS_REGISTRY_FQN,
    template=V1_INSTRUCTIONS,
    commit_message="V1: minimal baseline — no citation, scope, or format guidance",
)
mlflow.genai.set_prompt_alias(
    name=INSTRUCTIONS_REGISTRY_FQN,
    alias="v1",
    version=v1_version.version,
)

print(f"Registered '{INSTRUCTIONS_REGISTRY_FQN}@v1' (version {v1_version.version})")
print()
print("V1 weaknesses (expected failures in 02c):")
print("  - No citation requirement")
print("  - No scope constraint")
print("  - No format guidance")
print("  - No completeness requirement")

# COMMAND ----------

# DBTITLE 1,Section 4 — Test KA Endpoint
# MAGIC %md
# MAGIC ## Section 4: Test the KA Endpoint

# COMMAND ----------

# DBTITLE 1,Test KA endpoint
from mlflow.deployments import get_deploy_client

def test_ka_endpoint(endpoint_name: str, question: str) -> str:
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint_name,
        inputs={"input": [{"role": "user", "content": question}]},
    )
    # KA returns Responses API format: output[].content[].text
    for item in response.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                return part["text"]
    return str(response)


TEST_QUESTION = "What PPE is required for workers entering a refinery process unit?"

try:
    print(f"Testing endpoint: {KA_ENDPOINT}")
    answer = test_ka_endpoint(KA_ENDPOINT, TEST_QUESTION)
    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("\nKA is responding. Proceed to 02b_tracing_deep_dive.")
except Exception as e:
    print(f"Could not reach '{KA_ENDPOINT}': {e}")
    print("KA may still be indexing. Check Agents UI for state.")

# COMMAND ----------

# DBTITLE 1,Section 5 — Grant Experiment Access
# MAGIC %md
# MAGIC ## Section 5: Grant Experiment Access to KA Service Principal

# COMMAND ----------

# DBTITLE 1,Grant experiment permissions
try:
    from databricks.sdk.service.ml import (
        ExperimentAccessControlRequest,
        ExperimentPermissionLevel,
    )

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
        print(f"Granted CAN_EDIT on '{EXPERIMENT_PATH}' to '{sp_id}'.")
    else:
        print("Could not determine KA service principal — skipping.")
except Exception as e:
    print(f"Permission grant skipped (non-fatal): {e}")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Status |
# MAGIC |------|--------|
# MAGIC | Documents uploaded to UC Volume | Done in `01_setup_data` |
# MAGIC | KA created via API | Done |
# MAGIC | Knowledge source (docs volume) attached | Done |
# MAGIC | KA reached ACTIVE state | Done |
# MAGIC | V1 instructions registered | Done |
# MAGIC | KA endpoint tested | Done |
# MAGIC | Experiment permissions granted | Done |
# MAGIC
# MAGIC **Next:** Run `02b_tracing_deep_dive`.
