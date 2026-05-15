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
# MAGIC 2. Create the KA via the Knowledge Assistants REST API
# MAGIC 3. Wait for the KA to finish provisioning
# MAGIC 4. Register V1 (baseline) KA instructions in MLflow Prompt Registry
# MAGIC 5. Grant the KA endpoint access to the MLflow experiment

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
# MAGIC ## Section 1: Create the Knowledge Assistant via API
# MAGIC
# MAGIC We use the Databricks Knowledge Assistants REST API to create the KA programmatically.
# MAGIC The KA will index all documents in the UC Volume and expose a chat endpoint.

# COMMAND ----------

# DBTITLE 1,Create KA via API
w = WorkspaceClient()

KA_PAYLOAD = {
    "display_name": KA_NAME,
    "description": "PEMEX operational Q&A assistant for safety, environmental, and operational procedures",
    "instructions": (
        "You are a helpful assistant for PEMEX employees. "
        "Answer questions about PEMEX procedures and policies."
    ),
    "knowledge_sources": [
        {
            "type": "VOLUME",
            "volume_path": DOCS_PATH,
        }
    ],
}

print(f"Creating Knowledge Assistant: {KA_NAME}")
print(f"  Documents path: {DOCS_PATH}")
print()

try:
    response = w.api_client.do(
        "POST",
        "/api/2.0/knowledge-assistants",
        body=KA_PAYLOAD,
    )
    KA_TILE_ID = response.get("id") or response.get("tile_id")
    print(f"KA created successfully!")
    print(f"  Tile ID       : {KA_TILE_ID}")
    print(f"  Status        : {response.get('status', response.get('endpoint_status', 'PROVISIONING'))}")
    print()
    print("The KA is now indexing documents. This may take a few minutes.")
except Exception as e:
    err = str(e)
    if "already exists" in err.lower() or "conflict" in err.lower():
        print(f"KA '{KA_NAME}' already exists — looking it up...")
        # Find existing KA by name
        all_kas = w.api_client.do("GET", "/api/2.0/knowledge-assistants")
        kas = all_kas.get("knowledge_assistants", all_kas.get("items", []))
        match = next((k for k in kas if k.get("display_name") == KA_NAME or k.get("name") == KA_NAME), None)
        if match:
            KA_TILE_ID = match.get("id") or match.get("tile_id")
            print(f"  Found existing KA. Tile ID: {KA_TILE_ID}")
        else:
            raise RuntimeError(f"KA '{KA_NAME}' exists but could not be found in list. Check the Agents UI.")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Wait for KA to be online
# MAGIC %md
# MAGIC ## Waiting for KA provisioning
# MAGIC
# MAGIC The KA needs to index all documents before it can answer questions.
# MAGIC This typically takes 2–10 minutes depending on document volume.

# COMMAND ----------

# DBTITLE 1,Poll KA status until ONLINE
MAX_WAIT_SECONDS = 1200  # 20 minutes max
POLL_INTERVAL = 30

print(f"Polling KA status (tile_id={KA_TILE_ID})...")
print(f"Max wait: {MAX_WAIT_SECONDS // 60} minutes, polling every {POLL_INTERVAL}s")
print()

start = time.time()
status = "PROVISIONING"

while time.time() - start < MAX_WAIT_SECONDS:
    try:
        ka = w.api_client.do("GET", f"/api/2.0/knowledge-assistants/{KA_TILE_ID}")
        status = ka.get("status") or ka.get("endpoint_status", "UNKNOWN")
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>4}s] Status: {status}")

        if status == "ONLINE":
            print()
            print(f"KA is ONLINE and ready!")
            break
        elif status in ("ERROR", "FAILED"):
            raise RuntimeError(f"KA provisioning failed with status: {status}. Check the Agents UI.")
    except Exception as e:
        print(f"  Poll error (will retry): {e}")

    time.sleep(POLL_INTERVAL)
else:
    print()
    print(f"WARNING: KA did not reach ONLINE status within {MAX_WAIT_SECONDS // 60} minutes.")
    print("You can continue — the KA may still be indexing.")
    print("Re-run the 'Test KA endpoint' cell once the Agents UI shows ONLINE.")

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
V1_INSTRUCTIONS = (
    "You are a helpful assistant for PEMEX employees. "
    "Answer questions about PEMEX procedures and policies."
)

v1_version = mlflow.genai.register_prompt(
    name=INSTRUCTIONS_REGISTRY_FQN,
    template=V1_INSTRUCTIONS,
    commit_message="V1: minimal baseline instructions — no citation, scope, or format guidance",
)
mlflow.genai.set_prompt_alias(
    name=INSTRUCTIONS_REGISTRY_FQN,
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

# DBTITLE 1,Section 3 — Test KA Endpoint
# MAGIC %md
# MAGIC ## Section 3: Test the KA Endpoint

# COMMAND ----------

# DBTITLE 1,Test KA endpoint
from mlflow.deployments import get_deploy_client

def test_ka_endpoint(endpoint_name: str, question: str) -> str:
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=endpoint_name,
        inputs={"messages": [{"role": "user", "content": question}]},
    )
    if "choices" in response:
        return response["choices"][0]["message"]["content"]
    elif "content" in response:
        return response["content"]
    return str(response)


TEST_QUESTION = "What PPE is required for workers entering a refinery process unit?"

try:
    print(f"Testing KA endpoint: {KA_ENDPOINT}")
    answer = test_ka_endpoint(KA_ENDPOINT, TEST_QUESTION)
    print("KA Response:")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    print()
    print("KA endpoint is responding. Proceed to 02b_tracing_deep_dive.")
except Exception as e:
    print(f"Could not reach KA endpoint '{KA_ENDPOINT}': {e}")
    print("The KA may still be indexing. Check the Agents UI for status.")

# COMMAND ----------

# DBTITLE 1,Section 4 — Grant Experiment Access
# MAGIC %md
# MAGIC ## Section 4: Grant Experiment Access to KA Service Principal

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
        print(f"Granted CAN_EDIT on experiment '{EXPERIMENT_PATH}' to '{sp_id}'.")
    else:
        print("Could not determine KA service principal — skipping ACL grant.")
except Exception as e:
    print(f"Permission grant skipped (non-fatal): {e}")
    print("KA traces will land in the KA's auto-created experiment.")

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Status |
# MAGIC |------|--------|
# MAGIC | Documents uploaded to UC Volume | Done in `01_setup_data` |
# MAGIC | Knowledge Assistant created via API | Done — check Agents UI for indexing status |
# MAGIC | V1 instructions registered in Prompt Registry | Done |
# MAGIC | KA endpoint tested | Done |
# MAGIC | Experiment permissions granted | Done |
# MAGIC
# MAGIC **Next:** Run `02b_tracing_deep_dive` to explore KA traces.
