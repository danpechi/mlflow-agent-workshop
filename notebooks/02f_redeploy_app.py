# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Apply Optimized Instructions to the PEMEX Knowledge Assistant
# MAGIC
# MAGIC We've confirmed that the GEPA-optimized instructions score better in evaluation.
# MAGIC Now we apply them to the live Knowledge Assistant so users immediately benefit.
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC 1. Load the optimized instructions from the Prompt Registry (`@optimized` alias)
# MAGIC 2. Update the KA configuration via the Databricks SDK
# MAGIC 3. Verify the updated KA endpoint responds with improved quality
# MAGIC 4. Run a final spot-check evaluation on the live KA endpoint
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02b_config` (loaded via `%run`)
# MAGIC - `02d_optimize_prompt` must have been run (alias `optimized` exists)
# MAGIC - `02e_evaluate_and_compare` should have confirmed improvement

# COMMAND ----------

# DBTITLE 1,Load shared config
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,Load optimized instructions
# MAGIC %md
# MAGIC ## Step 1: Load Optimized Instructions

# COMMAND ----------

# DBTITLE 1,Load from registry
optimized = mlflow.genai.load_prompt(f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@optimized")
optimized_text = optimized.format()

print(f"Loaded optimized instructions from: {optimized.uri}")
print()
print("Optimized Instructions:")
print("=" * 70)
print(optimized_text)
print("=" * 70)
print()
print(f"Length: {len(optimized_text):,} chars  (V1 was {len(V1_INSTRUCTIONS):,} chars)")

# COMMAND ----------

# DBTITLE 1,Update KA header
# MAGIC %md
# MAGIC ## Step 2: Update the Knowledge Assistant
# MAGIC
# MAGIC Knowledge Assistant instructions can be updated via the Databricks SDK or UI.
# MAGIC Below we use the SDK. If the API call fails (e.g., the KA is still being provisioned),
# MAGIC the cell also prints the UI steps as a fallback.

# COMMAND ----------

# DBTITLE 1,Update KA via SDK
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
update_succeeded = False

try:
    # The Knowledge Assistant API path — update with your workspace's API version if needed
    import requests

    token = w.config.token
    host  = w.config.host.rstrip("/")

    # List KAs to find the one matching KA_NAME
    list_resp = requests.get(
        f"{host}/api/2.0/agent-bricks/knowledge-assistants",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    list_resp.raise_for_status()
    kas = list_resp.json().get("knowledge_assistants", [])
    ka_record = next((k for k in kas if k.get("name") == KA_NAME), None)

    if ka_record is None:
        raise ValueError(
            f"Knowledge Assistant '{KA_NAME}' not found. "
            f"Available KAs: {[k.get('name') for k in kas]}"
        )

    ka_id = ka_record["knowledge_assistant_id"]
    print(f"Found KA '{KA_NAME}' with ID: {ka_id}")

    # Update instructions
    update_resp = requests.patch(
        f"{host}/api/2.0/agent-bricks/knowledge-assistants/{ka_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"instructions": optimized_text},
        timeout=30,
    )
    update_resp.raise_for_status()
    print(f"Successfully updated KA instructions.")
    update_succeeded = True

except Exception as e:
    print(f"SDK update failed: {e}")
    print()

if not update_succeeded:
    print("=" * 60)
    print("  MANUAL UPDATE (UI fallback)")
    print("=" * 60)
    print()
    print("1. Go to Machine Learning → Agents in the Databricks sidebar.")
    print(f"2. Click your KA: '{KA_NAME}'.")
    print("3. Click 'Edit' → paste the instructions below into the Instructions field.")
    print("4. Click 'Save'.")
    print()
    print("Instructions to paste:")
    print("-" * 60)
    print(optimized_text)
    print("-" * 60)

# COMMAND ----------

# DBTITLE 1,Verify updated KA
# MAGIC %md
# MAGIC ## Step 3: Verify the Updated KA Endpoint

# COMMAND ----------

# DBTITLE 1,Spot-check updated KA
from mlflow.deployments import get_deploy_client

SPOT_CHECK_QUESTIONS = [
    "What PPE is required for workers entering a refinery process unit?",
    "What are the six steps of the LOTO procedure?",
    "What approval levels are required for a major Management of Change?",
]

print(f"Spot-checking KA endpoint: {KA_ENDPOINT}")
print()

all_passed = True
for question in SPOT_CHECK_QUESTIONS:
    try:
        client = get_deploy_client("databricks")
        response = client.predict(
            endpoint=KA_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": question}]},
        )
        answer = response.get("choices", [{}])[0].get("message", {}).get("content", str(response))

        # Quick quality checks
        has_citation = any(
            phrase in answer.lower()
            for phrase in ["hsse", "operational standards", "environmental", "contractor",
                           "according to", "per the", "as specified", "procedure manual"]
        )
        is_complete = len(answer) > 100

        status = "PASS" if (has_citation and is_complete) else "PARTIAL"
        if not (has_citation and is_complete):
            all_passed = False

        print(f"[{status}] {question[:65]}...")
        print(f"        Response ({len(answer)} chars, cited={has_citation}): {answer[:150]}...")
        print()

    except Exception as e:
        print(f"[ERROR] Could not reach KA endpoint: {e}")
        print(f"        Question: {question[:65]}...")
        print()
        all_passed = False

if all_passed:
    print("All spot checks passed — the optimized KA is live and responding well.")
else:
    print("Some checks flagged. Open the Agents UI to inspect the latest traces for details.")

# COMMAND ----------

# DBTITLE 1,Workshop summary
# MAGIC %md
# MAGIC ## Workshop Summary
# MAGIC
# MAGIC | Notebook | Topic | Key Takeaways |
# MAGIC |----------|-------|---------------|
# MAGIC | `01_setup_data` | **Data Setup** | Upload PEMEX documents, create hand-crafted and LLM-generated Q&A eval dataset |
# MAGIC | `02a` | **KA Setup** | Create Knowledge Assistant (UI), register V1 instructions in Prompt Registry |
# MAGIC | `02b` | **Tracing** | KA traces are automatic — explore in UI, search programmatically, snapshot to Delta for SQL |
# MAGIC | `02c` | **Evaluate V1** | Run baseline eval, quantify V1 failures: missing citations, incomplete answers |
# MAGIC | `02d` | **Optimize Instructions** | Use GEPA to auto-generate improved KA instructions; register as `optimized` alias |
# MAGIC | `02e` | **Compare** | Verify improvement — V1 vs Optimized side-by-side metric table |
# MAGIC | `02f` | **Update KA** | Apply optimized instructions to the live KA; spot-check improved quality |
# MAGIC
# MAGIC **Key capabilities demonstrated:**
# MAGIC
# MAGIC 1. **Zero-code tracing** — KA traces are generated automatically; no `start_span()` needed.
# MAGIC 2. **SQL observability** — Snapshot traces to Delta and join with your eval data in standard SQL.
# MAGIC 3. **Systematic evaluation** — `mlflow.genai.evaluate()` + custom scorers across 30 Q&A pairs.
# MAGIC 4. **Instruction optimization** — GEPA iterates on the instructions text and registers the result.
# MAGIC 5. **Prompt Registry** — Version KA instructions with aliases (`v1`, `optimized`) for rollback.
# MAGIC
# MAGIC **Next steps for PEMEX:**
# MAGIC - Add more documents to the KA knowledge sources as they become available.
# MAGIC - Expand the eval dataset with domain-expert labeled Q&A pairs for higher-coverage testing.
# MAGIC - Set up scheduled evaluation runs to detect quality regressions as documents change.
# MAGIC - Build a Genie space over the `ka_traces` Delta table for self-service latency analytics.
