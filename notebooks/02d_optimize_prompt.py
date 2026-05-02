# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Optimize the Prompt with GEPA
# MAGIC
# MAGIC We use **MLflow GEPA** (`mlflow.genai.optimize_prompts()`) to automatically
# MAGIC generate an improved prompt based on the V1 evaluation results.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02b_config` (loaded via `%run`)
# MAGIC - `02c_evaluate_v1` should have been run first so the V1 baseline exists

# COMMAND ----------

# DBTITLE 1,Load shared agent foundation
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,GEPA header
# MAGIC %md
# MAGIC ## Run GEPA Prompt Optimization

# COMMAND ----------

# DBTITLE 1,GEPA optimization
from mlflow.genai.optimize import GepaPromptOptimizer

# Load the V1 prompt URI from the registry -- GEPA will mutate it
v1_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_REGISTRY_NAME}@v1")
print(f"V1 prompt URI: {v1_prompt.uri}")

# GEPA predict_fn must reload the prompt on each call (GEPA swaps versions)
def gepa_predict_fn(*, query: str, **kwargs) -> str:
    prompt = mlflow.genai.load_prompt(v1_prompt.uri)
    agent = build_agent(prompt.format())
    return run_agent(agent, query)

# GEPA aggregates scorer outputs into a single optimization signal and only
# accepts numeric scorers. The Guidelines / Safety scorers in SCORERS return
# mlflow `Feedback` objects with value="yes"/"no", which GEPA cannot aggregate.
# Use a numeric-only subset for optimization; keep the full SCORERS list for
# the standalone V1 / optimized evaluation runs in 02b / 02d.
GEPA_SCORERS = [verdict_accuracy, structured_output]
print(f"GEPA scorers (numeric-only): {[getattr(s, 'name', repr(s)) for s in GEPA_SCORERS]}")

# Run GEPA optimization
optimization_result = mlflow.genai.optimize_prompts(
    predict_fn=gepa_predict_fn,
    train_data=eval_dataset,  # Has inputs + expectations
    prompt_uris=[v1_prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=f"databricks:/{LLM_ENDPOINT}",
        max_metric_calls=50,
        display_progress_bar=True,
    ),
    scorers=GEPA_SCORERS,
)

optimized_prompt = optimization_result.optimized_prompts[0].template

print("GEPA Optimized Prompt:")
print("=" * 80)
print(optimized_prompt)
print("=" * 80)
print(f"\nInitial score: {optimization_result.initial_eval_score}")
print(f"Final score:   {optimization_result.final_eval_score}")

# COMMAND ----------

# DBTITLE 1,Register header
# MAGIC %md
# MAGIC ### Register the optimized prompt in MLflow Prompt Registry
# MAGIC
# MAGIC The V1 and optimized prompts are versioned in the registry. The app loads
# MAGIC the prompt by alias (`v1` or `optimized`) at startup.

# COMMAND ----------

# DBTITLE 1,Register optimized prompt
# GEPA already registered the optimized prompt as a new version. Set the alias.
# Find the latest version
all_versions = mlflow.MlflowClient().search_prompt_versions(PROMPT_REGISTRY_NAME)
latest_version = max(v.version for v in all_versions)

mlflow.genai.set_prompt_alias(name=PROMPT_REGISTRY_NAME, alias="optimized", version=latest_version)

print(f"\nPrompt Registry: {PROMPT_REGISTRY_NAME}")
print(f"  v1       -> version 1 (pre-registered)")
print(f"  optimized -> version {latest_version}")
print(f"\nOptimized prompt length: {len(optimized_prompt)} chars")
print()
print("Key themes discovered by GEPA:")
for theme in ["verdict", "confidence", "evidence", "PII", "inject", "JSON", "tool", "benign", "suspicious", "malicious"]:
    present = theme.lower() in optimized_prompt.lower()
    print(f"  {theme:15s} -> {'FOUND' if present else 'missing'}")