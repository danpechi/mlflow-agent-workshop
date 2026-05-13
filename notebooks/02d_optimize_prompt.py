# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Optimize KA Instructions with GEPA
# MAGIC
# MAGIC We use **MLflow GEPA** (`mlflow.genai.optimize_prompts()`) to automatically generate
# MAGIC improved KA instructions based on the V1 evaluation results.
# MAGIC
# MAGIC GEPA works by:
# MAGIC 1. Running the `predict_fn` with the current instructions
# MAGIC 2. Scoring each response with the provided scorers
# MAGIC 3. Reflecting on failures and proposing instruction improvements
# MAGIC 4. Repeating until the score plateaus or `max_metric_calls` is reached
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02b_config` (loaded via `%run`)
# MAGIC - `02c_evaluate_v1` should have been run first so the V1 baseline is established

# COMMAND ----------

# DBTITLE 1,Load shared foundation
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,GEPA header
# MAGIC %md
# MAGIC ## Run GEPA Instruction Optimization

# COMMAND ----------

# DBTITLE 1,Configure GEPA
from mlflow.genai.optimize import GepaPromptOptimizer

# Load the V1 instructions URI — GEPA will iterate on this
v1_instructions = mlflow.genai.load_prompt(f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@v1")
print(f"Starting point: {v1_instructions.uri}")
print(f"V1 template: {v1_instructions.template}")
print()

# The predict_fn for GEPA must reload the instructions on each call
# (GEPA swaps instruction versions internally during optimization)
def gepa_predict_fn(*, question: str, **kwargs) -> str:
    instructions = mlflow.genai.load_prompt(v1_instructions.uri)
    context = simple_retrieve(question)
    response = model.invoke([
        {
            "role": "system",
            "content": (
                f"{instructions.format()}\n\n"
                "Use ONLY the following PEMEX document excerpts to answer. "
                "If the answer is not in the excerpts, say so.\n\n"
                f"{context}"
            ),
        },
        {"role": "user", "content": question},
    ])
    return response.content

# GEPA requires numeric scorers only — Guidelines/Safety return Feedback objects.
# Use the numeric subset for optimization; keep full SCORERS for standalone eval runs.
GEPA_SCORERS = [answer_quality]
print(f"GEPA scorers (numeric-only): {[getattr(s, 'name', repr(s)) for s in GEPA_SCORERS]}")

# COMMAND ----------

# DBTITLE 1,Run GEPA optimization
optimization_result = mlflow.genai.optimize_prompts(
    predict_fn=gepa_predict_fn,
    train_data=eval_dataset,
    prompt_uris=[v1_instructions.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=f"databricks:/{LLM_ENDPOINT}",
        max_metric_calls=50,
        display_progress_bar=True,
    ),
    scorers=GEPA_SCORERS,
)

optimized_template = optimization_result.optimized_prompts[0].template

print("GEPA Optimized Instructions:")
print("=" * 80)
print(optimized_template)
print("=" * 80)
print()
print(f"Initial score : {optimization_result.initial_eval_score:.3f}")
print(f"Final score   : {optimization_result.final_eval_score:.3f}")
print(f"Improvement   : {optimization_result.final_eval_score - optimization_result.initial_eval_score:+.3f}")

# COMMAND ----------

# DBTITLE 1,Register header
# MAGIC %md
# MAGIC ### Register the Optimized Instructions in MLflow Prompt Registry
# MAGIC
# MAGIC The optimized instructions are versioned alongside V1. The alias `optimized`
# MAGIC points to the new version. Both remain accessible for comparison.

# COMMAND ----------

# DBTITLE 1,Register optimized instructions
# GEPA registered a new version automatically. We just set the alias.
all_versions = mlflow.MlflowClient().search_prompt_versions(INSTRUCTIONS_REGISTRY_NAME)
latest_version = max(v.version for v in all_versions)

mlflow.genai.set_prompt_alias(
    name=INSTRUCTIONS_REGISTRY_NAME,
    alias="optimized",
    version=latest_version,
)

print(f"Prompt Registry: {INSTRUCTIONS_REGISTRY_NAME}")
print(f"  v1        → version 1 (minimal baseline)")
print(f"  optimized → version {latest_version} (GEPA-generated)")
print()
print(f"Optimized instructions length: {len(optimized_template):,} chars")
print()
print("Key themes GEPA should have added:")
for theme in ["cite", "source", "document", "complete", "step", "all", "specific", "according", "per"]:
    present = theme.lower() in optimized_template.lower()
    mark = "FOUND" if present else "missing"
    print(f"  {theme:15s} -> {mark}")
