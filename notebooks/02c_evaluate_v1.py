# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Evaluate V1 Instructions (Baseline)
# MAGIC
# MAGIC Run the full evaluation suite against the **V1 instructions** — a minimal two-sentence
# MAGIC prompt with no citation guidance, scope constraints, or format requirements.
# MAGIC
# MAGIC We expect V1 to produce:
# MAGIC - Answers without document citations
# MAGIC - Incomplete multi-part answers
# MAGIC - Off-topic responses for edge-case questions
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02b_config` (loaded via `%run`)
# MAGIC - `02b_tracing_deep_dive` recommended first (to understand the trace structure)

# COMMAND ----------

# DBTITLE 1,Load shared foundation
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,Eval run header
# MAGIC %md
# MAGIC ## Full Evaluation Run (V1 Instructions)
# MAGIC
# MAGIC We evaluate all 30 Q&A pairs from the eval dataset against the V1 instructions.
# MAGIC Each prediction call:
# MAGIC 1. Loads V1 instructions from the Prompt Registry (`@v1` alias)
# MAGIC 2. Retrieves relevant PEMEX document paragraphs for the question
# MAGIC 3. Calls the LLM and returns the answer
# MAGIC
# MAGIC MLflow logs a trace for every prediction, plus aggregate metrics for the run.

# COMMAND ----------

# DBTITLE 1,Run V1 evaluation
v1_predict = make_predict_fn("v1")

with mlflow.start_run(run_name="eval_v1_instructions"):
    mlflow.log_param("instructions_version", "v1")
    mlflow.log_param("instructions_alias",   "v1")
    mlflow.log_text(V1_INSTRUCTIONS, "instructions.txt")

    v1_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=v1_predict,
        scorers=SCORERS,
    )

print("V1 evaluation complete. Check the MLflow Experiment UI for per-example results.")
print()
print("Aggregate Metrics:")
print("-" * 50)
for k, v in sorted(v1_results.metrics.items()):
    bar = ""
    if isinstance(v, float) and 0.0 <= v <= 1.0:
        filled = int(v * 20)
        bar = f"  [{'#' * filled}{'.' * (20 - filled)}]"
    val_str = f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"  {k:<30} {val_str}{bar}")

# COMMAND ----------

# DBTITLE 1,Expected V1 weaknesses
# MAGIC %md
# MAGIC ## Expected V1 Weaknesses
# MAGIC
# MAGIC | Weakness | Root Cause | Scorer That Catches It |
# MAGIC |----------|------------|------------------------|
# MAGIC | **No document citations** | V1 instructions say nothing about citing sources | `groundedness` |
# MAGIC | **Incomplete answers** | V1 has no completeness requirement | `completeness` |
# MAGIC | **Vague answer quality** | V1 instructions too general to guide factual specificity | `answer_quality` |
# MAGIC | **Off-topic responses** | V1 doesn't constrain scope to PEMEX documentation | `groundedness`, `completeness` |
# MAGIC
# MAGIC **Next:** Run `02d_optimize_prompt` to use GEPA to automatically generate
# MAGIC improved instructions based on these evaluation results.
