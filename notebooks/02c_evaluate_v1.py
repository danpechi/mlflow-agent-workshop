# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Evaluate V1 Agent (Baseline)
# MAGIC
# MAGIC This notebook runs the full evaluation suite against the **V1 prompt** — a deliberately
# MAGIC minimal two-sentence instruction. We expect it to produce unstructured output,
# MAGIC inconsistent verdicts, PII leakage, and injection vulnerability.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02a_setup_and_agent` (loaded via `%run`)
# MAGIC - `02b_tracing_deep_dive` recommended first to understand trace inspection

# COMMAND ----------

# DBTITLE 1,Load setup and agent foundation
# MAGIC %run ./02a_setup_and_agent

# COMMAND ----------

# DBTITLE 1,Full eval header
# MAGIC %md
# MAGIC ## Full Evaluation Run (V1)

# COMMAND ----------

# DBTITLE 1,Evaluate V1 agent
v1_predict = make_predict_fn("v1")

with mlflow.start_run(run_name="eval_v1_prompt"):
    mlflow.log_param("prompt_version", "v1")
    mlflow.log_text(V1_PROMPT, "system_prompt.txt")

    v1_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=v1_predict,
        scorers=SCORERS,
    )

print("V1 evaluation complete. Check the MLflow Experiment UI for detailed results.")
print(f"\nMetrics:")
for k, v in v1_results.metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,Expected V1 weaknesses
# MAGIC %md
# MAGIC ## Expected V1 Weaknesses
# MAGIC
# MAGIC | Weakness | Description |
# MAGIC |----------|-------------|
# MAGIC | **Unstructured output** | Free-form text instead of JSON with required fields |
# MAGIC | **Inconsistent verdicts** | No clear criteria for benign vs suspicious vs malicious |
# MAGIC | **PII leakage** | May echo SSNs, DOBs, account numbers from alert data |
# MAGIC | **Injection vulnerability** | Follows adversarial instructions embedded in alert fields |
# MAGIC | **No confidence calibration** | Does not express certainty level |
# MAGIC | **Shallow reasoning** | Verdicts without step-by-step evidence correlation |
# MAGIC
# MAGIC **Next:** Run `02d_optimize_prompt` to use GEPA to auto-generate an improved prompt.