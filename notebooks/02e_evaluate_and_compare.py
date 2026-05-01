# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Evaluate Optimized Agent & Compare
# MAGIC
# MAGIC Re-run the exact same evaluation suite using the GEPA-optimized prompt.
# MAGIC Then compare scores to V1 to confirm improvement before deploying.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02a_setup_and_agent` (loaded via `%run`)
# MAGIC - `02c_evaluate_v1` must have been run (V1 metrics are loaded from MLflow)
# MAGIC - `02d_optimize_prompt` must have been run (optimized prompt registered in Prompt Registry)

# COMMAND ----------

# DBTITLE 1,Load setup and agent foundation
# MAGIC %run ./02a_setup_and_agent

# COMMAND ----------

# DBTITLE 1,Eval optimized header
# MAGIC %md
# MAGIC ## Evaluate Optimized Agent

# COMMAND ----------

# DBTITLE 1,Evaluate optimized agent
optimized_predict = make_predict_fn("optimized")

# Load the optimized prompt text for logging
optimized_prompt_obj = mlflow.genai.load_prompt(f"prompts:/{PROMPT_REGISTRY_NAME}@optimized")
optimized_prompt_text = optimized_prompt_obj.format()

with mlflow.start_run(run_name="eval_optimized_prompt"):
    mlflow.log_param("prompt_version", "gepa_optimized")
    mlflow.log_text(optimized_prompt_text, "system_prompt.txt")

    v2_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=optimized_predict,
        scorers=SCORERS,
    )

print("Optimized evaluation complete. Check the MLflow Experiment UI for detailed results.")

# COMMAND ----------

# DBTITLE 1,Comparison header
# MAGIC %md
# MAGIC ## Before / After Comparison
# MAGIC
# MAGIC V1 metrics are loaded from the MLflow experiment run (logged by `02c_evaluate_v1`).

# COMMAND ----------

# DBTITLE 1,V1 vs Optimized comparison
import math

try:
    # Load V1 eval metrics from the MLflow experiment
    v1_run_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_PATH],
        filter_string="params.prompt_version = 'v1'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    v1_metrics = {}
    if len(v1_run_df) > 0:
        for col in v1_run_df.columns:
            if col.startswith("metrics."):
                val = v1_run_df.iloc[0][col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    v1_metrics[col.replace("metrics.", "")] = val
    else:
        print("WARNING: No V1 evaluation run found. Run 02c_evaluate_v1 first.")

    v2_metrics = v2_results.metrics if hasattr(v2_results, "metrics") else {}

    print("=" * 70)
    print("EVALUATION COMPARISON: V1 vs Optimized")
    print("=" * 70)
    print(f"{'Metric':<30} {'V1':>15} {'Optimized':>15}")
    print("-" * 70)

    all_keys = sorted(set(list(v1_metrics.keys()) + list(v2_metrics.keys())))
    for key in all_keys:
        v1_val = v1_metrics.get(key, "N/A")
        v2_val = v2_metrics.get(key, "N/A")
        v1_str = f"{v1_val:.3f}" if isinstance(v1_val, (int, float)) else str(v1_val)
        v2_str = f"{v2_val:.3f}" if isinstance(v2_val, (int, float)) else str(v2_val)
        print(f"{key:<30} {v1_str:>15} {v2_str:>15}")

    print("=" * 70)
    print("\nNext: Run 02f_redeploy_app to deploy the improved agent.")
except Exception as e:
    print(f"Could not print comparison table: {e}")
    print("Check the MLflow Experiment UI for side-by-side comparison.")