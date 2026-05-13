# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Evaluate Optimized Instructions & Compare
# MAGIC
# MAGIC Re-run the exact same evaluation suite using the GEPA-optimized instructions.
# MAGIC Then compare scores to V1 to confirm improvement before applying them to the KA.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - `02b_config` (loaded via `%run`)
# MAGIC - `02c_evaluate_v1` must have been run (V1 metrics are loaded from MLflow)
# MAGIC - `02d_optimize_prompt` must have been run (optimized instructions registered in Prompt Registry)

# COMMAND ----------

# DBTITLE 1,Load shared foundation
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,Eval optimized header
# MAGIC %md
# MAGIC ## Evaluate with Optimized Instructions

# COMMAND ----------

# DBTITLE 1,Evaluate optimized instructions
optimized_predict = make_predict_fn("optimized")

# Load the optimized instructions text for logging
optimized_instructions_obj = mlflow.genai.load_prompt(f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@optimized")
optimized_instructions_text = optimized_instructions_obj.format()

with mlflow.start_run(run_name="eval_optimized_instructions"):
    mlflow.log_param("instructions_version", "optimized")
    mlflow.log_param("instructions_alias",   "optimized")
    mlflow.log_text(optimized_instructions_text, "instructions.txt")

    v2_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=optimized_predict,
        scorers=SCORERS,
    )

print("Optimized evaluation complete. Check the MLflow Experiment UI for per-example results.")

# COMMAND ----------

# DBTITLE 1,Comparison header
# MAGIC %md
# MAGIC ## Before / After Comparison
# MAGIC
# MAGIC V1 metrics are loaded from the MLflow experiment run logged by `02c_evaluate_v1`.

# COMMAND ----------

# DBTITLE 1,V1 vs Optimized comparison
import math

try:
    v1_run_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_PATH],
        filter_string="params.instructions_version = 'v1'",
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

    print("=" * 75)
    print("EVALUATION COMPARISON: V1 Instructions vs Optimized Instructions")
    print("=" * 75)
    print(f"{'Metric':<35} {'V1':>12} {'Optimized':>12} {'Delta':>8}")
    print("-" * 75)

    all_keys = sorted(set(list(v1_metrics.keys()) + list(v2_metrics.keys())))
    for key in all_keys:
        v1_val = v1_metrics.get(key, None)
        v2_val = v2_metrics.get(key, None)
        v1_str = f"{v1_val:.3f}" if isinstance(v1_val, (int, float)) else str(v1_val or "N/A")
        v2_str = f"{v2_val:.3f}" if isinstance(v2_val, (int, float)) else str(v2_val or "N/A")

        # Compute delta
        if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
            delta = v2_val - v1_val
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "N/A"

        print(f"{key:<35} {v1_str:>12} {v2_str:>12} {delta_str:>8}")

    print("=" * 75)
    print()
    print("Next: Run 02f_redeploy_app to apply the optimized instructions to the live KA.")

except Exception as e:
    print(f"Could not print comparison table: {e}")
    print("Check the MLflow Experiment UI for side-by-side metric comparison.")
