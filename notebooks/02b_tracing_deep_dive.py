# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # MLflow Tracing for Knowledge Assistant Observability
# MAGIC
# MAGIC Every query to your PEMEX Knowledge Assistant generates a full MLflow trace —
# MAGIC automatically, with no instrumentation code required. This notebook shows you
# MAGIC how to find, read, and act on those traces at scale.
# MAGIC
# MAGIC **Three objectives (~35 min):**
# MAGIC 1. **Explore KA traces in the UI** — understand the retrieval-augmented trace structure
# MAGIC 2. **Programmatic trace search** — use `search_traces()` to find slow or low-quality responses
# MAGIC 3. **Production observability** — snapshot traces to a Delta table, run SQL analytics

# COMMAND ----------

# DBTITLE 1,Load setup
# MAGIC %run ./02b_config

# COMMAND ----------

# DBTITLE 1,Setup overview
# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC We enable LangChain `autolog` so that our local LLM calls (used throughout this
# MAGIC notebook when demonstrating tracing) are captured automatically.
# MAGIC The actual KA endpoint also generates traces automatically — no setup needed there.

# COMMAND ----------

# DBTITLE 1,Enable autolog
mlflow.langchain.autolog(log_traces=True)
print("MLflow LangChain autolog enabled.")
print(f"Local LLM traces will be logged to: {EXPERIMENT_PATH}")
print()
print("Note: KA endpoint traces are auto-generated and visible in the Agents UI.")
print("      They can also be searched with mlflow.search_traces() using the KA's experiment.")

# COMMAND ----------

# DBTITLE 1,Objective 1 — Explore KA traces
# MAGIC %md
# MAGIC ## Objective 1: Explore KA Traces in the MLflow UI
# MAGIC
# MAGIC When a user asks the PEMEX KA a question, Databricks automatically records:
# MAGIC - The **user's question** (root span input)
# MAGIC - The **retrieval step** — which document chunks were returned
# MAGIC - The **LLM call** — the model input/output with token counts
# MAGIC - The **final response** — the answer sent back to the user, with citations
# MAGIC
# MAGIC This gives you full observability without writing a single line of tracing code.

# COMMAND ----------

# DBTITLE 1,Query KA and generate traces
# Generate a few traces by querying the KA endpoint (if it's live)
# Falls back gracefully to local LLM if the KA is still being set up.

from mlflow.deployments import get_deploy_client

TEST_QUESTIONS = [
    "What PPE is required for workers entering a refinery process unit?",
    "What actions must be taken in the first 15 minutes of discovering an oil spill?",
    "How long can hazardous waste be stored on-site?",
    "What are the six steps of the LOTO procedure?",
    "What TRIR threshold must contractors meet for pre-qualification?",
]

ka_traces_generated = 0
for i, question in enumerate(TEST_QUESTIONS, 1):
    try:
        client = get_deploy_client("databricks")
        response = client.predict(
            endpoint=KA_ENDPOINT,
            inputs={"messages": [{"role": "user", "content": question}]},
        )
        answer = response.get("choices", [{}])[0].get("message", {}).get("content", str(response))
        print(f"[{i}/{len(TEST_QUESTIONS)}] KA answered: {question[:60]}...")
        print(f"   Response: {answer[:120]}...")
        ka_traces_generated += 1
    except Exception as e:
        # KA not live yet — generate a local LLM trace as a stand-in
        instructions = mlflow.genai.load_prompt(f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@v1")
        context = simple_retrieve(question)
        response = model.invoke([
            {"role": "system", "content": f"{instructions.format()}\n\nContext:\n{context}"},
            {"role": "user",   "content": question},
        ])
        mlflow.update_current_trace(tags={"source": "local_llm_fallback", "question_id": f"OBJ1-{i:03d}"})
        print(f"[{i}/{len(TEST_QUESTIONS)}] Local LLM (KA not live): {question[:60]}...")

print(f"\nGenerated {ka_traces_generated} KA traces + {len(TEST_QUESTIONS) - ka_traces_generated} local LLM traces.")

# COMMAND ----------

# DBTITLE 1,How to view KA traces
# MAGIC %md
# MAGIC ### View KA Traces in the MLflow UI
# MAGIC
# MAGIC **For KA-generated traces:**
# MAGIC 1. In the left sidebar, go to **Machine Learning** → **Agents**.
# MAGIC 2. Click your PEMEX KA → **View traces** (or navigate to the linked experiment).
# MAGIC 3. Click any trace to see the full span waterfall.
# MAGIC
# MAGIC **What you'll see in a KA trace:**
# MAGIC
# MAGIC ```
# MAGIC ► knowledge_assistant  (root — AGENT span)
# MAGIC   ├── retrieve          (RETRIEVER span — documents fetched)
# MAGIC   │     inputs:  {"query": "What PPE is required..."}
# MAGIC   │     outputs: [{"content": "...", "source": "hsse_procedures.md"}, ...]
# MAGIC   └── llm_generate      (LLM span — model call with context)
# MAGIC         inputs:  {"messages": [...system with context..., ...user question...]}
# MAGIC         outputs: {"content": "Workers entering refinery process units must wear..."}
# MAGIC ```
# MAGIC
# MAGIC **For local LLM traces (fallback):**
# MAGIC 1. Click the **Experiment** icon in the left sidebar → select your experiment.
# MAGIC 2. Click the **Traces** tab → click the most recent trace.
# MAGIC
# MAGIC **What to look for in either case:**
# MAGIC - The **retrieval span** shows which chunks were returned — poor retrieval = poor answers
# MAGIC - The **LLM span** shows token counts (useful for cost monitoring)
# MAGIC - **Trace-level tags** let you filter: `tags.source = 'local_llm_fallback'`

# COMMAND ----------

# DBTITLE 1,Objective 2 — Programmatic trace search
# MAGIC %md
# MAGIC ## Objective 2: Programmatic Trace Search and Diagnosis
# MAGIC
# MAGIC `mlflow.search_traces()` turns your trace history into a queryable Pandas DataFrame.
# MAGIC Use it to find slow responses, error traces, or responses with specific tags.

# COMMAND ----------

# DBTITLE 1,Search traces and show overview
from mlflow import MlflowClient

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
all_traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    max_results=30,
    order_by=["timestamp DESC"],
)

print(f"Found {len(all_traces)} traces in experiment\n")
print(f"  {'Trace name':<40} {'State':<6} {'Duration (ms)':>14} {'Tags'}")
print(f"  {'-'*40} {'-'*6} {'-'*14} {'-'*30}")

for _, row in all_traces.iterrows():
    tags = row.get("tags", {}) or {}
    user_tags = {k: v for k, v in tags.items() if not k.startswith("mlflow.")}
    name = tags.get("mlflow.traceName", "unknown")[:40]
    state = str(row.get("state", "?"))[:6]
    dur = row.get("execution_duration")
    dur_str = f"{int(dur):>12,}" if dur is not None else "           ?"
    tag_str = str(user_tags)[:50] if user_tags else ""
    print(f"  {name:<40} {state:<6} {dur_str} ms  {tag_str}")

# COMMAND ----------

# DBTITLE 1,Find slow traces
# Find traces slower than a threshold — useful for latency SLO monitoring
SLOW_THRESHOLD_MS = 3000

slow_traces = all_traces[
    all_traces["execution_duration"].fillna(0) > SLOW_THRESHOLD_MS
].sort_values("execution_duration", ascending=False)

print(f"Slow traces (> {SLOW_THRESHOLD_MS} ms): {len(slow_traces)}")
if len(slow_traces) > 0:
    for _, row in slow_traces.iterrows():
        tags = row.get("tags", {}) or {}
        name = tags.get("mlflow.traceName", "unknown")
        dur = row.get("execution_duration", 0)
        print(f"  {name:<45}  {int(dur):>8,} ms  trace_id={row['trace_id'][:16]}...")

print()
print("These are your latency outliers. Drill into their spans in the next cell.")

# COMMAND ----------

# DBTITLE 1,Drill into spans of the slowest trace
# Drill into the slowest trace to find the bottleneck span
if len(all_traces) > 0:
    client = MlflowClient()

    # Pick the slowest trace we have
    sorted_traces = all_traces.sort_values("execution_duration", ascending=False)
    slowest_row = sorted_traces.iloc[0]
    trace_id = slowest_row["trace_id"]
    trace = client.get_trace(trace_id)

    total_ms = slowest_row.get("execution_duration", 0) or 0
    print(f"Slowest trace: {trace_id}")
    print(f"  Total duration : {int(total_ms):,} ms")
    print(f"  Status         : {trace.info.status}")
    print(f"  Spans          : {len(trace.data.spans)}")
    print()

    def _span_ms(span):
        return ((span.end_time_ns or 0) - (span.start_time_ns or 0)) / 1e6

    spans_sorted = sorted(trace.data.spans, key=lambda s: -_span_ms(s))
    print(f"  {'Type':<12} {'Span name':<38} {'Duration':>10}")
    print(f"  {'-'*12} {'-'*38} {'-'*10}")
    for span in spans_sorted:
        span_type = getattr(span, "span_type", "UNKNOWN")
        if isinstance(span_type, (list, tuple)):
            span_type = next((str(v) for v in span_type if v), "UNKNOWN")
        elif hasattr(span_type, "value"):
            span_type = str(span_type.value)
        span_type = str(span_type)[:12]
        dur_ms = _span_ms(span)
        print(f"  {span_type:<12} {span.name[:38]:<38} {dur_ms:>8.1f} ms")

    print()
    print("Top span is the bottleneck. Check:")
    print("  - RETRIEVER spans: slow retrieval = vector search or index issue")
    print("  - LLM spans      : slow generation = token count or endpoint cold start")
else:
    print("No traces found yet. Run the cells above to generate traces first.")

# COMMAND ----------

# DBTITLE 1,Diagnose a poor response
# MAGIC %md
# MAGIC ## Diagnose a Poor Response
# MAGIC
# MAGIC Generate a trace for a question where V1 instructions are likely to produce a
# MAGIC vague or incomplete answer, then inspect the trace to understand why.

# COMMAND ----------

# DBTITLE 1,Run a tricky question with manual span tagging
from mlflow.entities import SpanType

TRICKY_QUESTION = (
    "What approval levels are required for a major Management of Change, "
    "and what is the minimum review period?"
)

# Wrap the local LLM call with a diagnostic span so we can tag it and find it later
with mlflow.start_span(name="ka_diagnosis", span_type=SpanType.AGENT) as root_span:
    root_span.set_inputs({"question": TRICKY_QUESTION, "prompt_version": "v1"})

    # Retrieve relevant context
    with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as ret_span:
        context = simple_retrieve(TRICKY_QUESTION, top_k=3)
        ret_span.set_inputs({"query": TRICKY_QUESTION})
        ret_span.set_outputs({"context_length": len(context), "context_preview": context[:300]})

    # Call LLM with V1 instructions
    instructions = mlflow.genai.load_prompt(f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@v1")
    response = model.invoke([
        {"role": "system", "content": f"{instructions.format()}\n\nContext:\n{context}"},
        {"role": "user",   "content": TRICKY_QUESTION},
    ])
    answer = response.content
    root_span.set_outputs({"answer": answer[:500]})

    mlflow.update_current_trace(tags={
        "diagnosis": "v1_completeness_test",
        "prompt_version": "v1",
        "question_type": "multi_part",
    })

print(f"Question: {TRICKY_QUESTION}")
print()
print("V1 Response:")
print("=" * 60)
print(answer)
print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Analyze V1 weaknesses
# Check V1 response for known failure modes
print("V1 FAILURE ANALYSIS")
print("=" * 60)

has_citation = any(
    phrase in answer.lower()
    for phrase in ["operational standards", "hsse", "according to", "per the", "procedure manual"]
)
mentions_approval_levels = all(
    term in answer.lower()
    for term in ["plant manager", "hsse director", "vp"]
)
mentions_timeline = "30" in answer and ("day" in answer.lower() or "business" in answer.lower())
has_complete_answer = mentions_approval_levels and mentions_timeline

checks = [
    ("Cites source document",        has_citation,       not has_citation),
    ("Lists all 3 approvers",        mentions_approval_levels, not mentions_approval_levels),
    ("Mentions 30-day review period",mentions_timeline,  not mentions_timeline),
    ("Complete multi-part answer",   has_complete_answer, not has_complete_answer),
]

for label, passed, failed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")

print()
print("DIAGNOSIS:")
if not has_citation:
    print("  -> V1 lacks citation instructions: response doesn't reference source documents")
if not mentions_approval_levels:
    print("  -> V1 lacks completeness guidance: misses one or more required approval levels")
if not mentions_timeline:
    print("  -> V1 lacks completeness guidance: misses the 30-day review period")
print()
print("-> Next: Run 02c (evaluate V1 at scale) to quantify these failures across all 30 Q&A pairs.")

# COMMAND ----------

# DBTITLE 1,Objective 3 — Production observability
# MAGIC %md
# MAGIC ## Objective 3: Production Observability — Traces as Delta Tables
# MAGIC
# MAGIC Every KA trace is already governed data in Unity Catalog. Snapshot your traces
# MAGIC to a managed Delta table and you can:
# MAGIC - Ask SQL questions like *p95 latency by document category*
# MAGIC - JOIN traces with your Q&A eval dataset to find systematically wrong answers
# MAGIC - Build dashboards or Genie spaces over trace data
# MAGIC - Avoid second-vendor lock-in — MLflow is OSS, traces are Delta, lineage stays in UC

# COMMAND ----------

# DBTITLE 1,Snapshot traces to Delta
import logging as _logging
import warnings as _warnings

_warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
_logging.getLogger("urllib3.connectionpool").setLevel(_logging.ERROR)
_logging.getLogger("mlflow.tracing.client").setLevel(_logging.ERROR)

TRACES_TABLE        = f"{_short_name}_ka_traces"
TRACES_TABLE_FQN    = f"{CATALOG}.{SCHEMA}.{TRACES_TABLE}"
TRACES_TABLE_BT_FQN = f"{CATALOG_BT}.{SCHEMA_BT}.{_bt(TRACES_TABLE)}"

exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
trace_df = mlflow.search_traces(
    locations=[exp.experiment_id],
    max_results=500,
    order_by=["timestamp DESC"],
)

# Flatten complex columns (dicts/lists → JSON strings) for Spark write
def _to_json(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return v

def _to_ms(v):
    if v is None:
        return None
    if hasattr(v, "total_seconds"):
        return int(v.total_seconds() * 1000)
    try:
        return int(v)
    except Exception:
        return None

flat = trace_df.copy()
for col in ("tags", "request", "response", "spans", "request_metadata", "assessments", "info"):
    if col in flat.columns:
        flat[col] = flat[col].apply(_to_json)
if "execution_duration" in flat.columns:
    flat["execution_duration"] = flat["execution_duration"].apply(_to_ms)

(spark.createDataFrame(flat)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TRACES_TABLE_FQN))

_logging.getLogger("urllib3.connectionpool").setLevel(_logging.WARNING)
_logging.getLogger("mlflow.tracing.client").setLevel(_logging.WARNING)

print(f"Persisted {len(flat)} traces to {TRACES_TABLE_FQN}")
print("Traces are now queryable as a standard Delta table — same SQL, same governance.")

# COMMAND ----------

# DBTITLE 1,SQL latency analysis
# Latency distribution — answers "is my KA meeting its response time SLO?"
display(spark.sql(f"""
  SELECT
    COUNT(*)                                       AS n_traces,
    ROUND(AVG(execution_duration), 1)              AS avg_ms,
    ROUND(PERCENTILE(execution_duration, 0.50), 1) AS p50_ms,
    ROUND(PERCENTILE(execution_duration, 0.95), 1) AS p95_ms,
    MAX(execution_duration)                        AS max_ms
  FROM {TRACES_TABLE_BT_FQN}
  WHERE execution_duration IS NOT NULL
"""))

# COMMAND ----------

# DBTITLE 1,SQL error and slow trace query
# Find traces in ERROR state or slower than 5 seconds
display(spark.sql(f"""
  SELECT
    trace_id,
    state                                          AS status,
    execution_duration                             AS duration_ms,
    get_json_object(tags, '$.mlflow.traceName')    AS trace_name,
    get_json_object(tags, '$.diagnosis')           AS diagnosis_tag
  FROM {TRACES_TABLE_BT_FQN}
  WHERE state = 'ERROR' OR execution_duration > 5000
  ORDER BY execution_duration DESC
  LIMIT 20
"""))

# COMMAND ----------

# DBTITLE 1,SQL join with eval dataset
# JOIN traces with the Q&A eval dataset — identify systematically missed question categories
# This works because both tables live in Unity Catalog: same governance, same lineage.
display(spark.sql(f"""
  SELECT
    q.category,
    COUNT(*)                                        AS n_questions,
    ROUND(AVG(t.execution_duration), 1)             AS avg_latency_ms,
    COUNT(CASE WHEN t.state = 'ERROR' THEN 1 END)   AS error_count
  FROM {TRACES_TABLE_BT_FQN}   t
  CROSS JOIN {EVAL_TABLE_BT_FQN} q
  WHERE t.state IS NOT NULL
  GROUP BY q.category
  ORDER BY avg_latency_ms DESC
  LIMIT 20
"""))

# COMMAND ----------

# DBTITLE 1,Summary
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Objective | APIs Used | Key Insight |
# MAGIC |-----------|-----------|-------------|
# MAGIC | **1. KA traces (UI)** | Auto-generated by Databricks | Zero instrumentation needed; every query is traced |
# MAGIC | **2. Programmatic search** | `search_traces()`, `MlflowClient().get_trace()` | Find slow/failed traces at scale; drill into spans |
# MAGIC | **3. Production observability** | `search_traces()` → Delta → SQL | Traces are just rows — same SQL, same lineage, no second vendor |
# MAGIC
# MAGIC **Key takeaway:** You don't write any tracing code for a Knowledge Assistant.
# MAGIC You spend your time *reading* and *acting on* the traces — using SQL and Python
# MAGIC to identify where your KA is falling short.
# MAGIC
# MAGIC **Next steps:**
# MAGIC - `02c_evaluate_v1` — Run systematic evaluation to quantify V1 instruction failures
# MAGIC - `02d_optimize_prompt` — Use GEPA to auto-generate improved instructions
# MAGIC - `02e_evaluate_and_compare` — Verify improvements with the same eval suite
