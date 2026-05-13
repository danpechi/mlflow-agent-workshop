# Databricks notebook source
# DBTITLE 1,Intro
# MAGIC %md
# MAGIC # Agent Foundation (Shared Config)
# MAGIC
# MAGIC This notebook is the **shared foundation** for all evaluation and optimization notebooks.
# MAGIC All other notebooks (`02b_tracing` through `02f`) `%run` this notebook to inherit:
# MAGIC
# MAGIC - Dependencies and imports
# MAGIC - Workshop configuration (`00_config`)
# MAGIC - Evaluation dataset and sample Q&A
# MAGIC - PEMEX documents loaded into memory for local retrieval
# MAGIC - `predict_fn` builder (queries KA endpoint or local LLM fallback)
# MAGIC - Custom scorers (`answer_quality`, `Safety`, `groundedness`, `completeness`)
# MAGIC - V1 instructions registered in the Prompt Registry
# MAGIC
# MAGIC > **Note:** This notebook does NOT create the KA. Use `02a_setup_and_agent` for that.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install mlflow[databricks] databricks-sdk databricks-langchain "gepa>=0.0.26" langchain-core --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports
import json
import os
import re

import mlflow
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# DBTITLE 1,Load config
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Set experiment
INSTRUCTIONS_REGISTRY_NAME = INSTRUCTIONS_REGISTRY_FQN
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"MLflow experiment: {EXPERIMENT_PATH}")

# COMMAND ----------

# DBTITLE 1,Load eval dataset
# MAGIC %md
# MAGIC ### Load Evaluation Dataset and Sample Q&A

# COMMAND ----------

# DBTITLE 1,Load data from volume
with open(EVAL_DATASET_PATH) as f:
    eval_qas = json.load(f)

with open(SAMPLE_QA_PATH) as f:
    sample_qas = json.load(f)

print(f"Loaded {len(eval_qas)} evaluation Q&A pairs and {len(sample_qas)} sample Q&A pairs.")

# COMMAND ----------

# DBTITLE 1,Build eval dataset
# Build eval_dataset in the format mlflow.genai.evaluate() expects
eval_dataset = []
for qa in eval_qas:
    eval_dataset.append({
        "inputs": {"question": qa["question"]},
        "expected": {"expected_answer": qa["expected_answer"]},
        "expectations": {
            "expected_response": (
                f"The response should accurately answer the question using information from the "
                f"PEMEX documentation. The answer should be factually correct, cite the relevant "
                f"document or section, and cover all key points: {qa['expected_answer'][:200]}"
            )
        },
    })

print(f"Eval dataset ready: {len(eval_dataset)} rows")
print(f"Example input: {eval_dataset[0]['inputs']['question']}")

# COMMAND ----------

# DBTITLE 1,Load PEMEX documents
# MAGIC %md
# MAGIC ### Load PEMEX Documents for Local Retrieval
# MAGIC
# MAGIC For evaluation, we use a local LLM + keyword retrieval as a stand-in for the KA.
# MAGIC This lets us run evaluations quickly and vary instructions without redeploying the KA.
# MAGIC The actual KA endpoint is used in `02b_tracing_deep_dive` for the observability demo.

# COMMAND ----------

# DBTITLE 1,Load docs and build retriever
DOCS = {}
for fname in sorted(os.listdir(DOCS_PATH)):
    if fname.endswith(".md"):
        fpath = os.path.join(DOCS_PATH, fname)
        with open(fpath) as f:
            DOCS[fname] = f.read()

print(f"Loaded {len(DOCS)} PEMEX documents:")
for fname, content in DOCS.items():
    print(f"  - {fname}  ({len(content):,} chars)")


def simple_retrieve(question: str, top_k: int = 4) -> str:
    """Keyword-based paragraph retrieval from PEMEX documents.

    Scores each paragraph by the number of question words it contains,
    then returns the top-k paragraphs with their source document labels.
    """
    q_words = set(question.lower().split())
    # Remove common stopwords to improve relevance
    stopwords = {"what", "is", "the", "a", "an", "for", "of", "in", "to", "how",
                 "are", "should", "must", "does", "do", "when", "where", "which",
                 "can", "be", "it", "that", "this", "with", "at", "by", "from"}
    q_words -= stopwords

    candidates = []
    for doc_name, doc_text in DOCS.items():
        paragraphs = [p.strip() for p in doc_text.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs:
            para_words = set(para.lower().split())
            score = len(q_words & para_words)
            if score > 0:
                candidates.append((score, doc_name, para))

    candidates.sort(reverse=True)
    if not candidates:
        return "No relevant content found in PEMEX documentation."

    results = []
    for score, doc_name, para in candidates[:top_k]:
        label = doc_name.replace("_", " ").replace(".md", "").title()
        results.append(f"[Source: {label}]\n{para}")
    return "\n\n---\n\n".join(results)


# Quick smoke test
test_context = simple_retrieve("What PPE is required for refinery entry?")
print(f"\nRetrieval smoke test — found {len(test_context)} chars of context")
print(test_context[:300] + "...")

# COMMAND ----------

# DBTITLE 1,Build predict_fn
# MAGIC %md
# MAGIC ### Predict Function
# MAGIC
# MAGIC `make_predict_fn(alias)` returns a function that:
# MAGIC 1. Loads the KA instructions from the MLflow Prompt Registry using the given alias
# MAGIC 2. Retrieves relevant PEMEX document paragraphs for the question
# MAGIC 3. Calls the LLM with the instructions + retrieved context + question
# MAGIC
# MAGIC This "offline" approach lets us evaluate different instruction versions quickly
# MAGIC and is the same pattern GEPA uses during optimization.

# COMMAND ----------

# DBTITLE 1,Define predict_fn builder
from databricks_langchain import ChatDatabricks

model = ChatDatabricks(endpoint=LLM_ENDPOINT)


def make_predict_fn(instructions_alias: str):
    """Create a predict_fn compatible with mlflow.genai.evaluate().

    Loads the KA instructions from the Prompt Registry inside the traced call
    so MLflow automatically links the instruction version to each trace.
    """
    def predict_fn(*, question: str, **kwargs) -> str:
        instructions = mlflow.genai.load_prompt(
            f"prompts:/{INSTRUCTIONS_REGISTRY_NAME}@{instructions_alias}"
        )
        context = simple_retrieve(question)
        system_content = (
            f"{instructions.format()}\n\n"
            "Use ONLY the following excerpts from PEMEX documentation to answer. "
            "If the answer is not in the excerpts, say so.\n\n"
            f"{context}"
        )
        response = model.invoke([
            {"role": "system", "content": system_content},
            {"role": "user",   "content": question},
        ])
        return response.content

    return predict_fn


print(f"predict_fn builder ready — instructions registry: {INSTRUCTIONS_REGISTRY_NAME}")
print(f"LLM endpoint: {LLM_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Scorers header
# MAGIC %md
# MAGIC ### Custom Scorers
# MAGIC
# MAGIC Four scorers cover the key quality dimensions for a document Q&A system:
# MAGIC
# MAGIC | Scorer | Type | What it checks |
# MAGIC |--------|------|---------------|
# MAGIC | `answer_quality` | Custom (keyword) | Does the response contain the expected key facts? |
# MAGIC | `Safety` | Built-in LLM-judge | Is the response safe and appropriate? |
# MAGIC | `groundedness` | Built-in LLM-judge (Guidelines) | Does the response cite or reference source documents? |
# MAGIC | `completeness` | Built-in LLM-judge (Guidelines) | Does the response address all aspects of the question? |

# COMMAND ----------

# DBTITLE 1,Define custom scorer
@mlflow.genai.scorer
def answer_quality(inputs, outputs, expectations) -> float:
    """Check if the response contains key facts from the expected answer.

    Scoring:
    1.0 = response contains ≥ 70% of the key terms from the expected answer
    0.5 = response contains 40–69% of key terms
    0.0 = response contains < 40% of key terms or is empty
    """
    expected = (expectations or {}).get("expected_answer", "")
    if not expected:
        return 1.0

    response_text = outputs if isinstance(outputs, str) else str(outputs)
    if not response_text.strip():
        return 0.0

    # Extract meaningful terms from the expected answer (ignore stopwords)
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "to", "of", "in",
                 "for", "on", "with", "at", "by", "from", "as", "into", "or",
                 "and", "but", "not", "no", "it", "its", "this", "that", "these"}
    expected_words = [
        w.lower().strip(".,;:()") for w in expected.split()
        if len(w) > 3 and w.lower() not in stopwords
    ]
    if not expected_words:
        return 1.0

    response_lower = response_text.lower()
    matched = sum(1 for w in expected_words if w in response_lower)
    ratio = matched / len(expected_words)

    if ratio >= 0.70:
        return 1.0
    elif ratio >= 0.40:
        return 0.5
    return 0.0

# COMMAND ----------

# DBTITLE 1,Assemble scorer list
from mlflow.genai.scorers import Guidelines, Safety

SCORERS = [
    answer_quality,
    Safety(),
    Guidelines(
        name="groundedness",
        guidelines=[
            "The response MUST cite or reference the relevant PEMEX document or section "
            "when providing procedural or regulatory information. Acceptable citations include "
            "phrases like 'According to the HSSE Procedures Manual', 'Per the Environmental "
            "Compliance Procedures', 'As specified in the Operational Standards', or similar "
            "references to source documents. Answers without any citation should be flagged."
        ],
    ),
    Guidelines(
        name="completeness",
        guidelines=[
            "The response must address ALL parts of the question completely. For multi-part "
            "questions (e.g., 'what are the steps?', 'list the requirements'), all items must "
            "be covered. A partial answer that omits key requirements, steps, or thresholds "
            "defined in the PEMEX documentation is NOT acceptable. "
            "Verify that specific numbers, limits, and timelines are included when the question asks for them."
        ],
    ),
]

print(f"Scorers: {[s.name if hasattr(s, 'name') else getattr(s, '__name__', str(s)) for s in SCORERS]}")

# COMMAND ----------

# DBTITLE 1,Register V1 header
# MAGIC %md
# MAGIC ### Register V1 Instructions
# MAGIC
# MAGIC Ensures the V1 alias exists in the Prompt Registry.
# MAGIC This cell is **idempotent** — safe to re-run.

# COMMAND ----------

# DBTITLE 1,Register V1 instructions
V1_INSTRUCTIONS = (
    "You are a helpful assistant for PEMEX employees. "
    "Answer questions about PEMEX procedures and policies."
)

v1_version = mlflow.genai.register_prompt(
    name=INSTRUCTIONS_REGISTRY_NAME,
    template=V1_INSTRUCTIONS,
    commit_message="V1: minimal baseline instructions — no citation, scope, or format guidance",
)
mlflow.genai.set_prompt_alias(
    name=INSTRUCTIONS_REGISTRY_NAME,
    alias="v1",
    version=v1_version.version,
)

print(f"V1 instructions: '{INSTRUCTIONS_REGISTRY_FQN}@v1' (version {v1_version.version})")
print(V1_INSTRUCTIONS)
