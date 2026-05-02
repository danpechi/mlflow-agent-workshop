import logging
import os
from typing import Any, AsyncGenerator, Sequence, TypedDict

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)
from typing_extensions import Annotated

from agent_server.prompts import SYSTEM_PROMPT
from agent_server.tools import triage_tools
from agent_server.utils import (
    _get_or_create_thread_id,
    process_agent_astream_events,
)

logger = logging.getLogger(__name__)
mlflow.langchain.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

# Pin traces to a specific MLflow experiment when MLFLOW_EXPERIMENT_NAME is set.
# Without this, the deployed app logs to the App SP's default experiment, which
# is separate from the workshop eval experiment.
_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
if _EXPERIMENT_NAME:
    try:
        mlflow.set_experiment(_EXPERIMENT_NAME)
        logger.info("MLflow experiment pinned to %s", _EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("Could not set MLflow experiment %s: %s", _EXPERIMENT_NAME, e)

# LLM endpoint is configurable via the LLM_ENDPOINT_NAME env var (set in
# app.yaml). The default mirrors the widget default in notebooks/00_config.py.
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5")
PROMPT_ALIAS = os.getenv("AGENT_PROMPT_VERSION", "v1")
DATABRICKS_HOST = (os.getenv("DATABRICKS_HOST") or "").rstrip("/")


def _build_trace_url(trace_id: str) -> str | None:
    """Build a deep link to the trace in the Databricks MLflow UI."""
    if not trace_id or not DATABRICKS_HOST:
        return None
    host = DATABRICKS_HOST if DATABRICKS_HOST.startswith("http") else f"https://{DATABRICKS_HOST}"
    try:
        if _EXPERIMENT_NAME:
            exp = mlflow.get_experiment_by_name(_EXPERIMENT_NAME)
            if exp:
                return f"{host}/ml/experiments/{exp.experiment_id}/traces?selectedTraceId={trace_id}"
    except Exception:
        pass
    return f"{host}/ml/traces/{trace_id}"


def _capture_trace_id() -> str | None:
    """Capture trace ID using the correct API for the current execution context.

    get_last_active_trace_id() returns None when called INSIDE an active trace
    (e.g., within @invoke()). Use get_current_active_span() first — it returns
    the span from the current context, which has the trace_id we need.
    Fall back to get_last_active_trace_id(thread_local=True) for cases where
    the trace has already completed.
    """
    # Strategy 1: Get trace ID from the currently active span (works INSIDE a trace)
    try:
        span = mlflow.get_current_active_span()
        if span and hasattr(span, "trace_id") and span.trace_id:
            return span.trace_id
    except Exception:
        pass

    # Strategy 2: Get the last completed trace (works AFTER a trace finishes)
    try:
        fn = getattr(mlflow, "get_last_active_trace_id", None)
        if callable(fn):
            trace_id = fn(thread_local=True)
            if trace_id:
                return trace_id
    except Exception:
        pass

    return None


_checkpointer = MemorySaver()
_store = InMemoryStore()


class StatefulAgentState(TypedDict, total=False):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    remaining_steps: int
    custom_inputs: dict[str, Any]
    custom_outputs: dict[str, Any]


async def init_agent():
    tools = triage_tools()
    model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer,
        store=_store,
        state_schema=StatefulAgentState,
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in stream_handler(request)
        if event.type == "response.output_item.done"
    ]

    # Surface the trace ID + a deep-link to the workshop UI so the user can jump
    # straight to the trace in MLflow without searching.
    custom_outputs: dict[str, Any] = {"prompt_alias": PROMPT_ALIAS}
    try:
        trace_id = _capture_trace_id()
        if trace_id:
            custom_outputs["trace_id"] = trace_id
            url = _build_trace_url(trace_id)
            if url:
                custom_outputs["trace_url"] = url
        if _EXPERIMENT_NAME:
            custom_outputs["experiment_name"] = _EXPERIMENT_NAME
    except Exception as e:
        logger.warning("trace metadata capture failed: %s", e)

    return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    thread_id = _get_or_create_thread_id(request)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    input_state: dict[str, Any] = {
        "messages": to_chat_completions_input([i.model_dump() for i in request.input]),
        "custom_inputs": dict(request.custom_inputs or {}),
    }

    agent = await init_agent()
    async for event in process_agent_astream_events(
        agent.astream(input_state, config, stream_mode=["updates", "messages"])
    ):
        yield event

    # Tag the trace with session ID AFTER the agent has run (trace now exists)
    try:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})
    except Exception:
        pass
