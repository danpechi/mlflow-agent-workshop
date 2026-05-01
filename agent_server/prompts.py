"""System prompts for the security alert triage agent.

Uses MLflow Prompt Registry for versioned prompt management.
The AGENT_PROMPT_VERSION env var maps to a registry alias (e.g. "v1", "optimized").
Falls back to the inline V1 prompt if the registry is unavailable.
"""

import logging
import os

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_V1 = """You are a helpful assistant. Look at the alert and tell me what you think."""

PROMPT_REGISTRY_NAME = os.getenv("PROMPT_REGISTRY_NAME", "")
PROMPT_ALIAS = os.getenv("AGENT_PROMPT_VERSION", "v1")


def _load_prompt() -> str:
    if PROMPT_REGISTRY_NAME:
        try:
            import mlflow
            prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_REGISTRY_NAME}@{PROMPT_ALIAS}")
            logger.info("Loaded prompt from registry: %s@%s", PROMPT_REGISTRY_NAME, PROMPT_ALIAS)
            return prompt.template
        except Exception as e:
            logger.warning("Failed to load prompt from registry: %s. Using V1 fallback.", e)
    return SYSTEM_PROMPT_V1


SYSTEM_PROMPT = _load_prompt()
