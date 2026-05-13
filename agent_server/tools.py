"""Document retrieval tools for the PEMEX Knowledge Assistant."""

import json
import os
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

# Bundled fixtures path (legacy; DOCS_PATH env var points to the PEMEX documents volume)
_BUNDLED_FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "tool_fixtures.json")


@lru_cache(maxsize=1)
def _load_fixtures() -> dict[str, Any]:
    path = os.getenv("FIXTURES_PATH", _BUNDLED_FIXTURES_PATH)
    # Try env var path first, fall back to bundled
    for candidate in [path, _BUNDLED_FIXTURES_PATH]:
        try:
            with open(candidate) as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Fixtures not found at {path} or {_BUNDLED_FIXTURES_PATH}")


@tool
def lookup_threat_intel(indicator: str) -> dict:
    """Look up threat intelligence for an IP address or file hash.

    Args:
        indicator: An IP address (e.g. '185.220.101.47') or file hash (e.g. SHA-256) to check.

    Returns:
        Threat intel record with reputation, categories, first_seen, and notes.
    """
    fixtures = _load_fixtures()
    data = fixtures.get("threat_intel", {})
    result = data.get(indicator)
    if result:
        return result
    return {"reputation": "unknown", "categories": [], "notes": f"No threat intel data found for {indicator}"}


@tool
def get_user_history(username: str) -> dict:
    """Get behavioral history and risk profile for a user account.

    Args:
        username: The user's email address (e.g. 'j.smith@customer-acme.com').

    Returns:
        User profile with role, typical_hours, last_login, anomaly_score, mfa_enabled, and notes.
    """
    fixtures = _load_fixtures()
    data = fixtures.get("user_history", {})
    result = data.get(username)
    if result:
        return result
    return {"role": "unknown", "anomaly_score": 0.5, "mfa_enabled": None, "notes": f"No user history found for {username}"}


@tool
def get_asset_criticality(hostname: str) -> dict:
    """Get the criticality classification and data sensitivity of a host asset.

    Args:
        hostname: The hostname to look up (e.g. 'PROD-DB-NYC-03').

    Returns:
        Asset record with type, criticality, data_classification, owner, and notes.
    """
    fixtures = _load_fixtures()
    data = fixtures.get("asset_criticality", {})
    result = data.get(hostname)
    if result:
        return result
    return {"criticality": "unknown", "data_classification": "unknown", "notes": f"No asset data found for {hostname}"}


@tool
def search_logs(hostname: str) -> list[dict]:
    """Search recent security logs for a given host.

    Args:
        hostname: The hostname to search logs for (e.g. 'WIN-LAPTOP-4421').

    Returns:
        List of recent log entries with timestamps, event types, and details.
    """
    fixtures = _load_fixtures()
    data = fixtures.get("log_search", {})
    result = data.get(hostname)
    if result:
        return result
    return [{"message": f"No log data found for {hostname}"}]


def triage_tools() -> list:
    """Return the list of triage tools for the agent."""
    return [lookup_threat_intel, get_user_history, get_asset_criticality, search_logs]
