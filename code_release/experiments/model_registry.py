#!/usr/bin/env python3
"""Formal model registry helpers.

Keeps paper runs off ad hoc OPENAI_* edits by translating a model alias into
the legacy environment variables used by the current runners.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.utils.openai_client import normalize_base_url


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "experiments" / "config" / "model_endpoints.local.json"


def load_env_file(*, override: bool = False) -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path).expanduser().resolve()
    return json.loads(registry_path.read_text(encoding="utf-8"))


def registry_by_alias(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(entry["alias"]): entry for entry in registry.get("models", [])}


def get_model_entry(alias: str, registry: dict[str, Any]) -> dict[str, Any]:
    entries = registry_by_alias(registry)
    if alias not in entries:
        raise KeyError(f"model alias not found in registry: {alias}")
    entry = entries[alias]
    if not bool(entry.get("enabled", True)):
        raise RuntimeError(f"model alias is disabled: {alias}")
    return entry


def masked(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def safe_run_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    return cleaned.strip("._-") or "unknown"


def env_overlay_for_alias(
    alias: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    request_timeout_s: int | None = None,
    max_retries: int | None = None,
) -> dict[str, str]:
    load_env_file(override=False)
    registry = load_registry(registry_path)
    entry = get_model_entry(alias, registry)

    api_key = os.getenv(str(entry["api_key_env"]), "")
    raw_base_url = os.getenv(str(entry["base_url_env"]), "")
    base_url = normalize_base_url(raw_base_url) or ""
    if not api_key:
        raise RuntimeError(f"missing secret env for {alias}: {entry['api_key_env']}")
    if not base_url:
        raise RuntimeError(f"missing base-url env for {alias}: {entry['base_url_env']}")

    overlay = {
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_BASE": base_url,
        "OPENAI_MODEL": str(entry["model_id"]),
        "EXP_MODEL_ALIAS": alias,
        "EXP_MODEL_ID": str(entry["model_id"]),
        "EXP_MODEL_ROLE": str(entry.get("role", "")),
    }
    if request_timeout_s is not None:
        overlay["REQUEST_TIMEOUT"] = str(request_timeout_s)
    if max_retries is not None:
        overlay["MAX_RETRIES"] = str(max_retries)
    return overlay


def endpoint_summary_for_alias(alias: str, registry_path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    load_env_file(override=False)
    registry = load_registry(registry_path)
    entry = get_model_entry(alias, registry)
    api_key = os.getenv(str(entry["api_key_env"]), "")
    raw_base_url = os.getenv(str(entry["base_url_env"]), "")
    return {
        "alias": alias,
        "model_id": entry["model_id"],
        "role": entry.get("role", ""),
        "api_key_env": entry["api_key_env"],
        "api_key_present": bool(api_key),
        "api_key_masked": masked(api_key),
        "base_url_env": entry["base_url_env"],
        "base_url_present": bool(raw_base_url),
        "base_url_normalized": normalize_base_url(raw_base_url),
    }
