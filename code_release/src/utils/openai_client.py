from __future__ import annotations

import os
from urllib.parse import urlparse


def normalize_base_url(raw_url: str | None) -> str | None:
    """
    Normalize endpoint-style URLs to OpenAI SDK base_url.

    Supports both:
    - https://host/v1
    - https://host/v1/chat/completions
    """
    if not raw_url:
        return None

    url = raw_url.strip().rstrip("/")

    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return raw_url

    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def create_openai_client():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    raw_base = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    base_url = normalize_base_url(raw_base)

    timeout_env = os.getenv("REQUEST_TIMEOUT", "120")
    max_retries_env = os.getenv("MAX_RETRIES", "1")

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    try:
        kwargs["timeout"] = int(timeout_env)
    except ValueError:
        pass
    try:
        kwargs["max_retries"] = int(max_retries_env)
    except ValueError:
        pass

    return OpenAI(**kwargs)


def get_default_model(default: str = "deepseek-v3.2") -> str:
    return os.getenv("OPENAI_MODEL", default)
