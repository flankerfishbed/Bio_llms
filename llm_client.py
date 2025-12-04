from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Central model registry: map logical role keys to OpenRouter model names.
# These are easy to change later or expose via the UI if desired.
MODEL_REGISTRY: Dict[str, str] = {
    "BIO_SUMMARISER_MODEL": "anthropic/claude-3.5-sonnet",
    "BIO_METHODS_REVIEWER_MODEL": "anthropic/claude-3.5-sonnet",
    "BIO_STATS_CHECKER_MODEL": "anthropic/claude-3.5-sonnet",
    "BIO_NEXT_EXPERIMENTS_MODEL": "anthropic/claude-3.5-sonnet",
    "BIO_CLASSIFIER_MODEL": "anthropic/claude-3.5-sonnet",
    "BIO_JUDGE_MODEL": "openai/gpt-4.1",
}


class OpenRouterError(Exception):
    """Custom error type for OpenRouter-related issues."""


def get_openrouter_api_key(raise_on_missing: bool = True) -> Optional[str]:
    """Fetch the OpenRouter API key from the environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key and raise_on_missing:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Please set it in your environment."
        )
    return api_key


def call_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Call the OpenRouter chat completions API.

    Returns a dict with:
        - text: assistant message content (or empty string on failure)
        - model: model identifier actually used (if available)
        - usage: token usage information (if available)
        - error: error message (if any)
    """
    api_key = get_openrouter_api_key(raise_on_missing=False)
    if not api_key:
        return {
            "text": "",
            "model": model,
            "usage": None,
            "error": "Missing OPENROUTER_API_KEY.",
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended headers for OpenRouter
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Biological Paper Critic"),
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {
            "text": "",
            "model": model,
            "usage": None,
            "error": f"HTTP error calling OpenRouter: {exc}",
        }

    try:
        data = response.json()
    except ValueError as exc:  # JSONDecodeError is a subclass
        return {
            "text": "",
            "model": model,
            "usage": None,
            "error": f"Failed to parse OpenRouter response JSON: {exc}",
        }

    # Expected OpenAI-compatible schema
    try:
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        used_model = data.get("model", model)
        usage = data.get("usage")
        return {
            "text": content,
            "model": used_model,
            "usage": usage,
            "error": None,
        }
    except (KeyError, IndexError) as exc:
        return {
            "text": "",
            "model": model,
            "usage": None,
            "error": f"Unexpected OpenRouter response structure: {exc}",
        }


def get_model_for_key(model_key: str) -> str:
    """Resolve a logical model key to an actual model identifier."""
    return MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["BIO_SUMMARISER_MODEL"])


