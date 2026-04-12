"""Centralized LLM client using OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import logging

import openai

log = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_client(api_key: str) -> openai.OpenAI:
    """Create an OpenAI client pointed at OpenRouter."""
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def chat_completion(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 500,
) -> str | None:
    """Send a chat completion request. Returns content string or None on error."""
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception:
        log.warning("LLM API call failed", exc_info=True)
        return None
