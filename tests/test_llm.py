"""Tests for the centralized LLM client."""

from unittest.mock import MagicMock, patch
from weatheredge.llm import get_llm_client, chat_completion, OPENROUTER_BASE_URL


def test_get_llm_client_uses_openrouter_base_url():
    with patch("weatheredge.llm.openai.OpenAI") as mock_cls:
        get_llm_client("test-key")
        mock_cls.assert_called_once_with(base_url=OPENROUTER_BASE_URL, api_key="test-key")


def test_chat_completion_returns_content():
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello from LLM"
    client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    result = chat_completion(client, "anthropic/claude-sonnet-4-20250514", "Say hello")
    assert result == "Hello from LLM"
    client.chat.completions.create.assert_called_once()


def test_chat_completion_passes_model_and_prompt():
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    chat_completion(client, "anthropic/claude-haiku-4-5-20251001", "Test prompt", max_tokens=200)

    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "anthropic/claude-haiku-4-5-20251001"
    assert call_kwargs["max_tokens"] == 200
    assert call_kwargs["messages"][0]["content"] == "Test prompt"


def test_chat_completion_returns_none_on_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("API down")

    result = chat_completion(client, "model", "prompt")
    assert result is None
