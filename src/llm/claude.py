"""Claude client implementation using the Anthropic SDK."""

import os
import time
from typing import Any

import anthropic

from .protocol import LLMClient


class ClaudeClient(LLMClient):
    """
    LLM client for Claude via the Anthropic API.

    Supports both simple completions and tool use.
    """

    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the Claude client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model to use. Defaults to claude-sonnet-4-5-20250929.
            max_tokens: Maximum tokens in response.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API key required. Pass api_key or set ANTHROPIC_API_KEY env var."
            )

        self._client = anthropic.Anthropic(api_key=self._api_key)
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens

    def set_model(self, model: str) -> None:
        """Set the default model for this client."""
        self._model = model

    def get_model(self) -> str:
        """Get the current default model."""
        return self._model

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """
        Send messages and get completion text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional options:
                - model: Override default model
                - max_tokens: Override default max tokens
                - system: System prompt
                - temperature: Sampling temperature

        Returns:
            The completion text response.
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        system = kwargs.pop("system", None)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._normalize_messages(messages),
        }

        if system:
            create_kwargs["system"] = system

        # Pass through other kwargs like temperature
        create_kwargs.update(kwargs)

        response = self._call_with_retry(create_kwargs)
        return self._extract_text(response)

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """
        Send messages with tool definitions, get completion with tool calls.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: List of tool definitions. Each tool should have:
                - name: Tool name
                - description: What the tool does
                - parameters: JSON Schema for parameters (or input_schema)
            **kwargs: Additional options (model, max_tokens, system, temperature).

        Returns:
            Dict with:
                - "content": Text response (may be empty)
                - "tool_calls": List of tool call dicts with "id", "name", "arguments"
                - "stop_reason": Why generation stopped
        """
        model = kwargs.pop("model", self._model)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        system = kwargs.pop("system", None)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._normalize_messages(messages),
            "tools": self._normalize_tools(tools),
        }

        if system:
            create_kwargs["system"] = system

        create_kwargs.update(kwargs)

        response = self._call_with_retry(create_kwargs)
        return self._parse_tool_response(response)

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages to Anthropic format."""
        normalized = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map 'system' role - Anthropic handles system separately
            if role == "system":
                continue  # System prompts handled via system parameter

            # Map assistant tool calls
            if role == "assistant" and "tool_calls" in msg:
                content_blocks = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"call_{len(content_blocks)}"),
                            "name": tc["name"],
                            "input": tc.get("arguments", tc.get("input", {})),
                        }
                    )
                normalized.append({"role": "assistant", "content": content_blocks})
                continue

            # Map tool results
            if role == "tool":
                normalized.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", "unknown"),
                                "content": str(msg.get("content", "")),
                            }
                        ],
                    }
                )
                continue

            normalized.append({"role": role, "content": content})

        return normalized

    def _normalize_tools(self, tools: list[dict]) -> list[dict]:
        """Normalize tool definitions to Anthropic format."""
        normalized = []
        for tool in tools:
            normalized.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get(
                        "input_schema",
                        tool.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    ),
                }
            )
        return normalized

    def _call_with_retry(self, create_kwargs: dict) -> Any:
        """Call the API with retry logic for transient errors."""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                return self._client.messages.create(**create_kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
            except anthropic.APIConnectionError as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
            except anthropic.APIStatusError as e:
                # Don't retry client errors (4xx except rate limit)
                if 400 <= e.status_code < 500 and e.status_code != 429:
                    raise
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        raise last_error  # type: ignore

    def _extract_text(self, response: Any) -> str:
        """Extract text content from API response."""
        texts = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
        return "\n".join(texts)

    def _parse_tool_response(self, response: Any) -> dict:
        """Parse API response into normalized tool response format."""
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return {
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": response.stop_reason,
        }
