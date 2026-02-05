"""Mock LLM client for deterministic testing."""

from typing import Any

from .protocol import LLMClient


class MockLLMClient(LLMClient):
    """
    Mock LLM client that returns predetermined responses.

    Useful for testing without actual API calls.
    """

    def __init__(
        self, responses: list[str | dict] | None = None, model: str = "mock-model"
    ) -> None:
        """
        Initialize with predetermined responses.

        Args:
            responses: List of responses to return in order.
                Can be strings (for complete()) or dicts (for complete_with_tools()).
                Cycles through responses if more calls are made than responses provided.
            model: Default model identifier.
        """
        self._responses = responses or ["Mock response"]
        self._call_index = 0
        self._call_history: list[dict] = []
        self._model = model

    def get_model(self) -> str:
        """Get the current model."""
        return self._model

    def add_response(self, response: str | dict) -> None:
        """Add a response to the queue."""
        self._responses.append(response)

    def set_responses(self, responses: list[str | dict]) -> None:
        """Set the response queue."""
        self._responses = responses
        self._call_index = 0

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """Return next predetermined string response."""
        self._call_history.append(
            {
                "method": "complete",
                "messages": messages,
                "kwargs": kwargs,
            }
        )

        response = self._get_next_response()

        if isinstance(response, dict):
            return response.get("content", "")
        return str(response)

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> dict:
        """Return next predetermined tool response."""
        self._call_history.append(
            {
                "method": "complete_with_tools",
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )

        response = self._get_next_response()

        if isinstance(response, dict):
            return {
                "content": response.get("content", ""),
                "tool_calls": response.get("tool_calls", []),
                "stop_reason": response.get("stop_reason", "end_turn"),
            }

        return {
            "content": str(response),
            "tool_calls": [],
            "stop_reason": "end_turn",
        }

    def _get_next_response(self) -> str | dict:
        """Get the next response, cycling if needed."""
        if not self._responses:
            return "Mock response"

        response = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        return response

    def get_call_history(self) -> list[dict]:
        """Get the history of all calls made."""
        return self._call_history.copy()

    def get_call_count(self) -> int:
        """Get the total number of calls made."""
        return len(self._call_history)

    def reset(self) -> None:
        """Reset call index and history."""
        self._call_index = 0
        self._call_history.clear()

    def set_model(self, model: str) -> None:
        """Set the default model for this client."""
        self._model = model

    def get_model(self) -> str:
        """Get the current default model."""
        return self._model
