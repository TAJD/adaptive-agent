"""Protocol definition for LLM clients."""

from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM chat completion clients."""

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """
        Send messages and get completion text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Provider-specific options (model, temperature, etc.)

        Returns:
            The completion text response.
        """
        pass

    @abstractmethod
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
            tools: List of tool definitions in provider-agnostic format.
            **kwargs: Provider-specific options.

        Returns:
            Dict with:
                - "content": Text response (may be empty)
                - "tool_calls": List of tool call dicts with "name" and "arguments"
                - "stop_reason": Why generation stopped ("tool_use", "end_turn", etc.)
        """
        pass

    @abstractmethod
    def set_model(self, model: str) -> None:
        """
        Set the default model for this client.

        Args:
            model: The model identifier to use as default.
        """
        pass

    @abstractmethod
    def get_model(self) -> str:
        """
        Get the current default model for this client.

        Returns:
            The current default model identifier.
        """
        pass
