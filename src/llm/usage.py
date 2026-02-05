"""Token counting and cost tracking for LLM usage."""

from dataclasses import dataclass, field
from typing import Dict, Any

from .protocol import LLMClient


@dataclass
class TokenUsage:
    """Token usage information for an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost(self) -> float:
        """Calculate cost (placeholder, needs rates)."""
        return 0.0  # Will be calculated by CostTracker


@dataclass
class LLMResponse:
    """Extended response including usage information."""

    content: str
    usage: TokenUsage | None = None
    tool_calls: list[dict] | None = None
    stop_reason: str | None = None


class CostTracker:
    """Tracks costs for LLM usage across different models."""

    # Cost rates in USD per 1K tokens (example rates, should be configurable)
    COST_RATES: Dict[str, Dict[str, float]] = {
        "anthropic": {
            "input": 0.015,  # $0.015 per 1K input tokens
            "output": 0.075,  # $0.075 per 1K output tokens
        },
        "openai": {
            "input": 0.002,  # $0.002 per 1K input tokens
            "output": 0.002,  # $0.002 per 1K output tokens
        },
        "mock": {
            "input": 0.001,  # Test rate
            "output": 0.001,  # Test rate
        },
    }

    def __init__(self) -> None:
        self.total_cost: float = 0.0
        self.usage_by_model: Dict[str, TokenUsage] = {}

    def calculate_cost(self, provider: str, usage: TokenUsage) -> float:
        """Calculate cost for given usage."""
        # Handle model keys that include provider
        if "/" in provider:
            provider = provider.split("/")[0]

        # Special case for mock
        if provider == "mock-model":
            provider = "mock"

        if provider not in self.COST_RATES:
            return 0.0  # Unknown provider

        rates = self.COST_RATES[provider]
        input_cost = (usage.input_tokens / 1000) * rates.get("input", 0)
        output_cost = (usage.output_tokens / 1000) * rates.get("output", 0)
        return input_cost + output_cost

    def record_usage(self, model_key: str, usage: TokenUsage) -> float:
        """Record usage and return cost."""
        if model_key not in self.usage_by_model:
            self.usage_by_model[model_key] = TokenUsage()

        existing = self.usage_by_model[model_key]
        existing.input_tokens += usage.input_tokens
        existing.output_tokens += usage.output_tokens
        existing.total_tokens += usage.total_tokens

        cost = self.calculate_cost(model_key.split("/")[0], usage)
        self.total_cost += cost
        return cost

    def get_total_cost(self) -> float:
        """Get total accumulated cost."""
        return self.total_cost

    def get_usage_summary(self) -> Dict[str, TokenUsage]:
        """Get usage summary by model."""
        return self.usage_by_model.copy()

    def reset(self) -> None:
        """Reset all tracking."""
        self.total_cost = 0.0
        self.usage_by_model.clear()


class UsageTrackingClient:
    """Wrapper LLM client that tracks token usage and costs."""

    def __init__(self, client: LLMClient, cost_tracker: CostTracker | None = None):
        self._client = client
        self._cost_tracker = cost_tracker or CostTracker()
        self._current_model = (
            client.get_model() if hasattr(client, "get_model") else "unknown"
        )

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """Complete with usage tracking."""
        result = self._client.complete(messages, **kwargs)
        # For now, mock usage (in real implementation, get from API response)
        usage = TokenUsage(
            input_tokens=100,  # Placeholder
            output_tokens=50,  # Placeholder
            total_tokens=150,
        )
        self._cost_tracker.record_usage(self._current_model, usage)
        return result

    def complete_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs: Any
    ) -> dict:
        """Complete with tools and usage tracking."""
        result = self._client.complete_with_tools(messages, tools, **kwargs)
        # Mock usage
        usage = TokenUsage(input_tokens=120, output_tokens=60, total_tokens=180)
        self._cost_tracker.record_usage(self._current_model, usage)
        return result

    def set_model(self, model: str) -> None:
        """Set model and update tracking."""
        if hasattr(self._client, "set_model"):
            self._client.set_model(model)
        self._current_model = model

    def get_model(self) -> str:
        """Get current model."""
        return self._current_model

    def get_cost_tracker(self) -> CostTracker:
        """Get the cost tracker."""
        return self._cost_tracker
