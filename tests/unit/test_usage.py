"""Tests for usage tracking."""

from src.llm.usage import TokenUsage, CostTracker, UsageTrackingClient
from src.llm.mock import MockLLMClient


class TestTokenUsage:
    """Tests for TokenUsage."""

    def test_creation(self) -> None:
        """Test creating TokenUsage."""
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost == 0.0  # Placeholder

    def test_defaults(self) -> None:
        """Test default values."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


class TestCostTracker:
    """Tests for CostTracker."""

    def test_calculate_cost_anthropic(self) -> None:
        """Test cost calculation for Anthropic."""
        tracker = CostTracker()
        usage = TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000)

        cost = tracker.calculate_cost("anthropic", usage)
        # 1000 input tokens * 0.015 / 1000 = 0.015
        # 1000 output tokens * 0.075 / 1000 = 0.075
        # Total: 0.09
        assert cost == 0.09

    def test_calculate_cost_openai(self) -> None:
        """Test cost calculation for OpenAI."""
        tracker = CostTracker()
        usage = TokenUsage(input_tokens=2000, output_tokens=2000, total_tokens=4000)

        cost = tracker.calculate_cost("openai", usage)
        # 2000 * 0.002 / 1000 = 0.004 each
        # Total: 0.008
        assert cost == 0.008

    def test_calculate_cost_unknown_provider(self) -> None:
        """Test cost for unknown provider is 0."""
        tracker = CostTracker()
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)

        cost = tracker.calculate_cost("unknown", usage)
        assert cost == 0.0

    def test_record_usage(self) -> None:
        """Test recording usage and accumulating costs."""
        tracker = CostTracker()

        usage1 = TokenUsage(input_tokens=1000, output_tokens=500)
        cost1 = tracker.record_usage("anthropic/claude", usage1)
        assert cost1 == 0.0525  # 0.015 + 0.0375

        usage2 = TokenUsage(input_tokens=1000, output_tokens=500)
        cost2 = tracker.record_usage("anthropic/claude", usage2)
        assert cost2 == 0.0525

        assert tracker.get_total_cost() == 0.105

        summary = tracker.get_usage_summary()
        assert "anthropic/claude" in summary
        assert summary["anthropic/claude"].input_tokens == 2000
        assert summary["anthropic/claude"].output_tokens == 1000

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = CostTracker()
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        tracker.record_usage("anthropic/model", usage)

        assert tracker.get_total_cost() > 0
        assert tracker.get_usage_summary()

        tracker.reset()
        assert tracker.get_total_cost() == 0.0
        assert not tracker.get_usage_summary()


class TestUsageTrackingClient:
    """Tests for UsageTrackingClient."""

    def test_complete_with_tracking(self) -> None:
        """Test complete with usage tracking."""
        mock_client = MockLLMClient(["Hello"])
        tracker = CostTracker()
        client = UsageTrackingClient(mock_client, tracker)

        result = client.complete([{"role": "user", "content": "Hi"}])

        assert result == "Hello"
        assert tracker.get_total_cost() > 0  # Should have recorded cost
        summary = tracker.get_usage_summary()
        assert len(summary) == 1

    def test_complete_with_tools_tracking(self) -> None:
        """Test complete_with_tools with tracking."""
        mock_client = MockLLMClient(
            [
                {
                    "content": "I'll help",
                    "tool_calls": [{"name": "tool", "arguments": {}}],
                    "stop_reason": "tool_use",
                }
            ]
        )
        tracker = CostTracker()
        client = UsageTrackingClient(mock_client, tracker)

        result = client.complete_with_tools(
            [{"role": "user", "content": "Use tool"}],
            [{"name": "tool", "description": "A tool"}],
        )

        assert result["content"] == "I'll help"
        assert len(result["tool_calls"]) == 1
        assert tracker.get_total_cost() > 0

    def test_set_and_get_model(self) -> None:
        """Test setting and getting model."""
        mock_client = MockLLMClient()
        client = UsageTrackingClient(mock_client)

        assert client.get_model() == "mock-model"

        client.set_model("new-model")
        assert client.get_model() == "new-model"

    def test_get_cost_tracker(self) -> None:
        """Test getting cost tracker."""
        mock_client = MockLLMClient()
        tracker = CostTracker()
        client = UsageTrackingClient(mock_client, tracker)

        assert client.get_cost_tracker() is tracker
