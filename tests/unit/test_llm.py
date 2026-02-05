"""Tests for LLM clients and executor."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.types import Task, ExecutionResult
from src.llm.mock import MockLLMClient
from src.llm.claude import ClaudeClient
from src.executor.llm import LLMExecutor
from src.executor.code_runner import CodeRunner


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_complete_returns_string_response(self) -> None:
        """Test complete() returns string responses."""
        mock = MockLLMClient(["response one", "response two"])

        result1 = mock.complete([{"role": "user", "content": "test"}])
        result2 = mock.complete([{"role": "user", "content": "test"}])

        assert result1 == "response one"
        assert result2 == "response two"

    def test_complete_cycles_responses(self) -> None:
        """Test responses cycle when exhausted."""
        mock = MockLLMClient(["only one"])

        result1 = mock.complete([{"role": "user", "content": "a"}])
        result2 = mock.complete([{"role": "user", "content": "b"}])

        assert result1 == "only one"
        assert result2 == "only one"

    def test_complete_with_tools_returns_dict(self) -> None:
        """Test complete_with_tools() returns proper dict structure."""
        mock = MockLLMClient(
            [
                {
                    "content": "I'll use the tool",
                    "tool_calls": [{"name": "calc", "arguments": {"x": 1}}],
                    "stop_reason": "tool_use",
                }
            ]
        )

        result = mock.complete_with_tools(
            [{"role": "user", "content": "test"}],
            [{"name": "calc", "description": "Calculator"}],
        )

        assert result["content"] == "I'll use the tool"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "calc"

    def test_set_and_get_model(self) -> None:
        """Test setting and getting the model."""
        mock = MockLLMClient()
        assert mock.get_model() == "mock-model"

        mock.set_model("new-model")
        assert mock.get_model() == "new-model"

    def test_complete_with_tools_string_response(self) -> None:
        """Test complete_with_tools() handles string responses."""
        mock = MockLLMClient(["Just a string"])

        result = mock.complete_with_tools(
            [{"role": "user", "content": "test"}],
            [{"name": "calc", "description": "Calculator"}],
        )

        assert result["content"] == "Just a string"
        assert result["tool_calls"] == []
        assert result["stop_reason"] == "end_turn"

    def test_tracks_call_history(self) -> None:
        """Test call history is recorded."""
        mock = MockLLMClient(["response"])

        mock.complete([{"role": "user", "content": "hello"}], temperature=0.5)
        mock.complete_with_tools(
            [{"role": "user", "content": "hi"}],
            [{"name": "tool"}],
        )

        history = mock.get_call_history()
        assert len(history) == 2
        assert history[0]["method"] == "complete"
        assert history[0]["messages"] == [{"role": "user", "content": "hello"}]
        assert history[0]["kwargs"]["temperature"] == 0.5
        assert history[1]["method"] == "complete_with_tools"

    def test_get_call_count(self) -> None:
        """Test call count tracking."""
        mock = MockLLMClient(["r"])

        assert mock.get_call_count() == 0
        mock.complete([{"role": "user", "content": "a"}])
        assert mock.get_call_count() == 1
        mock.complete([{"role": "user", "content": "b"}])
        assert mock.get_call_count() == 2

    def test_reset_clears_state(self) -> None:
        """Test reset clears index and history."""
        mock = MockLLMClient(["first", "second"])

        mock.complete([{"role": "user", "content": "a"}])
        mock.reset()

        assert mock.get_call_count() == 0
        assert mock.get_call_history() == []
        assert mock.complete([{"role": "user", "content": "b"}]) == "first"

    def test_add_response(self) -> None:
        """Test adding responses dynamically."""
        mock = MockLLMClient(["initial"])

        mock.add_response("added")

        mock.complete([{"role": "user", "content": "1"}])
        result = mock.complete([{"role": "user", "content": "2"}])

        assert result == "added"

    def test_set_responses(self) -> None:
        """Test replacing all responses."""
        mock = MockLLMClient(["old"])

        mock.set_responses(["new1", "new2"])

        assert mock.complete([{"role": "user", "content": "a"}]) == "new1"
        assert mock.complete([{"role": "user", "content": "b"}]) == "new2"


class TestLLMExecutor:
    """Tests for LLMExecutor."""

    @pytest.fixture
    def simple_task(self) -> Task:
        return Task(id="test-1", query="Calculate 2+2", expected_answer=4)

    def test_executes_generated_code(self, simple_task: Task) -> None:
        """Test executor runs LLM-generated code."""
        mock_llm = MockLLMClient(["```python\nresult = 2 + 2\n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output == 4
        assert result.code_generated == "result = 2 + 2"

    def test_extracts_code_from_python_block(self, simple_task: Task) -> None:
        """Test code extraction from ```python blocks."""
        mock_llm = MockLLMClient(
            ["Here's the solution:\n```python\nresult = 42\n```\nThat's it!"]
        )
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output == 42
        assert result.code_generated == "result = 42"

    def test_extracts_code_from_generic_block(self, simple_task: Task) -> None:
        """Test code extraction from generic ``` blocks."""
        mock_llm = MockLLMClient(["```\nresult = 123\n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output == 123

    def test_handles_no_code_block(self, simple_task: Task) -> None:
        """Test handling when LLM returns no code block."""
        mock_llm = MockLLMClient(["Just some text without code"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output is None
        assert result.code_generated is None
        assert "No code block" in result.metadata.get("error", "")

    def test_handles_execution_error(self, simple_task: Task) -> None:
        """Test handling when code execution fails."""
        mock_llm = MockLLMClient(["```python\nraise ValueError('oops')\n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output is None
        assert "ValueError" in result.metadata.get("error", "")

    def test_handles_syntax_error(self, simple_task: Task) -> None:
        """Test handling syntax errors in generated code."""
        mock_llm = MockLLMClient(["```python\nresult = \n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert result.output is None
        assert "SyntaxError" in result.metadata.get("error", "")

    def test_passes_hints_to_prompt(self, simple_task: Task) -> None:
        """Test hints are included in prompt."""
        mock_llm = MockLLMClient(["```python\nresult = 4\n```"])
        executor = LLMExecutor(mock_llm)

        executor.execute(simple_task, {"hints": ["Use addition"]})

        history = mock_llm.get_call_history()
        prompt = history[0]["messages"][0]["content"]
        assert "Hints:" in prompt
        assert "Use addition" in prompt

    def test_passes_constraints_to_prompt(self, simple_task: Task) -> None:
        """Test constraints are included in prompt."""
        mock_llm = MockLLMClient(["```python\nresult = 4\n```"])
        executor = LLMExecutor(mock_llm)

        executor.execute(simple_task, {"constraints": ["No loops"]})

        history = mock_llm.get_call_history()
        prompt = history[0]["messages"][0]["content"]
        assert "Constraints:" in prompt
        assert "No loops" in prompt

    def test_passes_examples_to_prompt(self, simple_task: Task) -> None:
        """Test examples are included in prompt."""
        mock_llm = MockLLMClient(["```python\nresult = 4\n```"])
        executor = LLMExecutor(mock_llm)

        executor.execute(simple_task, {"examples": ["result = 1 + 1"]})

        history = mock_llm.get_call_history()
        prompt = history[0]["messages"][0]["content"]
        assert "Examples" in prompt
        assert "result = 1 + 1" in prompt

    def test_injects_data_into_globals(self, simple_task: Task) -> None:
        """Test data from context is available in execution."""
        mock_llm = MockLLMClient(["```python\nresult = sum(data)\n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {"data": [1, 2, 3, 4]})

        assert result.output == 10

    def test_records_trajectory(self, simple_task: Task) -> None:
        """Test execution trajectory is recorded."""
        mock_llm = MockLLMClient(["```python\nresult = 1\n```"])
        executor = LLMExecutor(mock_llm)

        result = executor.execute(simple_task, {})

        assert len(result.trajectory) >= 3
        steps = [t["step"] for t in result.trajectory]
        assert "prompt_built" in steps
        assert "llm_response" in steps
        assert "code_extracted" in steps
        assert "code_executed" in steps

    def test_uses_custom_system_prompt(self, simple_task: Task) -> None:
        """Test custom system prompt is used."""
        mock_llm = MockLLMClient(["```python\nresult = 1\n```"])
        executor = LLMExecutor(mock_llm, system_prompt="Custom prompt")

        executor.execute(simple_task, {})

        history = mock_llm.get_call_history()
        assert history[0]["kwargs"].get("system") == "Custom prompt"

    def test_uses_custom_code_runner(self, simple_task: Task) -> None:
        """Test custom CodeRunner is used."""
        mock_llm = MockLLMClient(["```python\nresult = numpy.array([1])\n```"])
        runner = CodeRunner(allowed_modules=["numpy"])
        executor = LLMExecutor(mock_llm, code_runner=runner)

        result = executor.execute(simple_task, {})

        # Should have access to numpy
        assert result.output is not None or "numpy" in str(result.metadata)


class TestClaudeClient:
    """Tests for ClaudeClient with mocked API."""

    def test_init_requires_api_key(self) -> None:
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                ClaudeClient(api_key=None)

    def test_init_uses_env_var(self) -> None:
        """Test initialization uses ANTHROPIC_API_KEY env var."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic"):
                client = ClaudeClient()
                assert client._api_key == "test-key"

    def test_init_prefers_explicit_key(self) -> None:
        """Test explicit API key takes precedence."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("anthropic.Anthropic"):
                client = ClaudeClient(api_key="explicit-key")
                assert client._api_key == "explicit-key"

    def test_complete_calls_api(self) -> None:
        """Test complete() calls the Anthropic API correctly."""
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hello!")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            result = client.complete([{"role": "user", "content": "Hi"}])

            assert result == "Hello!"
            mock_client.messages.create.assert_called_once()

    def test_complete_passes_system_prompt(self) -> None:
        """Test system prompt is passed to API."""
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            client.complete([{"role": "user", "content": "Hi"}], system="Be helpful")

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["system"] == "Be helpful"

    def test_complete_with_tools_parses_response(self) -> None:
        """Test complete_with_tools() parses tool calls."""
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Using tool"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "calculator"
        tool_block.input = {"x": 5}

        mock_response = Mock()
        mock_response.content = [text_block, tool_block]
        mock_response.stop_reason = "tool_use"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            result = client.complete_with_tools(
                [{"role": "user", "content": "Calculate"}],
                [{"name": "calculator", "description": "Math"}],
            )

            assert result["content"] == "Using tool"
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["name"] == "calculator"
            assert result["tool_calls"][0]["arguments"] == {"x": 5}
            assert result["stop_reason"] == "tool_use"

    def test_normalizes_tool_definitions(self) -> None:
        """Test tool definitions are normalized to Anthropic format."""
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Ok")]
        mock_response.stop_reason = "end_turn"

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            client.complete_with_tools(
                [{"role": "user", "content": "Hi"}],
                [
                    {
                        "name": "tool",
                        "description": "A tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "number"}},
                        },
                    }
                ],
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            tools = call_kwargs["tools"]
            assert tools[0]["name"] == "tool"
            assert "input_schema" in tools[0]

    def test_retries_on_rate_limit(self) -> None:
        """Test retry logic on rate limit errors."""
        import anthropic as anthropic_module

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Success")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            # Fail twice, then succeed
            mock_client.messages.create.side_effect = [
                anthropic_module.RateLimitError(
                    message="Rate limited",
                    response=Mock(status_code=429),
                    body=None,
                ),
                mock_response,
            ]
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            client.RETRY_DELAY = 0.01  # Speed up test

            result = client.complete([{"role": "user", "content": "Hi"}])

            assert result == "Success"
            assert mock_client.messages.create.call_count == 2

    def test_normalizes_tool_result_messages(self) -> None:
        """Test tool result messages are normalized."""
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Done")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key")
            client.complete(
                [
                    {"role": "user", "content": "Use tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "call_1", "name": "calc", "arguments": {}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "42"},
                ]
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            messages = call_kwargs["messages"]

            # Tool result should be converted to user message with tool_result block
            tool_result_msg = messages[-1]
            assert tool_result_msg["role"] == "user"
            assert tool_result_msg["content"][0]["type"] == "tool_result"

    def test_set_and_get_model(self) -> None:
        """Test setting and getting the model."""
        with patch("anthropic.Anthropic"):
            client = ClaudeClient(api_key="test-key")
            assert client.get_model() == ClaudeClient.DEFAULT_MODEL

            client.set_model("claude-sonnet-3.5")
            assert client.get_model() == "claude-sonnet-3.5"

    def test_model_override_in_complete(self) -> None:
        """Test model can be overridden per call."""
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Response")]

        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            client = ClaudeClient(api_key="test-key", model="default-model")
            client.complete([{"role": "user", "content": "Hi"}], model="override-model")

            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "override-model"
