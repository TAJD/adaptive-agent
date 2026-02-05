"""Tests for the LLM-based evaluator."""

import pytest

from src.core.types import Task, ExecutionResult
from src.evaluator.llm import LLMEvaluator, EVALUATION_TOOL


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, tool_response: dict | None = None):
        self.tool_response = tool_response or {
            "is_correct": True,
            "score": 1.0,
            "error_type": "none",
            "feedback": "Correct answer!",
            "reasoning": "The values match.",
        }
        self.calls: list[dict] = []

    def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return "Evaluation complete."

    def complete_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> dict:
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return {
            "content": "",
            "tool_calls": [
                {
                    "name": "submit_evaluation",
                    "arguments": self.tool_response,
                }
            ],
            "stop_reason": "tool_use",
        }


class TestLLMEvaluator:
    """Tests for LLMEvaluator."""

    def test_correct_answer(self) -> None:
        """Test evaluation of correct answer."""
        client = MockLLMClient({
            "is_correct": True,
            "score": 1.0,
            "error_type": "none",
            "feedback": "The answer is correct.",
            "reasoning": "Values match exactly.",
        })
        evaluator = LLMEvaluator(llm_client=client)

        task = Task(id="t1", query="What is 2+2?", expected_answer=4)
        result = ExecutionResult(output=4, code_generated="result = 2 + 2")

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True
        assert evaluation.score == 1.0
        assert evaluation.error_type is None
        assert "correct" in evaluation.feedback.lower()

    def test_incorrect_answer_with_error_type(self) -> None:
        """Test evaluation of incorrect answer with error classification."""
        client = MockLLMClient({
            "is_correct": False,
            "score": 0.3,
            "error_type": "filter_error",
            "feedback": "Missing time period filter.",
            "reasoning": "The query should filter by Q1 2023.",
        })
        evaluator = LLMEvaluator(llm_client=client)

        task = Task(id="t1", query="Revenue for Q1 2023", expected_answer=1000)
        result = ExecutionResult(output=5000, code_generated="result = df.sum()")

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.score == 0.3
        assert evaluation.error_type == "filter_error"
        assert "filter" in evaluation.feedback.lower()

    def test_no_output(self) -> None:
        """Test evaluation when no output is produced."""
        client = MockLLMClient()
        evaluator = LLMEvaluator(llm_client=client)

        task = Task(id="t1", query="Calculate something", expected_answer=100)
        result = ExecutionResult(output=None, code_generated="x = 100")

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.score == 0.0
        assert evaluation.error_type == "no_output"

    def test_no_expected_answer(self) -> None:
        """Test evaluation when no expected answer is defined."""
        client = MockLLMClient()
        evaluator = LLMEvaluator(llm_client=client)

        task = Task(id="t1", query="Generate something", expected_answer=None)
        result = ExecutionResult(output="Some output", code_generated="result = 'text'")

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True
        assert evaluation.score == 1.0

    def test_includes_code_in_prompt(self) -> None:
        """Test that generated code is included in evaluation prompt."""
        client = MockLLMClient()
        evaluator = LLMEvaluator(llm_client=client, include_code_in_eval=True)

        task = Task(id="t1", query="Test", expected_answer=1)
        result = ExecutionResult(output=1, code_generated="result = 1")

        evaluator.evaluate(task, result)

        # Check the prompt included the code
        call = client.calls[0]
        prompt = call["messages"][0]["content"]
        assert "result = 1" in prompt
        assert "```python" in prompt

    def test_excludes_code_when_disabled(self) -> None:
        """Test that code can be excluded from evaluation."""
        client = MockLLMClient()
        evaluator = LLMEvaluator(llm_client=client, include_code_in_eval=False)

        task = Task(id="t1", query="Test", expected_answer=1)
        result = ExecutionResult(output=1, code_generated="secret_code = 1")

        evaluator.evaluate(task, result)

        call = client.calls[0]
        prompt = call["messages"][0]["content"]
        assert "secret_code" not in prompt

    def test_fallback_on_llm_error(self) -> None:
        """Test fallback evaluation when LLM fails."""

        class FailingClient:
            def complete_with_tools(self, *args, **kwargs):
                raise Exception("API Error")

        evaluator = LLMEvaluator(llm_client=FailingClient())

        task = Task(id="t1", query="Test", expected_answer=100)
        result = ExecutionResult(output=100)

        # Should fallback to exact match
        evaluation = evaluator.evaluate(task, result)
        assert evaluation.passed is True

    def test_fallback_numeric_tolerance(self) -> None:
        """Test fallback uses numeric tolerance."""

        class FailingClient:
            def complete_with_tools(self, *args, **kwargs):
                raise Exception("API Error")

        evaluator = LLMEvaluator(llm_client=FailingClient())

        task = Task(id="t1", query="Test", expected_answer=100.0)
        result = ExecutionResult(output=100.00001)  # Very close

        evaluation = evaluator.evaluate(task, result)
        assert evaluation.passed is True

    def test_fallback_wrong_answer(self) -> None:
        """Test fallback with clearly wrong answer."""

        class FailingClient:
            def complete_with_tools(self, *args, **kwargs):
                raise Exception("API Error")

        evaluator = LLMEvaluator(llm_client=FailingClient())

        task = Task(id="t1", query="Test", expected_answer=100)
        result = ExecutionResult(output=999)

        evaluation = evaluator.evaluate(task, result)
        assert evaluation.passed is False

    def test_json_string_arguments(self) -> None:
        """Test handling of JSON string arguments from tool call."""

        class JsonStringClient:
            def complete_with_tools(self, *args, **kwargs) -> dict:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "submit_evaluation",
                            "arguments": '{"is_correct": true, "score": 0.9, "error_type": "none", "feedback": "Good!", "reasoning": "Close enough."}',
                        }
                    ],
                    "stop_reason": "tool_use",
                }

        evaluator = LLMEvaluator(llm_client=JsonStringClient())

        task = Task(id="t1", query="Test", expected_answer=100)
        result = ExecutionResult(output=99)

        evaluation = evaluator.evaluate(task, result)
        assert evaluation.passed is True
        assert evaluation.score == 0.9

    def test_no_tool_calls_in_response(self) -> None:
        """Test handling when no tool calls are returned."""

        class NoToolClient:
            def complete_with_tools(self, *args, **kwargs) -> dict:
                return {
                    "content": "I think it's correct.",
                    "tool_calls": [],
                    "stop_reason": "end_turn",
                }

        evaluator = LLMEvaluator(llm_client=NoToolClient())

        task = Task(id="t1", query="Test", expected_answer=100)
        result = ExecutionResult(output=100)

        evaluation = evaluator.evaluate(task, result)
        # Should handle gracefully
        assert evaluation.score == 0.5
        assert "incomplete" in evaluation.feedback.lower()


class TestEvaluationTool:
    """Tests for the evaluation tool definition."""

    def test_tool_has_required_fields(self) -> None:
        """Test that tool definition has all required fields."""
        assert "name" in EVALUATION_TOOL
        assert "description" in EVALUATION_TOOL
        assert "input_schema" in EVALUATION_TOOL

    def test_schema_has_required_properties(self) -> None:
        """Test that schema has all required properties."""
        schema = EVALUATION_TOOL["input_schema"]
        props = schema["properties"]

        assert "is_correct" in props
        assert "score" in props
        assert "error_type" in props
        assert "feedback" in props
        assert "reasoning" in props

    def test_error_types_defined(self) -> None:
        """Test that error types enum is properly defined."""
        schema = EVALUATION_TOOL["input_schema"]
        error_type_def = schema["properties"]["error_type"]

        assert "enum" in error_type_def
        error_types = error_type_def["enum"]

        # Check key error types are present
        assert "logic_error" in error_types
        assert "filter_error" in error_types
        assert "calculation_error" in error_types
        assert "no_output" in error_types
