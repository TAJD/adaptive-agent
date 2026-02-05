"""Tests for evaluators."""

import pytest

from src.core.types import Task, ExecutionResult
from src.evaluator.exact_match import ExactMatchEvaluator


class TestExactMatchEvaluator:
    """Tests for ExactMatchEvaluator."""

    @pytest.fixture
    def evaluator(self) -> ExactMatchEvaluator:
        return ExactMatchEvaluator()

    def test_exact_match_passes(self, evaluator: ExactMatchEvaluator) -> None:
        """Test that exact numeric match passes."""
        task = Task(id="1", query="...", expected_answer=42.0)
        result = ExecutionResult(output=42.0, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True
        assert evaluation.score == 1.0

    def test_exact_integer_match(self, evaluator: ExactMatchEvaluator) -> None:
        """Test that integer matches work."""
        task = Task(id="1", query="...", expected_answer=42)
        result = ExecutionResult(output=42, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True
        assert evaluation.score == 1.0

    def test_close_match_with_tolerance(self, evaluator: ExactMatchEvaluator) -> None:
        """Test numeric tolerance."""
        task = Task(id="1", query="...", expected_answer=1.0)
        result = ExecutionResult(output=1.0 + 1e-8, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True
        assert evaluation.score == 1.0

    def test_wrong_answer_fails(self, evaluator: ExactMatchEvaluator) -> None:
        """Test that wrong answer fails."""
        task = Task(id="1", query="...", expected_answer=42)
        result = ExecutionResult(output=43, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.score < 1.0

    def test_no_output_fails(self, evaluator: ExactMatchEvaluator) -> None:
        """Test that no output fails."""
        task = Task(id="1", query="...", expected_answer=42)
        result = ExecutionResult(output=None, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.error_type == "no_output"

    def test_string_exact_match(self, evaluator: ExactMatchEvaluator) -> None:
        """Test string exact match."""
        task = Task(id="1", query="...", expected_answer="hello")
        result = ExecutionResult(output="hello", trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True

    def test_string_mismatch(self, evaluator: ExactMatchEvaluator) -> None:
        """Test string mismatch."""
        task = Task(id="1", query="...", expected_answer="hello")
        result = ExecutionResult(output="world", trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.error_type == "string_mismatch"

    def test_whitespace_tolerance(self, evaluator: ExactMatchEvaluator) -> None:
        """Test whitespace is handled."""
        task = Task(id="1", query="...", expected_answer="hello")
        result = ExecutionResult(output=" hello ", trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        # Should pass but with minor penalty
        assert evaluation.passed is True
        assert evaluation.score >= 0.9

    def test_list_exact_match(self, evaluator: ExactMatchEvaluator) -> None:
        """Test list exact match."""
        task = Task(id="1", query="...", expected_answer=[1, 2, 3])
        result = ExecutionResult(output=[1, 2, 3], trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True

    def test_list_partial_match(self, evaluator: ExactMatchEvaluator) -> None:
        """Test list partial match."""
        task = Task(id="1", query="...", expected_answer=[1, 2, 3])
        result = ExecutionResult(output=[1, 2, 4], trajectory=[])  # One wrong

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is False
        assert evaluation.score > 0  # Partial credit

    def test_type_mismatch(self, evaluator: ExactMatchEvaluator) -> None:
        """Test type mismatch detection."""
        task = Task(id="1", query="...", expected_answer=42)
        result = ExecutionResult(output="42", trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        # Should still pass due to string coercion
        assert evaluation.passed is True

    def test_classifies_error_types(self, evaluator: ExactMatchEvaluator) -> None:
        """Test error classification."""
        # Large numeric error
        task = Task(id="1", query="...", expected_answer=100)
        result = ExecutionResult(output=10, trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.error_type is not None
        assert "numeric" in evaluation.error_type.lower()

    def test_no_expected_answer(self, evaluator: ExactMatchEvaluator) -> None:
        """Test when no expected answer is defined."""
        task = Task(id="1", query="...")  # No expected_answer
        result = ExecutionResult(output="something", trajectory=[])

        evaluation = evaluator.evaluate(task, result)

        assert evaluation.passed is True  # Has output, so passes
