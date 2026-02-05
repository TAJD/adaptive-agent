"""Integration tests for the agent loop."""

import pytest

from src.core.types import Task, ExecutionResult
from src.agent.runner import AgentRunner, AgentConfig
from src.executor.mock import MockExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.storage.memory import InMemoryStorage


class TestAgentLoop:
    """Tests for the agent loop."""

    @pytest.fixture
    def evaluator(self) -> ExactMatchEvaluator:
        return ExactMatchEvaluator()

    @pytest.fixture
    def task(self) -> Task:
        return Task(
            id="test_task",
            query="Calculate 1 + 1",
            expected_answer=2,
            difficulty="easy",
        )

    def test_passes_on_first_try(self, evaluator: ExactMatchEvaluator, task: Task) -> None:
        """Test agent passes when executor returns correct answer."""
        executor = MockExecutor()
        executor.add_response(task.id, output=2)

        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=NoImprovementStrategy(),
        )

        result = runner.run(task)

        assert result.passed is True
        assert result.attempts == 1
        assert result.final_score == 1.0

    def test_fails_after_max_attempts(self, evaluator: ExactMatchEvaluator, task: Task) -> None:
        """Test agent fails after max attempts with wrong answer."""
        executor = MockExecutor()
        executor.add_response(task.id, output=999)  # Always wrong

        config = AgentConfig(max_attempts=3)
        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=NoImprovementStrategy(),
            config=config,
        )

        result = runner.run(task)

        assert result.passed is False
        assert result.attempts == 3
        assert executor.get_call_count(task.id) == 3

    def test_improvement_applied_on_retry(self, evaluator: ExactMatchEvaluator, task: Task) -> None:
        """Test that improvements modify subsequent attempts."""
        executor = MockExecutor()
        # First attempt wrong, second attempt correct
        executor.add_response(task.id, output=999, attempt=1)
        executor.add_response(task.id, output=2, attempt=2)

        storage = InMemoryStorage()
        strategy = ReflectionStrategy(storage=storage)

        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=strategy,
        )

        result = runner.run(task)

        assert result.passed is True
        assert result.attempts == 2
        assert len(result.score_progression) == 2
        assert result.score_progression[0] < result.score_progression[1]

    def test_strategy_receives_correct_context(
        self, evaluator: ExactMatchEvaluator, task: Task
    ) -> None:
        """Test that ImprovementContext is populated correctly."""
        executor = MockExecutor()
        executor.add_response(task.id, output=999)  # Always wrong

        # Custom strategy to capture context
        captured_contexts = []

        class CapturingStrategy:
            def improve(self, context):
                captured_contexts.append(context)
                return {"captured": True}

            def persist(self, context):
                pass

            def load_priors(self, task):
                return {}

        config = AgentConfig(max_attempts=2)
        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=CapturingStrategy(),
            config=config,
        )

        runner.run(task)

        # Should have captured one context (after first failure)
        assert len(captured_contexts) == 1
        ctx = captured_contexts[0]
        assert ctx.task.id == task.id
        assert ctx.attempt_number == 1
        assert ctx.evaluation.passed is False

    def test_score_progression_tracked(self, evaluator: ExactMatchEvaluator, task: Task) -> None:
        """Test that score progression is tracked correctly."""
        executor = MockExecutor()
        # Gradually improving scores
        executor.add_response(task.id, output=0, attempt=1)   # Very wrong
        executor.add_response(task.id, output=1, attempt=2)   # Close
        executor.add_response(task.id, output=2, attempt=3)   # Correct

        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=ReflectionStrategy(),
        )

        result = runner.run(task)

        assert result.passed is True
        assert result.attempts == 3
        assert len(result.score_progression) == 3
        # Scores should generally improve (or reach 1.0)
        assert result.score_progression[-1] == 1.0

    def test_run_batch(self, evaluator: ExactMatchEvaluator) -> None:
        """Test running multiple tasks."""
        tasks = [
            Task(id="t1", query="1+1", expected_answer=2),
            Task(id="t2", query="2+2", expected_answer=4),
            Task(id="t3", query="3+3", expected_answer=6),
        ]

        executor = MockExecutor()
        executor.add_response("t1", output=2)
        executor.add_response("t2", output=4)
        executor.add_response("t3", output=999)  # This one fails

        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=NoImprovementStrategy(),
        )

        results = runner.run_batch(tasks)

        assert len(results) == 3
        assert results[0].passed is True
        assert results[1].passed is True
        assert results[2].passed is False

    def test_context_passed_to_executor(self, evaluator: ExactMatchEvaluator, task: Task) -> None:
        """Test that context from strategy is passed to executor."""
        executor = MockExecutor()
        executor.add_response(task.id, output=999, attempt=1)
        executor.add_response(task.id, output=2, attempt=2)

        storage = InMemoryStorage()
        # Pre-store some priors
        storage.save(f"reflection/tasks/{task.id}", {"learnings": ["prior_hint"]})

        strategy = ReflectionStrategy(storage=storage)

        config = AgentConfig(max_attempts=2)
        runner = AgentRunner(
            executor=executor,
            evaluator=evaluator,
            strategy=strategy,
            config=config,
        )

        result = runner.run(task)

        # The metadata in final result should show context was passed
        assert result.final_result is not None
        assert result.final_result.metadata.get("context") is not None
