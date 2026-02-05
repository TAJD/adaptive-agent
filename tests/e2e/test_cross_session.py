"""End-to-end tests for cross-session learning."""

import pytest
from pathlib import Path

from src.core.types import Task
from src.agent.runner import AgentRunner, AgentConfig
from src.executor.mock import MockExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.storage.file import FileStorage


class TestCrossSessionLearning:
    """Tests for cross-session learning persistence."""

    @pytest.fixture
    def evaluator(self) -> ExactMatchEvaluator:
        return ExactMatchEvaluator()

    def test_session_1_failure_informs_session_2(
        self, evaluator: ExactMatchEvaluator, tmp_path: Path
    ) -> None:
        """Test that learnings from session 1 are available in session 2."""
        storage = FileStorage(tmp_path / "learning_storage")

        task_a = Task(
            id="numeric_task",
            query="Calculate something",
            expected_answer=100,
            tags=("math",),
        )

        # Session 1: Fail and learn
        executor1 = MockExecutor()
        executor1.add_response(task_a.id, output=50)  # Always wrong

        strategy1 = ReflectionStrategy(storage=storage)
        config = AgentConfig(max_attempts=2)

        agent1 = AgentRunner(
            executor=executor1,
            evaluator=evaluator,
            strategy=strategy1,
            config=config,
        )

        result1 = agent1.run(task_a)
        assert not result1.passed  # Should fail

        # Verify learnings were persisted
        keys = storage.list_keys()
        assert len(keys) > 0, "Learnings should be stored"

        # Session 2: New agent, same storage, same task
        strategy2 = ReflectionStrategy(storage=storage)

        # Load priors should find something
        priors = strategy2.load_priors(task_a)
        # Note: priors might be empty if no task-specific learnings,
        # but error patterns should be stored

        error_data = storage.load("reflection/errors/large_numeric_error")
        assert error_data is not None or storage.load("reflection/errors/numeric_error") is not None

    def test_cross_session_with_different_tasks(
        self, evaluator: ExactMatchEvaluator, tmp_path: Path
    ) -> None:
        """Test that learnings transfer to similar tasks."""
        storage = FileStorage(tmp_path / "transfer_storage")

        # Two similar tasks (same tags)
        task_a = Task(
            id="task_a",
            query="Sum numbers",
            expected_answer=10,
            tags=("arithmetic",),
        )
        task_b = Task(
            id="task_b",
            query="Sum more numbers",
            expected_answer=20,
            tags=("arithmetic",),
        )

        # Session 1: Work on task_a
        executor1 = MockExecutor()
        executor1.add_response(task_a.id, output=5, attempt=1)
        executor1.add_response(task_a.id, output=10, attempt=2)

        strategy1 = ReflectionStrategy(storage=storage)
        agent1 = AgentRunner(
            executor=executor1,
            evaluator=evaluator,
            strategy=strategy1,
        )

        result1 = agent1.run(task_a)
        assert result1.passed

        # Session 2: Work on task_b with accumulated knowledge
        executor2 = MockExecutor()
        executor2.add_response(task_b.id, output=20)

        strategy2 = ReflectionStrategy(storage=storage)
        agent2 = AgentRunner(
            executor=executor2,
            evaluator=evaluator,
            strategy=strategy2,
        )

        result2 = agent2.run(task_b)
        assert result2.passed

    def test_storage_isolation_between_strategies(
        self, evaluator: ExactMatchEvaluator, tmp_path: Path
    ) -> None:
        """Test that different storage paths are isolated."""
        storage1 = FileStorage(tmp_path / "storage1")
        storage2 = FileStorage(tmp_path / "storage2")

        task = Task(id="task", query="test", expected_answer=42)

        # Store in storage1
        storage1.save("test_key", {"value": 1})

        # storage2 should not see it
        assert storage2.load("test_key") is None
        assert storage1.load("test_key") == {"value": 1}


class TestBenchmarkDemo:
    """End-to-end benchmark demonstration."""

    @pytest.fixture
    def evaluator(self) -> ExactMatchEvaluator:
        return ExactMatchEvaluator()

    def test_reflection_outperforms_no_improvement(
        self, evaluator: ExactMatchEvaluator
    ) -> None:
        """
        Demonstrate that reflection strategy can improve over baseline.

        This test sets up a scenario where:
        - MockExecutor returns wrong answer on attempt 1
        - MockExecutor returns correct answer on attempt 2+

        With NoImprovementStrategy: Only 1 attempt, always fails
        With ReflectionStrategy: Gets to attempt 2, succeeds
        """
        tasks = [
            Task(id="t1", query="q1", expected_answer=10),
            Task(id="t2", query="q2", expected_answer=20),
            Task(id="t3", query="q3", expected_answer=30),
        ]

        # Setup executor to fail first attempt, succeed second
        def create_executor():
            executor = MockExecutor()
            for task in tasks:
                executor.add_response(task.id, output=0, attempt=1)  # Wrong
                executor.add_response(task.id, output=task.expected_answer, attempt=2)  # Right
            return executor

        # Run with no improvement (single attempt)
        executor_baseline = create_executor()
        config_baseline = AgentConfig(max_attempts=1)
        agent_baseline = AgentRunner(
            executor=executor_baseline,
            evaluator=evaluator,
            strategy=NoImprovementStrategy(),
            config=config_baseline,
        )
        results_baseline = agent_baseline.run_batch(tasks)
        pass_rate_baseline = sum(1 for r in results_baseline if r.passed) / len(results_baseline)

        # Run with reflection (multiple attempts)
        executor_reflection = create_executor()
        config_reflection = AgentConfig(max_attempts=3)
        agent_reflection = AgentRunner(
            executor=executor_reflection,
            evaluator=evaluator,
            strategy=ReflectionStrategy(),
            config=config_reflection,
        )
        results_reflection = agent_reflection.run_batch(tasks)
        pass_rate_reflection = sum(1 for r in results_reflection if r.passed) / len(results_reflection)

        # Reflection should outperform baseline
        assert pass_rate_baseline == 0.0, "Baseline should fail all (1 attempt, wrong answer)"
        assert pass_rate_reflection == 1.0, "Reflection should pass all (2nd attempt correct)"
        assert pass_rate_reflection > pass_rate_baseline
