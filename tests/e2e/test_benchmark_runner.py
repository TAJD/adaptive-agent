"""End-to-end tests for the benchmark runner."""

import pytest
from pathlib import Path

from src.core.types import Task
from src.executor.mock import MockExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.benchmark.tasks import EASY_TASKS


class TestBenchmarkRunner:
    """Tests for the benchmark runner."""

    @pytest.fixture
    def tasks(self) -> list[Task]:
        """Simple task suite for testing."""
        return [
            Task(id="t1", query="1+1", expected_answer=2),
            Task(id="t2", query="2+2", expected_answer=4),
        ]

    @pytest.fixture
    def executor(self, tasks: list[Task]) -> MockExecutor:
        """Executor that fails first attempt, succeeds second."""
        executor = MockExecutor()
        for task in tasks:
            executor.add_response(task.id, output=0, attempt=1)
            executor.add_response(task.id, output=task.expected_answer, attempt=2)
        return executor

    def test_run_single_strategy(self, tasks: list[Task], executor: MockExecutor) -> None:
        """Test running a single strategy."""
        config = BenchmarkConfig(
            max_attempts=3,
            strategies={"reflection": ReflectionStrategy()},
            task_suite=tasks,
            evaluator=ExactMatchEvaluator(),
            executor=executor,
        )

        runner = BenchmarkRunner(config)
        metrics = runner.run_single("reflection")

        assert metrics.strategy_name == "reflection"
        assert metrics.pass_rate == 1.0  # All should pass on 2nd attempt
        assert metrics.avg_attempts_to_pass == 2.0

    def test_run_all_strategies(self, tasks: list[Task]) -> None:
        """Test running multiple strategies."""
        # Create separate executors for each strategy run
        def create_executor():
            executor = MockExecutor()
            for task in tasks:
                executor.add_response(task.id, output=0, attempt=1)
                executor.add_response(task.id, output=task.expected_answer, attempt=2)
            return executor

        config = BenchmarkConfig(
            max_attempts=3,
            strategies={
                "none": NoImprovementStrategy(),
                "reflection": ReflectionStrategy(),
            },
            task_suite=tasks,
            evaluator=ExactMatchEvaluator(),
            executor=create_executor(),  # Will be reused, but that's OK for this test
        )

        runner = BenchmarkRunner(config)
        results = runner.run()

        assert "none" in results
        assert "reflection" in results

    def test_compare_strategies(self, tasks: list[Task]) -> None:
        """Test strategy comparison."""
        executor = MockExecutor()
        for task in tasks:
            executor.add_response(task.id, output=0, attempt=1)
            executor.add_response(task.id, output=task.expected_answer, attempt=2)

        config = BenchmarkConfig(
            max_attempts=3,
            strategies={
                "none": NoImprovementStrategy(),
                "reflection": ReflectionStrategy(),
            },
            task_suite=tasks,
            evaluator=ExactMatchEvaluator(),
            executor=executor,
        )

        runner = BenchmarkRunner(config)
        results = runner.run()
        report = runner.compare(results)

        assert report.best_strategy in ["none", "reflection"]
        assert report.summary != ""
        assert "Pass Rate" in report.summary

    def test_export_report(self, tasks: list[Task], executor: MockExecutor, tmp_path: Path) -> None:
        """Test exporting benchmark report."""
        config = BenchmarkConfig(
            max_attempts=3,
            strategies={"reflection": ReflectionStrategy()},
            task_suite=tasks,
            evaluator=ExactMatchEvaluator(),
            executor=executor,
        )

        runner = BenchmarkRunner(config)
        results = runner.run()

        report_path = tmp_path / "report.json"
        runner.export_report(results, str(report_path))

        assert report_path.exists()

        import json
        with open(report_path) as f:
            data = json.load(f)

        assert "config" in data
        assert "results" in data
        assert "reflection" in data["results"]


class TestBenchmarkWithRealTasks:
    """Tests using the actual task suite."""

    def test_easy_tasks_suite(self) -> None:
        """Test with the easy tasks suite."""
        # Create executor that always returns correct answers
        executor = MockExecutor()
        for task in EASY_TASKS:
            executor.add_response(task.id, output=task.expected_answer)

        config = BenchmarkConfig(
            max_attempts=1,
            strategies={"none": NoImprovementStrategy()},
            task_suite=EASY_TASKS,
            evaluator=ExactMatchEvaluator(),
            executor=executor,
        )

        runner = BenchmarkRunner(config)
        results = runner.run()

        assert results["none"].pass_rate == 1.0
        assert len(results["none"].per_task_results) == len(EASY_TASKS)
