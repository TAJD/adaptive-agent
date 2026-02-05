"""Tests for MatrixBenchmarkRunner."""

import pytest

from src.core.types import BenchmarkConfig, Task
from src.executor.mock import MockExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.benchmark.matrix_runner import MatrixBenchmarkRunner, MatrixBenchmarkResult
from src.benchmark.model_config import ModelConfig
from src.storage.memory import InMemoryStorage


class TestMatrixBenchmarkRunner:
    """Tests for MatrixBenchmarkRunner."""

    @pytest.fixture
    def base_config(self) -> BenchmarkConfig:
        """Create a base benchmark config for testing."""
        tasks = [
            Task(id="task1", query="What is 1+1?", expected_answer=2),
            Task(id="task2", query="What is 2+2?", expected_answer=4),
        ]

        models = [
            ModelConfig(name="gpt-4", provider="openai"),
            ModelConfig(name="claude", provider="anthropic"),
        ]

        strategies = {
            "none": NoImprovementStrategy(),
        }

        return BenchmarkConfig(
            max_attempts=3,
            strategies=strategies,
            task_suite=tasks,
            models=models,
            evaluator=ExactMatchEvaluator(),
            executor=MockExecutor(),
        )

    def test_run_matrix(self, base_config: BenchmarkConfig) -> None:
        """Test running matrix benchmarks."""
        # Configure mock executor to succeed
        executor = base_config.executor
        assert isinstance(executor, MockExecutor)

        # Add responses for both tasks (succeed on attempt 1)
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        result = runner.run_matrix()

        assert isinstance(result, MatrixBenchmarkResult)
        assert len(result.results) == 2  # 1 strategy * 2 models

        # Check results
        strategies_found = set()
        models_found = set()

        for matrix_result in result.results:
            strategies_found.add(matrix_result.strategy_name)
            models_found.add(matrix_result.model_config.model_key)

            assert matrix_result.metrics.pass_rate == 1.0  # All tasks pass
            assert len(matrix_result.metrics.per_task_results) == 2

        assert strategies_found == {"none"}
        assert models_found == {"openai/gpt-4", "anthropic/claude"}

    def test_get_matrix(self, base_config: BenchmarkConfig) -> None:
        """Test getting results as matrix."""
        # Setup similar to above
        executor = base_config.executor
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        result = runner.run_matrix()

        matrix = result.get_matrix()
        assert "none" in matrix
        assert "openai/gpt-4" in matrix["none"]
        assert "anthropic/claude" in matrix["none"]

    def test_get_results_for_strategy(self, base_config: BenchmarkConfig) -> None:
        """Test filtering results by strategy."""
        executor = base_config.executor
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        result = runner.run_matrix()

        strategy_results = result.get_results_for_strategy("none")
        assert len(strategy_results) == 2  # One for each model

        for r in strategy_results:
            assert r.strategy_name == "none"

    def test_get_results_for_model(self, base_config: BenchmarkConfig) -> None:
        """Test filtering results by model."""
        executor = base_config.executor
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        result = runner.run_matrix()

        model_results = result.get_results_for_model("openai/gpt-4")
        assert len(model_results) == 1  # One strategy

        assert model_results[0].model_config.model_key == "openai/gpt-4"

    def test_get_best_performing(self, base_config: BenchmarkConfig) -> None:
        """Test getting best performing combinations."""
        # Configure different performance levels
        executor = base_config.executor

        # Task 1: succeed on attempt 1 for both models
        executor.add_response("task1", output=2, attempt=1)
        executor.add_response("task1", output=2, attempt=1)

        # Task 2: succeed on attempt 1 for gpt-4, attempt 2 for claude
        executor.add_response("task2", output=4, attempt=1)  # gpt-4
        executor.add_response("task2", output=4, attempt=2)  # claude

        runner = MatrixBenchmarkRunner(base_config)
        result = runner.run_matrix()

        best = result.get_best_performing("pass_rate")
        assert len(best) == 2

        # Should be sorted by pass_rate descending
        assert best[0].metrics.pass_rate >= best[1].metrics.pass_rate

    def test_run_strategy_comparison(self, base_config: BenchmarkConfig) -> None:
        """Test running all strategies against one model."""
        # Add multiple strategies to config
        base_config.strategies["none2"] = NoImprovementStrategy()

        executor = base_config.executor
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        model_config = ModelConfig(name="test-model", provider="test")
        results = runner.run_strategy_comparison(model_config)

        assert "none" in results
        assert "none2" in results
        assert results["none"].pass_rate == 1.0
        assert results["none2"].pass_rate == 1.0

    def test_storage_integration(self, base_config: BenchmarkConfig) -> None:
        """Test that results are saved to and loaded from storage."""
        storage = InMemoryStorage()

        # Run benchmark with storage
        runner = MatrixBenchmarkRunner(base_config, storage)
        result = runner.run_matrix()

        # Verify results were saved
        keys = storage.list_keys("matrix_results/")
        assert len(keys) > 1  # At least summary + results

        # Load results back
        loaded_result = runner.load_results()
        assert len(loaded_result.results) == len(result.results)

        # Verify summaries
        summaries = runner.get_stored_summaries()
        assert len(summaries) == 1
        assert summaries[0]["total_results"] == len(result.results)

    def test_run_model_comparison(self, base_config: BenchmarkConfig) -> None:
        """Test running one strategy against all models."""
        executor = base_config.executor
        for task in base_config.task_suite:
            executor.add_response(task.id, output=task.expected_answer, attempt=1)
            executor.add_response(task.id, output=task.expected_answer, attempt=1)

        runner = MatrixBenchmarkRunner(base_config)
        results = runner.run_model_comparison("none")

        assert "openai/gpt-4" in results
        assert "anthropic/claude" in results
        assert results["openai/gpt-4"].pass_rate == 1.0
        assert results["anthropic/claude"].pass_rate == 1.0
