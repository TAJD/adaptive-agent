"""Matrix benchmark runner for strategy x model grid evaluation."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.core.types import BenchmarkConfig, StrategyMetrics, TaskResult
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.model_config import ModelConfig
from src.core.protocols import Storage
from src.executor.llm import LLMExecutor
from src.llm.claude import ClaudeClient


@dataclass
class MatrixResult:
    """Result for a strategy-model combination."""

    strategy_name: str
    model_config: ModelConfig
    metrics: StrategyMetrics
    cost_info: Dict[str, Any] = field(default_factory=dict)  # Token usage, costs, etc.


@dataclass
class MatrixBenchmarkResult:
    """Complete results from a matrix benchmark run."""

    results: List[MatrixResult] = field(default_factory=list)
    config: Optional[BenchmarkConfig] = field(default=None)

    def get_results_for_strategy(self, strategy: str) -> List[MatrixResult]:
        """Get all results for a specific strategy."""
        return [r for r in self.results if r.strategy_name == strategy]

    def get_results_for_model(self, model_key: str) -> List[MatrixResult]:
        """Get all results for a specific model."""
        return [r for r in self.results if r.model_config.model_key == model_key]

    def get_matrix(self) -> Dict[str, Dict[str, MatrixResult]]:
        """Get results as a strategy x model matrix."""
        matrix = {}
        for result in self.results:
            if result.strategy_name not in matrix:
                matrix[result.strategy_name] = {}
            matrix[result.strategy_name][result.model_config.model_key] = result
        return matrix

    def get_best_performing(self, metric: str = "pass_rate") -> List[MatrixResult]:
        """Get results sorted by performance metric."""
        return sorted(
            self.results, key=lambda r: getattr(r.metrics, metric), reverse=True
        )


class MatrixBenchmarkRunner:
    """
    Runs benchmarks across a grid of strategies and models.

    This allows comparing how different improvement strategies perform
    when using different LLM models.
    """

    def __init__(self, base_config: BenchmarkConfig, storage: Storage | None = None):
        """
        Initialize with base configuration and optional storage.

        Args:
            base_config: BenchmarkConfig with strategies, tasks, evaluator, executor
            storage: Optional storage for persisting results
        """
        self.base_config = base_config
        self.storage = storage

    def run_matrix(self) -> MatrixBenchmarkResult:
        """
        Run benchmarks for all strategy-model combinations.

        Returns:
            MatrixBenchmarkResult with results for each combination
        """
        results = []

        # For each model
        for model_config in self.base_config.models:
            print(
                f"Running benchmarks for model: {model_config.name} ({model_config.provider})"
            )

            # For each strategy
            for strategy_name, strategy in self.base_config.strategies.items():
                print(f"  Testing strategy: {strategy_name}")

                # Create model-specific executor for this model
                model_client = ClaudeClient(model=model_config.name)
                model_executor = LLMExecutor(llm_client=model_client)

                config = BenchmarkConfig(
                    max_attempts=self.base_config.max_attempts,
                    strategies={strategy_name: strategy},  # Only this strategy
                    task_suite=self.base_config.task_suite,
                    models=[model_config],  # Only this model
                    evaluator=self.base_config.evaluator,
                    executor=model_executor,  # Use model-specific executor
                )

                # Run the benchmark
                runner = BenchmarkRunner(
                    config, storage=self.storage, save_conversations=True
                )
                strategy_results = runner.run()

                # Extract the result for this strategy
                if strategy_name in strategy_results:
                    matrix_result = MatrixResult(
                        strategy_name=strategy_name,
                        model_config=model_config,
                        metrics=strategy_results[strategy_name],
                    )
                    results.append(matrix_result)

        matrix_result = MatrixBenchmarkResult(results=results, config=self.base_config)

        # Save results if storage is available
        if self.storage:
            self._save_results(matrix_result)

        return matrix_result

    def run_strategy_comparison(
        self, model_config: ModelConfig
    ) -> Dict[str, StrategyMetrics]:
        """
        Run all strategies against a single model.

        Args:
            model_config: The model to test strategies against

        Returns:
            Dict mapping strategy names to their metrics
        """
        config = BenchmarkConfig(
            max_attempts=self.base_config.max_attempts,
            strategies=self.base_config.strategies,
            task_suite=self.base_config.task_suite,
            models=[model_config],
            evaluator=self.base_config.evaluator,
            executor=self.base_config.executor,
        )

        runner = BenchmarkRunner(config)
        return runner.run()

    def run_model_comparison(self, strategy_name: str) -> Dict[str, StrategyMetrics]:
        """
        Run a single strategy against all models.

        Args:
            strategy_name: The strategy to test against all models

        Returns:
            Dict mapping model keys to strategy metrics
        """
        results = {}

        for model_config in self.base_config.models:
            config = BenchmarkConfig(
                max_attempts=self.base_config.max_attempts,
                strategies={strategy_name: self.base_config.strategies[strategy_name]},
                task_suite=self.base_config.task_suite,
                models=[model_config],
                evaluator=self.base_config.evaluator,
                executor=self.base_config.executor,
            )

            runner = BenchmarkRunner(config)
            strategy_results = runner.run()

            # Store with model key
            results[model_config.model_key] = strategy_results[strategy_name]

        return results

    def export_matrix_results(
        self, matrix_result: MatrixBenchmarkResult, path: str
    ) -> None:
        """
        Export matrix results to a JSON file.

        Args:
            matrix_result: Results to export
            path: Path to export to
        """
        import json
        from pathlib import Path

        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "config": {
                "max_attempts": matrix_result.config.max_attempts
                if matrix_result.config
                else None,
                "num_tasks": len(matrix_result.config.task_suite)
                if matrix_result.config
                else 0,
                "strategies": list(matrix_result.config.strategies.keys())
                if matrix_result.config
                else [],
                "models": [m.to_dict() for m in matrix_result.config.models]
                if matrix_result.config
                else [],
            },
            "results": [],
        }

        for result in matrix_result.results:
            result_data = {
                "strategy": result.strategy_name,
                "model": result.model_config.to_dict(),
                "metrics": {
                    "pass_rate": result.metrics.pass_rate,
                    "avg_attempts_to_pass": result.metrics.avg_attempts_to_pass,
                    "avg_final_score": result.metrics.avg_final_score,
                    "improvement_curve": result.metrics.improvement_curve,
                    "per_task": [
                        {
                            "task_id": tr.task.id,
                            "passed": tr.passed,
                            "attempts": tr.attempts,
                            "final_score": tr.final_score,
                            "score_progression": tr.score_progression,
                        }
                        for tr in result.metrics.per_task_results
                    ],
                },
            }
            data["results"].append(result_data)

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _save_results(self, matrix_result: MatrixBenchmarkResult) -> None:
        """Save matrix benchmark results to storage."""
        if not self.storage:
            return

        # Save each result with a unique key
        for result in matrix_result.results:
            key = (
                f"matrix_results/{result.strategy_name}/{result.model_config.model_key}"
            )
            self.storage.save(key, result)

        # Save summary metadata
        from datetime import datetime

        summary_key = "matrix_results/_summary"
        summary = {
            "total_results": len(matrix_result.results),
            "strategies": list(set(r.strategy_name for r in matrix_result.results)),
            "models": list(
                set(r.model_config.model_key for r in matrix_result.results)
            ),
            "timestamp": datetime.now().isoformat(),
        }
        self.storage.save(summary_key, summary)

    def load_results(
        self, strategy_filter: str | None = None, model_filter: str | None = None
    ) -> MatrixBenchmarkResult:
        """
        Load matrix benchmark results from storage.

        Args:
            strategy_filter: Optional strategy name to filter by
            model_filter: Optional model key to filter by

        Returns:
            MatrixBenchmarkResult with loaded results
        """
        if not self.storage:
            return MatrixBenchmarkResult()

        results = []
        keys = self.storage.list_keys("matrix_results/")

        for key in keys:
            if key == "matrix_results/_summary":
                continue  # Skip summary

            parts = key.split("/")
            if len(parts) >= 3:
                strategy_name = parts[1]
                model_key = parts[2]

                # Apply filters
                if strategy_filter and strategy_name != strategy_filter:
                    continue
                if model_filter and model_key != model_filter:
                    continue

                result_data = self.storage.load(key)
                if result_data:
                    results.append(result_data)

        return MatrixBenchmarkResult(results=results)

    def get_stored_summaries(self) -> list[dict]:
        """Get summaries of all stored matrix benchmark runs."""
        if not self.storage:
            return []

        summaries = []
        keys = self.storage.list_keys("matrix_results/_summary")

        for key in keys:
            summary = self.storage.load(key)
            if summary:
                summaries.append(summary)

        return summaries
