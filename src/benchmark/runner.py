"""Benchmark runner for comparing strategies."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.core.types import (
    Task,
    StrategyMetrics,
    ComparisonReport,
    TaskResult,
    BenchmarkConfig,
)
from src.core.protocols import Executor, Evaluator, ImprovementStrategy, Storage
from src.agent.runner import AgentRunner, AgentConfig
from src.benchmark.metrics import MetricsCollector


class BenchmarkRunner:
    """
    Clean API for comparing improvement strategies.

    Usage:
        runner = BenchmarkRunner(config)
        results = runner.run()
        report = runner.compare(results)
        runner.export_report(results, "benchmark_report.json")
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        storage: Storage | None = None,
        save_conversations: bool = False,
    ) -> None:
        self.config = config
        self.storage = storage
        self.save_conversations = save_conversations
        self._metrics_by_strategy: dict[str, MetricsCollector] = {}

    def run(self) -> dict[str, StrategyMetrics]:
        """Run all strategies against task suite."""
        results: dict[str, StrategyMetrics] = {}

        for strategy_name, strategy in self.config.strategies.items():
            metrics = self.run_single(strategy_name, strategy)
            results[strategy_name] = metrics

        return results

    def run_single(
        self,
        strategy_name: str,
        strategy: ImprovementStrategy | None = None,
    ) -> StrategyMetrics:
        """Run a single strategy (useful for incremental testing)."""
        if strategy is None:
            strategy = self.config.strategies.get(strategy_name)
            if strategy is None:
                raise ValueError(f"Unknown strategy: {strategy_name}")

        if self.config.executor is None:
            raise ValueError("No executor configured")
        if self.config.evaluator is None:
            raise ValueError("No evaluator configured")

        # Reset executor state if possible (for mock executors)
        if hasattr(self.config.executor, "reset_counts"):
            self.config.executor.reset_counts()

        # Create fresh metrics collector
        metrics = MetricsCollector()
        self._metrics_by_strategy[strategy_name] = metrics

        # Create agent runner with this strategy
        # Enable trajectory collection for detailed conversation logging
        agent_config = AgentConfig(
            max_attempts=self.config.max_attempts,
            collect_trajectory=self.save_conversations,
        )
        agent = AgentRunner(
            executor=self.config.executor,
            evaluator=self.config.evaluator,
            strategy=strategy,
            config=agent_config,
        )

        # Run all tasks
        task_results: list[TaskResult] = []
        for task in self.config.task_suite:
            # Inject data context for real data tasks
            task_context = None
            if "real_data" in task.tags:
                from .data_loader import create_task_context

                task_context = create_task_context(task.id)

            result = agent.run(task, task_context)
            task_results.append(result)
            metrics.record_task_complete(result)

            # Save conversation if requested
            if self.save_conversations and self.storage:
                self._save_conversation(task, result)

        # Calculate strategy metrics
        summary = metrics.get_summary()
        improvement_curve = metrics.get_improvement_curve()

        # Save strategy metrics if storage is available
        if self.storage:
            self._save_strategy_metrics(strategy_name, summary)

        return StrategyMetrics(
            strategy_name=strategy_name,
            pass_rate=summary["pass_rate"],
            avg_attempts_to_pass=summary["avg_attempts_to_pass"],
            avg_final_score=summary["avg_final_score"],
            improvement_curve=improvement_curve,
            per_task_results=task_results,
        )

    def compare(self, results: dict[str, StrategyMetrics]) -> ComparisonReport:
        """Generate comparison with analysis."""
        if not results:
            return ComparisonReport()

        # Find baseline (none strategy or first strategy)
        baseline_name = "none" if "none" in results else list(results.keys())[0]
        baseline = results[baseline_name]

        # Find best strategy by pass rate
        best_name = max(results.keys(), key=lambda k: results[k].pass_rate)
        best = results[best_name]

        # Calculate improvement over baseline
        improvement = best.pass_rate - baseline.pass_rate

        # Generate summary
        lines = ["Strategy Comparison:", "=" * 50]
        for name, metrics in sorted(results.items(), key=lambda x: -x[1].pass_rate):
            lines.append(
                f"  {name:20} | Pass Rate: {metrics.pass_rate:5.1%} | "
                f"Avg Attempts: {metrics.avg_attempts_to_pass:.1f} | "
                f"Avg Score: {metrics.avg_final_score:.2f}"
            )
        lines.append("=" * 50)
        lines.append(f"Best strategy: {best_name}")
        if best_name != baseline_name:
            lines.append(
                f"Improvement over baseline ({baseline_name}): "
                f"{improvement:+.1%} pass rate"
            )

        return ComparisonReport(
            strategies=results,
            best_strategy=best_name,
            improvement_over_baseline=improvement,
            summary="\n".join(lines),
        )

    def export_report(self, results: dict[str, StrategyMetrics], path: str) -> None:
        """Export results to JSON for visualization."""
        report_data = {
            "config": {
                "max_attempts": self.config.max_attempts,
                "num_tasks": len(self.config.task_suite),
                "strategies": list(results.keys()),
            },
            "results": {},
        }

        for name, metrics in results.items():
            report_data["results"][name] = {
                "pass_rate": metrics.pass_rate,
                "avg_attempts_to_pass": metrics.avg_attempts_to_pass,
                "avg_final_score": metrics.avg_final_score,
                "improvement_curve": metrics.improvement_curve,
                "per_task": [
                    {
                        "task_id": r.task.id,
                        "passed": r.passed,
                        "attempts": r.attempts,
                        "final_score": r.final_score,
                        "score_progression": r.score_progression,
                    }
                    for r in metrics.per_task_results
                ],
            }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    def _save_conversation(self, task: Task, result: TaskResult) -> None:
        """Save the conversation trajectory for a task result."""
        if not self.storage:
            return

        conversation_data = {
            "task_id": task.id,
            "task_query": task.query,
            "strategy": getattr(result, "strategy_name", "unknown"),
            "passed": result.passed,
            "attempts": result.attempts,
            "final_score": result.final_score,
            "timestamp": None,  # Would need to add timestamp tracking
            "trajectory": [],
        }

        # Extract trajectory from final result if available
        if result.final_result and result.final_result.trajectory:
            conversation_data["trajectory"] = result.final_result.trajectory

        # Add evaluation info if available
        if result.final_evaluation:
            conversation_data["evaluation"] = {
                "score": result.final_evaluation.score,
                "passed": result.final_evaluation.passed,
                "feedback": result.final_evaluation.feedback,
                "error_type": result.final_evaluation.error_type,
            }

        # Save with unique key
        import hashlib

        key = f"conversations/{task.id}/{hashlib.md5(task.query.encode()).hexdigest()[:8]}"
        self.storage.save(key, conversation_data)

    def _save_strategy_metrics(self, strategy_name: str, metrics: dict) -> None:
        """Save strategy performance metrics."""
        if not self.storage:
            return

        from datetime import datetime

        key = f"strategy_metrics/{strategy_name}"
        self.storage.save(
            key,
            {
                "strategy": strategy_name,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            },
        )

    def get_saved_conversations(self, task_id: str | None = None) -> list[dict]:
        """Retrieve saved conversations, optionally filtered by task_id."""
        if not self.storage:
            return []

        conversations = []
        prefix = "conversations"
        if task_id:
            prefix = f"conversations/{task_id}"

        for key in self.storage.list_keys(prefix):
            data = self.storage.load(key)
            if data:
                conversations.append(data)

        return conversations
