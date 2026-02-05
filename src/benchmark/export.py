"""Export functions for benchmark results."""

import json
from pathlib import Path
from typing import Any

from src.core.types import BenchmarkRunRecord, StrategyMetrics, ComparisonReport


def export_run_to_json(record: BenchmarkRunRecord, path: str | Path) -> None:
    """Export a benchmark run record to JSON."""
    # Convert to dict for JSON serialization
    data = {
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "version": record.version,
        "config": {
            "max_attempts": record.config.max_attempts,
            "strategies": list(record.config.strategies.keys()),
            "num_tasks": len(record.config.task_suite),
            "seed": record.config.seed,
        },
        "results": {},
        "metadata": record.metadata,
    }

    for strategy_name, metrics in record.results.items():
        data["results"][strategy_name] = {
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_strategies_to_csv(
    results: dict[str, StrategyMetrics], path: str | Path
) -> None:
    """Export strategy metrics to CSV."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            [
                "Strategy",
                "Pass Rate",
                "Avg Attempts to Pass",
                "Avg Final Score",
                "Improvement Curve",
                "Total Tasks",
                "Passed Tasks",
                "Failed Tasks",
            ]
        )

        for strategy_name, metrics in sorted(results.items()):
            writer.writerow(
                [
                    strategy_name,
                    f"{metrics.pass_rate:.3f}",
                    f"{metrics.avg_attempts_to_pass:.2f}",
                    f"{metrics.avg_final_score:.3f}",
                    ";".join(f"{x:.3f}" for x in metrics.improvement_curve),
                    len(metrics.per_task_results),
                    sum(1 for r in metrics.per_task_results if r.passed),
                    sum(1 for r in metrics.per_task_results if not r.passed),
                ]
            )


def export_comparison_to_json(report: ComparisonReport, path: str | Path) -> None:
    """Export comparison report to JSON."""
    data = {
        "best_strategy": report.best_strategy,
        "improvement_over_baseline": report.improvement_over_baseline,
        "strategies": {},
        "summary": report.summary,
    }

    for strategy_name, metrics in report.strategies.items():
        data["strategies"][strategy_name] = {
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

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_tasks_to_csv(tasks: list, path: str | Path) -> None:
    """Export task results to CSV."""
    import csv

    if not tasks:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            [
                "Task ID",
                "Query",
                "Expected Answer",
                "Difficulty",
                "Tags",
                "Passed",
                "Attempts",
                "Final Score",
                "Score Progression",
            ]
        )

        for task_result in tasks:
            writer.writerow(
                [
                    task_result.task.id,
                    task_result.task.query,
                    str(task_result.task.expected_answer),
                    task_result.task.difficulty,
                    ";".join(task_result.task.tags),
                    task_result.passed,
                    task_result.attempts,
                    f"{task_result.final_score:.3f}",
                    ";".join(f"{x:.3f}" for x in task_result.score_progression),
                ]
            )
