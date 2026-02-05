"""Tests for export functions."""

import json
import csv
from pathlib import Path

import pytest

from src.core.types import (
    BenchmarkConfig,
    BenchmarkRunRecord,
    StrategyMetrics,
    TaskResult,
    Task,
    ComparisonReport,
)
from src.benchmark.export import (
    export_run_to_json,
    export_strategies_to_csv,
    export_comparison_to_json,
    export_tasks_to_csv,
)


class TestExport:
    """Tests for export functions."""

    def test_export_run_to_json(self, tmp_path: Path) -> None:
        """Test exporting a run to JSON."""
        config = BenchmarkConfig(max_attempts=3)
        results = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7, 0.9],
                per_task_results=[
                    TaskResult(
                        task=Task(id="task1", query="test", expected_answer=1),
                        passed=True,
                        attempts=1,
                        final_score=1.0,
                        score_progression=[1.0],
                    )
                ],
            )
        }

        record = BenchmarkRunRecord(
            run_id="test_run",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata={"model": "gpt-4"},
        )

        json_path = tmp_path / "test_run.json"
        export_run_to_json(record, json_path)

        assert json_path.exists()

        # Check content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["run_id"] == "test_run"
        assert data["timestamp"] == "2023-01-01T12:00:00Z"
        assert data["version"] == "1.0"
        assert data["config"]["max_attempts"] == 3
        assert data["metadata"]["model"] == "gpt-4"
        assert "strategy1" in data["results"]
        assert data["results"]["strategy1"]["pass_rate"] == 0.8

    def test_export_strategies_to_csv(self, tmp_path: Path) -> None:
        """Test exporting strategies to CSV."""
        results = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[
                    TaskResult(
                        task=Task(id="t1", query="q", expected_answer=1),
                        passed=True,
                        attempts=1,
                        final_score=1.0,
                        score_progression=[1.0],
                    )
                ],
            ),
            "strategy2": StrategyMetrics(
                strategy_name="strategy2",
                pass_rate=0.6,
                avg_attempts_to_pass=2.0,
                avg_final_score=0.7,
                improvement_curve=[0.4, 0.6],
                per_task_results=[
                    TaskResult(
                        task=Task(id="t1", query="q", expected_answer=1),
                        passed=False,
                        attempts=2,
                        final_score=0.5,
                        score_progression=[0.0, 0.5],
                    )
                ],
            ),
        }

        csv_path = tmp_path / "strategies.csv"
        export_strategies_to_csv(results, csv_path)

        assert csv_path.exists()

        # Check content
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # header + 2 strategies
        assert rows[0] == [
            "Strategy",
            "Pass Rate",
            "Avg Attempts to Pass",
            "Avg Final Score",
            "Improvement Curve",
            "Total Tasks",
            "Passed Tasks",
            "Failed Tasks",
        ]
        assert rows[1][0] == "strategy1"
        assert rows[1][1] == "0.800"
        assert rows[2][0] == "strategy2"
        assert rows[2][1] == "0.600"

    def test_export_comparison_to_json(self, tmp_path: Path) -> None:
        """Test exporting comparison to JSON."""
        strategies = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            )
        }

        report = ComparisonReport(
            strategies=strategies,
            best_strategy="strategy1",
            improvement_over_baseline=0.2,
            summary="Strategy1 is best",
        )

        json_path = tmp_path / "comparison.json"
        export_comparison_to_json(report, json_path)

        assert json_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["best_strategy"] == "strategy1"
        assert data["improvement_over_baseline"] == 0.2
        assert "strategy1" in data["strategies"]

    def test_export_tasks_to_csv(self, tmp_path: Path) -> None:
        """Test exporting tasks to CSV."""
        tasks = [
            TaskResult(
                task=Task(
                    id="task1",
                    query="What is 1+1?",
                    expected_answer=2,
                    difficulty="easy",
                    tags=("math",),
                ),
                passed=True,
                attempts=1,
                final_score=1.0,
                score_progression=[1.0],
            ),
            TaskResult(
                task=Task(
                    id="task2",
                    query="What is 2+2?",
                    expected_answer=4,
                    difficulty="easy",
                    tags=("math",),
                ),
                passed=False,
                attempts=2,
                final_score=0.5,
                score_progression=[0.0, 0.5],
            ),
        ]

        csv_path = tmp_path / "tasks.csv"
        export_tasks_to_csv(tasks, csv_path)

        assert csv_path.exists()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # header + 2 tasks
        assert rows[0][0] == "Task ID"
        assert rows[1][0] == "task1"
        assert rows[1][1] == "What is 1+1?"
        assert rows[1][5] == "True"
        assert rows[2][0] == "task2"
        assert rows[2][5] == "False"
