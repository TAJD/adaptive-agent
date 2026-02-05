"""Tests for PerformanceMatrix."""

import pytest
from pathlib import Path

from src.core.types import BenchmarkConfig, BenchmarkRunRecord, StrategyMetrics
from src.benchmark.performance_matrix import PerformanceMatrix, PerformanceCell


class TestPerformanceMatrix:
    """Tests for PerformanceMatrix."""

    def test_build_matrix_single_run(self) -> None:
        """Test building matrix from a single run."""
        config = BenchmarkConfig()
        results = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7, 0.9],
                per_task_results=[],
            ),
            "strategy2": StrategyMetrics(
                strategy_name="strategy2",
                pass_rate=0.6,
                avg_attempts_to_pass=2.0,
                avg_final_score=0.7,
                improvement_curve=[0.4, 0.6, 0.8],
                per_task_results=[],
            ),
        }

        record = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record])

        assert matrix.strategies == ["strategy1", "strategy2"]
        assert matrix.models == ["gpt-4"]

        cell1 = matrix.get_cell("strategy1", "gpt-4")
        assert cell1 is not None
        assert cell1.pass_rate == 0.8
        assert cell1.avg_attempts_to_pass == 1.5
        assert cell1.avg_final_score == 0.9
        assert cell1.improvement_curve == [0.5, 0.7, 0.9]
        assert cell1.num_runs == 1
        assert cell1.run_ids == ["run_001"]

    def test_build_matrix_multiple_runs(self) -> None:
        """Test aggregating multiple runs."""
        config = BenchmarkConfig()

        # Run 1: gpt-4, strategy1
        results1 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            )
        }
        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results1,
            metadata={"model": "gpt-4"},
        )

        # Run 2: gpt-4, strategy1 (different performance)
        results2 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.9,
                avg_attempts_to_pass=1.2,
                avg_final_score=0.95,
                improvement_curve=[0.6, 0.8],
                per_task_results=[],
            )
        }
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T13:00:00Z",
            version="1.0",
            config=config,
            results=results2,
            metadata={"model": "gpt-4"},
        )

        # Run 3: claude, strategy1
        results3 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.7,
                avg_attempts_to_pass=1.8,
                avg_final_score=0.8,
                improvement_curve=[0.4, 0.6],
                per_task_results=[],
            )
        }
        record3 = BenchmarkRunRecord(
            run_id="run_003",
            timestamp="2023-01-01T14:00:00Z",
            version="1.0",
            config=config,
            results=results3,
            metadata={"model": "claude"},
        )

        matrix = PerformanceMatrix([record1, record2, record3])

        assert set(matrix.strategies) == {"strategy1"}
        assert set(matrix.models) == {"claude", "gpt-4"}

        # Check gpt-4 aggregation (average of run1 and run2)
        cell_gpt4 = matrix.get_cell("strategy1", "gpt-4")
        assert cell_gpt4 is not None
        assert cell_gpt4.pass_rate == pytest.approx(0.85)  # (0.8 + 0.9) / 2
        assert cell_gpt4.avg_attempts_to_pass == pytest.approx(1.35)  # (1.5 + 1.2) / 2
        assert cell_gpt4.avg_final_score == pytest.approx(0.925)  # (0.9 + 0.95) / 2
        assert cell_gpt4.improvement_curve == [
            pytest.approx(0.55),
            pytest.approx(0.75),
        ]  # averages
        assert cell_gpt4.num_runs == 2
        assert set(cell_gpt4.run_ids) == {"run_001", "run_002"}

        # Check claude
        cell_claude = matrix.get_cell("strategy1", "claude")
        assert cell_claude is not None
        assert cell_claude.pass_rate == 0.7
        assert cell_claude.num_runs == 1

    def test_get_strategy_performance(self) -> None:
        """Test getting performance for a strategy across models."""
        # Similar setup as above
        config = BenchmarkConfig()
        results1 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            )
        }
        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results1,
            metadata={"model": "gpt-4"},
        )

        results2 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.7,
                avg_attempts_to_pass=1.8,
                avg_final_score=0.8,
                improvement_curve=[0.4, 0.6],
                per_task_results=[],
            )
        }
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T13:00:00Z",
            version="1.0",
            config=config,
            results=results2,
            metadata={"model": "claude"},
        )

        matrix = PerformanceMatrix([record1, record2])

        perf = matrix.get_strategy_performance("strategy1")
        assert len(perf) == 2
        assert "gpt-4" in perf
        assert "claude" in perf

    def test_get_model_performance(self) -> None:
        """Test getting performance for a model across strategies."""
        config = BenchmarkConfig()
        results1 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            ),
            "strategy2": StrategyMetrics(
                strategy_name="strategy2",
                pass_rate=0.6,
                avg_attempts_to_pass=2.0,
                avg_final_score=0.7,
                improvement_curve=[0.4, 0.6],
                per_task_results=[],
            ),
        }
        record = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results1,
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record])

        perf = matrix.get_model_performance("gpt-4")
        assert len(perf) == 2
        assert "strategy1" in perf
        assert "strategy2" in perf

    def test_best_methods(self) -> None:
        """Test finding best strategy/model combinations."""
        config = BenchmarkConfig()

        # strategy1 better on gpt-4
        results1 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.9,
                avg_attempts_to_pass=1.2,
                avg_final_score=0.95,
                improvement_curve=[0.6, 0.8],
                per_task_results=[],
            ),
            "strategy2": StrategyMetrics(
                strategy_name="strategy2",
                pass_rate=0.7,
                avg_attempts_to_pass=1.8,
                avg_final_score=0.8,
                improvement_curve=[0.4, 0.6],
                per_task_results=[],
            ),
        }
        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results1,
            metadata={"model": "gpt-4"},
        )

        # strategy2 better on claude
        results2 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.6,
                avg_attempts_to_pass=2.0,
                avg_final_score=0.7,
                improvement_curve=[0.3, 0.5],
                per_task_results=[],
            ),
            "strategy2": StrategyMetrics(
                strategy_name="strategy2",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.85,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            ),
        }
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T13:00:00Z",
            version="1.0",
            config=config,
            results=results2,
            metadata={"model": "claude"},
        )

        matrix = PerformanceMatrix([record1, record2])

        # Best strategy for gpt-4 should be strategy1
        assert matrix.get_best_strategy_for_model("gpt-4") == "strategy1"

        # Best strategy for claude should be strategy2
        assert matrix.get_best_strategy_for_model("claude") == "strategy2"

        # Best model for strategy1 should be gpt-4
        assert matrix.get_best_model_for_strategy("strategy1") == "gpt-4"

        # Best model for strategy2 should be claude
        assert matrix.get_best_model_for_strategy("strategy2") == "claude"

        # Overall best should be strategy1 on gpt-4 (pass_rate 0.9)
        best = matrix.get_overall_best()
        assert best == ("strategy1", "gpt-4")

    def test_to_csv(self, tmp_path: Path) -> None:
        """Test exporting to CSV."""
        config = BenchmarkConfig()
        results = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7],
                per_task_results=[],
            )
        }
        record = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record])

        csv_path = tmp_path / "test_matrix.csv"
        matrix.to_csv(str(csv_path))

        assert csv_path.exists()

        # Check content
        content = csv_path.read_text()
        assert "Strategy,gpt-4" in content
        assert "strategy1,0.8" in content

    def test_get_strategy_trend(self) -> None:
        """Test getting trend for a strategy-model pair."""
        config = BenchmarkConfig()

        # Earlier run
        results1 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.6,
                avg_attempts_to_pass=2.0,
                avg_final_score=0.7,
                improvement_curve=[0.4, 0.6],
                per_task_results=[],
            )
        }
        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results=results1,
            metadata={"model": "gpt-4"},
        )

        # Later run
        results2 = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.8],
                per_task_results=[],
            )
        }
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-02T10:00:00Z",
            version="1.0",
            config=config,
            results=results2,
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record1, record2])

        trend = matrix.get_strategy_trend("strategy1", "gpt-4", "pass_rate")
        assert len(trend) == 2
        assert trend[0]["timestamp"] == "2023-01-01T10:00:00Z"
        assert trend[0]["value"] == 0.6
        assert trend[1]["timestamp"] == "2023-01-02T10:00:00Z"
        assert trend[1]["value"] == 0.8

    def test_get_overall_trends(self) -> None:
        """Test getting overall trends."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.6, 2.0, 0.7, [0.4, 0.6], [])
            },
            metadata={"model": "gpt-4"},
        )

        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.8, 1.5, 0.9, [0.5, 0.8], [])
            },
            metadata={"model": "claude"},
        )

        matrix = PerformanceMatrix([record1, record2])

        trends = matrix.get_overall_trends("pass_rate")
        assert "strategy1_gpt-4" in trends
        assert "strategy1_claude" in trends
        assert len(trends["strategy1_gpt-4"]) == 1
        assert len(trends["strategy1_claude"]) == 1

    def test_compare_runs_over_time(self) -> None:
        """Test comparing runs over time."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.6, 2.0, 0.7, [], []),
                "strategy2": StrategyMetrics("strategy2", 0.5, 2.5, 0.6, [], []),
            },
            metadata={"model": "gpt-4"},
        )

        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-02T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.8, 1.5, 0.9, [], []),
                "strategy2": StrategyMetrics("strategy2", 0.7, 2.0, 0.8, [], []),
            },
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record1, record2])

        comparison = matrix.compare_runs_over_time("pass_rate")
        assert "strategy1" in comparison
        assert "strategy2" in comparison
        assert comparison["strategy1"] == [0.6, 0.8]
        assert comparison["strategy2"] == [0.5, 0.7]

    def test_get_improvement_trends(self) -> None:
        """Test getting improvement trends."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.6, 2.0, 0.7, [0.4, 0.6], [])
            },
            metadata={"model": "gpt-4"},
        )

        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-02T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.8, 1.5, 0.9, [0.5, 0.8], [])
            },
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record1, record2])

        trends = matrix.get_improvement_trends()
        assert "strategy1" in trends
        assert trends["strategy1"] == pytest.approx([0.2, 0.3])  # 0.6-0.4, 0.8-0.5
