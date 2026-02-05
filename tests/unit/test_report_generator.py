"""Tests for SummaryReportGenerator."""

from src.core.types import BenchmarkConfig, BenchmarkRunRecord, StrategyMetrics
from src.benchmark.performance_matrix import PerformanceMatrix
from src.benchmark.report_generator import SummaryReportGenerator


class TestSummaryReportGenerator:
    """Tests for SummaryReportGenerator."""

    def test_generate_run_summary(self) -> None:
        """Test generating run summary."""
        config = BenchmarkConfig(max_attempts=3)
        results = {
            "strategy1": StrategyMetrics(
                strategy_name="strategy1",
                pass_rate=0.8,
                avg_attempts_to_pass=1.5,
                avg_final_score=0.9,
                improvement_curve=[0.5, 0.7, 0.9],
                per_task_results=[],
            )
        }

        record = BenchmarkRunRecord(
            run_id="test_run",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata={"model": "gpt-4", "env": "test"},
        )

        generator = SummaryReportGenerator()
        report = generator.generate_run_summary(record)

        assert "# Benchmark Run Summary: test_run" in report
        assert "2023-01-01T12:00:00Z" in report
        assert "strategy1" in report
        assert "80.0%" in report
        assert "gpt-4" in report

    def test_generate_comparison_summary(self) -> None:
        """Test generating comparison summary."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={"strategy1": StrategyMetrics("strategy1", 0.8, 1.5, 0.9, [], [])},
            metadata={"model": "gpt-4"},
        )

        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={"strategy1": StrategyMetrics("strategy1", 0.7, 1.8, 0.8, [], [])},
            metadata={"model": "claude"},
        )

        generator = SummaryReportGenerator()
        report = generator.generate_comparison_summary([record1, record2])

        assert "# Benchmark Comparison Summary" in report
        assert "**Total Runs:** 2" in report
        assert "strategy1" in report
        assert "gpt-4" in report
        assert "claude" in report

    def test_generate_matrix_summary(self) -> None:
        """Test generating matrix summary."""
        config = BenchmarkConfig()

        record = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={
                "strategy1": StrategyMetrics("strategy1", 0.8, 1.5, 0.9, [], []),
                "strategy2": StrategyMetrics("strategy2", 0.6, 2.0, 0.7, [], []),
            },
            metadata={"model": "gpt-4"},
        )

        matrix = PerformanceMatrix([record])
        generator = SummaryReportGenerator()
        report = generator.generate_matrix_summary(matrix)

        assert "# Performance Matrix Summary" in report
        assert "**Strategies:** 2" in report
        assert "**Models:** 1" in report
        assert "strategy1" in report
        assert "strategy2" in report

    def test_generate_trend_summary(self) -> None:
        """Test generating trend summary."""
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
        generator = SummaryReportGenerator()
        report = generator.generate_trend_summary(matrix)

        assert "# Trend Analysis Summary" in report
        assert "strategy1_gpt-4" in report
        assert "+20.0%" in report  # 0.6 to 0.8

    def test_empty_comparison(self) -> None:
        """Test comparison with no records."""
        generator = SummaryReportGenerator()
        report = generator.generate_comparison_summary([])
        assert report == "# No Records to Compare"
