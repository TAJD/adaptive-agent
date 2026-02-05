"""Tests for core type definitions."""

import pytest
from datetime import datetime

from src.core.types import (
    Task,
    BenchmarkConfig,
    ExecutionResult,
    Evaluation,
    TaskResult,
    StrategyMetrics,
    BenchmarkRunRecord,
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test BenchmarkConfig with default values."""
        config = BenchmarkConfig()
        assert config.max_attempts == 5
        assert config.strategies == {}
        assert config.task_suite == []
        assert config.models == []
        assert config.evaluator is None
        assert config.executor is None
        assert config.seed is None

    def test_custom_values(self) -> None:
        """Test BenchmarkConfig with custom values."""
        from src.benchmark.model_config import ModelConfig

        model = ModelConfig(name="test-model", provider="test")
        config = BenchmarkConfig(max_attempts=10, models=[model], seed=42)
        assert config.max_attempts == 10
        assert len(config.models) == 1
        assert config.models[0].name == "test-model"
        assert config.seed == 42


class TestBenchmarkRunRecord:
    """Tests for BenchmarkRunRecord dataclass."""

    def test_creation(self) -> None:
        """Test creating a BenchmarkRunRecord."""
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
        metadata = {"version": "1.0", "git_commit": "abc123"}

        record = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata=metadata,
        )

        assert record.run_id == "run_001"
        assert record.timestamp == "2023-01-01T12:00:00Z"
        assert record.version == "1.0"
        assert record.config == config
        assert record.results == results
        assert record.metadata == metadata

    def test_default_metadata(self) -> None:
        """Test BenchmarkRunRecord with default metadata."""
        config = BenchmarkConfig()
        results = {}

        record = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-02T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
        )

        assert record.metadata == {}
