"""Tests for SQLiteResultsStore."""

import pytest
from pathlib import Path

from src.core.types import (
    BenchmarkConfig,
    BenchmarkRunRecord,
    StrategyMetrics,
    TaskResult,
    Task,
)
from src.benchmark.results_store import SQLiteResultsStore


class TestSQLiteResultsStore:
    """Tests for SQLiteResultsStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> SQLiteResultsStore:
        """Create a SQLiteResultsStore with temporary database."""
        db_path = tmp_path / "test_results.db"
        return SQLiteResultsStore(db_path)

    def test_save_and_load_run(self, store: SQLiteResultsStore) -> None:
        """Test saving and loading a benchmark run."""
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
                        task=Task(id="task1", query="test query", expected_answer=42),
                        passed=True,
                        attempts=1,
                        final_score=1.0,
                        score_progression=[1.0],
                    )
                ],
            )
        }

        record = BenchmarkRunRecord(
            run_id="test_run_001",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results=results,
            metadata={"env": "test"},
        )

        # Save
        store.save_run(record)

        # Load
        loaded = store.load_run("test_run_001")
        assert loaded is not None
        assert loaded.run_id == "test_run_001"
        assert loaded.timestamp == "2023-01-01T12:00:00Z"
        assert loaded.version == "1.0"
        assert loaded.metadata == {"env": "test"}
        assert loaded.config.max_attempts == 3
        assert "strategy1" in loaded.results
        assert loaded.results["strategy1"].pass_rate == 0.8
        assert len(loaded.results["strategy1"].per_task_results) == 1

    def test_load_nonexistent_run(self, store: SQLiteResultsStore) -> None:
        """Test loading a nonexistent run returns None."""
        result = store.load_run("nonexistent")
        assert result is None

    def test_list_runs_empty(self, store: SQLiteResultsStore) -> None:
        """Test listing runs when empty."""
        runs = store.list_runs()
        assert runs == []

    def test_list_runs(self, store: SQLiteResultsStore) -> None:
        """Test listing runs after saving some."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.5, 1.0, 0.8, [], [])},
            metadata={},
        )
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.6, 1.2, 0.9, [], [])},
            metadata={},
        )

        store.save_run(record1)
        store.save_run(record2)

        runs = store.list_runs()
        assert runs == ["run_001", "run_002"]

    def test_delete_run(self, store: SQLiteResultsStore) -> None:
        """Test deleting a run."""
        config = BenchmarkConfig()
        record = BenchmarkRunRecord(
            run_id="run_to_delete",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.5, 1.0, 0.8, [], [])},
            metadata={},
        )

        store.save_run(record)
        assert store.load_run("run_to_delete") is not None

        # Delete
        deleted = store.delete_run("run_to_delete")
        assert deleted is True

        # Should be gone
        assert store.load_run("run_to_delete") is None

        # Delete nonexistent
        deleted = store.delete_run("nonexistent")
        assert deleted is False

    def test_query_runs(self, store: SQLiteResultsStore) -> None:
        """Test querying runs."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.5, 1.0, 0.8, [], [])},
            metadata={"env": "prod", "version": "1.0"},
        )
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.6, 1.2, 0.9, [], [])},
            metadata={"env": "test", "version": "1.0"},
        )

        store.save_run(record1)
        store.save_run(record2)

        # Query by metadata
        prod_runs = store.query_runs({"env": "prod"})
        assert len(prod_runs) == 1
        assert prod_runs[0].run_id == "run_001"

        # Query by version (field)
        v1_runs = store.query_runs({"version": "1.0"})
        assert len(v1_runs) == 2

        # Query with no matches
        no_matches = store.query_runs({"env": "dev"})
        assert no_matches == []

    def test_save_run_twice(self, store: SQLiteResultsStore) -> None:
        """Test saving the same run twice (should replace)."""
        config = BenchmarkConfig()
        record = BenchmarkRunRecord(
            run_id="run_twice",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results={"s1": StrategyMetrics("s1", 0.5, 1.0, 0.8, [], [])},
            metadata={"first": True},
        )

        store.save_run(record)
        loaded = store.load_run("run_twice")
        assert loaded.metadata["first"] is True

        # Save again with different metadata
        record.metadata = {"second": True}
        store.save_run(record)

        loaded = store.load_run("run_twice")
        assert loaded.metadata["second"] is True
        assert "first" not in loaded.metadata
