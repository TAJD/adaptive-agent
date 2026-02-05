"""Tests for ResultsStore implementations."""

import pytest
from pathlib import Path

from src.core.types import BenchmarkConfig, BenchmarkRunRecord, StrategyMetrics
from src.benchmark.results_store import FileResultsStore


class TestFileResultsStore:
    """Tests for FileResultsStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileResultsStore:
        """Create a FileResultsStore with temporary path."""
        return FileResultsStore(tmp_path / "test_results")

    def test_save_and_load_run(self, store: FileResultsStore) -> None:
        """Test saving and loading a benchmark run."""
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

    def test_load_nonexistent_run(self, store: FileResultsStore) -> None:
        """Test loading a nonexistent run returns None."""
        result = store.load_run("nonexistent")
        assert result is None

    def test_list_runs_empty(self, store: FileResultsStore) -> None:
        """Test listing runs when empty."""
        runs = store.list_runs()
        assert runs == []

    def test_list_runs(self, store: FileResultsStore) -> None:
        """Test listing runs after saving some."""
        # Save multiple runs
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={},
        )
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={},
        )

        store.save_run(record1)
        store.save_run(record2)

        runs = store.list_runs()
        assert set(runs) == {"run_001", "run_002"}

    def test_delete_run(self, store: FileResultsStore) -> None:
        """Test deleting a run."""
        config = BenchmarkConfig()
        record = BenchmarkRunRecord(
            run_id="run_to_delete",
            timestamp="2023-01-01T12:00:00Z",
            version="1.0",
            config=config,
            results={},
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

    def test_query_runs(self, store: FileResultsStore) -> None:
        """Test querying runs."""
        config = BenchmarkConfig()

        record1 = BenchmarkRunRecord(
            run_id="run_001",
            timestamp="2023-01-01T10:00:00Z",
            version="1.0",
            config=config,
            results={},
            metadata={"env": "prod", "version": "1.0"},
        )
        record2 = BenchmarkRunRecord(
            run_id="run_002",
            timestamp="2023-01-01T11:00:00Z",
            version="1.0",
            config=config,
            results={},
            metadata={"env": "test", "version": "1.0"},
        )
        record3 = BenchmarkRunRecord(
            run_id="run_003",
            timestamp="2023-01-01T12:00:00Z",
            version="2.0",
            config=config,
            results={},
            metadata={"env": "prod", "version": "2.0"},
        )

        store.save_run(record1)
        store.save_run(record2)
        store.save_run(record3)

        # Query by metadata
        prod_runs = store.query_runs({"env": "prod"})
        assert len(prod_runs) == 2
        run_ids = {r.run_id for r in prod_runs}
        assert run_ids == {"run_001", "run_003"}

        # Query by version (field)
        v1_runs = store.query_runs({"version": "1.0"})
        assert len(v1_runs) == 2
        run_ids = {r.run_id for r in v1_runs}
        assert run_ids == {"run_001", "run_002"}

        # Query with no matches
        no_matches = store.query_runs({"env": "dev"})
        assert no_matches == []
