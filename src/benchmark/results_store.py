"""Results storage for benchmark persistence."""

import json
import pickle
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from src.core.types import BenchmarkRunRecord, StrategyMetrics, TaskResult


class ResultsStore(Protocol):
    """Protocol for storing and retrieving benchmark results."""

    @abstractmethod
    def save_run(self, record: BenchmarkRunRecord) -> None:
        """Save a benchmark run record."""
        ...

    @abstractmethod
    def load_run(self, run_id: str) -> BenchmarkRunRecord | None:
        """Load a benchmark run record by ID."""
        ...

    @abstractmethod
    def list_runs(self) -> list[str]:
        """List all run IDs."""
        ...

    @abstractmethod
    def delete_run(self, run_id: str) -> bool:
        """Delete a benchmark run record. Returns True if deleted."""
        ...

    @abstractmethod
    def query_runs(self, filter_dict: dict) -> list[BenchmarkRunRecord]:
        """Query runs matching the filter criteria."""
        ...


class FileResultsStore:
    """File-based implementation of ResultsStore using pickle."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        if base_path is None:
            base_path = Path(".benchmark_results")
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, run_id: str) -> Path:
        """Get file path for a run ID."""
        return self._base_path / f"{run_id}.pkl"

    def save_run(self, record: BenchmarkRunRecord) -> None:
        """Save a benchmark run record."""
        path = self._get_path(record.run_id)
        with open(path, "wb") as f:
            pickle.dump(record, f)

    def load_run(self, run_id: str) -> BenchmarkRunRecord | None:
        """Load a benchmark run record by ID."""
        path = self._get_path(run_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError):
            return None

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        runs = []
        for path in self._base_path.glob("*.pkl"):
            runs.append(path.stem)
        return sorted(runs)

    def delete_run(self, run_id: str) -> bool:
        """Delete a benchmark run record."""
        path = self._get_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def query_runs(self, filter_dict: dict) -> list[BenchmarkRunRecord]:
        """Query runs matching the filter criteria."""
        # Load all runs and filter
        all_runs = []
        for run_id in self.list_runs():
            record = self.load_run(run_id)
            if record:
                all_runs.append(record)

        # Simple filtering on metadata and top-level fields
        results = []
        for record in all_runs:
            match = True
            for key, value in filter_dict.items():
                if key in record.metadata:
                    if record.metadata[key] != value:
                        match = False
                        break
                elif hasattr(record, key):
                    if getattr(record, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                results.append(record)

        return results


class SQLiteResultsStore:
    """SQLite-based implementation of ResultsStore for complex queries."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    version TEXT,
                    config TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    strategy_name TEXT,
                    pass_rate REAL,
                    avg_attempts_to_pass REAL,
                    avg_final_score REAL,
                    improvement_curve TEXT,
                    num_runs INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT,
                    strategy_name TEXT,
                    task_id TEXT,
                    passed BOOLEAN,
                    attempts INTEGER,
                    final_score REAL,
                    score_progression TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)
            conn.commit()

    def save_run(self, record: BenchmarkRunRecord) -> None:
        """Save a benchmark run record."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert run
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, timestamp, version, config, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    record.run_id,
                    record.timestamp,
                    record.version,
                    json.dumps(
                        {
                            "max_attempts": record.config.max_attempts,
                            "strategies": list(record.config.strategies.keys()),
                            "num_tasks": len(record.config.task_suite),
                            "seed": record.config.seed,
                        }
                    ),
                    json.dumps(record.metadata),
                ),
            )

            # Insert strategies
            for strategy_name, metrics in record.results.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO strategies
                    (run_id, strategy_name, pass_rate, avg_attempts_to_pass, avg_final_score, improvement_curve, num_runs)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.run_id,
                        strategy_name,
                        metrics.pass_rate,
                        metrics.avg_attempts_to_pass,
                        metrics.avg_final_score,
                        json.dumps(metrics.improvement_curve),
                        len(metrics.per_task_results),
                    ),
                )

                # Insert tasks
                for task_result in metrics.per_task_results:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO tasks
                        (run_id, strategy_name, task_id, passed, attempts, final_score, score_progression)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record.run_id,
                            strategy_name,
                            task_result.task.id,
                            task_result.passed,
                            task_result.attempts,
                            task_result.final_score,
                            json.dumps(task_result.score_progression),
                        ),
                    )

            conn.commit()

    def load_run(self, run_id: str) -> BenchmarkRunRecord | None:
        """Load a benchmark run record by ID."""
        with sqlite3.connect(self.db_path) as conn:
            # Load run
            run_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not run_row:
                return None

            run_id, timestamp, version, config_json, metadata_json = run_row
            config_data = json.loads(config_json)
            metadata = json.loads(metadata_json)

            # Load strategies
            results = {}
            strategy_rows = conn.execute(
                "SELECT * FROM strategies WHERE run_id = ?", (run_id,)
            ).fetchall()
            for row in strategy_rows:
                (
                    _,
                    _,
                    strategy_name,
                    pass_rate,
                    avg_attempts,
                    avg_score,
                    curve_json,
                    num_runs,
                ) = row
                improvement_curve = json.loads(curve_json)

                # Load tasks for this strategy
                task_rows = conn.execute(
                    """
                    SELECT task_id, passed, attempts, final_score, score_progression
                    FROM tasks WHERE run_id = ? AND strategy_name = ?
                """,
                    (run_id, strategy_name),
                ).fetchall()

                per_task_results = []
                for t_row in task_rows:
                    task_id, passed, attempts, final_score, prog_json = t_row
                    score_progression = json.loads(prog_json)
                    # Note: we don't have full Task object, just reconstruct minimal
                    from src.core.types import Task

                    task = Task(id=task_id, query="", expected_answer=None)  # Minimal
                    per_task_results.append(
                        TaskResult(
                            task=task,
                            passed=bool(passed),
                            attempts=attempts,
                            final_score=final_score,
                            score_progression=score_progression,
                        )
                    )

                results[strategy_name] = StrategyMetrics(
                    strategy_name=strategy_name,
                    pass_rate=pass_rate,
                    avg_attempts_to_pass=avg_attempts,
                    avg_final_score=avg_score,
                    improvement_curve=improvement_curve,
                    per_task_results=per_task_results,
                )

            # Reconstruct config (minimal)
            from src.core.types import BenchmarkConfig

            config = BenchmarkConfig(
                max_attempts=config_data.get("max_attempts", 5),
                strategies={},  # Can't reconstruct full
                task_suite=[],  # Can't reconstruct
                seed=config_data.get("seed"),
            )

            return BenchmarkRunRecord(
                run_id=run_id,
                timestamp=timestamp,
                version=version,
                config=config,
                results=results,
                metadata=metadata,
            )

    def list_runs(self) -> list[str]:
        """List all run IDs."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT run_id FROM runs ORDER BY timestamp").fetchall()
            return [row[0] for row in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a benchmark run record."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if exists
            exists = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not exists:
                return False

            # Delete (cascade via foreign keys would be better, but manual)
            conn.execute("DELETE FROM tasks WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM strategies WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return True

    def query_runs(self, filter_dict: dict) -> list[BenchmarkRunRecord]:
        """Query runs matching the filter criteria."""
        # For simplicity, load all and filter in Python
        # For complex queries, could use SQL
        all_runs = []
        for run_id in self.list_runs():
            record = self.load_run(run_id)
            if record:
                all_runs.append(record)

        results = []
        for record in all_runs:
            match = True
            for key, value in filter_dict.items():
                if key in record.metadata:
                    if record.metadata[key] != value:
                        match = False
                        break
                elif hasattr(record, key):
                    if getattr(record, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                results.append(record)

        return results
