"""Shared test fixtures."""

import pytest
from pathlib import Path

from src.core.types import Task, ExecutionResult, Evaluation
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.executor.mock import MockExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy


@pytest.fixture
def simple_task() -> Task:
    """A simple task for testing."""
    return Task(
        id="test_sum",
        query="Calculate the sum of [1, 2, 3]",
        expected_answer=6,
        difficulty="easy",
        tags=("arithmetic",),
    )


@pytest.fixture
def string_task() -> Task:
    """A string-based task for testing."""
    return Task(
        id="test_string",
        query="Concatenate 'hello' and 'world' with a space",
        expected_answer="hello world",
        difficulty="easy",
        tags=("string",),
    )


@pytest.fixture
def memory_storage() -> InMemoryStorage:
    """In-memory storage for testing."""
    return InMemoryStorage()


@pytest.fixture
def file_storage(tmp_path: Path) -> FileStorage:
    """File-based storage for testing."""
    return FileStorage(tmp_path / "storage")


@pytest.fixture
def mock_executor() -> MockExecutor:
    """Mock executor for testing."""
    return MockExecutor()


@pytest.fixture
def exact_evaluator() -> ExactMatchEvaluator:
    """Exact match evaluator for testing."""
    return ExactMatchEvaluator()


@pytest.fixture
def no_improvement_strategy() -> NoImprovementStrategy:
    """No-improvement baseline strategy."""
    return NoImprovementStrategy()


@pytest.fixture
def reflection_strategy(memory_storage: InMemoryStorage) -> ReflectionStrategy:
    """Reflection strategy with in-memory storage."""
    return ReflectionStrategy(storage=memory_storage)


@pytest.fixture
def correct_result() -> ExecutionResult:
    """A correct execution result."""
    return ExecutionResult(
        output=6,
        code_generated="result = sum([1, 2, 3])",
        trajectory=[{"step": "calculation", "action": "sum"}],
    )


@pytest.fixture
def incorrect_result() -> ExecutionResult:
    """An incorrect execution result."""
    return ExecutionResult(
        output=5,  # Wrong answer
        code_generated="result = 5",
        trajectory=[{"step": "calculation", "action": "wrong"}],
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    """A set of sample tasks for testing."""
    return [
        Task(id="t1", query="Sum [1,2,3]", expected_answer=6, difficulty="easy"),
        Task(id="t2", query="Mean [10,20]", expected_answer=15.0, difficulty="easy"),
        Task(id="t3", query="Max [1,5,3]", expected_answer=5, difficulty="easy"),
    ]
