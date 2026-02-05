"""Tests for improvement strategies."""

import pytest

from src.core.types import Task, ExecutionResult, Evaluation, ImprovementContext
from src.storage.memory import InMemoryStorage
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy


class TestNoImprovementStrategy:
    """Tests for NoImprovementStrategy."""

    @pytest.fixture
    def strategy(self) -> NoImprovementStrategy:
        return NoImprovementStrategy()

    @pytest.fixture
    def context(self) -> ImprovementContext:
        return ImprovementContext(
            task=Task(id="t1", query="test"),
            result=ExecutionResult(output=None, trajectory=[]),
            evaluation=Evaluation(score=0.0, passed=False, feedback="Wrong"),
            attempt_number=1,
        )

    def test_improve_returns_empty(
        self, strategy: NoImprovementStrategy, context: ImprovementContext
    ) -> None:
        """Test that improve returns empty dict."""
        result = strategy.improve(context)
        assert result == {}

    def test_load_priors_returns_empty(
        self, strategy: NoImprovementStrategy
    ) -> None:
        """Test that load_priors returns empty dict."""
        task = Task(id="t1", query="test")
        result = strategy.load_priors(task)
        assert result == {}

    def test_persist_does_nothing(
        self, strategy: NoImprovementStrategy, context: ImprovementContext
    ) -> None:
        """Test that persist does nothing (no error)."""
        strategy.persist(context)  # Should not raise


class TestReflectionStrategy:
    """Tests for ReflectionStrategy."""

    @pytest.fixture
    def storage(self) -> InMemoryStorage:
        return InMemoryStorage()

    @pytest.fixture
    def strategy(self, storage: InMemoryStorage) -> ReflectionStrategy:
        return ReflectionStrategy(storage=storage)

    @pytest.fixture
    def context(self) -> ImprovementContext:
        return ImprovementContext(
            task=Task(id="t1", query="test", tags=("math",)),
            result=ExecutionResult(output=5, trajectory=[]),
            evaluation=Evaluation(
                score=0.5,
                passed=False,
                feedback="Expected 10, got 5",
                error_type="numeric_error",
            ),
            attempt_number=1,
        )

    def test_improve_returns_hints(
        self, strategy: ReflectionStrategy, context: ImprovementContext
    ) -> None:
        """Test that improve returns hints."""
        result = strategy.improve(context)

        assert "hints" in result
        assert len(result["hints"]) > 0
        assert "constraints" in result

    def test_improve_includes_feedback(
        self, strategy: ReflectionStrategy, context: ImprovementContext
    ) -> None:
        """Test that improvements include previous feedback."""
        result = strategy.improve(context)

        constraints = result.get("constraints", [])
        assert any("feedback" in c.lower() for c in constraints)

    def test_improve_tracks_score(
        self, strategy: ReflectionStrategy, context: ImprovementContext
    ) -> None:
        """Test that improvements reference previous score."""
        result = strategy.improve(context)

        assert "previous_score" in result
        assert result["previous_score"] == 0.5

    def test_persist_stores_error_patterns(
        self, strategy: ReflectionStrategy, storage: InMemoryStorage, context: ImprovementContext
    ) -> None:
        """Test that persist stores error patterns."""
        strategy.persist(context)

        # Check error was stored
        error_data = storage.load("reflection/errors/numeric_error")
        assert error_data is not None
        assert error_data["count"] == 1
        assert "t1" in error_data["tasks"]

    def test_persist_accumulates_errors(
        self, strategy: ReflectionStrategy, storage: InMemoryStorage, context: ImprovementContext
    ) -> None:
        """Test that errors accumulate."""
        strategy.improve(context)  # Generate learnings
        strategy.persist(context)
        strategy.persist(context)  # Persist twice

        error_data = storage.load("reflection/errors/numeric_error")
        assert error_data["count"] == 2

    def test_load_priors_retrieves_learnings(
        self, strategy: ReflectionStrategy, storage: InMemoryStorage
    ) -> None:
        """Test that load_priors retrieves stored learnings."""
        # Store some learnings
        storage.save("reflection/tasks/t1", {"learnings": ["hint1", "hint2"]})

        task = Task(id="t1", query="test")
        priors = strategy.load_priors(task)

        assert "hints" in priors
        assert "hint1" in priors["hints"]

    def test_strategy_without_storage(self) -> None:
        """Test strategy works without storage (session-only)."""
        strategy = ReflectionStrategy(storage=None)
        context = ImprovementContext(
            task=Task(id="t1", query="test"),
            result=ExecutionResult(output=None, trajectory=[]),
            evaluation=Evaluation(
                score=0.0,
                passed=False,
                feedback="Error",
                error_type="no_output",
            ),
            attempt_number=1,
        )

        result = strategy.improve(context)
        assert "hints" in result

        # Persist should not raise
        strategy.persist(context)

    def test_pattern_analysis_with_history(
        self, strategy: ReflectionStrategy
    ) -> None:
        """Test that patterns are analyzed from history."""
        # Build history with multiple attempts
        history = [
            (
                ExecutionResult(output=5, trajectory=[]),
                Evaluation(score=0.3, passed=False, feedback="Wrong", error_type="numeric_error"),
            ),
            (
                ExecutionResult(output=7, trajectory=[]),
                Evaluation(score=0.5, passed=False, feedback="Wrong", error_type="numeric_error"),
            ),
        ]

        context = ImprovementContext(
            task=Task(id="t1", query="test"),
            result=ExecutionResult(output=8, trajectory=[]),
            evaluation=Evaluation(score=0.6, passed=False, feedback="Close", error_type="numeric_error"),
            attempt_number=3,
            history=history,
        )

        result = strategy.improve(context)

        # Should detect repeated error pattern
        hints = result.get("hints", [])
        assert any("progress" in h.lower() or "repeated" in h.lower() for h in hints)
