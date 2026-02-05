"""Tests for the EpisodicMemoryStrategy."""

import pytest

from src.core.types import Task, ExecutionResult, Evaluation, ImprovementContext
from src.storage.memory import InMemoryStorage
from src.strategies.episodic_memory import (
    EpisodicMemoryStrategy,
    Episode,
    extract_keywords,
    compute_similarity,
)


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_extracts_financial_terms(self) -> None:
        """Test extraction of financial terms."""
        query = "What is the total revenue for Q1 2023?"
        keywords = extract_keywords(query)

        assert "revenue" in keywords
        assert "total" in keywords
        assert "q1" in keywords
        assert "2023" in keywords

    def test_extracts_product_terms(self) -> None:
        """Test extraction of product references."""
        query = "Calculate margin for Product A in United States"
        keywords = extract_keywords(query)

        assert "margin" in keywords
        assert "product a" in keywords
        assert "united states" in keywords

    def test_extracts_comparison_terms(self) -> None:
        """Test extraction of comparison terms."""
        query = "What is the year-over-year growth in OPEX?"
        keywords = extract_keywords(query)

        assert "growth" in keywords
        assert "opex" in keywords

    def test_case_insensitive(self) -> None:
        """Test that extraction is case-insensitive."""
        query = "REVENUE for PRODUCT A in Q1"
        keywords = extract_keywords(query)

        assert "revenue" in keywords
        assert "product a" in keywords
        assert "q1" in keywords


class TestComputeSimilarity:
    """Tests for similarity computation."""

    def test_identical_keywords(self) -> None:
        """Test that identical keywords give similarity of 1."""
        kw = ["revenue", "q1", "2023", "product a"]
        assert compute_similarity(kw, kw) == 1.0

    def test_no_overlap(self) -> None:
        """Test that no overlap gives similarity of 0."""
        kw1 = ["revenue", "q1"]
        kw2 = ["opex", "q2"]
        assert compute_similarity(kw1, kw2) == 0.0

    def test_partial_overlap(self) -> None:
        """Test partial overlap gives expected similarity."""
        kw1 = ["revenue", "q1", "2023"]
        kw2 = ["revenue", "q1", "2024"]
        # Intersection: revenue, q1 (2)
        # Union: revenue, q1, 2023, 2024 (4)
        assert compute_similarity(kw1, kw2) == 0.5

    def test_empty_lists(self) -> None:
        """Test that empty lists give similarity of 0."""
        assert compute_similarity([], []) == 0.0
        assert compute_similarity(["revenue"], []) == 0.0


class TestEpisode:
    """Tests for the Episode dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        episode = Episode(
            query="What is the revenue?",
            failed_code="result = 0",
            error_message="Wrong value",
            task_id="task-1",
            version="1.1",
            change_description="Fixed revenue calculation",
            tags=["finance"],
        )
        d = episode.to_dict()

        assert d["query"] == "What is the revenue?"
        assert d["failed_code"] == "result = 0"
        assert d["task_id"] == "task-1"
        assert d["version"] == "1.1"
        assert d["change_description"] == "Fixed revenue calculation"
        assert d["tags"] == ["finance"]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "query": "What is the revenue?",
            "failed_code": "result = 0",
            "error_message": "Wrong",
            "fixed_code": "result = 100",
            "error_type": "numeric_error",
            "keywords": ["revenue"],
            "task_id": "task-1",
            "timestamp": "",
            "version": "2.0",
            "parent_version": "1.0",
            "change_description": "Updated calculation logic",
            "author": "test_user",
            "tags": ["improvement"],
            "metadata": {"priority": "high"},
        }
        episode = Episode.from_dict(d)

        assert episode.query == "What is the revenue?"
        assert episode.fixed_code == "result = 100"
        assert episode.version == "2.0"
        assert episode.parent_version == "1.0"
        assert episode.change_description == "Updated calculation logic"
        assert episode.author == "test_user"
        assert episode.tags == ["improvement"]
        assert episode.metadata == {"priority": "high"}

    def test_versioning_defaults(self) -> None:
        """Test that versioning fields have proper defaults."""
        episode = Episode(
            query="Test query",
            failed_code="bad code",
            error_message="error",
        )

        assert episode.version == "1.0"
        assert episode.parent_version is None
        assert episode.change_description == ""
        assert episode.author == ""
        assert episode.tags == []
        assert episode.metadata == {}

    def test_versioning_fields(self) -> None:
        """Test setting versioning fields."""
        episode = Episode(
            query="Test query",
            failed_code="bad code",
            error_message="error",
            version="2.1",
            parent_version="2.0",
            change_description="Improved error handling",
            author="developer",
            tags=["bugfix", "performance"],
            metadata={"reviewed": True, "priority": 1},
        )

        assert episode.version == "2.1"
        assert episode.parent_version == "2.0"
        assert episode.change_description == "Improved error handling"
        assert episode.author == "developer"
        assert episode.tags == ["bugfix", "performance"]
        assert episode.metadata == {"reviewed": True, "priority": 1}


class TestEpisodicMemoryStrategy:
    """Tests for the EpisodicMemoryStrategy."""

    @pytest.fixture
    def storage(self) -> InMemoryStorage:
        """Create a fresh storage instance."""
        return InMemoryStorage()

    @pytest.fixture
    def strategy(self, storage: InMemoryStorage) -> EpisodicMemoryStrategy:
        """Create a strategy with storage."""
        return EpisodicMemoryStrategy(storage=storage)

    @pytest.fixture
    def failed_context(self) -> ImprovementContext:
        """Create a context for a failed attempt."""
        task = Task(
            id="task-revenue-q1",
            query="What was the total Gross Revenue for Product A in Q1 2023?",
            expected_answer=150000.0,
        )
        result = ExecutionResult(
            output=100000.0,
            code_generated="result = df[df['Product'] == 'Product A']['Amount in USD'].sum()",
        )
        evaluation = Evaluation(
            score=0.5,
            passed=False,
            feedback="Wrong value. Expected 150000.0, got 100000.0",
            error_type="numeric_error",
        )
        return ImprovementContext(
            task=task,
            result=result,
            evaluation=evaluation,
            attempt_number=1,
        )

    def test_improve_returns_hints(
        self, strategy: EpisodicMemoryStrategy, failed_context: ImprovementContext
    ) -> None:
        """Test that improve returns hints for the error."""
        improvements = strategy.improve(failed_context)

        assert "hints" in improvements
        assert len(improvements["hints"]) > 0
        assert improvements["attempt"] == 2

    def test_persist_stores_episode(
        self,
        strategy: EpisodicMemoryStrategy,
        storage: InMemoryStorage,
        failed_context: ImprovementContext,
    ) -> None:
        """Test that persist stores the failure episode."""
        strategy.persist(failed_context)

        # Check something was stored
        keys = storage.list_keys("episodes")
        assert len(keys) > 0

    def test_persist_updates_episode_on_success(
        self,
        strategy: EpisodicMemoryStrategy,
        storage: InMemoryStorage,
        failed_context: ImprovementContext,
    ) -> None:
        """Test that a successful attempt updates the episode with the fix."""
        # First, record the failure
        strategy.improve(failed_context)

        # Now simulate success
        success_context = ImprovementContext(
            task=failed_context.task,
            result=ExecutionResult(
                output=150000.0,
                code_generated="result = df[(df['Product'] == 'Product A') & (df['FSLine Statement L2'] == 'Gross Revenue')]['Amount in USD'].sum()",
            ),
            evaluation=Evaluation(
                score=1.0,
                passed=True,
                feedback="Correct!",
            ),
            attempt_number=2,
            history=[(failed_context.result, failed_context.evaluation)],
        )
        strategy.persist(success_context)

        # Check the episode has the fix
        keys = storage.list_keys("episodes")
        assert len(keys) > 0
        episode_data = storage.load(keys[0])
        assert episode_data["fixed_code"] is not None

    def test_load_priors_returns_empty_without_storage(self) -> None:
        """Test that load_priors returns empty without storage."""
        strategy = EpisodicMemoryStrategy(storage=None)
        task = Task(id="t1", query="What is revenue?")

        priors = strategy.load_priors(task)

        assert priors == {}

    def test_load_priors_returns_similar_episodes(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """Test that load_priors returns fixes from similar episodes."""
        # Store an episode with a fix
        episode = Episode(
            query="What was the total Gross Revenue for Product B in Q2 2023?",
            failed_code="result = df[df['Product'] == 'Product B']['Amount in USD'].sum()",
            error_message="Wrong value",
            fixed_code="result = df[(df['Product'] == 'Product B') & (df['FSLine Statement L2'] == 'Gross Revenue') & (df['Fiscal Quarter'] == 'Q2') & (df['Fiscal Year'] == 2023)]['Amount in USD'].sum()",
            keywords=["revenue", "gross revenue", "product b", "q2", "2023", "total"],
            task_id="task-old",
        )
        storage.save("episodes/task-old/abc123", episode.to_dict())

        # Query with similar task
        task = Task(
            id="task-new",
            query="What was the total Gross Revenue for Product C in Q3 2023?",
        )

        priors = strategy.load_priors(task)

        # Should find the similar episode and return its fix
        assert "examples" in priors
        assert len(priors["examples"]) > 0

    def test_find_similar_excludes_same_task(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """Test that similar episode search excludes the same task."""
        # Store an episode
        episode = Episode(
            query="What is revenue for Q1?",
            failed_code="result = 0",
            error_message="Wrong",
            keywords=["revenue", "q1"],
            task_id="task-1",
        )
        storage.save("episodes/task-1/abc", episode.to_dict())

        # Search for similar with the same task id
        target = Episode(
            query="What is revenue for Q1?",
            failed_code="",
            error_message="",
            keywords=["revenue", "q1"],
            task_id="task-1",  # Same task
        )
        similar = strategy._find_similar_episodes(target)

        # Should not return the same task's episode
        assert len(similar) == 0

    def test_get_episode_count(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """Test counting stored episodes."""
        assert strategy.get_episode_count() == 0

        storage.save("episodes/t1/a", {"query": "q1"})
        storage.save("episodes/t2/b", {"query": "q2"})

        assert strategy.get_episode_count() == 2

    def test_clear_episodes(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """Test clearing all episodes."""
        storage.save("episodes/t1/a", {"query": "q1"})
        storage.save("episodes/t2/b", {"query": "q2"})

        strategy.clear_episodes()

        assert strategy.get_episode_count() == 0

    def test_strategy_without_storage(self, failed_context: ImprovementContext) -> None:
        """Test strategy works without storage (session-only mode)."""
        strategy = EpisodicMemoryStrategy(storage=None)

        # Should not raise
        improvements = strategy.improve(failed_context)
        assert "hints" in improvements

        # Persist should be a no-op
        strategy.persist(failed_context)

    def test_improve_with_history(self, strategy: EpisodicMemoryStrategy) -> None:
        """Test that improve uses attempt history."""
        task = Task(id="t1", query="Calculate revenue")

        # First attempt
        ctx1 = ImprovementContext(
            task=task,
            result=ExecutionResult(output=100, code_generated="result = 100"),
            evaluation=Evaluation(score=0.3, passed=False, feedback="Too low"),
            attempt_number=1,
        )

        # Second attempt shows progress
        ctx2 = ImprovementContext(
            task=task,
            result=ExecutionResult(output=140, code_generated="result = 140"),
            evaluation=Evaluation(score=0.7, passed=False, feedback="Closer"),
            attempt_number=2,
            history=[(ctx1.result, ctx1.evaluation)],
        )

        improvements = strategy.improve(ctx2)

        # Should mention progress
        assert any("progress" in h.lower() for h in improvements["hints"])


class TestEffectivenessTracking:
    """Tests for episode effectiveness tracking."""

    def test_episode_has_effectiveness_fields(self) -> None:
        """Test that Episode dataclass has effectiveness tracking fields."""
        episode = Episode(
            query="test query", failed_code="test code", error_message="test error"
        )

        assert hasattr(episode, "effectiveness_score")
        assert hasattr(episode, "times_applied")
        assert hasattr(episode, "times_succeeded")
        assert episode.effectiveness_score == 0.0
        assert episode.times_applied == 0
        assert episode.times_succeeded == 0

    def test_load_priors_tracks_applied_episodes(self) -> None:
        """Test that load_priors records which episodes were applied."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Create and save an episode
        episode = Episode(
            query="Calculate total revenue",
            failed_code="result = 100",
            error_message="Wrong value",
            fixed_code="result = df['revenue'].sum()",
            keywords=["revenue", "total"],
            task_id="task1",
        )
        strategy._save_episode(episode)

        # Load priors for a similar task
        task = Task(id="task2", query="What is total revenue?")
        priors = strategy.load_priors(task)

        # Should have recorded the applied episode
        assert "task2" in strategy._applied_episodes
        assert len(strategy._applied_episodes["task2"]) == 1
        assert strategy._applied_episodes["task2"][0].query == episode.query

        # Should return the fixed code as example
        assert len(priors["examples"]) == 1
        assert priors["examples"][0] == episode.fixed_code

    def test_effectiveness_update_on_success(self) -> None:
        """Test that effectiveness scores are updated on success."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Create and save an episode
        episode = Episode(
            query="Calculate total revenue",
            failed_code="result = 100",
            error_message="Wrong value",
            fixed_code="result = df['revenue'].sum()",
            keywords=["revenue", "total"],
            task_id="task1",
            effectiveness_score=0.5,
            times_applied=2,
            times_succeeded=1,
        )
        strategy._save_episode(episode)

        # Simulate applying the episode to a task
        task = Task(id="task2", query="What is total revenue?")
        strategy.load_priors(task)  # This records applied episodes

        # Create success context
        success_context = ImprovementContext(
            task=task,
            result=ExecutionResult(
                output=1000, code_generated="result = df['revenue'].sum()"
            ),
            evaluation=Evaluation(score=1.0, passed=True, feedback="Correct"),
            attempt_number=1,
        )

        # Persist the success
        strategy.persist(success_context)

        # Check that episode effectiveness was updated
        # Reload the episode from storage (keys now include timestamp)
        query_hash = __import__("hashlib").md5(episode.query.encode()).hexdigest()[:8]
        prefix = f"{strategy.STORAGE_PREFIX}/task1/{query_hash}"
        matching_keys = storage.list_keys(prefix)
        assert len(matching_keys) == 1
        updated_data = storage.load(matching_keys[0])

        assert updated_data is not None
        assert updated_data["effectiveness_score"] == 0.7 * 0.5 + 0.3 * 1.0  # 0.65
        assert updated_data["times_applied"] == 3
        assert updated_data["times_succeeded"] == 2

    def test_effectiveness_update_on_failure(self) -> None:
        """Test that effectiveness scores are updated on failure."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Create and save an episode
        episode = Episode(
            query="Calculate total revenue",
            failed_code="result = 100",
            error_message="Wrong value",
            fixed_code="result = df['revenue'].sum()",
            keywords=["revenue", "total"],
            task_id="task1",
            effectiveness_score=0.8,
            times_applied=3,
            times_succeeded=2,
        )
        strategy._save_episode(episode)

        # Simulate applying the episode to a task
        task = Task(id="task2", query="What is total revenue?")
        strategy.load_priors(task)

        # Create failure context
        failure_context = ImprovementContext(
            task=task,
            result=ExecutionResult(
                output=500, code_generated="result = df['revenue'].sum() / 2"
            ),
            evaluation=Evaluation(
                score=0.5, passed=False, feedback="Wrong calculation"
            ),
            attempt_number=1,
        )

        # Persist the failure
        strategy.persist(failure_context)

        # Check that episode effectiveness was updated (keys now include timestamp)
        query_hash = __import__("hashlib").md5(episode.query.encode()).hexdigest()[:8]
        prefix = f"{strategy.STORAGE_PREFIX}/task1/{query_hash}"
        matching_keys = storage.list_keys(prefix)
        assert len(matching_keys) == 1
        updated_data = storage.load(matching_keys[0])

        assert updated_data is not None
        assert updated_data["effectiveness_score"] == 0.7 * 0.8 + 0.3 * 0.0  # 0.56
        assert updated_data["times_applied"] == 4
        assert updated_data["times_succeeded"] == 2  # No increase

    def test_no_effectiveness_update_without_applied_episodes(self) -> None:
        """Test that effectiveness is not updated when no episodes were applied."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Create and save an episode
        episode = Episode(
            query="Calculate total revenue",
            failed_code="result = 100",
            error_message="Wrong value",
            fixed_code="result = df['revenue'].sum()",
            keywords=["revenue", "total"],
            task_id="task1",
            effectiveness_score=0.5,
            times_applied=1,
            times_succeeded=0,
        )
        strategy._save_episode(episode)

        # Create success context without loading priors first
        task = Task(id="task2", query="Different query without similarity")
        success_context = ImprovementContext(
            task=task,
            result=ExecutionResult(output=1000, code_generated="result = 1000"),
            evaluation=Evaluation(score=1.0, passed=True, feedback="Correct"),
            attempt_number=1,
        )

        # Persist the success
        strategy.persist(success_context)

        # Check that episode effectiveness was NOT updated (no applied episodes)
        # Keys now include timestamp, so use prefix search
        query_hash = __import__("hashlib").md5(episode.query.encode()).hexdigest()[:8]
        prefix = f"{strategy.STORAGE_PREFIX}/task1/{query_hash}"
        matching_keys = storage.list_keys(prefix)
        assert len(matching_keys) == 1
        data = storage.load(matching_keys[0])

        assert data["effectiveness_score"] == 0.5  # Unchanged
        assert data["times_applied"] == 1  # Unchanged
        assert data["times_succeeded"] == 0  # Unchanged

    def test_applied_episodes_cleanup(self) -> None:
        """Test that applied episodes are cleaned up after persist."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Create and save an episode
        episode = Episode(
            query="Calculate total revenue",
            failed_code="result = 100",
            error_message="Wrong value",
            fixed_code="result = df['revenue'].sum()",
            keywords=["revenue", "total"],
            task_id="task1",
        )
        strategy._save_episode(episode)

        # Load priors
        task = Task(id="task2", query="What is total revenue?")
        strategy.load_priors(task)

        # Verify applied episodes are recorded
        assert "task2" in strategy._applied_episodes

        # Persist success
        success_context = ImprovementContext(
            task=task,
            result=ExecutionResult(
                output=1000, code_generated="result = df['revenue'].sum()"
            ),
            evaluation=Evaluation(score=1.0, passed=True, feedback="Correct"),
            attempt_number=1,
        )
        strategy.persist(success_context)

        # Applied episodes should be cleaned up
        assert "task2" not in strategy._applied_episodes
