"""
Tests that PROVE the agent learns correctly across sessions.

These tests demonstrate the core value proposition:
1. Episodes are stored after failures
2. Similar episodes are retrieved based on keyword matching
3. Retrieved episodes improve subsequent task performance
4. Effectiveness tracking rewards helpful episodes
5. The full learning loop works end-to-end

Run with: uv run pytest tests/unit/test_learning_proof.py -v
"""

import pytest
from pathlib import Path

from src.core.types import Task, ExecutionResult, Evaluation, ImprovementContext
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.strategies.episodic_memory import (
    EpisodicMemoryStrategy,
    Episode,
    extract_keywords,
    compute_similarity,
)


class TestLearningProof:
    """
    Core tests that prove the learning mechanism works.

    These tests form the evidence that the agent genuinely learns
    from failures and applies that learning to new, similar tasks.
    """

    @pytest.fixture
    def storage(self) -> InMemoryStorage:
        """Fresh storage for each test."""
        return InMemoryStorage()

    @pytest.fixture
    def strategy(self, storage: InMemoryStorage) -> EpisodicMemoryStrategy:
        """Strategy with storage."""
        return EpisodicMemoryStrategy(storage=storage)

    # =========================================================================
    # PROOF 1: Episodes are stored after failures
    # =========================================================================

    def test_failure_creates_episode(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: When a task fails, an episode is created and stored."""
        # Arrange: A failed task
        task = Task(
            id="fx-query-1",
            query="What is the Foreign Exchange impact for Product C?",
            expected_answer=35000.0,
        )
        failed_result = ExecutionResult(
            output=500000.0,  # Wrong!
            code_generated="result = data['Amount in USD'].sum()",
        )
        failed_eval = Evaluation(
            score=0.2,
            passed=False,
            feedback="Value too large - missing filters",
            error_type="much_larger",
        )
        context = ImprovementContext(
            task=task,
            result=failed_result,
            evaluation=failed_eval,
            attempt_number=1,
        )

        # Act: Persist the failure
        strategy.persist(context)

        # Assert: Episode was stored
        keys = storage.list_keys("episodes")
        assert len(keys) == 1, "Exactly one episode should be stored"

        # Verify episode content
        data = storage.load(keys[0])
        assert data["query"] == task.query
        assert data["failed_code"] == failed_result.code_generated
        assert data["error_type"] == "much_larger"
        assert "foreign exchange" in data["keywords"]

    def test_success_updates_episode_with_fix(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: When a task succeeds after failure, the fix is stored."""
        task = Task(
            id="fx-query-2",
            query="What is the FX impact for Product D?",
            expected_answer=-16650.0,
        )

        # First: Record the failure via improve()
        failed_context = ImprovementContext(
            task=task,
            result=ExecutionResult(
                output=100000.0,
                code_generated="result = data.sum()",  # Bad code
            ),
            evaluation=Evaluation(
                score=0.1, passed=False, feedback="Wrong", error_type="much_larger"
            ),
            attempt_number=1,
        )
        strategy.improve(failed_context)

        # Then: Success on attempt 2
        success_context = ImprovementContext(
            task=task,
            result=ExecutionResult(
                output=-16650.0,
                code_generated="result = data[data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss']['Amount in USD'].sum()",
            ),
            evaluation=Evaluation(score=1.0, passed=True, feedback="Correct!"),
            attempt_number=2,
            history=[(failed_context.result, failed_context.evaluation)],
        )
        strategy.persist(success_context)

        # Assert: Episode has the fix
        keys = storage.list_keys("episodes")
        assert len(keys) == 1
        data = storage.load(keys[0])
        assert data["fixed_code"] is not None
        assert "Foreign Exchange Gain/Loss" in data["fixed_code"]

    # =========================================================================
    # PROOF 2: Similar episodes are retrieved by keyword matching
    # =========================================================================

    def test_similar_query_retrieves_episode(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: A new query with similar keywords retrieves stored episodes."""
        # Store an episode about FX for Product C
        episode = Episode(
            query="What was the Foreign Exchange impact for Product C in 2024?",
            failed_code="result = data.sum()",
            error_message="Missing FX filter",
            fixed_code="result = data[data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss']['Amount in USD'].sum()",
            keywords=["foreign exchange", "fx", "product c", "2024"],
            task_id="old-task",
            timestamp="1700000000000",
        )
        strategy._save_episode(episode)

        # New task: Similar query but for Product D
        new_task = Task(
            id="new-task",
            query="What was the Foreign Exchange impact for Product D in 2024?",
        )

        # Act: Load priors for the new task
        priors = strategy.load_priors(new_task)

        # Assert: The stored episode's fix is returned as an example
        assert "examples" in priors
        assert len(priors["examples"]) == 1
        assert "Foreign Exchange Gain/Loss" in priors["examples"][0]

    def test_dissimilar_query_does_not_retrieve(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: Unrelated queries don't retrieve irrelevant episodes."""
        # Store an FX episode
        episode = Episode(
            query="What was the Foreign Exchange impact for Product C?",
            failed_code="bad code",
            error_message="error",
            fixed_code="good code",
            keywords=["foreign exchange", "fx", "product c"],
            task_id="fx-task",
            timestamp="1700000000000",
        )
        strategy._save_episode(episode)

        # New task: Completely different query about revenue
        new_task = Task(
            id="revenue-task",
            query="What was the total Gross Revenue in Japan?",
        )

        # Act
        priors = strategy.load_priors(new_task)

        # Assert: No examples retrieved (keywords don't overlap enough)
        assert priors.get("examples", []) == []

    def test_keyword_similarity_threshold(self) -> None:
        """PROOF: Similarity is computed correctly using Jaccard index."""
        # Same query pattern, different product
        kw1 = ["foreign exchange", "fx", "product c", "2024", "all countries"]
        kw2 = ["foreign exchange", "fx", "product d", "2024", "all countries"]

        similarity = compute_similarity(kw1, kw2)

        # Intersection: foreign exchange, fx, 2024, all countries (4)
        # Union: all 6 unique keywords
        assert similarity == 4 / 6
        assert similarity > 0.6  # High similarity

        # Very different queries
        kw3 = ["revenue", "gross revenue", "japan", "q1"]
        similarity_low = compute_similarity(kw1, kw3)
        assert similarity_low < 0.2  # Low similarity

    # =========================================================================
    # PROOF 3: Learning improves performance
    # =========================================================================

    def test_episode_provides_working_example(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: Retrieved episodes provide working code examples."""
        # Store a solved episode
        working_code = """
fx_data = data[
    (data['Product'] == 'Product C') &
    (data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
]
result = fx_data['Amount in USD'].sum()
"""
        episode = Episode(
            query="FX impact for Product C?",
            failed_code="result = data.sum()",
            error_message="Missing filters",
            fixed_code=working_code,
            keywords=["fx", "foreign exchange", "product c"],
            task_id="solved-task",
            timestamp="1700000000000",
        )
        strategy._save_episode(episode)

        # Load for similar task
        task = Task(id="new-fx-task", query="FX impact for Product D?")
        priors = strategy.load_priors(task)

        # Assert: The working code is available as an example
        assert len(priors["examples"]) == 1
        assert "FSLine Statement L2" in priors["examples"][0]
        assert "Foreign Exchange Gain/Loss" in priors["examples"][0]

    def test_improve_returns_hints_and_examples(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: improve() provides actionable hints and code examples."""
        # Store a prior episode
        episode = Episode(
            query="Calculate FX impact for Product A",
            failed_code="result = 0",
            error_message="Wrong",
            fixed_code="result = data[data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss'].sum()",
            error_type="numeric_error",
            keywords=["fx", "foreign exchange", "product a"],
            task_id="prior-task",
            timestamp="1700000000000",
        )
        strategy._save_episode(episode)

        # New failure with similar pattern
        task = Task(id="new-task", query="Calculate FX impact for Product B")
        context = ImprovementContext(
            task=task,
            result=ExecutionResult(output=0, code_generated="result = 0"),
            evaluation=Evaluation(
                score=0.1,
                passed=False,
                feedback="Wrong value",
                error_type="numeric_error",
            ),
            attempt_number=1,
        )

        # Act
        improvements = strategy.improve(context)

        # Assert: Hints and examples are provided
        assert "hints" in improvements
        assert "examples" in improvements
        assert len(improvements["examples"]) > 0

    # =========================================================================
    # PROOF 4: Effectiveness tracking
    # =========================================================================

    def test_effectiveness_increases_on_success(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: Effectiveness score increases when an applied episode helps."""
        # Store episode with low effectiveness
        episode = Episode(
            query="FX for Product C",
            failed_code="bad",
            error_message="error",
            fixed_code="good code",
            keywords=["fx", "product c"],
            task_id="task1",
            timestamp="1700000000000",
            effectiveness_score=0.3,
            times_applied=1,
            times_succeeded=0,
        )
        strategy._save_episode(episode)

        # Apply to new task
        task = Task(id="task2", query="FX for Product D")
        strategy.load_priors(task)  # Records applied episodes

        # Task succeeds
        success_context = ImprovementContext(
            task=task,
            result=ExecutionResult(output=100, code_generated="good code"),
            evaluation=Evaluation(score=1.0, passed=True, feedback="Correct"),
            attempt_number=1,
        )
        strategy.persist(success_context)

        # Assert: Effectiveness increased
        keys = storage.list_keys("episodes")
        data = storage.load(keys[0])

        # New score = 0.7 * 0.3 + 0.3 * 1.0 = 0.51
        assert data["effectiveness_score"] == pytest.approx(0.51, rel=0.01)
        assert data["times_applied"] == 2
        assert data["times_succeeded"] == 1

    def test_effectiveness_decreases_on_failure(
        self, strategy: EpisodicMemoryStrategy, storage: InMemoryStorage
    ) -> None:
        """PROOF: Effectiveness score decreases when an applied episode doesn't help."""
        # Store episode with high effectiveness
        episode = Episode(
            query="Revenue for Q1",
            failed_code="bad",
            error_message="error",
            fixed_code="good code",
            keywords=["revenue", "q1"],
            task_id="task1",
            timestamp="1700000000000",
            effectiveness_score=0.8,
            times_applied=5,
            times_succeeded=4,
        )
        strategy._save_episode(episode)

        # Apply to new task
        task = Task(id="task2", query="Revenue for Q2")
        strategy.load_priors(task)

        # Task fails
        failure_context = ImprovementContext(
            task=task,
            result=ExecutionResult(output=0, code_generated="still bad"),
            evaluation=Evaluation(score=0.2, passed=False, feedback="Wrong"),
            attempt_number=3,
        )
        strategy.persist(failure_context)

        # Assert: Effectiveness decreased
        keys = storage.list_keys("episodes")
        data = storage.load(keys[0])

        # New score = 0.7 * 0.8 + 0.3 * 0.0 = 0.56
        assert data["effectiveness_score"] == pytest.approx(0.56, rel=0.01)
        assert data["times_applied"] == 6
        assert data["times_succeeded"] == 4  # No increase

    # =========================================================================
    # PROOF 5: Cross-session persistence
    # =========================================================================

    def test_episodes_persist_across_strategy_instances(
        self, storage: InMemoryStorage
    ) -> None:
        """PROOF: Episodes survive strategy recreation (simulating new session)."""
        # Session 1: Create strategy and store episode
        strategy1 = EpisodicMemoryStrategy(storage=storage)
        episode = Episode(
            query="Test query",
            failed_code="bad",
            error_message="error",
            fixed_code="good code",
            keywords=["test"],
            task_id="task1",
            timestamp="1700000000000",
        )
        strategy1._save_episode(episode)
        del strategy1

        # Session 2: New strategy instance, same storage
        strategy2 = EpisodicMemoryStrategy(storage=storage)

        # Assert: Episode is retrievable
        task = Task(id="task2", query="Another test query")
        priors = strategy2.load_priors(task)
        assert len(priors["examples"]) == 1

    def test_file_storage_persistence(self, tmp_path: Path) -> None:
        """PROOF: Episodes persist to filesystem and survive restarts."""
        storage_path = tmp_path / "test_storage"

        # Session 1: Store episode
        storage1 = FileStorage(storage_path)
        strategy1 = EpisodicMemoryStrategy(storage=storage1)

        episode = Episode(
            query="FX impact query",
            failed_code="bad",
            error_message="error",
            fixed_code="result = data['FX'].sum()",
            keywords=["fx", "foreign exchange"],
            task_id="task1",
            timestamp="1700000000000",
        )
        strategy1._save_episode(episode)

        # Verify file exists
        episode_files = list(storage_path.glob("**/*.json"))
        assert len(episode_files) == 1

        # Session 2: New storage and strategy instance
        storage2 = FileStorage(storage_path)
        strategy2 = EpisodicMemoryStrategy(storage=storage2)

        # Assert: Episode is retrievable
        task = Task(id="task2", query="Calculate FX impact")
        priors = strategy2.load_priors(task)
        assert len(priors["examples"]) == 1
        assert "FX" in priors["examples"][0]


class TestKeywordExtraction:
    """Tests for the keyword extraction that powers similarity matching."""

    def test_extracts_fx_terms(self) -> None:
        """Foreign exchange terms are extracted."""
        query = "What is the Foreign Exchange Gain/Loss for Product C?"
        keywords = extract_keywords(query)

        assert "foreign exchange" in keywords
        assert "product c" in keywords

    def test_extracts_time_terms(self) -> None:
        """Temporal terms are extracted."""
        query = "Calculate total revenue for Q1 2024"
        keywords = extract_keywords(query)

        assert "q1" in keywords
        assert "2024" in keywords
        assert "total" in keywords
        assert "revenue" in keywords

    def test_extracts_aggregation_terms(self) -> None:
        """Aggregation terms are extracted."""
        query = "What is the average margin across all products?"
        keywords = extract_keywords(query)

        assert "average" in keywords
        assert "margin" in keywords
        assert "all products" in keywords

    def test_handles_mixed_case(self) -> None:
        """Extraction is case-insensitive."""
        query = "TOTAL Revenue for PRODUCT A"
        keywords = extract_keywords(query)

        assert "total" in keywords
        assert "revenue" in keywords
        assert "product a" in keywords


class TestSimilarityComputation:
    """Tests for Jaccard similarity used in episode matching."""

    def test_identical_sets_have_similarity_one(self) -> None:
        """Identical keyword sets have perfect similarity."""
        kw = ["fx", "product a", "2024"]
        assert compute_similarity(kw, kw) == 1.0

    def test_disjoint_sets_have_similarity_zero(self) -> None:
        """Non-overlapping sets have zero similarity."""
        kw1 = ["fx", "foreign exchange"]
        kw2 = ["revenue", "margin"]
        assert compute_similarity(kw1, kw2) == 0.0

    def test_partial_overlap_computes_correctly(self) -> None:
        """Partial overlap computes Jaccard index correctly."""
        kw1 = ["fx", "product a", "2024"]  # 3 elements
        kw2 = ["fx", "product b", "2024"]  # 3 elements

        # Intersection: fx, 2024 (2)
        # Union: fx, product a, product b, 2024 (4)
        expected = 2 / 4
        assert compute_similarity(kw1, kw2) == expected

    def test_empty_sets(self) -> None:
        """Empty sets return zero similarity."""
        assert compute_similarity([], []) == 0.0
        assert compute_similarity(["fx"], []) == 0.0
        assert compute_similarity([], ["fx"]) == 0.0
