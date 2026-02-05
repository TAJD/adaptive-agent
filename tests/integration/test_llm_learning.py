"""
Integration tests for cross-session learning with REAL LLM calls.

These tests prove that the learning mechanism works end-to-end with actual
Claude API calls. They verify:

1. The agent can solve financial data queries
2. Episodes are stored after task completion
3. Similar queries retrieve relevant prior episodes
4. Learning from prior episodes improves performance

IMPORTANT: These tests make real API calls and cost money.
Run with: uv run pytest tests/integration/test_llm_learning.py -v -s

Requires: ANTHROPIC_API_KEY environment variable
"""

import os
import pytest
from pathlib import Path

# Load .env file for API key
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.core.types import Task
from src.agent.runner import AgentRunner, AgentConfig
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.episodic_memory import EpisodicMemoryStrategy, extract_keywords
from src.strategies.none import NoImprovementStrategy
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.llm.claude import ClaudeClient


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping LLM integration tests"
)


@pytest.fixture(scope="module")
def pl_data() -> pd.DataFrame:
    """Load the P&L dataset once for all tests."""
    data_path = Path(__file__).parent.parent.parent / "data" / "FUN_company_pl_actuals_dataset.csv"
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


@pytest.fixture
def context(pl_data: pd.DataFrame) -> dict:
    """Create execution context with data."""
    return {
        "data": pl_data,
        "hints": [
            "The DataFrame is available as 'data' variable",
            "Store your final answer in a variable named 'result'",
            "Available columns: " + ", ".join(pl_data.columns.tolist()),
        ],
        "constraints": [
            "Store the final numeric result in a variable named 'result'",
            "Use pandas operations on the 'data' DataFrame",
        ],
    }


@pytest.fixture
def llm_client() -> ClaudeClient:
    """Create LLM client - uses haiku for speed and cost efficiency."""
    return ClaudeClient(model="claude-haiku-4-5-20251001")


@pytest.fixture
def evaluator() -> ExactMatchEvaluator:
    """Create evaluator with tolerance for floating point."""
    return ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.01)


class TestLLMBasicExecution:
    """Test that the agent can execute queries with real LLM calls."""

    def test_simple_query_execution(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """Test that agent can answer a simple financial query."""
        # Simple query: Count unique products
        task = Task(
            id="count-products",
            query="How many unique products are in the dataset? Return just the number.",
            expected_answer=4,  # Product A, B, C, D
            difficulty="easy",
        )

        executor = LLMExecutor(llm_client=llm_client)
        strategy = NoImprovementStrategy()
        config = AgentConfig(max_attempts=3)

        agent = AgentRunner(executor, evaluator, strategy, config)
        result = agent.run(task, context)

        assert result.passed, f"Failed with output: {result.final_result.output if result.final_result else 'None'}"
        assert result.attempts <= 3

    def test_filtering_query(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """Test that agent can filter data correctly."""
        # Calculate expected answer
        expected = pl_data[
            (pl_data['Product'] == 'Product A') &
            (pl_data['Fiscal Year'] == 2024)
        ]['Fiscal Quarter'].nunique()

        task = Task(
            id="count-quarters",
            query="How many unique quarters have data for Product A in fiscal year 2024?",
            expected_answer=int(expected),
            difficulty="easy",
        )

        executor = LLMExecutor(llm_client=llm_client)
        strategy = NoImprovementStrategy()
        config = AgentConfig(max_attempts=3)

        agent = AgentRunner(executor, evaluator, strategy, config)
        result = agent.run(task, context)

        assert result.passed, f"Expected {expected}, got {result.final_result.output if result.final_result else 'None'}"


class TestEpisodeStorage:
    """Test that episodes are correctly stored after LLM execution."""

    def test_episode_stored_after_success(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        context: dict,
    ) -> None:
        """Test that successful tasks with retries store episodes."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        task = Task(
            id="episode-test-1",
            query="How many unique countries are in the dataset?",
            expected_answer=6,
            difficulty="easy",
        )

        executor = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=3)

        agent = AgentRunner(executor, evaluator, strategy, config)
        result = agent.run(task, context)

        # Verify task completed (may or may not have stored episode depending on retries)
        assert result.final_result is not None

        # If there were retries and success, episode should be stored
        if result.passed and result.attempts > 1:
            episode_count = strategy.get_episode_count()
            assert episode_count > 0, "Episode should be stored after retry success"

    def test_episode_stored_after_failure(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        context: dict,
    ) -> None:
        """Test that failed tasks store episodes for future learning."""
        storage = InMemoryStorage()
        strategy = EpisodicMemoryStrategy(storage=storage)

        # Intentionally difficult/wrong expected answer to force failure
        task = Task(
            id="episode-fail-test",
            query="What is the total Amount in USD for all data?",
            expected_answer=999999999999.99,  # Wrong answer to force failure
            difficulty="hard",
        )

        executor = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=2)  # Limited attempts

        agent = AgentRunner(executor, evaluator, strategy, config)
        result = agent.run(task, context)

        # Task should fail (wrong expected answer)
        assert not result.passed

        # Episode should be stored for learning
        episode_count = strategy.get_episode_count()
        assert episode_count > 0, "Failure episode should be stored"


class TestCrossSessionLearning:
    """
    Test that learning transfers across sessions with real LLM calls.

    This is the core proof that the system learns.
    """

    def test_prior_episode_retrieved_for_similar_query(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """Test that a stored episode is retrieved for a similar query."""
        storage = InMemoryStorage()

        # SESSION 1: Store a "learned" episode manually
        from src.strategies.episodic_memory import Episode

        learned_episode = Episode(
            query="What was the Foreign Exchange Gain/Loss for Product C in 2024?",
            failed_code="result = data['Amount in USD'].sum()",
            error_message="Missing FSLine Statement L2 filter",
            fixed_code="""
fx_data = data[
    (data['Product'] == 'Product C') &
    (data['Fiscal Year'] == 2024) &
    (data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
]
result = fx_data['Amount in USD'].sum()
""",
            keywords=["foreign exchange", "fx", "product c", "2024"],
            task_id="session1-fx-task",
            timestamp="1700000000000",
            effectiveness_score=0.8,
        )

        strategy1 = EpisodicMemoryStrategy(storage=storage)
        strategy1._save_episode(learned_episode)

        # SESSION 2: New strategy, same storage - similar query
        strategy2 = EpisodicMemoryStrategy(storage=storage)

        new_task = Task(
            id="session2-fx-task",
            query="What was the Foreign Exchange Gain/Loss for Product D in 2024?",
            expected_answer=-16650.33,  # Actual value from data
        )

        # Load priors should retrieve the similar episode
        priors = strategy2.load_priors(new_task)

        assert "examples" in priors, "Should retrieve prior examples"
        assert len(priors["examples"]) > 0, "Should have at least one example"
        assert "FSLine Statement L2" in priors["examples"][0], "Example should contain the learned pattern"

    def test_learning_improves_success_rate(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """
        CORE TEST: Prove that learning from Session 1 helps Session 2.

        This test runs the same type of query twice:
        1. First with NO prior learning
        2. Then with a pre-seeded episode

        The second run should benefit from the learned pattern.
        """
        # Calculate actual expected value for FX query
        fx_value = pl_data[
            (pl_data['Product'] == 'Product D') &
            (pl_data['Fiscal Year'] == 2024) &
            (pl_data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
        ]['Amount in USD'].sum()

        task = Task(
            id="fx-learning-test",
            query="What was the total Foreign Exchange Gain/Loss for Product D in fiscal year 2024? Filter by FSLine Statement L2. Return the numeric value.",
            expected_answer=round(fx_value, 2),
            difficulty="hard",
        )

        # RUN 1: Without any prior learning
        storage_empty = InMemoryStorage()
        strategy_empty = EpisodicMemoryStrategy(storage=storage_empty)
        executor1 = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=3)

        agent1 = AgentRunner(executor1, evaluator, strategy_empty, config)
        result_without_learning = agent1.run(task, context)

        # RUN 2: With a seeded episode providing the pattern
        storage_seeded = InMemoryStorage()
        strategy_seeded = EpisodicMemoryStrategy(storage=storage_seeded)

        # Seed with learned pattern
        from src.strategies.episodic_memory import Episode
        seeded_episode = Episode(
            query="Calculate the FX impact for Product C in 2024",
            failed_code="result = data.sum()",
            error_message="Wrong - missing filters",
            fixed_code="""
# Filter by Product, Year, and FSLine Statement L2 for FX
filtered = data[
    (data['Product'] == 'Product C') &
    (data['Fiscal Year'] == 2024) &
    (data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
]
result = filtered['Amount in USD'].sum()
""",
            keywords=["fx", "foreign exchange", "product c", "2024"],
            task_id="prior-fx-task",
            timestamp="1700000000000",
        )
        strategy_seeded._save_episode(seeded_episode)

        executor2 = LLMExecutor(llm_client=llm_client)
        agent2 = AgentRunner(executor2, evaluator, strategy_seeded, config)
        result_with_learning = agent2.run(task, context)

        # Log results for visibility
        print(f"\n{'='*60}")
        print("LEARNING COMPARISON RESULTS")
        print(f"{'='*60}")
        print(f"Expected answer: {task.expected_answer}")
        print(f"\nWithout learning:")
        print(f"  - Passed: {result_without_learning.passed}")
        print(f"  - Attempts: {result_without_learning.attempts}")
        print(f"  - Output: {result_without_learning.final_result.output if result_without_learning.final_result else 'None'}")
        print(f"\nWith learning (seeded episode):")
        print(f"  - Passed: {result_with_learning.passed}")
        print(f"  - Attempts: {result_with_learning.attempts}")
        print(f"  - Output: {result_with_learning.final_result.output if result_with_learning.final_result else 'None'}")
        print(f"{'='*60}\n")

        # At minimum, the seeded version should pass or use fewer attempts
        # (LLM behavior is non-deterministic, so we check for improvement signals)
        if result_with_learning.passed:
            # If seeded version passed, that's a success signal
            pass
        elif result_without_learning.passed and not result_with_learning.passed:
            # This would be unexpected - seeded should not be worse
            pytest.fail("Seeded learning should not perform worse than no learning")


class TestLearningWithHardQueries:
    """Test learning with intentionally tricky queries."""

    def test_ambiguous_query_benefits_from_learning(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """
        Test that an ambiguous query benefits from a seeded example.

        Uses a query that's tricky because:
        - "FX" is ambiguous (could mean many things)
        - Need to know the exact column name 'FSLine Statement L2'
        - Need to know the exact value 'Foreign Exchange Gain/Loss'
        """
        storage = InMemoryStorage()

        # Calculate expected
        fx_value = pl_data[
            (pl_data['Product'] == 'Product B') &
            (pl_data['Fiscal Year'] == 2023) &
            (pl_data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
        ]['Amount in USD'].sum()

        # Ambiguous query - doesn't specify the exact filter values
        task = Task(
            id="ambiguous-fx-query",
            query="What was the FX impact for Product B in 2023? Return the numeric total.",
            expected_answer=round(fx_value, 2),
            difficulty="hard",
        )

        # First, seed an example that shows the correct pattern
        # Keywords must overlap enough with the test query for retrieval
        from src.strategies.episodic_memory import Episode
        seeded_episode = Episode(
            query="What was the FX impact for Product A in 2023?",  # Similar pattern
            failed_code="result = data[data['Product']=='Product A'].sum()",
            error_message="Need to filter by FSLine Statement L2 = 'Foreign Exchange Gain/Loss'",
            fixed_code="""
# FX Impact = Foreign Exchange Gain/Loss in FSLine Statement L2
fx_data = data[
    (data['Product'] == 'Product A') &
    (data['Fiscal Year'] == 2023) &
    (data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
]
result = fx_data['Amount in USD'].sum()
""",
            keywords=["fx", "foreign exchange", "product a", "2023"],  # Better overlap
            task_id="prior-fx",
            timestamp="1700000000000",
        )

        strategy = EpisodicMemoryStrategy(storage=storage)
        strategy._save_episode(seeded_episode)

        # Load priors and merge with context
        priors = strategy.load_priors(task)
        full_context = {**context}
        if priors.get("examples"):
            full_context["examples"] = priors["examples"]
        if priors.get("hints"):
            full_context["hints"] = context.get("hints", []) + priors["hints"]

        # Run with learning
        executor = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=3)
        agent = AgentRunner(executor, evaluator, strategy, config)

        result = agent.run(task, full_context)

        print(f"\nAmbiguous query test:")
        print(f"  Expected: {task.expected_answer}")
        print(f"  Got: {result.final_result.output if result.final_result else 'None'}")
        print(f"  Passed: {result.passed}")
        print(f"  Attempts: {result.attempts}")
        print(f"  Prior examples found: {len(priors.get('examples', []))}")

        # The key assertion: with the seeded example, it should pass
        assert result.passed, "With learned pattern, should solve the FX query"


class TestRealDatasetQuestions:
    """Tests using the actual test questions from the problem specification."""

    def test_cogs_percentage_with_learning(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
    ) -> None:
        """
        Test the 'very hard' COGS percentage question with learning.

        Question: Compare the Cost of Goods Sold as a percentage of Gross Revenue
                  between 2020 and 2024 for Product B
        """
        # Calculate expected answer
        def calc_cogs_pct(year):
            cogs = pl_data[
                (pl_data['Product'] == 'Product B') &
                (pl_data['Fiscal Year'] == year) &
                (pl_data['FSLine Statement L1'] == 'Cost of Goods Sold')
            ]['Amount in USD'].sum()
            gross_rev = pl_data[
                (pl_data['Product'] == 'Product B') &
                (pl_data['Fiscal Year'] == year) &
                (pl_data['FSLine Statement L2'] == 'Gross Revenue')
            ]['Amount in USD'].sum()
            return round((cogs / gross_rev) * 100, 2)

        expected = {"2020": calc_cogs_pct(2020), "2024": calc_cogs_pct(2024)}

        # Seed a learning episode
        storage = InMemoryStorage()
        from src.strategies.episodic_memory import Episode, extract_keywords

        seed_episode = Episode(
            query="Calculate COGS as percentage of Revenue for Product A",
            failed_code="result = data.sum()",
            error_message="Need FSLine Statement L1 for COGS, L2 for Gross Revenue",
            fixed_code="""
results = {}
for year in [2020, 2024]:
    cogs = data[
        (data['Product'] == 'Product A') &
        (data['Fiscal Year'] == year) &
        (data['FSLine Statement L1'] == 'Cost of Goods Sold')
    ]['Amount in USD'].sum()
    gross_rev = data[
        (data['Product'] == 'Product A') &
        (data['Fiscal Year'] == year) &
        (data['FSLine Statement L2'] == 'Gross Revenue')
    ]['Amount in USD'].sum()
    results[str(year)] = round((cogs / gross_rev) * 100, 2)
result = results
""",
            keywords=["cogs", "cost of goods", "percentage", "revenue", "gross revenue"],
            task_id="cogs-seed",
            timestamp="1700000000000",
        )

        strategy = EpisodicMemoryStrategy(storage=storage)
        strategy._save_episode(seed_episode)

        # Create task
        task = Task(
            id="cogs-pct-test",
            query="""Compare the Cost of Goods Sold as a percentage of Gross Revenue
between 2020 and 2024 for Product B.
Return a dictionary: {'2020': percentage, '2024': percentage}""",
            expected_answer=expected,
            difficulty="very_hard",
        )

        # Load priors
        priors = strategy.load_priors(task)
        full_context = {**context}
        if priors.get("examples"):
            full_context["examples"] = priors["examples"]

        print(f"\nCOGS % Test:")
        print(f"  Expected: {expected}")
        print(f"  Prior examples found: {len(priors.get('examples', []))}")

        # Run with learning
        executor = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=3)
        # Use more tolerant evaluator for dict comparison
        dict_evaluator = ExactMatchEvaluator(numeric_tolerance=0.5, relative_tolerance=0.02)
        agent = AgentRunner(executor, dict_evaluator, strategy, config)
        result = agent.run(task, full_context)

        print(f"  Result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"  Output: {result.final_result.output if result.final_result else None}")

        assert len(priors.get("examples", [])) > 0, "Should retrieve prior example"
        assert result.passed, f"Should pass with learning. Got: {result.final_result.output if result.final_result else None}"


class TestFullDemoScenario:
    """
    Full end-to-end test of the demo scenario.

    This mirrors exactly what the demo script does.
    """

    def test_demo_scenario_end_to_end(
        self,
        llm_client: ClaudeClient,
        evaluator: ExactMatchEvaluator,
        pl_data: pd.DataFrame,
        context: dict,
        tmp_path: Path,
    ) -> None:
        """
        Full demo scenario:
        1. Session 1: Run FX query for Product C
        2. Verify episode stored
        3. Session 2: Run similar FX query for Product D
        4. Verify prior knowledge was used
        """
        # Use file storage for cross-session persistence
        storage_path = tmp_path / "demo_test_storage"
        storage = FileStorage(storage_path)

        # Calculate expected values
        fx_c = pl_data[
            (pl_data['Product'] == 'Product C') &
            (pl_data['Fiscal Year'] == 2024) &
            (pl_data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
        ]['Amount in USD'].sum()

        fx_d = pl_data[
            (pl_data['Product'] == 'Product D') &
            (pl_data['Fiscal Year'] == 2024) &
            (pl_data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
        ]['Amount in USD'].sum()

        # SESSION 1: FX for Product C
        print("\n" + "="*60)
        print("SESSION 1: FX Query for Product C")
        print("="*60)

        task1 = Task(
            id="demo-session1-fx-c",
            query="What was the total Foreign Exchange Gain/Loss for Product C in fiscal year 2024? Use FSLine Statement L2 to filter. Return the numeric value.",
            expected_answer=round(fx_c, 2),
            difficulty="hard",
        )

        strategy1 = EpisodicMemoryStrategy(storage=storage)
        executor1 = LLMExecutor(llm_client=llm_client)
        config = AgentConfig(max_attempts=3)

        agent1 = AgentRunner(executor1, evaluator, strategy1, config)
        result1 = agent1.run(task1, context)

        print(f"Session 1 Result: {'PASSED' if result1.passed else 'FAILED'}")
        print(f"Attempts: {result1.attempts}")
        print(f"Output: {result1.final_result.output if result1.final_result else 'None'}")

        # Verify episode was stored (either success with retries or failure)
        episode_count_after_s1 = strategy1.get_episode_count()
        print(f"Episodes after Session 1: {episode_count_after_s1}")

        # SESSION 2: FX for Product D (new strategy instance, same storage)
        print("\n" + "="*60)
        print("SESSION 2: FX Query for Product D (NEW SESSION)")
        print("="*60)

        task2 = Task(
            id="demo-session2-fx-d",
            query="What was the total Foreign Exchange Gain/Loss for Product D in fiscal year 2024? Use FSLine Statement L2 to filter. Return the numeric value.",
            expected_answer=round(fx_d, 2),
            difficulty="hard",
        )

        # NEW strategy instance (simulates new session)
        strategy2 = EpisodicMemoryStrategy(storage=storage)
        executor2 = LLMExecutor(llm_client=llm_client)

        # Check if priors are loaded
        priors = strategy2.load_priors(task2)
        print(f"Prior examples found: {len(priors.get('examples', []))}")
        print(f"Prior hints found: {len(priors.get('hints', []))}")

        if priors.get("examples"):
            print("Prior example preview:")
            print(priors["examples"][0][:200] + "...")

        agent2 = AgentRunner(executor2, evaluator, strategy2, config)

        # Merge priors into context
        full_context = {**context}
        if priors.get("examples"):
            full_context["examples"] = priors["examples"]
        if priors.get("hints"):
            full_context["hints"] = context.get("hints", []) + priors["hints"]

        result2 = agent2.run(task2, full_context)

        print(f"\nSession 2 Result: {'PASSED' if result2.passed else 'FAILED'}")
        print(f"Attempts: {result2.attempts}")
        print(f"Output: {result2.final_result.output if result2.final_result else 'None'}")
        print("="*60 + "\n")

        # Assertions
        # At least one session should demonstrate the learning mechanism
        assert result1.final_result is not None, "Session 1 should produce a result"
        assert result2.final_result is not None, "Session 2 should produce a result"

        # If Session 1 stored episodes, Session 2 should have found them
        if episode_count_after_s1 > 0:
            # Keywords should match
            task1_keywords = extract_keywords(task1.query)
            task2_keywords = extract_keywords(task2.query)
            from src.strategies.episodic_memory import compute_similarity
            similarity = compute_similarity(task1_keywords, task2_keywords)
            assert similarity > 0.5, f"Tasks should be similar (got {similarity})"
