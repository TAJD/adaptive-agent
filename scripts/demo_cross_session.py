#!/usr/bin/env python3
"""
Cross-Session Learning Demo (Real LLM Calls)

This script demonstrates how the EpisodicMemoryStrategy enables
true cross-session learning with actual Claude API calls:

1. SESSION 1: Agent attempts a financial data query. If it fails initially,
   it learns from the failure and retries. The failure and fix are stored.

2. SESSION 2: A completely new session encounters a similar query.
   The agent retrieves the relevant past episode and uses the
   learned pattern to solve the new query more effectively.

Requires: ANTHROPIC_API_KEY environment variable

Run with: uv run python scripts/demo_cross_session.py
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Configure logging to show learning events
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s',
)
# Enable agent runner logging
logging.getLogger("src.agent.runner").setLevel(logging.INFO)

import pandas as pd

from src.core.types import Task
from src.storage.file import FileStorage
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.strategies.none import NoImprovementStrategy
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.llm.claude import ClaudeClient
from src.agent.runner import AgentRunner, AgentConfig


def print_header(text: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def print_section(text: str) -> None:
    """Print a section header."""
    print()
    print(f"--- {text} ---")
    print()


def print_stored_episode(storage_path: Path) -> None:
    """Display the most recently stored episode for demo evidence."""
    import json
    episode_files = sorted(storage_path.glob("episodes/**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not episode_files:
        print("No episodes stored yet.")
        return

    latest = episode_files[0]
    with open(latest) as f:
        data = json.load(f)

    print(f"File: {latest.relative_to(storage_path)}")
    print()
    print("Episode Contents:")
    print(f"  Query: {data.get('query', '')[:70]}...")
    print(f"  Keywords: {data.get('keywords', [])[:6]}")
    print(f"  Error Type: {data.get('error_type', 'N/A')}")
    print(f"  Has Fix: {data.get('fixed_code') is not None}")
    print(f"  Effectiveness: {data.get('effectiveness_score', 0):.0%}")

    if data.get('fixed_code'):
        print()
        print("  Fixed Code (learned pattern):")
        for line in data['fixed_code'].split('\n')[:6]:
            print(f"    {line}")
    print()


def load_pl_data() -> pd.DataFrame:
    """Load the P&L dataset."""
    data_path = Path(__file__).parent.parent / "data" / "FUN_company_pl_actuals_dataset.csv"
    return pd.read_csv(data_path)


def create_context(df: pd.DataFrame) -> dict:
    """Create execution context with data and hints."""
    return {
        "data": df,
        "hints": [
            "The DataFrame is available as 'data' variable",
            "Store your final answer in a variable named 'result'",
            "Available columns: " + ", ".join(df.columns.tolist()),
            "Products: " + ", ".join(df['Product'].unique()),
            "Countries: " + ", ".join(df['Country'].unique()),
            "Years: " + ", ".join(str(y) for y in sorted(df['Fiscal Year'].unique())),
        ],
        "constraints": [
            "Store the final numeric result in a variable named 'result'",
            "Use pandas operations on the 'data' DataFrame",
        ],
    }


def run_session(
    task: Task,
    context: dict,
    strategy,
    session_name: str,
    max_attempts: int = 3,
) -> dict:
    """Run a single session and return results."""

    print_header(f"{session_name}")
    print(f"Task: {task.query}")
    print(f"Expected Answer: ${task.expected_answer:,.2f}")

    # Check for prior knowledge
    priors = strategy.load_priors(task)
    if priors.get("examples"):
        print_section("CROSS-SESSION LEARNING IN ACTION")
        print("*" * 50)
        print("  PRIOR KNOWLEDGE RETRIEVED FROM PREVIOUS SESSION!")
        print("*" * 50)
        print()
        print(f"Found {len(priors.get('examples', []))} relevant examples from past sessions")
        print(f"Found {len(priors.get('hints', []))} hints from similar problems")
        print()
        print("This is the learned pattern being applied:")
        for i, example in enumerate(priors.get("examples", [])[:2]):
            print(f"```python\n{example[:400]}{'...' if len(example) > 400 else ''}\n```")
        print()
        print("The agent will use this pattern to solve the new query!")
    else:
        print_section("No Prior Knowledge")
        print("Starting fresh - no relevant episodes found.")

    # Create LLM client and executor
    llm_client = ClaudeClient(model="claude-sonnet-4-5-20250929")
    executor = LLMExecutor(llm_client=llm_client)

    # Create evaluator with tolerance for floating point
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.001)

    # Create agent
    config = AgentConfig(max_attempts=max_attempts, collect_trajectory=True)
    agent = AgentRunner(executor, evaluator, strategy, config)

    # Merge priors into context
    full_context = {**context}
    if priors.get("examples"):
        full_context["examples"] = priors["examples"]
    if priors.get("hints"):
        existing_hints = full_context.get("hints", [])
        full_context["hints"] = existing_hints + priors["hints"]

    # Run the agent
    print_section("Running Agent")
    result = agent.run(task, full_context)

    # Show results
    print(f"Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"Attempts: {result.attempts}")
    print(f"Score progression: {result.score_progression}")

    # Show generated code from final attempt
    if result.final_result and result.final_result.code_generated:
        print_section("Final Generated Code")
        code = result.final_result.code_generated
        print(f"```python\n{code}\n```")

    # Show output
    if result.final_result:
        print(f"\nOutput: {result.final_result.output}")

    return {
        "passed": result.passed,
        "attempts": result.attempts,
        "score_progression": result.score_progression,
    }


def demo_cross_session_learning():
    """
    Demonstrate cross-session learning with episodic memory using real LLM calls.
    """
    print_header("CROSS-SESSION LEARNING DEMO (Real LLM)")
    print("This demo uses actual Claude API calls to demonstrate learning.")
    print()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in .env file or export it in your shell.")
        return

    # Load data
    df = load_pl_data()
    context = create_context(df)

    print(f"Loaded P&L dataset: {len(df)} rows")
    print(f"Products: {df['Product'].unique().tolist()}")
    print(f"Countries: {df['Country'].unique().tolist()}")
    print()

    # Create persistent storage (unified with chat.py --persist-default)
    storage_path = Path(__file__).parent.parent / ".agent_memory"
    storage_path.mkdir(parents=True, exist_ok=True)
    storage = FileStorage(storage_path)

    print(f"Storage location: {storage_path}")
    print(f"Existing episodes: {len(storage.list_keys('episodes'))}")
    print()

    # ============================================================
    # SESSION 1: First query - agent learns (HARDER QUESTION)
    # ============================================================

    # Task 1: Foreign Exchange impact - requires finding specific line items,
    # aggregating across quarters/countries, and handling negative values
    task1 = Task(
        id="fx-impact-product-c-2024",
        query="What was the total Foreign Exchange Gain/Loss (FX impact) for Product C across ALL countries in fiscal year 2024? This is found in FSLine Statement L2. Return just the numeric value (can be negative).",
        expected_answer=35095.60,
        difficulty="hard",
    )

    strategy1 = EpisodicMemoryStrategy(storage=storage)

    result1 = run_session(
        task=task1,
        context=context,
        strategy=strategy1,
        session_name="SESSION 1: Initial Learning (FX Impact Query)",
        max_attempts=3,
    )

    print_section("Learning Status After Session 1")
    episode_count = strategy1.get_episode_count()
    print(f"Episodes stored: {episode_count}")

    # DEMO EVIDENCE: Show what was actually stored
    print_section("EVIDENCE: Stored Episode (The Learning)")
    print_stored_episode(storage_path)

    # ============================================================
    # SESSION BREAK - Simulating "days later"
    # ============================================================
    print()
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  SESSION BREAK - SIMULATING A NEW SESSION (DAYS LATER)".center(66) + "#")
    print("#" + " " * 68 + "#")
    print("#" + "  The agent from Session 1 is gone.".center(66) + "#")
    print("#" + "  A NEW agent instance will be created.".center(66) + "#")
    print("#" + "  Only the PERSISTENT STORAGE survives.".center(66) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print()

    # ============================================================
    # SESSION 2: Similar query - agent applies learning
    # ============================================================

    # Task 2: Similar FX query for Product D (tests if learning transfers)
    # This one has a NEGATIVE value, testing handling of negative numbers
    task2 = Task(
        id="fx-impact-product-d-2024",
        query="What was the total Foreign Exchange Gain/Loss (FX impact) for Product D across ALL countries in fiscal year 2024? This is found in FSLine Statement L2. Return just the numeric value (can be negative).",
        expected_answer=-16650.33,
        difficulty="hard",
    )

    # Create NEW strategy instance (simulating new session)
    # but with the SAME persistent storage
    strategy2 = EpisodicMemoryStrategy(storage=storage)

    result2 = run_session(
        task=task2,
        context=context,
        strategy=strategy2,
        session_name="SESSION 2: Applying Learned Knowledge",
        max_attempts=3,
    )

    # ============================================================
    # COMPARISON: Run without episodic memory
    # ============================================================

    print_header("COMPARISON: Without Episodic Memory")
    print("Running the same Session 2 task WITHOUT prior learning...")

    # Use NoImprovementStrategy (no learning)
    no_learn_strategy = NoImprovementStrategy()

    result_no_learn = run_session(
        task=task2,
        context=context,
        strategy=no_learn_strategy,
        session_name="SESSION 2 (No Learning Baseline)",
        max_attempts=3,
    )

    # ============================================================
    # Summary
    # ============================================================
    print_header("DEMO SUMMARY")

    print("Cross-Session Learning Results:")
    print()
    print("  Session 1 (Initial Learning):")
    print(f"    * Task: FX Impact for Product C, all countries, 2024")
    print(f"    * Expected: $35,095.60 (positive)")
    print(f"    * Result: {'PASSED' if result1['passed'] else 'FAILED'}")
    print(f"    * Attempts: {result1['attempts']}")
    print()
    print("  Session 2 (With Episodic Memory):")
    print(f"    * Task: FX Impact for Product D, all countries, 2024")
    print(f"    * Expected: -$16,650.33 (NEGATIVE - harder!)")
    print(f"    * Result: {'PASSED' if result2['passed'] else 'FAILED'}")
    print(f"    * Attempts: {result2['attempts']}")
    print()
    print("  Session 2 (Without Learning - Baseline):")
    print(f"    * Task: FX Impact for Product D, all countries, 2024")
    print(f"    * Result: {'PASSED' if result_no_learn['passed'] else 'FAILED'}")
    print(f"    * Attempts: {result_no_learn['attempts']}")
    print()

    if result2['attempts'] < result_no_learn['attempts']:
        improvement = result_no_learn['attempts'] - result2['attempts']
        print(f"Episodic Memory saved {improvement} attempt(s)!")
    elif result2['passed'] and not result_no_learn['passed']:
        print("Episodic Memory enabled success where baseline failed!")
    else:
        print("Note: Results may vary - LLMs can be unpredictable.")
        print("Run multiple times to see the learning effect.")


def demo_similarity_matching():
    """Show how the similarity matching works."""
    print_header("BONUS: Similarity Matching Explained")

    from src.strategies.episodic_memory import extract_keywords, compute_similarity

    query1 = "What was the total Foreign Exchange Gain/Loss for Product C across all countries in 2024?"
    query2 = "What was the total Foreign Exchange Gain/Loss for Product D across all countries in 2024?"
    query3 = "Calculate the total Gross Revenue for Product A in the United States"

    print("Query 1:", query1)
    kw1 = extract_keywords(query1)
    print("Keywords:", kw1)
    print()

    print("Query 2:", query2)
    kw2 = extract_keywords(query2)
    print("Keywords:", kw2)
    print()

    print("Query 3:", query3)
    kw3 = extract_keywords(query3)
    print("Keywords:", kw3)
    print()

    print("Similarity Scores (Jaccard):")
    print(f"  Query 1 vs Query 2: {compute_similarity(kw1, kw2):.2f} (HIGH - both FX queries)")
    print(f"  Query 1 vs Query 3: {compute_similarity(kw1, kw3):.2f} (LOW - FX vs Revenue)")
    print(f"  Query 2 vs Query 3: {compute_similarity(kw2, kw3):.2f} (LOW - FX vs Revenue)")
    print()
    print("This is why Session 2 can retrieve and apply learning from Session 1!")
    print()


def main():
    print()
    print("+" + "=" * 68 + "+")
    print("|   SELF-IMPROVING DATA ANALYSIS AGENT - CROSS-SESSION DEMO (REAL)  |")
    print("+" + "=" * 68 + "+")

    demo_cross_session_learning()
    demo_similarity_matching()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
