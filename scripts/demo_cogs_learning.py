#!/usr/bin/env python3
"""
Demo: Learning on COGS Percentage Calculation

Demonstrates cross-session learning on a "very hard" question:
"Compare the Cost of Goods Sold as a percentage of Gross Revenue
between 2020 and 2024 for Product B"

This tests:
- Multi-year aggregation
- Percentage calculations
- Understanding financial relationships (COGS vs Gross Revenue)

Usage:
    uv run python scripts/demo_cogs_learning.py
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.core.types import Task
from src.storage.file import FileStorage
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.strategies.none import NoImprovementStrategy
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.llm.claude import ClaudeClient
from src.agent.runner import AgentRunner, AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def load_data() -> pd.DataFrame:
    """Load the P&L dataset."""
    data_path = Path(__file__).parent.parent / "data" / "FUN_company_pl_actuals_dataset.csv"
    return pd.read_csv(data_path)


def calculate_expected_answer(df: pd.DataFrame) -> dict:
    """Calculate the expected COGS % for Product B in 2020 and 2024."""
    results = {}

    for year in [2020, 2024]:
        # COGS = sum of 'Cost of Goods Sold' L1 category
        cogs = df[
            (df['Product'] == 'Product B') &
            (df['Fiscal Year'] == year) &
            (df['FSLine Statement L1'] == 'Cost of Goods Sold')
        ]['Amount in USD'].sum()

        # Gross Revenue from L2
        gross_rev = df[
            (df['Product'] == 'Product B') &
            (df['Fiscal Year'] == year) &
            (df['FSLine Statement L2'] == 'Gross Revenue')
        ]['Amount in USD'].sum()

        pct = round((cogs / gross_rev) * 100, 2)
        results[year] = {"cogs": cogs, "revenue": gross_rev, "percentage": pct}

    return results


def create_context(df: pd.DataFrame) -> dict:
    """Create execution context."""
    return {
        "data": df,
        "hints": [
            "The DataFrame is available as 'data' variable",
            "Store your final answer in a variable named 'result'",
            "Available columns: " + ", ".join(df.columns.tolist()),
            "FSLine Statement L1 contains: " + ", ".join(df['FSLine Statement L1'].unique()),
            "FSLine Statement L2 contains detailed line items like 'Gross Revenue', 'Direct Labor', etc.",
        ],
        "constraints": [
            "Return result as a dictionary with keys '2020' and '2024', values are the COGS percentages",
            "Example format: {'2020': 45.5, '2024': 48.2}",
        ],
    }


def run_agent(task: Task, context: dict, strategy, llm_client, max_attempts: int = 3) -> dict:
    """Run the agent and return results."""
    executor = LLMExecutor(llm_client=llm_client)
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.5, relative_tolerance=0.01)
    config = AgentConfig(max_attempts=max_attempts, collect_trajectory=True)

    agent = AgentRunner(executor, evaluator, strategy, config)

    # Load and merge priors
    priors = strategy.load_priors(task)
    full_context = {**context}
    if priors.get("examples"):
        full_context["examples"] = priors["examples"]
        print(f"  [LEARNING] Found {len(priors['examples'])} prior examples!")
    if priors.get("hints"):
        full_context["hints"] = context.get("hints", []) + priors["hints"]

    result = agent.run(task, full_context)

    return {
        "passed": result.passed,
        "attempts": result.attempts,
        "output": result.final_result.output if result.final_result else None,
        "code": result.final_result.code_generated if result.final_result else None,
    }


def main():
    print_banner("COGS PERCENTAGE LEARNING DEMO")

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    # Load data and calculate expected answer
    df = load_data()
    expected = calculate_expected_answer(df)

    print("Question: Compare the Cost of Goods Sold as a percentage of")
    print("          Gross Revenue between 2020 and 2024 for Product B")
    print()
    print("Expected Answer:")
    print(f"  2020: COGS={expected[2020]['cogs']:,.0f}, Rev={expected[2020]['revenue']:,.0f} => {expected[2020]['percentage']}%")
    print(f"  2024: COGS={expected[2024]['cogs']:,.0f}, Rev={expected[2024]['revenue']:,.0f} => {expected[2024]['percentage']}%")

    # Expected as dict for evaluation
    expected_dict = {
        "2020": expected[2020]["percentage"],
        "2024": expected[2024]["percentage"],
    }

    context = create_context(df)
    llm_client = ClaudeClient(model="claude-sonnet-4-5-20250929")

    # Storage for learning (unified with chat.py --persist-default)
    storage_path = Path(__file__).parent.parent / ".agent_memory"
    storage_path.mkdir(parents=True, exist_ok=True)
    storage = FileStorage(storage_path)

    # =========================================================================
    # RUN 1: Without prior learning (baseline)
    # =========================================================================
    print_banner("RUN 1: WITHOUT LEARNING (Baseline)")

    task1 = Task(
        id="cogs-pct-baseline",
        query="""Compare the Cost of Goods Sold as a percentage of Gross Revenue
between 2020 and 2024 for Product B.

COGS is found in FSLine Statement L1 = 'Cost of Goods Sold'.
Gross Revenue is found in FSLine Statement L2 = 'Gross Revenue'.

Return a dictionary with format: {'2020': percentage, '2024': percentage}""",
        expected_answer=expected_dict,
        difficulty="very_hard",
    )

    strategy1 = NoImprovementStrategy()
    result1 = run_agent(task1, context, strategy1, llm_client, max_attempts=3)

    print(f"Result: {'PASSED' if result1['passed'] else 'FAILED'}")
    print(f"Attempts: {result1['attempts']}")
    print(f"Output: {result1['output']}")

    if result1['code']:
        print("\nGenerated Code:")
        print(result1['code'][:500])

    # =========================================================================
    # SEED LEARNING: Store a helpful episode
    # =========================================================================
    print_banner("SEEDING LEARNING EPISODE")

    from src.strategies.episodic_memory import Episode, extract_keywords

    seed_episode = Episode(
        query="Calculate COGS as percentage of Revenue for Product A in 2021 and 2023",
        failed_code="""
# Wrong: summed all amounts instead of filtering correctly
result = data.groupby('Fiscal Year')['Amount in USD'].sum()
""",
        error_message="Need to filter by FSLine Statement L1 for COGS and FSLine Statement L2 for Gross Revenue separately",
        fixed_code="""
# Correct approach: Calculate COGS % of Gross Revenue by year
results = {}
for year in [2021, 2023]:
    # COGS from FSLine Statement L1
    cogs = data[
        (data['Product'] == 'Product A') &
        (data['Fiscal Year'] == year) &
        (data['FSLine Statement L1'] == 'Cost of Goods Sold')
    ]['Amount in USD'].sum()

    # Gross Revenue from FSLine Statement L2
    gross_rev = data[
        (data['Product'] == 'Product A') &
        (data['Fiscal Year'] == year) &
        (data['FSLine Statement L2'] == 'Gross Revenue')
    ]['Amount in USD'].sum()

    results[str(year)] = round((cogs / gross_rev) * 100, 2)

result = results
""",
        keywords=extract_keywords("Calculate COGS percentage of Gross Revenue Product"),
        task_id="cogs-seed",
        timestamp="1700000000000",
        effectiveness_score=0.8,
    )

    # Clear old episodes and save new seed
    strategy_seed = EpisodicMemoryStrategy(storage=storage)
    strategy_seed.clear_episodes()
    strategy_seed._save_episode(seed_episode)

    print(f"Seeded episode with keywords: {seed_episode.keywords[:5]}...")
    print("Episode teaches: How to calculate COGS % of Gross Revenue correctly")

    # =========================================================================
    # RUN 2: With learning
    # =========================================================================
    print_banner("RUN 2: WITH LEARNING (Should use seeded pattern)")

    task2 = Task(
        id="cogs-pct-with-learning",
        query="""Compare the Cost of Goods Sold as a percentage of Gross Revenue
between 2020 and 2024 for Product B.

Return a dictionary with format: {'2020': percentage, '2024': percentage}""",
        expected_answer=expected_dict,
        difficulty="very_hard",
    )

    strategy2 = EpisodicMemoryStrategy(storage=storage)
    result2 = run_agent(task2, context, strategy2, llm_client, max_attempts=3)

    print(f"Result: {'PASSED' if result2['passed'] else 'FAILED'}")
    print(f"Attempts: {result2['attempts']}")
    print(f"Output: {result2['output']}")

    if result2['code']:
        print("\nGenerated Code:")
        print(result2['code'][:500])

    # =========================================================================
    # Summary
    # =========================================================================
    print_banner("SUMMARY")
    print(f"Expected: {expected_dict}")
    print()
    print(f"Baseline (no learning): {'PASSED' if result1['passed'] else 'FAILED'} in {result1['attempts']} attempts")
    print(f"With learning:          {'PASSED' if result2['passed'] else 'FAILED'} in {result2['attempts']} attempts")

    if result2['passed'] and (not result1['passed'] or result2['attempts'] < result1['attempts']):
        print("\n>>> LEARNING HELPED! <<<")
    elif result1['passed'] and result2['passed']:
        print("\nBoth passed - LLM is capable, but learning provides consistent patterns.")


if __name__ == "__main__":
    main()
