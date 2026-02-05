#!/usr/bin/env python3
"""
Demo script showing the agent benchmarking framework in action.

This script demonstrates:
1. How the framework compares different improvement strategies
2. How reflection strategy can improve pass rate over baseline
3. Cross-session learning persistence

Requires: ANTHROPIC_API_KEY environment variable

Run with: uv run python scripts/demo.py
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.core.types import Task
from src.agent.runner import AgentRunner, AgentConfig
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.llm.claude import ClaudeClient
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig


def check_api_key() -> bool:
    """Check if API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in .env file or export it in your shell.")
        return False
    return True


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
        ],
        "constraints": [
            "Store the final numeric result in a variable named 'result'",
            "Use pandas operations on the 'data' DataFrame",
        ],
    }


def demo_basic_comparison():
    """Demonstrate basic strategy comparison with real LLM."""
    print("=" * 60)
    print("DEMO 1: Basic Strategy Comparison (Real LLM)")
    print("=" * 60)
    print()

    if not check_api_key():
        return

    # Load data
    df = load_pl_data()
    context = create_context(df)

    # Create tasks with real expected answers
    tasks = [
        Task(
            id="gross_revenue_a_2023",
            query="What is the total Gross Revenue for Product A in 2023? Return the numeric value.",
            expected_answer=df[(df['Product'] == 'Product A') &
                              (df['FSLine Statement L2'] == 'Gross Revenue') &
                              (df['Fiscal Year'] == 2023)]['Amount in USD'].sum(),
        ),
        Task(
            id="gross_revenue_b_2023",
            query="What is the total Gross Revenue for Product B in 2023? Return the numeric value.",
            expected_answer=df[(df['Product'] == 'Product B') &
                              (df['FSLine Statement L2'] == 'Gross Revenue') &
                              (df['Fiscal Year'] == 2023)]['Amount in USD'].sum(),
        ),
    ]

    print("Task suite:")
    for t in tasks:
        print(f"  - {t.id}: {t.query}")
        print(f"    Expected: ${t.expected_answer:,.2f}")
    print()

    # Create LLM client
    llm_client = ClaudeClient(model="claude-sonnet-4-5-20250929")
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.001)

    # Run with NoImprovement (1 attempt only)
    print("Strategy: NoImprovement (baseline, 1 attempt)")
    executor1 = LLMExecutor(llm_client=llm_client)
    agent1 = AgentRunner(
        executor=executor1,
        evaluator=evaluator,
        strategy=NoImprovementStrategy(),
        config=AgentConfig(max_attempts=1),
    )
    results1 = []
    for task in tasks:
        result = agent1.run(task, context)
        results1.append(result)
        print(f"  {task.id}: {'PASS' if result.passed else 'FAIL'} (attempts: {result.attempts})")

    passed1 = sum(1 for r in results1 if r.passed)
    print(f"  Pass rate: {passed1}/{len(tasks)} ({100*passed1/len(tasks):.0f}%)")
    print()

    # Run with Reflection (3 attempts)
    print("Strategy: Reflection (3 attempts)")
    storage = InMemoryStorage()
    executor2 = LLMExecutor(llm_client=llm_client)
    agent2 = AgentRunner(
        executor=executor2,
        evaluator=evaluator,
        strategy=ReflectionStrategy(storage=storage),
        config=AgentConfig(max_attempts=3),
    )
    results2 = []
    for task in tasks:
        result = agent2.run(task, context)
        results2.append(result)
        print(f"  {task.id}: {'PASS' if result.passed else 'FAIL'} (attempts: {result.attempts})")

    passed2 = sum(1 for r in results2 if r.passed)
    print(f"  Pass rate: {passed2}/{len(tasks)} ({100*passed2/len(tasks):.0f}%)")
    print()

    improvement = passed2 - passed1
    if improvement > 0:
        print(f"Improvement: +{improvement} tasks passed with reflection strategy")
    else:
        print("Note: Both strategies performed similarly on these tasks.")
    print()


def demo_cross_session():
    """Demonstrate cross-session learning with real LLM."""
    print("=" * 60)
    print("DEMO 2: Cross-Session Learning (Real LLM)")
    print("=" * 60)
    print()

    if not check_api_key():
        return

    # Load data
    df = load_pl_data()
    context = create_context(df)

    # Use persistent file storage (unified with chat.py --persist-default)
    storage_path = Path(__file__).parent.parent / ".agent_memory"
    storage_path.mkdir(parents=True, exist_ok=True)
    storage = FileStorage(storage_path)

    print(f"Storage location: {storage_path}")
    print(f"Existing episodes: {len(storage.list_keys('episodes'))}")
    print()

    task = Task(
        id="opex_japan_2024",
        query="What is the total OPEX (Operating Expenses) for Japan in Q1 2024? OPEX includes: General & Administrative, Headcount Expenses, IT Expenses, Marketing Expenses, R&D Expenses, and Sales Expenses. Return just the numeric sum.",
        expected_answer=df[(df['Country'] == 'Japan') &
                          (df['Fiscal Year'] == 2024) &
                          (df['Fiscal Quarter'] == 'Q1') &
                          (df['FSLine Statement L1'] == 'Operating Expenses')]['Amount in USD'].sum(),
    )

    print(f"Task: {task.query[:80]}...")
    print(f"Expected: ${task.expected_answer:,.2f}")
    print()

    # Create LLM client
    llm_client = ClaudeClient(model="claude-sonnet-4-5-20250929")
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.001)

    # Session 1: First attempt
    print("Session 1: First attempt")
    executor1 = LLMExecutor(llm_client=llm_client)
    strategy1 = EpisodicMemoryStrategy(storage=storage)
    agent1 = AgentRunner(
        executor=executor1,
        evaluator=evaluator,
        strategy=strategy1,
        config=AgentConfig(max_attempts=2),
    )
    result1 = agent1.run(task, context)
    print(f"  Output: {result1.final_result.output if result1.final_result else None}")
    print(f"  Passed: {result1.passed}")
    print(f"  Attempts: {result1.attempts}")
    print()

    # Check what was learned
    episode_count = strategy1.get_episode_count()
    print(f"  Episodes stored: {episode_count}")
    print()

    # Session 2: Similar task, new session
    print("Session 2: Similar task (different quarter)")
    task2 = Task(
        id="opex_japan_q2_2024",
        query="What is the total OPEX (Operating Expenses) for Japan in Q2 2024? OPEX includes: General & Administrative, Headcount Expenses, IT Expenses, Marketing Expenses, R&D Expenses, and Sales Expenses. Return just the numeric sum.",
        expected_answer=df[(df['Country'] == 'Japan') &
                          (df['Fiscal Year'] == 2024) &
                          (df['Fiscal Quarter'] == 'Q2') &
                          (df['FSLine Statement L1'] == 'Operating Expenses')]['Amount in USD'].sum(),
    )

    executor2 = LLMExecutor(llm_client=llm_client)
    strategy2 = EpisodicMemoryStrategy(storage=storage)

    # Load priors
    priors = strategy2.load_priors(task2)
    print(f"  Loaded priors: {len(priors.get('examples', []))} examples, {len(priors.get('hints', []))} hints")

    agent2 = AgentRunner(
        executor=executor2,
        evaluator=evaluator,
        strategy=strategy2,
        config=AgentConfig(max_attempts=2),
    )
    result2 = agent2.run(task2, context)
    print(f"  Output: {result2.final_result.output if result2.final_result else None}")
    print(f"  Passed: {result2.passed}")
    print(f"  Attempts: {result2.attempts}")
    print()


def demo_benchmark_runner():
    """Demonstrate the benchmark runner with real LLM."""
    print("=" * 60)
    print("DEMO 3: Benchmark Runner (Real LLM)")
    print("=" * 60)
    print()

    if not check_api_key():
        return

    # Load data
    df = load_pl_data()
    context = create_context(df)

    # Create diverse tasks
    tasks = [
        Task(
            id="revenue_a_us",
            query="Total Gross Revenue for Product A in United States for 2023?",
            expected_answer=df[(df['Product'] == 'Product A') &
                              (df['Country'] == 'United States') &
                              (df['FSLine Statement L2'] == 'Gross Revenue') &
                              (df['Fiscal Year'] == 2023)]['Amount in USD'].sum(),
        ),
        Task(
            id="revenue_b_canada",
            query="Total Gross Revenue for Product B in Canada for 2023?",
            expected_answer=df[(df['Product'] == 'Product B') &
                              (df['Country'] == 'Canada') &
                              (df['FSLine Statement L2'] == 'Gross Revenue') &
                              (df['Fiscal Year'] == 2023)]['Amount in USD'].sum(),
        ),
        Task(
            id="cogs_c_germany",
            query="Total Cost of Goods Sold for Product C in Germany for 2024?",
            expected_answer=df[(df['Product'] == 'Product C') &
                              (df['Country'] == 'Germany') &
                              (df['FSLine Statement L1'] == 'Cost of Goods Sold') &
                              (df['Fiscal Year'] == 2024)]['Amount in USD'].sum(),
        ),
    ]

    print(f"Running benchmark with {len(tasks)} tasks...")
    print()

    # Create components
    llm_client = ClaudeClient(model="claude-sonnet-4-5-20250929")
    executor = LLMExecutor(llm_client=llm_client)
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.001)
    storage = InMemoryStorage()

    config = BenchmarkConfig(
        max_attempts=3,
        strategies={
            "none": NoImprovementStrategy(),
            "reflection": ReflectionStrategy(storage=storage),
        },
        task_suite=tasks,
        evaluator=evaluator,
        executor=executor,
        context=context,
    )

    runner = BenchmarkRunner(config)
    results = runner.run()

    report = runner.compare(results)
    print()
    print(report.summary)
    print()


def main():
    print()
    print("AGENT BENCHMARKING FRAMEWORK DEMO (Real LLM)")
    print("=" * 60)
    print()

    demo_basic_comparison()
    demo_cross_session()
    demo_benchmark_runner()

    print("Demo complete!")
    print()


if __name__ == "__main__":
    main()
