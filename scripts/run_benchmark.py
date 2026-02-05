#!/usr/bin/env python3
"""
CLI script to run benchmarks and compare strategies using real LLM calls.

Requires: ANTHROPIC_API_KEY environment variable

Examples:
    # Run with default settings (easy tasks, 2 strategies)
    uv run python scripts/run_benchmark.py

    # Run with medium difficulty tasks
    uv run python scripts/run_benchmark.py --tasks medium

    # Run with all strategies
    uv run python scripts/run_benchmark.py --strategies none,reflection,episodic

    # Run with specific model
    uv run python scripts/run_benchmark.py --model claude-haiku-4-5-20251001

    # Export results
    uv run python scripts/run_benchmark.py --output results/benchmark.json
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.core.types import Task
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.llm.claude import ClaudeClient
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.benchmark.tasks import create_task_suite


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


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks comparing improvement strategies with real LLM"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="none,reflection",
        help="Comma-separated list of strategies: none, reflection, episodic (default: none,reflection)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="easy",
        help="Task difficulty: easy, medium, hard, challenging, or all (default: easy)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Limit number of tasks (default: all in selected difficulty)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per task (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export JSON report (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed per-task results",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in .env file or export it in your shell.")
        sys.exit(1)

    # Load data
    df = load_pl_data()
    context = create_context(df)

    # Select tasks
    if args.tasks == "easy":
        tasks = create_task_suite(include_easy=True, include_medium=False, include_real_data=False, include_challenging=False)
    elif args.tasks == "medium":
        tasks = create_task_suite(include_easy=False, include_medium=True, include_real_data=False, include_challenging=False)
    elif args.tasks == "hard":
        tasks = create_task_suite(include_easy=False, include_medium=False, include_real_data=True, include_challenging=False)
    elif args.tasks == "challenging":
        tasks = create_task_suite(include_easy=False, include_medium=False, include_real_data=False, include_challenging=True)
    elif args.tasks == "all":
        tasks = create_task_suite(include_easy=True, include_medium=True, include_real_data=True, include_challenging=True)
    else:
        print(f"Unknown task set: {args.tasks}")
        sys.exit(1)

    if args.num_tasks:
        tasks = tasks[:args.num_tasks]

    print(f"Running benchmark with {len(tasks)} {args.tasks} tasks")
    print(f"Model: {args.model}")
    print(f"Max attempts: {args.max_attempts}")
    print()

    # Select strategies
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    strategies = {}
    storage = InMemoryStorage()

    for name in strategy_names:
        if name == "none":
            strategies[name] = NoImprovementStrategy()
        elif name == "reflection":
            strategies[name] = ReflectionStrategy(storage=storage)
        elif name == "episodic":
            strategies[name] = EpisodicMemoryStrategy(storage=storage)
        else:
            print(f"Unknown strategy: {name}")
            print("Available strategies: none, reflection, episodic")
            sys.exit(1)

    print(f"Strategies: {', '.join(strategies.keys())}")
    print("=" * 60)
    print()

    # Create LLM client and executor
    llm_client = ClaudeClient(model=args.model)
    executor = LLMExecutor(llm_client=llm_client)
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.001)

    config = BenchmarkConfig(
        max_attempts=args.max_attempts,
        strategies=strategies,
        task_suite=tasks,
        evaluator=evaluator,
        executor=executor,
        context=context,
    )

    # Run benchmark
    print("Running benchmark (this may take a while with real LLM calls)...")
    print()

    runner = BenchmarkRunner(config)
    results = runner.run()

    # Show comparison
    report = runner.compare(results)
    print()
    print(report.summary)
    print()

    # Verbose output
    if args.verbose:
        print("\nDetailed Results:")
        print("-" * 60)
        for strategy_name, metrics in results.items():
            print(f"\n{strategy_name}:")
            for task_result in metrics.per_task_results:
                status = "PASS" if task_result.passed else "FAIL"
                print(
                    f"  {task_result.task.id:30} [{status}] "
                    f"attempts={task_result.attempts} "
                    f"score={task_result.final_score:.2f}"
                )

    # Export report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        runner.export_report(results, args.output)
        print(f"\nReport exported to: {args.output}")


if __name__ == "__main__":
    main()
