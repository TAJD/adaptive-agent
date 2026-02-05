#!/usr/bin/env python3
"""
Multi-session benchmark comparing model × strategy × session count.

This script runs multiple benchmark sessions to measure how strategies
improve over time with accumulated learning.

Usage:
  # Run 5 sessions for each model × strategy combination
  uv run python scripts/run_multi_session_benchmark.py --sessions 5

  # Run with specific models
  uv run python scripts/run_multi_session_benchmark.py --sessions 5 --models haiku sonnet

  # Quick test with fewer tasks
  uv run python scripts/run_multi_session_benchmark.py --sessions 3 --tasks 3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.core.types import Task, BenchmarkConfig
from src.executor.llm import LLMExecutor
from src.evaluator.exact_match import ExactMatchEvaluator
from src.llm.claude import ClaudeClient
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.benchmark.matrix_runner import MatrixBenchmarkRunner
from src.benchmark.model_config import ModelConfig
from src.benchmark.tasks import create_task_suite
from src.agent.runner import AgentRunner, AgentConfig


# Model configurations
MODELS = {
    "haiku": ModelConfig(name="claude-haiku-4-5-20251001", provider="anthropic", temperature=0.0),
    "sonnet": ModelConfig(name="claude-sonnet-4-5-20250929", provider="anthropic", temperature=0.0),
}


def get_timestamp() -> str:
    """Get timestamp for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_single_session(
    model_config: ModelConfig,
    strategy_name: str,
    strategy,
    tasks: list[Task],
    max_attempts: int,
) -> dict:
    """Run a single benchmark session and return metrics."""

    # Create executor and evaluator
    llm_client = ClaudeClient(model=model_config.name)
    executor = LLMExecutor(llm_client=llm_client)
    evaluator = ExactMatchEvaluator(numeric_tolerance=0.01, relative_tolerance=0.01)

    # Run each task
    results = []
    passed_count = 0
    total_attempts = 0
    total_score = 0.0

    for task in tasks:
        config = AgentConfig(max_attempts=max_attempts, collect_trajectory=True)
        agent = AgentRunner(executor, evaluator, strategy, config)

        # Get task context if needed
        task_context = None
        if "real_data" in task.tags:
            from src.benchmark.data_loader import create_task_context
            task_context = create_task_context(task.id)

        result = agent.run(task, task_context)

        results.append({
            "task_id": task.id,
            "passed": result.passed,
            "attempts": result.attempts,
            "score": result.final_score,
        })

        if result.passed:
            passed_count += 1
        total_attempts += result.attempts
        total_score += result.final_score

    num_tasks = len(tasks)
    return {
        "pass_rate": passed_count / num_tasks if num_tasks > 0 else 0,
        "avg_attempts": total_attempts / num_tasks if num_tasks > 0 else 0,
        "avg_score": total_score / num_tasks if num_tasks > 0 else 0,
        "task_results": results,
    }


def run_multi_session_benchmark(
    model_names: list[str],
    strategy_names: list[str],
    num_sessions: int,
    num_tasks: int,
    max_attempts: int,
    output_path: Path,
    difficulty: str = "medium",
) -> dict:
    """
    Run benchmark across multiple sessions for each model × strategy combination.

    Returns a nested dict: results[model][strategy][session] = metrics
    """

    timestamp = get_timestamp()

    # Select tasks based on difficulty
    if difficulty == "easy":
        tasks = create_task_suite(include_easy=True, include_medium=False, include_real_data=False, include_challenging=False)
    elif difficulty == "medium":
        tasks = create_task_suite(include_easy=False, include_medium=True, include_real_data=False, include_challenging=False)
    elif difficulty == "hard":
        tasks = create_task_suite(include_easy=False, include_medium=False, include_real_data=True, include_challenging=False)
    elif difficulty == "challenging":
        tasks = create_task_suite(include_easy=False, include_medium=False, include_real_data=False, include_challenging=True)
    else:  # all
        tasks = create_task_suite(include_easy=True, include_medium=True, include_real_data=True, include_challenging=True)

    tasks = tasks[:num_tasks]

    print(f"Multi-Session Benchmark")
    print(f"=" * 70)
    print(f"Models: {model_names}")
    print(f"Strategies: {strategy_names}")
    print(f"Sessions per combination: {num_sessions}")
    print(f"Tasks per session: {num_tasks} ({difficulty} difficulty)")
    print(f"Max attempts per task: {max_attempts}")
    print(f"=" * 70)
    print()

    all_results = {}

    for model_name in model_names:
        model_config = MODELS[model_name]
        all_results[model_name] = {}

        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} ({model_config.name})")
        print(f"{'=' * 70}")

        for strategy_name in strategy_names:
            all_results[model_name][strategy_name] = {}

            # Create fresh storage for this model × strategy combination
            storage_path = output_path / f"memory_{model_name}_{strategy_name}"
            storage = FileStorage(storage_path)

            # Create strategy with this storage
            if strategy_name == "none":
                strategy = NoImprovementStrategy()
            elif strategy_name == "reflection":
                strategy = ReflectionStrategy(storage=storage)
            elif strategy_name == "episodic":
                strategy = EpisodicMemoryStrategy(storage=storage)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            print(f"\n  Strategy: {strategy_name}")
            print(f"  Storage: {storage_path}")
            print(f"  " + "-" * 50)

            for session in range(1, num_sessions + 1):
                # Count episodes at start
                episode_count = len(storage.list_keys("episodes"))

                print(f"    Session {session}/{num_sessions} (episodes: {episode_count})...", end=" ", flush=True)

                # Run the session
                metrics = run_single_session(
                    model_config=model_config,
                    strategy_name=strategy_name,
                    strategy=strategy,
                    tasks=tasks,
                    max_attempts=max_attempts,
                )

                # Count episodes at end
                episodes_end = len(storage.list_keys("episodes"))
                metrics["episodes_at_start"] = episode_count
                metrics["episodes_at_end"] = episodes_end
                metrics["episodes_created"] = episodes_end - episode_count

                all_results[model_name][strategy_name][session] = metrics

                print(f"pass_rate={metrics['pass_rate']:.1%}, episodes_created={metrics['episodes_created']}")

    return {
        "timestamp": timestamp,
        "config": {
            "models": model_names,
            "strategies": strategy_names,
            "num_sessions": num_sessions,
            "num_tasks": num_tasks,
            "max_attempts": max_attempts,
        },
        "results": all_results,
    }


def generate_matrix_report(data: dict, output_path: Path) -> Path:
    """Generate a markdown report with comparison matrices."""

    timestamp = data["timestamp"]
    config = data["config"]
    results = data["results"]

    filepath = output_path / f"{timestamp}_multi_session_report.md"

    lines = [
        "# Multi-Session Benchmark Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Models:** {', '.join(config['models'])}",
        f"**Strategies:** {', '.join(config['strategies'])}",
        f"**Sessions:** {config['num_sessions']}",
        f"**Tasks per session:** {config['num_tasks']}",
        "",
        "## Summary Matrix: Final Session Pass Rates",
        "",
        "This shows the pass rate at the final session for each model × strategy combination.",
        "",
    ]

    # Create summary matrix table
    strategies = config["strategies"]
    models = config["models"]
    final_session = config["num_sessions"]

    # Header row
    header = "| Model |"
    separator = "|-------|"
    for strat in strategies:
        header += f" {strat} |"
        separator += "--------|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for model in models:
        row = f"| {model} |"
        for strat in strategies:
            pass_rate = results[model][strat][final_session]["pass_rate"]
            row += f" {pass_rate:.1%} |"
        lines.append(row)

    lines.extend([
        "",
        "## Improvement Over Sessions",
        "",
        "This shows how each strategy's pass rate changes across sessions.",
        "",
    ])

    # For each model, show session-over-session improvement
    for model in models:
        lines.extend([
            f"### {model.upper()}",
            "",
            "| Session |" + "".join(f" {s} |" for s in strategies),
            "|---------|" + "".join("--------|" for _ in strategies),
        ])

        for session in range(1, final_session + 1):
            row = f"| {session} |"
            for strat in strategies:
                pass_rate = results[model][strat][session]["pass_rate"]
                row += f" {pass_rate:.1%} |"
            lines.append(row)

        # Add improvement row
        row = "| **Δ (1→N)** |"
        for strat in strategies:
            first = results[model][strat][1]["pass_rate"]
            last = results[model][strat][final_session]["pass_rate"]
            delta = last - first
            row += f" {delta:+.1%} |"
        lines.append(row)

        lines.append("")

    # Episode accumulation
    lines.extend([
        "## Episode Accumulation",
        "",
        "Shows how many episodes (learned experiences) accumulated for each strategy.",
        "",
    ])

    for model in models:
        lines.extend([
            f"### {model.upper()}",
            "",
            "| Session |" + "".join(f" {s} |" for s in strategies),
            "|---------|" + "".join("--------|" for _ in strategies),
        ])

        for session in range(1, final_session + 1):
            row = f"| {session} |"
            for strat in strategies:
                episodes = results[model][strat][session]["episodes_at_end"]
                row += f" {episodes} |"
            lines.append(row)

        lines.append("")

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    # Find best performing combination
    best_combo = None
    best_rate = 0
    for model in models:
        for strat in strategies:
            rate = results[model][strat][final_session]["pass_rate"]
            if rate > best_rate:
                best_rate = rate
                best_combo = (model, strat)

    if best_combo:
        lines.append(f"- **Best combination:** {best_combo[0]} + {best_combo[1]} ({best_rate:.1%} pass rate)")

    # Find biggest improvement
    best_improvement = None
    best_delta = -1
    for model in models:
        for strat in strategies:
            first = results[model][strat][1]["pass_rate"]
            last = results[model][strat][final_session]["pass_rate"]
            delta = last - first
            if delta > best_delta:
                best_delta = delta
                best_improvement = (model, strat)

    if best_improvement and best_delta > 0:
        lines.append(f"- **Most improved:** {best_improvement[0]} + {best_improvement[1]} ({best_delta:+.1%} over {final_session} sessions)")

    lines.extend([
        "",
        "---",
        "*Generated by run_multi_session_benchmark.py*",
    ])

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Multi-session benchmark comparing model × strategy × session count"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=5,
        help="Number of sessions to run for each combination (default: 5)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["haiku", "sonnet"],
        choices=["haiku", "sonnet"],
        help="Models to test (default: haiku sonnet)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["none", "reflection", "episodic"],
        choices=["none", "reflection", "episodic"],
        help="Strategies to test (default: none reflection episodic)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=5,
        help="Number of tasks per session (default: 5)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "challenging", "all"],
        default="medium",
        help="Task difficulty (default: medium). Use 'challenging' to test edge cases.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max attempts per task (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/multi_session",
        help="Output directory (default: results/multi_session)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in .env file or export it in your shell.")
        return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    data = run_multi_session_benchmark(
        model_names=args.models,
        strategy_names=args.strategies,
        num_sessions=args.sessions,
        num_tasks=args.tasks,
        max_attempts=args.max_attempts,
        output_path=output_path,
        difficulty=args.difficulty,
    )

    # Save raw results
    timestamp = data["timestamp"]
    json_path = output_path / f"{timestamp}_multi_session_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    # Generate report
    report_path = generate_matrix_report(data, output_path)

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print(f"Results saved to:")
    print(f"  - {json_path} (raw data)")
    print(f"  - {report_path} (report)")
    print()
    print("Quick summary:")
    print("-" * 70)

    # Print quick summary
    results = data["results"]
    final_session = args.sessions

    print(f"{'Model':<10} {'Strategy':<12} {'Session 1':<12} {'Session {}':<12} {'Delta':<10}".format(final_session))
    print("-" * 70)

    for model in args.models:
        for strat in args.strategies:
            first = results[model][strat][1]["pass_rate"]
            last = results[model][strat][final_session]["pass_rate"]
            delta = last - first
            print(f"{model:<10} {strat:<12} {first:<12.1%} {last:<12.1%} {delta:+.1%}")

    print()
    print(f"View full report: cat {report_path}")


if __name__ == "__main__":
    main()
