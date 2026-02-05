#!/usr/bin/env python3
"""
Learning Benchmark: Proves that episodic memory improves performance.

This benchmark specifically tests whether the EpisodicMemoryStrategy helps
the agent perform better on similar queries by:

1. Running "teacher" tasks first (agent learns patterns)
2. Running "learner" tasks second (agent should apply learned patterns)
3. Comparing performance with vs without episodic memory

Uses the same P&L dataset as the CLI and demo scripts.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import Task, BenchmarkConfig
from src.executor.llm import LLMExecutor
from src.evaluator.llm import LLMEvaluator
from src.llm.claude import ClaudeClient
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.agent.runner import AgentRunner, AgentConfig
from src.benchmark.tasks import create_learning_pairs, create_pl_task_suite
from src.benchmark.data_loader import create_task_context


def run_learning_pair_benchmark(
    model_name: str = "claude-haiku-4-5-20251001",
    max_attempts: int = 2,
    verbose: bool = True,
):
    """
    Run benchmark proving episodic memory helps on similar queries.

    Tests each strategy on learning pairs:
    - Teacher task first (agent learns)
    - Learner task second (agent should apply learning)
    """
    print("=" * 70)
    print("LEARNING PAIR BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Max attempts per task: {max_attempts}")
    print()

    # Get learning pairs
    pairs = create_learning_pairs()
    print(f"Testing {len(pairs)} learning pairs")
    print()

    # Setup LLM clients
    code_client = ClaudeClient(model=model_name)
    judge_client = ClaudeClient(model="claude-sonnet-4-5-20250929")

    # Results storage
    results = {
        "none": {"teacher_pass": 0, "learner_pass": 0, "teacher_attempts": [], "learner_attempts": []},
        "reflection": {"teacher_pass": 0, "learner_pass": 0, "teacher_attempts": [], "learner_attempts": []},
        "episodic": {"teacher_pass": 0, "learner_pass": 0, "teacher_attempts": [], "learner_attempts": []},
    }

    for pair_idx, (teacher, learner) in enumerate(pairs):
        print(f"\n{'='*70}")
        print(f"PAIR {pair_idx + 1}: {teacher.id} -> {learner.id}")
        print(f"{'='*70}")
        print(f"Teacher: {teacher.query[:60]}...")
        print(f"Learner: {learner.query[:60]}...")

        for strategy_name in ["none", "reflection", "episodic"]:
            print(f"\n--- Strategy: {strategy_name} ---")

            # Fresh storage for each strategy (isolate learning)
            storage = InMemoryStorage()

            if strategy_name == "none":
                strategy = NoImprovementStrategy()
            elif strategy_name == "reflection":
                strategy = ReflectionStrategy(storage=storage)
            else:
                strategy = EpisodicMemoryStrategy(storage=storage)

            # Create executor and evaluator
            executor = LLMExecutor(llm_client=code_client)
            evaluator = LLMEvaluator(llm_client=judge_client)

            # Agent config
            config = AgentConfig(max_attempts=max_attempts)
            agent = AgentRunner(executor, evaluator, strategy, config)

            # Run teacher task
            teacher_context = create_task_context(teacher.id)
            teacher_result = agent.run(teacher, teacher_context)

            results[strategy_name]["teacher_attempts"].append(teacher_result.attempts)
            if teacher_result.passed:
                results[strategy_name]["teacher_pass"] += 1

            if verbose:
                print(f"  Teacher: {'PASS' if teacher_result.passed else 'FAIL'} "
                      f"(attempts: {teacher_result.attempts}, score: {teacher_result.final_score:.2f})")

            # Run learner task (should benefit from teacher learning)
            learner_context = create_task_context(learner.id)
            learner_result = agent.run(learner, learner_context)

            results[strategy_name]["learner_attempts"].append(learner_result.attempts)
            if learner_result.passed:
                results[strategy_name]["learner_pass"] += 1

            if verbose:
                print(f"  Learner: {'PASS' if learner_result.passed else 'FAIL'} "
                      f"(attempts: {learner_result.attempts}, score: {learner_result.final_score:.2f})")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    num_pairs = len(pairs)
    print(f"\n{'Strategy':<12} | {'Teacher Pass':<12} | {'Learner Pass':<12} | {'Avg Teacher Att':<15} | {'Avg Learner Att':<15}")
    print("-" * 75)

    for strategy_name in ["none", "reflection", "episodic"]:
        r = results[strategy_name]
        teacher_pct = r["teacher_pass"] / num_pairs * 100 if num_pairs > 0 else 0
        learner_pct = r["learner_pass"] / num_pairs * 100 if num_pairs > 0 else 0
        avg_teacher = sum(r["teacher_attempts"]) / len(r["teacher_attempts"]) if r["teacher_attempts"] else 0
        avg_learner = sum(r["learner_attempts"]) / len(r["learner_attempts"]) if r["learner_attempts"] else 0

        print(f"{strategy_name:<12} | {r['teacher_pass']}/{num_pairs} ({teacher_pct:5.1f}%) | "
              f"{r['learner_pass']}/{num_pairs} ({learner_pct:5.1f}%) | "
              f"{avg_teacher:<15.2f} | {avg_learner:<15.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Compare episodic vs none on learner tasks
    episodic_learner = results["episodic"]["learner_pass"]
    none_learner = results["none"]["learner_pass"]

    if episodic_learner > none_learner:
        improvement = episodic_learner - none_learner
        print(f"\n✅ LEARNING PROVEN: Episodic memory improved learner task performance")
        print(f"   - Without learning: {none_learner}/{num_pairs} passed")
        print(f"   - With episodic memory: {episodic_learner}/{num_pairs} passed")
        print(f"   - Improvement: +{improvement} tasks ({improvement/num_pairs*100:.1f}%)")
    elif episodic_learner == none_learner:
        print(f"\n⚠️  NO DIFFERENCE: Episodic memory showed same performance as baseline")
        print(f"   Both passed {episodic_learner}/{num_pairs} learner tasks")
        print(f"   (Tasks may be too easy or learning not applicable)")
    else:
        print(f"\n❌ UNEXPECTED: Baseline outperformed episodic memory")
        print(f"   This suggests a bug in the learning implementation")

    # Check if attempts reduced
    episodic_avg = sum(results["episodic"]["learner_attempts"]) / len(results["episodic"]["learner_attempts"]) if results["episodic"]["learner_attempts"] else 0
    none_avg = sum(results["none"]["learner_attempts"]) / len(results["none"]["learner_attempts"]) if results["none"]["learner_attempts"] else 0

    if episodic_avg < none_avg:
        print(f"\n✅ EFFICIENCY GAIN: Episodic memory reduced attempts on learner tasks")
        print(f"   - Without learning: {none_avg:.2f} avg attempts")
        print(f"   - With episodic memory: {episodic_avg:.2f} avg attempts")

    return results


def run_pl_suite_benchmark(
    model_name: str = "claude-haiku-4-5-20251001",
    max_attempts: int = 3,
    include_easy: bool = True,
    include_medium: bool = True,
    include_hard: bool = False,
    verbose: bool = True,
):
    """
    Run benchmark on P&L task suite comparing all strategies.
    """
    print("=" * 70)
    print("P&L TASK SUITE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Max attempts per task: {max_attempts}")
    print()

    # Get tasks
    tasks = create_pl_task_suite(
        include_easy=include_easy,
        include_medium=include_medium,
        include_hard=include_hard,
    )
    print(f"Testing {len(tasks)} P&L tasks")
    print()

    # Setup LLM clients
    code_client = ClaudeClient(model=model_name)
    judge_client = ClaudeClient(model="claude-sonnet-4-5-20250929")

    # Results by strategy
    results = {}

    for strategy_name in ["none", "reflection", "episodic"]:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*50}")

        # Fresh storage for each strategy
        storage = InMemoryStorage()

        if strategy_name == "none":
            strategy = NoImprovementStrategy()
        elif strategy_name == "reflection":
            strategy = ReflectionStrategy(storage=storage)
        else:
            strategy = EpisodicMemoryStrategy(storage=storage)

        # Create executor and evaluator
        executor = LLMExecutor(llm_client=code_client)
        evaluator = LLMEvaluator(llm_client=judge_client)

        # Agent config
        config = AgentConfig(max_attempts=max_attempts)
        agent = AgentRunner(executor, evaluator, strategy, config)

        # Run all tasks
        passed = 0
        total_attempts = 0
        total_score = 0.0

        for task in tasks:
            context = create_task_context(task.id)
            result = agent.run(task, context)

            if result.passed:
                passed += 1
            total_attempts += result.attempts
            total_score += result.final_score

            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {task.id}: score={result.final_score:.2f}, attempts={result.attempts}")

        results[strategy_name] = {
            "passed": passed,
            "total": len(tasks),
            "pass_rate": passed / len(tasks) if tasks else 0,
            "avg_attempts": total_attempts / len(tasks) if tasks else 0,
            "avg_score": total_score / len(tasks) if tasks else 0,
        }

        print(f"\n  Summary: {passed}/{len(tasks)} passed ({results[strategy_name]['pass_rate']*100:.1f}%)")

    # Final comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Strategy':<12} | {'Pass Rate':<12} | {'Avg Attempts':<12} | {'Avg Score':<10}")
    print("-" * 55)

    for strategy_name in ["none", "reflection", "episodic"]:
        r = results[strategy_name]
        print(f"{strategy_name:<12} | {r['pass_rate']*100:>10.1f}% | {r['avg_attempts']:>12.2f} | {r['avg_score']:>10.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Learning Benchmark")
    parser.add_argument(
        "--mode",
        choices=["pairs", "suite", "both"],
        default="pairs",
        help="Benchmark mode: pairs (learning pairs), suite (full P&L suite), both",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model to use for code generation",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Max attempts per task",
    )
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help="Include hard P&L tasks in suite mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    if args.mode in ["pairs", "both"]:
        run_learning_pair_benchmark(
            model_name=args.model,
            max_attempts=args.max_attempts,
            verbose=not args.quiet,
        )

    if args.mode in ["suite", "both"]:
        if args.mode == "both":
            print("\n\n")
        run_pl_suite_benchmark(
            model_name=args.model,
            max_attempts=args.max_attempts,
            include_hard=args.include_hard,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
