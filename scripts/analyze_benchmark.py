#!/usr/bin/env python3
"""Detailed benchmark analysis showing LLM outputs and evaluations."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import Task
from src.executor.llm import LLMExecutor
from src.evaluator.llm import LLMEvaluator
from src.llm.claude import ClaudeClient
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.agent.runner import AgentRunner, AgentConfig
from src.benchmark.data_loader import create_task_context


def run_single_task_analysis(task, strategies, agent_models, judge_model):
    """Run a single task with all strategy/model combinations and show detailed outputs."""

    print(f"\n{'=' * 80}")
    print(f"Analyzing Task: {task.id}")
    print(f"Query: {task.query}")
    print(f"Expected: {task.expected_answer}")
    print(f"{'=' * 80}")

    # Setup clients
    judge_client = ClaudeClient(model=judge_model.name)

    results = {}

    for model_name, model_config in agent_models.items():
        print(f"\n🤖 Testing Model: {model_name}")
        print("-" * 50)

        code_client = ClaudeClient(model=model_config.name)

        for strategy_name, strategy in strategies.items():
            print(f"\n🧠 Strategy: {strategy_name}")
            print("-" * 30)

            # Create executor and evaluator
            executor = LLMExecutor(llm_client=code_client)
            evaluator = LLMEvaluator(llm_client=judge_client)

            # Get context for real data tasks
            context = {}
            if "real_data" in task.tags:
                context = create_task_context(task.id)

            # Setup agent
            config = AgentConfig(max_attempts=2, collect_trajectory=True)
            agent = AgentRunner(executor, evaluator, strategy, config)

            try:
                # Run the task
                result = agent.run(task, context)

                # Store result
                key = f"{model_name}_{strategy_name}"
                trajectory = None
                if result.final_result and hasattr(result.final_result, "trajectory"):
                    trajectory = result.final_result.trajectory
                results[key] = {"result": result, "trajectory": trajectory}

                print(f"Final Result: {'PASS' if result.passed else 'FAIL'}")
                print(f"Attempts: {result.attempts}")
                print(f"Final Score: {result.final_score:.2f}")

                # Show trajectory
                if trajectory:
                    for step in trajectory:
                        if step["step"] == "llm_response":
                            print(f"LLM Response: {step['response'][:200]}...")
                        elif step["step"] == "code_extracted":
                            print(f"Generated Code:\n{step['code']}")
                        elif step["step"] == "code_executed":
                            exec_result = step["result"]
                            if exec_result["success"]:
                                print(
                                    f"Execution Success: {exec_result['output'][:100]}..."
                                )
                            else:
                                print(
                                    f"Execution Failed: {exec_result['error'][:100]}..."
                                )

                # Show final evaluation if available
                if result.final_evaluation:
                    eval = result.final_evaluation
                    print(f"Evaluation Score: {eval.score:.2f}")
                    if hasattr(eval, "error_type") and eval.error_type:
                        print(f"Error Type: {eval.error_type}")
                    if hasattr(eval, "feedback"):
                        print(f"Feedback: {eval.feedback[:100]}...")

            except Exception as e:
                print(f"💥 Error running {strategy_name} with {model_name}: {e}")

    return results


def main():
    print("Detailed LLM Benchmark Analysis")
    print("=" * 80)

    # Setup tasks - use just one simple real data task for analysis
    from src.benchmark.tasks import create_task_suite

    tasks = create_task_suite(
        include_easy=False, include_medium=False, include_real_data=True
    )[:1]  # Just one task

    # Setup models
    from src.benchmark.model_config import ModelConfig

    agent_models = {
        "haiku": ModelConfig(name="claude-haiku-4-5-20251001", provider="anthropic"),
        "sonnet": ModelConfig(name="claude-sonnet-4-5-20250929", provider="anthropic"),
    }

    judge_model = ModelConfig(name="claude-sonnet-4-5-20250929", provider="anthropic")

    # Setup strategies
    storage = InMemoryStorage()
    strategies = {
        "none": NoImprovementStrategy(),
        "reflection": ReflectionStrategy(storage=storage),
        "episodic": EpisodicMemoryStrategy(storage=storage),
    }

    print(f"Task: {tasks[0].id}")
    print(f"Models: {list(agent_models.keys())}")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Judge: {judge_model.name}")

    # Run analysis
    task = tasks[0]
    results = run_single_task_analysis(task, strategies, agent_models, judge_model)

    # Summary
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY")
    print(f"{'=' * 80}")

    print(f"{'Model/Strategy':<15} {'Attempts':<8} {'Score':<6} {'Passed':<6}")
    print("-" * 40)

    for key, data in results.items():
        result = data["result"]
        model, strategy = key.split("_", 1)
        print(
            f"{f'{model[:6]}/{strategy[:6]}':<15} {result.attempts:<8} {result.final_score:<6.2f} {str(result.passed):<6}"
        )


if __name__ == "__main__":
    main()
