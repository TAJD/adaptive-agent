#!/usr/bin/env python3
"""Simple benchmark analysis using mock clients to demonstrate the system."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from src.core.types import Task
from src.executor.llm import LLMExecutor
from src.evaluator.llm import LLMEvaluator
from src.llm.claude import ClaudeClient
from src.strategies.none import NoImprovementStrategy
from src.strategies.reflection import ReflectionStrategy
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.storage.memory import InMemoryStorage
from src.agent.runner import AgentRunner, AgentConfig
from src.benchmark.tasks import create_task_suite


def demonstrate_strategies():
    """Demonstrate how different strategies work with a simple task."""

    print("Strategy Demonstration")
    print("=" * 50)

    # Create a challenging task that requires pandas operations and might fail initially
    task = Task(
        id="demo_pandas_complex",
        query="Given sales data in a pandas DataFrame called 'data', calculate the average sales growth rate (percentage) across all products from 2023 to 2024. The data has columns 'Product', 'Sales_2023', 'Sales_2024'. Return the result as a single float.",
        expected_answer=20.83,  # Average of the individual growth rates
        difficulty="hard",
    )

    print(f"Task: {task.query}")
    print(f"Expected Answer: {task.expected_answer}")
    print()

    # Load sales data for context
    from src.benchmark.data_loader import create_task_context

    task_context = create_task_context(
        "sales_growth_analysis"
    )  # This loads the sales data

    # Use Claude Haiku 4.5 for code generation (fast/cost-effective)
    code_client = ClaudeClient(model="claude-haiku-4-5-20251001")

    # Use Claude Sonnet 4.5 for evaluation
    evaluator_client = ClaudeClient(model="claude-sonnet-4-5-20250929")
    evaluator = LLMEvaluator(llm_client=evaluator_client)

    # Test each strategy
    storage = InMemoryStorage()
    strategies = {
        "none": NoImprovementStrategy(),
        "reflection": ReflectionStrategy(storage=storage),
        "episodic": EpisodicMemoryStrategy(storage=storage),
    }

    for strategy_name, strategy in strategies.items():
        print(f"\nTesting Strategy: {strategy_name}")
        print("-" * 40)

        executor = LLMExecutor(llm_client=code_client)
        config = AgentConfig(max_attempts=3, collect_trajectory=True)
        agent = AgentRunner(executor, evaluator, strategy, config)

        result = agent.run(task, task_context)

        print(f"Result: {'PASS' if result.passed else 'FAIL'}")
        print(f"Attempts: {result.attempts}")
        print(f"Final Score: {result.final_score}")
        print(f"Score Progression: {result.score_progression}")

        # Show trajectory with actual LLM responses
        if result.final_result and result.final_result.trajectory:
            print("\nLLM Interaction History:")
            for i, step in enumerate(result.final_result.trajectory, 1):
                if step["step"] == "prompt_built":
                    print(f"  Attempt {i}: Prompt sent to Claude...")
                    # Show first 200 chars of prompt
                    prompt = step["prompt"]
                    print(
                        f"    Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}"
                    )
                elif step["step"] == "llm_response":
                    response = step["response"]
                    print(
                        f"    Claude Response: {response[:150]}{'...' if len(response) > 150 else ''}"
                    )
                elif step["step"] == "code_extracted":
                    print(
                        f"    Extracted Code: {step['code'][:100]}{'...' if len(step['code']) > 100 else ''}"
                    )
                elif step["step"] == "code_executed":
                    exec_result = step["result"]
                    if exec_result.get("success"):
                        output = exec_result.get("output", "")
                        print(f"    Code Execution: SUCCESS - {str(output)[:50]}...")
                    else:
                        error = exec_result.get("error", "")
                        print(f"    Code Execution: FAILED - {str(error)[:50]}...")

        # Show improvements from strategy
        if result.attempts > 1:
            improvements = strategy.load_priors(task)
            if improvements:
                print(f"\nStrategy '{strategy_name}' provided context:")
                for key, value in improvements.items():
                    if key == "hints" and value:
                        print(f"  Hints: {value}")
                    elif key == "constraints" and value:
                        print(f"  Constraints: {value}")
                    elif isinstance(value, (int, float)):
                        print(f"  {key}: {value}")

        print("-" * 50)


def demonstrate_real_data_loading():
    """Demonstrate real data loading and context injection."""

    print("\nReal Data Integration Demo")
    print("=" * 50)

    # Get all tasks to see what's available
    all_tasks = create_task_suite(
        include_easy=True, include_medium=True, include_real_data=True
    )
    print(f"Total tasks available: {len(all_tasks)}")

    # Separate by type
    real_data_tasks = [t for t in all_tasks if "real_data" in t.tags]
    synthetic_tasks = [t for t in all_tasks if "real_data" not in t.tags]

    print(f"Real data tasks: {len(real_data_tasks)}")
    print(f"Synthetic tasks: {len(synthetic_tasks)}")
    print()

    # Show real data tasks
    for task in real_data_tasks[:2]:  # Show first 2
        print(f"Task: {task.id}")
        print(f"Query: {task.query[:100]}...")
        print(f"Expected: {task.expected_answer}")
        print(f"Tags: {list(task.tags)}")

        # Load context
        from src.benchmark.data_loader import create_task_context

        context = create_task_context(task.id)

        print(f"Data Available: {'Yes' if 'data' in context else 'No'}")
        if "data" in context:
            data = context["data"]
            if hasattr(data, "shape"):
                print(f"Data Shape: {data.shape}")
                print(f"Columns: {list(data.columns)}")
                print("Sample Data:")
                print(data.head(2).to_string())
            else:
                print(f"Data Type: {type(data)}")

        if "hints" in context:
            print(f"Hints Provided: {len(context['hints'])}")
            for hint in context["hints"][:2]:  # Show first 2 hints
                print(f"  - {hint}")

        print("-" * 50)


def main():
    demonstrate_strategies()
    demonstrate_real_data_loading()


if __name__ == "__main__":
    main()
