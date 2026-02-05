#!/usr/bin/env python3
"""
Interactive CLI for the Data Analysis Chatbot.

This script provides a REPL interface for asking natural language
questions about the financial dataset. It demonstrates:
- Code generation from natural language queries via LLM
- Execution of generated code against real data
- Cross-session learning via episodic memory
- Retry with improvement hints on failure

Run with: uv run python scripts/chat.py --persist-default

Commands:
  /help     - Show help
  /schema   - Show dataset schema
  /sample   - Show sample data
  /stats    - Show learning statistics
  /correct  - Mark last answer as CORRECT (saves pattern)
  /wrong    - Mark last answer as WRONG (saves for learning)
  /clear    - Clear learning history
  /quit     - Exit
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import argparse

from src.core.types import Task, ExecutionResult, Evaluation, ImprovementContext
from src.data import DatasetLoader
from src.prompts import build_data_analysis_prompt
from src.executor.code_runner import CodeRunner
from src.storage.file import FileStorage
from src.strategies.episodic_memory import EpisodicMemoryStrategy
from src.llm.claude import ClaudeClient
from src.evaluator.exact_match import ExactMatchEvaluator


def print_header() -> None:
    """Print the welcome header."""
    print()
    print("+" + "=" * 68 + "+")
    print("|              DATA ANALYSIS CHATBOT (LLM-powered)                  |")
    print("|              Ask questions about P&L data                        |")
    print("+" + "=" * 68 + "+")
    print()
    print("Type your question in natural language, or use commands:")
    print("  /help     - Show help")
    print("  /schema   - Show dataset schema")
    print("  /sample   - Show sample data")
    print("  /stats    - Show learning statistics")
    print("  /episodes - List stored episodes (learned patterns)")
    print("  /correct  - Mark last answer as CORRECT (saves pattern)")
    print("  /wrong    - Mark last answer as WRONG (saves for learning)")
    print("  /clear    - Clear learning history")
    print("  /quit     - Exit")
    print()


def print_schema() -> None:
    """Print the dataset schema."""
    print()
    print("Dataset Schema:")
    print("-" * 50)
    print("Columns:")
    print("  - Fiscal Year (int): 2020-2024")
    print("  - Fiscal Quarter (str): Q1, Q2, Q3, Q4")
    print("  - Fiscal Period (str): YYYY-MM format")
    print("  - FSLine Statement L1 (str): High-level category")
    print("  - FSLine Statement L2 (str): Detailed line item")
    print("  - Product (str): Product A, B, C, D")
    print("  - Country (str): Australia, Canada, Germany, Japan, UK, US")
    print("  - Currency (str): AUD, CAD, EUR, GBP, JPY, USD")
    print("  - Amount in Local Currency (float)")
    print("  - Amount in USD (float)")
    print("  - Version (str): Always 'Actuals'")
    print()
    print("L1 Categories: Net Revenue, Cost of Goods Sold, OPEX, Other Income/Expenses")
    print("L2 Examples: Gross Revenue, Marketing Expenses, R&D Expenses, Direct Labor")
    print()


def print_sample(df) -> None:
    """Print sample data."""
    print()
    print("Sample Data (first 5 rows):")
    print("-" * 80)
    sample = df.head()
    for _, row in sample.iterrows():
        print(f"  {row['Fiscal Year']} {row['Fiscal Quarter']} | "
              f"{row['FSLine Statement L2'][:20]:20s} | "
              f"{row['Product']:10s} | "
              f"{row['Country']:15s} | "
              f"${row['Amount in USD']:,.2f}")
    print()


def print_stats(strategy: EpisodicMemoryStrategy | None) -> None:
    """Print learning statistics."""
    print()
    print("Learning Statistics:")
    print("-" * 50)
    if strategy is None:
        print("  No persistence enabled. Use --persist-default to enable learning.")
    else:
        episode_count = strategy.get_episode_count()
        print(f"  Total episodes stored: {episode_count}")
        if episode_count > 0:
            print(f"  Episodes are stored for cross-session retrieval.")
            print(f"  Similar queries will benefit from past learnings.")
            print(f"  Use /episodes to list them.")
    print()


def print_episodes(strategy: EpisodicMemoryStrategy | None, storage) -> None:
    """List all stored episodes."""
    print()
    print("Stored Episodes:")
    print("-" * 60)
    if strategy is None or storage is None:
        print("  No persistence enabled.")
        return

    keys = storage.list_keys("episodes")
    if not keys:
        print("  No episodes stored yet.")
        print("  Ask questions and use /correct or /wrong to save learnings.")
        return

    from src.strategies.episodic_memory import Episode

    for i, key in enumerate(keys[:10], 1):  # Show max 10
        data = storage.load(key)
        if data:
            episode = Episode.from_dict(data)
            query_preview = episode.query[:50] + "..." if len(episode.query) > 50 else episode.query
            has_fix = "Yes" if episode.fixed_code else "No"
            print(f"  {i}. {query_preview}")
            print(f"     Has fix: {has_fix} | Keywords: {episode.keywords[:4]}")
            print()

    if len(keys) > 10:
        print(f"  ... and {len(keys) - 10} more")
    print()


class ChatSession:
    """Manages a chat session with LLM-powered code generation and learning."""

    def __init__(
        self,
        df,
        strategy: EpisodicMemoryStrategy | None,
        llm_client: ClaudeClient,
        max_attempts: int = 3,
    ):
        self.df = df
        self.strategy = strategy
        self.llm = llm_client
        self.max_attempts = max_attempts
        self.code_runner = CodeRunner()
        self.evaluator = ExactMatchEvaluator(
            numeric_tolerance=0.01,  # Allow small rounding differences
            relative_tolerance=0.001,
        )
        self.system_prompt = build_data_analysis_prompt()

        # Session stats
        self.queries_answered = 0
        self.first_attempt_successes = 0
        self.total_attempts = 0

        # Track last query for manual feedback
        self.last_task: Task | None = None
        self.last_result: ExecutionResult | None = None
        self.last_code: str | None = None
        self.last_history: list = []
        self.last_output = None

    def _build_prompt(self, query: str, context: dict, attempt: int = 1) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "You are a Python data analysis expert. Generate code to answer questions about a pandas DataFrame called 'df'.",
            "",
            "DataFrame columns:",
            "- Fiscal Year (int): 2020-2024",
            "- Fiscal Quarter (str): Q1, Q2, Q3, Q4",
            "- FSLine Statement L1 (str): Net Revenue, Cost of Goods Sold, OPEX, Other Income/Expenses",
            "- FSLine Statement L2 (str): Gross Revenue, Marketing Expenses, R&D Expenses, etc.",
            "- Product (str): Product A, Product B, Product C, Product D",
            "- Country (str): Australia, Canada, Germany, Japan, United Kingdom, United States",
            "- Amount in USD (float): The monetary value",
            "",
            "IMPORTANT: Store your final answer in a variable called 'result'.",
            "Return ONLY Python code in a ```python code block.",
            "",
        ]

        # Add examples from prior learning
        if examples := context.get("examples"):
            parts.append("Here are examples of similar successful queries:")
            for i, example in enumerate(examples[:2], 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"```python\n{example}\n```")
            parts.append("")

        # Add hints from prior learning or improvement
        if hints := context.get("hints"):
            parts.append("Hints from prior experience:")
            for hint in hints[:3]:
                parts.append(f"- {hint}")
            parts.append("")

        # Add attempt info if retrying
        if attempt > 1:
            parts.append(f"This is attempt {attempt}. Previous attempt failed.")
            if prev_error := context.get("previous_error"):
                parts.append(f"Previous error: {prev_error}")
            if prev_code := context.get("previous_code"):
                parts.append(f"Previous code that failed:\n```python\n{prev_code}\n```")
            parts.append("")

        parts.append(f"Question: {query}")

        return "\n".join(parts)

    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from LLM response."""
        import re

        # Try ```python blocks first
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try generic ``` blocks
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        return None

    def mark_correct(self) -> bool:
        """Mark the last response as correct and save the learned pattern."""
        if not self.last_task or not self.last_result:
            print("  No previous query to mark.")
            return False

        if not self.strategy:
            print("  No persistence enabled. Use --persist-default.")
            return False

        # Create success context and persist
        success_context = ImprovementContext(
            task=self.last_task,
            result=self.last_result,
            evaluation=Evaluation(score=1.0, passed=True, feedback="User confirmed correct"),
            attempt_number=1,
            history=self.last_history,
        )
        self.strategy.persist(success_context)
        print("  [Marked as CORRECT - pattern saved for future queries]")
        return True

    def mark_wrong(self, expected_answer=None) -> bool:
        """Mark the last response as wrong and save for learning."""
        if not self.last_task or not self.last_result:
            print("  No previous query to mark.")
            return False

        if not self.strategy:
            print("  No persistence enabled. Use --persist-default.")
            return False

        # Create failure context
        feedback = f"User marked as wrong. Got: {self.last_output}"
        if expected_answer:
            feedback += f", Expected: {expected_answer}"

        failure_eval = Evaluation(
            score=0.0,
            passed=False,
            feedback=feedback,
            error_type="user_marked_wrong",
        )

        failure_context = ImprovementContext(
            task=self.last_task,
            result=self.last_result,
            evaluation=failure_eval,
            attempt_number=1,
            history=[],
        )
        self.strategy.persist(failure_context)
        print("  [Marked as WRONG - failure saved for learning]")
        return True

    def answer_query(self, query: str, expected_answer=None) -> dict:
        """
        Answer a query using the LLM with retry and learning.

        Returns dict with: success, result, attempts, code
        """
        import uuid

        task_id = f"chat_{uuid.uuid4().hex[:8]}"
        task = Task(id=task_id, query=query, expected_answer=expected_answer)

        # Track for feedback
        self.last_task = task
        self.last_history = []

        # Load prior learnings
        context = {}
        if self.strategy:
            priors = self.strategy.load_priors(task)
            context.update(priors)
            if priors.get("examples"):
                print()
                print("  " + "*" * 50)
                print("  * LEARNING APPLIED - Found similar past query!")
                print("  " + "*" * 50)
                print(f"  Found {len(priors['examples'])} learned code pattern(s)")
                # Show preview of the first example
                example = priors['examples'][0]
                preview_lines = example.strip().split('\n')[:4]
                print("  Pattern preview:")
                for line in preview_lines:
                    print(f"    {line}")
                if len(example.strip().split('\n')) > 4:
                    print("    ...")
                print()

        history: list[tuple[ExecutionResult, Evaluation]] = []

        for attempt in range(1, self.max_attempts + 1):
            self.total_attempts += 1

            # Generate code via LLM
            prompt = self._build_prompt(query, context, attempt)

            try:
                response = self.llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    system="You are a Python code generator. Return only code in ```python blocks."
                )
            except Exception as e:
                print(f"  [LLM Error: {e}]")
                return {"success": False, "result": None, "attempts": attempt, "code": None, "error": str(e)}

            code = self._extract_code(response)
            if not code:
                print(f"  [No code found in LLM response]")
                context["previous_error"] = "No code block found in response"
                continue

            # Execute the code
            exec_result = self.code_runner.execute(code, {"df": self.df})

            result = ExecutionResult(
                output=exec_result.get("result"),
                code_generated=code,
                metadata={"stdout": exec_result.get("stdout", "")},
            )

            # Show the code
            print()
            print(f"  Attempt {attempt} - Generated Code:")
            print("  " + "-" * 40)
            for line in code.split("\n"):
                print(f"    {line}")
            print("  " + "-" * 40)

            if not exec_result["success"]:
                error_msg = exec_result.get("error", "Unknown error")
                print(f"  Execution Error: {error_msg}")

                # Create evaluation for the failure
                evaluation = Evaluation(
                    score=0.0,
                    passed=False,
                    feedback=f"Execution failed: {error_msg}",
                    error_type="execution_error",
                )
                history.append((result, evaluation))

                # Get improvement hints
                if self.strategy and attempt < self.max_attempts:
                    imp_context = ImprovementContext(
                        task=task,
                        result=result,
                        evaluation=evaluation,
                        attempt_number=attempt,
                        history=history,
                    )
                    improvements = self.strategy.improve(imp_context)
                    context.update(improvements)
                    context["previous_error"] = error_msg
                    context["previous_code"] = code
                    print(f"  [Retrying with hints: {improvements.get('hints', [])[:2]}]")
                continue

            # Code executed successfully - check result
            actual_result = exec_result["result"]
            print(f"  Output: {actual_result}")

            # Track for feedback
            self.last_result = result
            self.last_code = code
            self.last_output = actual_result

            # If no expected answer, we can't evaluate - assume success
            if expected_answer is None:
                self.queries_answered += 1
                if attempt == 1:
                    self.first_attempt_successes += 1

                # Persist successful pattern if we had prior failures
                if history and self.strategy:
                    success_context = ImprovementContext(
                        task=task,
                        result=result,
                        evaluation=Evaluation(score=1.0, passed=True, feedback="Success"),
                        attempt_number=attempt,
                        history=history,
                    )
                    self.strategy.persist(success_context)
                    print("  [Learned from this session - pattern saved]")

                return {
                    "success": True,
                    "result": actual_result,
                    "attempts": attempt,
                    "code": code,
                }

            # Evaluate against expected answer
            evaluation = self.evaluator.evaluate(task, result)

            if evaluation.passed:
                print(f"  [Correct!]")
                self.queries_answered += 1
                if attempt == 1:
                    self.first_attempt_successes += 1

                # Persist if we learned something (had failures before success)
                if history and self.strategy:
                    success_context = ImprovementContext(
                        task=task,
                        result=result,
                        evaluation=evaluation,
                        attempt_number=attempt,
                        history=history,
                    )
                    self.strategy.persist(success_context)
                    print("  [Learned from this session - pattern saved]")

                return {
                    "success": True,
                    "result": actual_result,
                    "attempts": attempt,
                    "code": code,
                }

            # Wrong answer
            print(f"  [Incorrect: {evaluation.feedback}]")
            history.append((result, evaluation))

            # Get improvement hints for retry
            if self.strategy and attempt < self.max_attempts:
                imp_context = ImprovementContext(
                    task=task,
                    result=result,
                    evaluation=evaluation,
                    attempt_number=attempt,
                    history=history,
                )
                improvements = self.strategy.improve(imp_context)
                context.update(improvements)
                context["previous_error"] = evaluation.feedback
                context["previous_code"] = code
                if hints := improvements.get("hints"):
                    print(f"  [Retrying with hints: {hints[:2]}]")

        # Failed after all attempts
        return {
            "success": False,
            "result": actual_result if 'actual_result' in dir() else None,
            "attempts": self.max_attempts,
            "code": code if 'code' in dir() else None,
            "error": "Max attempts reached",
        }


def run_interactive(storage_path: Path | None = None, model: str = "claude-sonnet-4-5-20250929") -> None:
    """Run the interactive CLI."""
    print_header()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it in .env file or export it in your shell.")
        return

    # Load dataset
    print("Loading dataset...")
    try:
        loader = DatasetLoader()
        data = loader.load()
        print(f"Loaded {len(data):,} rows of financial data.")
    except FileNotFoundError:
        print("ERROR: Dataset not found. Run the download script first.")
        print("  uv run python -c \"from src.data import DatasetLoader; DatasetLoader().load()\"")
        return

    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(data)

    # Set up storage and strategy
    storage = None
    strategy = None
    if storage_path:
        storage = FileStorage(storage_path)
        strategy = EpisodicMemoryStrategy(storage=storage)
        episode_count = strategy.get_episode_count()
        print(f"Loaded {episode_count} learned episodes from previous sessions.")
    else:
        print("Running without persistence (use --persist-default for cross-session learning).")

    # Set up LLM client
    print(f"Using model: {model}")
    try:
        llm_client = ClaudeClient(model=model)
    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {e}")
        return

    # Create chat session
    session = ChatSession(df=df, strategy=strategy, llm_client=llm_client)

    print()
    print("Ready! Ask a question about the financial data.")
    print()

    # REPL loop
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # Handle commands
        if query.startswith("/"):
            cmd = query.lower().split()[0]
            if cmd in ["/quit", "/exit", "/q"]:
                print("Goodbye!")
                break
            elif cmd == "/help":
                print_header()
            elif cmd == "/schema":
                print_schema()
            elif cmd == "/sample":
                print_sample(df)
            elif cmd == "/stats":
                print_stats(strategy)
                print(f"  Session: {session.queries_answered} queries, "
                      f"{session.first_attempt_successes} first-attempt successes, "
                      f"{session.total_attempts} total attempts")
            elif cmd == "/clear":
                if strategy:
                    strategy.clear_episodes()
                    print("Learning history cleared.")
                else:
                    print("No persistence enabled.")
            elif cmd == "/episodes":
                print_episodes(strategy, storage)
            elif cmd == "/correct":
                session.mark_correct()
            elif cmd == "/wrong":
                # Parse optional expected answer
                parts = query.split(maxsplit=1)
                expected = parts[1] if len(parts) > 1 else None
                session.mark_wrong(expected)
            else:
                print(f"Unknown command: {query}")
            continue

        # Answer the query
        print()
        print("Thinking...")
        result = session.answer_query(query)

        print()
        if result["success"]:
            value = result["result"]
            print(f"Answer: {value}")
            if isinstance(value, (int, float)) and abs(value) > 1:
                print(f"        (${value:,.2f})")
        else:
            print(f"Failed to answer after {result['attempts']} attempts.")
            if result.get("error"):
                print(f"Error: {result['error']}")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI for data analysis chatbot with LLM"
    )
    parser.add_argument(
        "--persist",
        type=str,
        default=None,
        help="Path to store learning data for cross-session persistence",
    )
    parser.add_argument(
        "--persist-default",
        action="store_true",
        help="Use default persistence path (.agent_memory/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Model to use for code generation (default: claude-sonnet-4-5-20250929)",
    )

    args = parser.parse_args()

    storage_path = None
    if args.persist_default:
        storage_path = Path(".agent_memory")
    elif args.persist:
        storage_path = Path(args.persist)

    run_interactive(storage_path, args.model)


if __name__ == "__main__":
    main()
