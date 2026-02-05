"""Main agent runner loop with retry and improvement logic."""

import logging
from dataclasses import dataclass, field
from typing import Any

from src.core.types import (
    Task,
    ExecutionResult,
    Evaluation,
    ImprovementContext,
    TaskResult,
)
from src.core.protocols import Executor, Evaluator, ImprovementStrategy

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent runner."""

    max_attempts: int = 5
    stop_on_pass: bool = True
    collect_trajectory: bool = True


class AgentRunner:
    """
    Main agent loop that executes tasks with retry and improvement.

    This is the core orchestrator that:
    1. Executes tasks using an Executor
    2. Evaluates results using an Evaluator
    3. On failure, uses an ImprovementStrategy to learn and retry
    4. Tracks progress and returns detailed results

    Usage:
        runner = AgentRunner(executor, evaluator, strategy)
        result = runner.run(task)
        print(f"Passed: {result.passed} in {result.attempts} attempts")
    """

    def __init__(
        self,
        executor: Executor,
        evaluator: Evaluator,
        strategy: ImprovementStrategy,
        config: AgentConfig | None = None,
    ) -> None:
        self.executor = executor
        self.evaluator = evaluator
        self.strategy = strategy
        self.config = config or AgentConfig()

    def run(self, task: Task, context: dict[str, Any] | None = None) -> TaskResult:
        """
        Run a task through the agent loop.

        Returns:
            TaskResult with pass/fail status, attempts, and score progression.
        """
        history: list[tuple[ExecutionResult, Evaluation]] = []
        score_progression: list[float] = []
        final_result: ExecutionResult | None = None
        final_evaluation: Evaluation | None = None

        # Load any prior learnings for this task
        strategy_context = self.strategy.load_priors(task)

        # Log learning retrieval
        if strategy_context.get("examples"):
            logger.info(
                f"[LEARNING] Retrieved {len(strategy_context['examples'])} prior examples for task {task.id}"
            )
        if strategy_context.get("hints"):
            logger.info(
                f"[LEARNING] Retrieved {len(strategy_context['hints'])} hints for task {task.id}"
            )

        # Merge with provided context (e.g., real data)
        if context is None:
            context = strategy_context
        else:
            # Start with provided context, then add strategy context
            merged_context = context.copy()
            merged_context.update(strategy_context)
            context = merged_context

        for attempt in range(1, self.config.max_attempts + 1):
            # Execute the task
            result = self.executor.execute(task, context)
            final_result = result

            # Evaluate the result
            evaluation = self.evaluator.evaluate(task, result)
            final_evaluation = evaluation
            score_progression.append(evaluation.score)

            # Check if passed
            if evaluation.passed:
                # Persist learning if we succeeded after failures
                if history:
                    success_context = ImprovementContext(
                        task=task,
                        result=result,
                        evaluation=evaluation,
                        attempt_number=attempt,
                        history=history,
                    )
                    self.strategy.persist(success_context)
                    logger.info(
                        f"[LEARNING] Persisted successful fix for task {task.id} after {attempt} attempts"
                    )

                return TaskResult(
                    task=task,
                    passed=True,
                    attempts=attempt,
                    final_score=evaluation.score,
                    score_progression=score_progression,
                    final_result=final_result,
                    final_evaluation=final_evaluation,
                )

            # Store in history for improvement context
            history.append((result, evaluation))

            # Don't try to improve on last attempt
            if attempt >= self.config.max_attempts:
                break

            # Generate improvements for next attempt
            improvement_context = ImprovementContext(
                task=task,
                result=result,
                evaluation=evaluation,
                attempt_number=attempt,
                history=history,
            )

            # Get improvements from strategy
            improvements = self.strategy.improve(improvement_context)
            context.update(improvements)

            # Log improvement generation
            logger.info(
                f"[LEARNING] Generated improvements after attempt {attempt}: "
                f"{len(improvements.get('hints', []))} hints, "
                f"{len(improvements.get('examples', []))} examples"
            )

        # Failed after all attempts - persist the final failure for learning
        if history:
            final_failure_context = ImprovementContext(
                task=task,
                result=final_result,
                evaluation=final_evaluation,
                attempt_number=self.config.max_attempts,
                history=history,
            )
            self.strategy.persist(final_failure_context)
            logger.info(
                f"[LEARNING] Persisted failure episode for task {task.id} after {self.config.max_attempts} attempts"
            )

        return TaskResult(
            task=task,
            passed=False,
            attempts=self.config.max_attempts,
            final_score=score_progression[-1] if score_progression else 0.0,
            score_progression=score_progression,
            final_result=final_result,
            final_evaluation=final_evaluation,
        )

    def run_batch(self, tasks: list[Task]) -> list[TaskResult]:
        """Run multiple tasks and return all results."""
        results = []
        for task in tasks:
            result = self.run(task)
            results.append(result)
            # Clear session state between tasks to prevent cross-task contamination
            if hasattr(self.strategy, "clear_session_state"):
                self.strategy.clear_session_state()
        return results
