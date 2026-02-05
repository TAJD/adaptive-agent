"""Metrics collection and reporting."""

from dataclasses import dataclass, field

from src.core.types import (
    Task,
    ExecutionResult,
    Evaluation,
    TaskResult,
)


@dataclass
class AttemptRecord:
    """Record of a single attempt."""

    task_id: str
    attempt: int
    score: float
    passed: bool
    error_type: str | None


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self) -> None:
        self._attempts: list[AttemptRecord] = []
        self._task_results: list[TaskResult] = []

    def record_attempt(
        self,
        task: Task,
        result: ExecutionResult,
        evaluation: Evaluation,
        attempt: int,
    ) -> None:
        """Record a single attempt."""
        self._attempts.append(
            AttemptRecord(
                task_id=task.id,
                attempt=attempt,
                score=evaluation.score,
                passed=evaluation.passed,
                error_type=evaluation.error_type,
            )
        )

    def record_task_complete(self, task_result: TaskResult) -> None:
        """Record task completion."""
        self._task_results.append(task_result)

    def get_summary(self) -> dict:
        """Get summary metrics."""
        if not self._task_results:
            return {
                "total_tasks": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_attempts": 0.0,
                "avg_final_score": 0.0,
            }

        passed = sum(1 for r in self._task_results if r.passed)
        total = len(self._task_results)

        # Average attempts for passed tasks only
        passed_attempts = [r.attempts for r in self._task_results if r.passed]
        avg_attempts = sum(passed_attempts) / len(passed_attempts) if passed_attempts else 0.0

        # Average final score
        avg_score = sum(r.final_score for r in self._task_results) / total

        # Error type distribution
        error_counts: dict[str, int] = {}
        for attempt in self._attempts:
            if attempt.error_type:
                error_counts[attempt.error_type] = error_counts.get(attempt.error_type, 0) + 1

        return {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total,
            "avg_attempts_to_pass": avg_attempts,
            "avg_final_score": avg_score,
            "total_attempts": len(self._attempts),
            "error_distribution": error_counts,
        }

    def get_improvement_curve(self) -> list[float]:
        """Get average score at each attempt number."""
        if not self._attempts:
            return []

        # Group by attempt number
        by_attempt: dict[int, list[float]] = {}
        for attempt in self._attempts:
            if attempt.attempt not in by_attempt:
                by_attempt[attempt.attempt] = []
            by_attempt[attempt.attempt].append(attempt.score)

        # Average score at each attempt
        max_attempt = max(by_attempt.keys())
        curve = []
        for i in range(1, max_attempt + 1):
            if i in by_attempt:
                curve.append(sum(by_attempt[i]) / len(by_attempt[i]))
            elif curve:
                curve.append(curve[-1])  # Carry forward

        return curve

    def reset(self) -> None:
        """Reset all metrics."""
        self._attempts.clear()
        self._task_results.clear()
