"""Protocol definitions for pluggable components."""

from typing import Protocol, Any

from .types import Task, ExecutionResult, Evaluation, ImprovementContext


class Executor(Protocol):
    """Protocol for task execution."""

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        """Execute a task and return the result."""
        ...


class Evaluator(Protocol):
    """Protocol for result evaluation."""

    def evaluate(self, task: Task, result: ExecutionResult) -> Evaluation:
        """Evaluate an execution result against the task."""
        ...


class ImprovementStrategy(Protocol):
    """Protocol for learning from failures."""

    def improve(self, context: ImprovementContext) -> dict:
        """
        Generate improvements based on failure context.

        Returns a dict of improvements to apply to the next attempt,
        e.g., {"hints": [...], "constraints": [...], "examples": [...]}.
        """
        ...

    def persist(self, context: ImprovementContext) -> None:
        """Persist learnings for cross-session use."""
        ...

    def load_priors(self, task: Task) -> dict:
        """Load any prior learnings relevant to this task."""
        ...


class Storage(Protocol):
    """Protocol for persistence - swappable implementations."""

    def save(self, key: str, data: Any) -> None:
        """Save data under a key."""
        ...

    def load(self, key: str) -> Any | None:
        """Load data by key. Returns None if not found."""
        ...

    def query(self, filter_dict: dict) -> list[Any]:
        """Query for items matching the filter."""
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted, False if not found."""
        ...


class MetricsCollector(Protocol):
    """Protocol for collecting performance metrics."""

    def record_attempt(
        self,
        task: Task,
        result: ExecutionResult,
        evaluation: Evaluation,
        attempt: int,
    ) -> None:
        """Record a single attempt."""
        ...

    def record_task_complete(self, task_result: "TaskResult") -> None:
        """Record task completion."""
        ...

    def get_summary(self) -> dict:
        """Get summary metrics."""
        ...

    def reset(self) -> None:
        """Reset all metrics."""
        ...


# Forward reference for TaskResult
from .types import TaskResult as TaskResult  # noqa: E402, F811
