"""Mock executor for deterministic testing."""

from typing import Any

from src.core.types import Task, ExecutionResult


class MockExecutor:
    """
    Mock executor that returns predetermined responses.

    Useful for testing and benchmarking without actual LLM calls.
    """

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        """
        Initialize with predetermined responses.

        Args:
            responses: Dict mapping task IDs to response dicts.
                Each response can have:
                - "output": The output value
                - "code": Generated code (optional)
                - "trajectory": List of steps (optional)
                - "error": If set, simulate an error
        """
        self._responses = responses or {}
        self._call_count: dict[str, int] = {}

    def add_response(
        self,
        task_id: str,
        output: Any,
        code: str | None = None,
        trajectory: list[dict] | None = None,
        attempt: int | None = None,
    ) -> None:
        """
        Add a response for a task.

        Args:
            task_id: The task ID to respond to
            output: The output value
            code: Generated code (optional)
            trajectory: List of execution steps (optional)
            attempt: If set, only return this response on the nth attempt
        """
        key = f"{task_id}:{attempt}" if attempt is not None else task_id
        self._responses[key] = {
            "output": output,
            "code": code,
            "trajectory": trajectory or [],
        }

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        """Execute a task and return the result."""
        # Track call count for this task
        self._call_count[task.id] = self._call_count.get(task.id, 0) + 1
        attempt = self._call_count[task.id]

        # Look for attempt-specific response first
        key = f"{task.id}:{attempt}"
        response = self._responses.get(key)

        # Fall back to general response for this task
        if response is None:
            response = self._responses.get(task.id)

        # Fall back to default response
        if response is None:
            response = self._responses.get("default", {
                "output": None,
                "code": None,
                "trajectory": [],
            })

        # Check for simulated error
        if "error" in response:
            raise RuntimeError(response["error"])

        return ExecutionResult(
            output=response.get("output"),
            code_generated=response.get("code"),
            trajectory=response.get("trajectory", []),
            metadata={
                "mock": True,
                "attempt": attempt,
                "context": context,
            },
        )

    def get_call_count(self, task_id: str) -> int:
        """Get the number of times a task was executed."""
        return self._call_count.get(task_id, 0)

    def reset_counts(self) -> None:
        """Reset call counts."""
        self._call_count.clear()
