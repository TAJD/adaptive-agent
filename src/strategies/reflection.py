"""Reflection-based improvement strategy."""

from src.core.types import Task, ImprovementContext
from src.core.protocols import Storage


class ReflectionStrategy:
    """
    Self-critique loop strategy for in-session learning.

    This strategy:
    1. Analyzes the error and evaluation feedback
    2. Generates hints based on what went wrong
    3. Tracks patterns within the session
    4. Optionally persists learnings for cross-session use

    This is a simple but effective baseline for iterative improvement.
    """

    def __init__(self, storage: Storage | None = None) -> None:
        """
        Initialize the reflection strategy.

        Args:
            storage: Optional storage for cross-session persistence.
                    If None, learnings are only used within the session.
        """
        self.storage = storage
        self._session_learnings: dict[str, list[str]] = {}

    def improve(self, context: ImprovementContext) -> dict:
        """
        Generate improvements based on failure analysis.

        Returns a dict with hints and constraints for the next attempt.
        """
        hints: list[str] = []
        constraints: list[str] = []

        # Extract information from evaluation
        error_type = context.evaluation.error_type
        feedback = context.evaluation.feedback
        score = context.evaluation.score

        # Generate hints based on error type
        if error_type:
            error_hints = self._hints_for_error_type(error_type)
            hints.extend(error_hints)

        # Generate hints based on attempt history
        if len(context.history) >= 2:
            pattern_hints = self._analyze_patterns(context.history)
            hints.extend(pattern_hints)

        # Add feedback as a constraint
        if feedback:
            constraints.append(f"Previous attempt feedback: {feedback}")

        # Track progress
        if score > 0:
            hints.append(f"Previous attempt scored {score:.2f} - build on partial success.")

        # Store in session learnings
        task_id = context.task.id
        if task_id not in self._session_learnings:
            self._session_learnings[task_id] = []
        self._session_learnings[task_id].extend(hints)

        return {
            "hints": hints,
            "constraints": constraints,
            "attempt": context.attempt_number + 1,
            "previous_score": score,
            "error_type": error_type,
        }

    def persist(self, context: ImprovementContext) -> None:
        """Persist learnings for cross-session use."""
        if self.storage is None:
            return

        task_id = context.task.id
        error_type = context.evaluation.error_type

        # Store error patterns
        if error_type:
            key = f"reflection/errors/{error_type}"
            existing = self.storage.load(key) or {"count": 0, "tasks": []}
            existing["count"] += 1
            if task_id not in existing["tasks"]:
                existing["tasks"].append(task_id)
            self.storage.save(key, existing)

        # Store task-specific learnings
        if self._session_learnings.get(task_id):
            key = f"reflection/tasks/{task_id}"
            existing = self.storage.load(key) or {"learnings": []}
            existing["learnings"].extend(self._session_learnings[task_id])
            # Deduplicate
            existing["learnings"] = list(set(existing["learnings"]))
            self.storage.save(key, existing)

        # Store tag mappings for cross-task learning lookup
        if context.task.tags:
            for tag in context.task.tags:
                tag_key = f"reflection/tags/{tag}/{task_id}"
                self.storage.save(tag_key, {"task_id": task_id})

    def load_priors(self, task: Task) -> dict:
        """Load any prior learnings relevant to this task."""
        if self.storage is None:
            return {}

        priors: dict = {"hints": [], "constraints": []}

        # Load task-specific learnings
        task_data = self.storage.load(f"reflection/tasks/{task.id}")
        if task_data and task_data.get("learnings"):
            priors["hints"].extend(task_data["learnings"][:5])  # Limit to avoid overload

        # Load learnings from tasks with matching tags
        # Tags are stored as "reflection/tags/{tag}/{task_id}" during persist
        if task.tags:
            for tag in task.tags:
                tag_keys = self.storage.list_keys(f"reflection/tags/{tag}/")
                for key in tag_keys[:3]:
                    # Extract task_id from key and load its learnings
                    parts = key.split("/")
                    if len(parts) >= 4:
                        other_task_id = parts[3]
                        if other_task_id != task.id:  # Don't load our own learnings twice
                            other_data = self.storage.load(
                                f"reflection/tasks/{other_task_id}"
                            )
                            if other_data and other_data.get("learnings"):
                                priors["hints"].extend(other_data["learnings"][:2])

        return priors

    def clear_session_state(self) -> None:
        """Clear session-only state between tasks to prevent cross-task contamination."""
        self._session_learnings.clear()

    def _hints_for_error_type(self, error_type: str) -> list[str]:
        """Generate hints based on error type."""
        hints_map = {
            "no_output": [
                "Ensure the code produces output.",
                "Check that 'result' variable is set.",
            ],
            "type_mismatch": [
                "Check the expected output type.",
                "Ensure numeric results are numbers, not strings.",
            ],
            "small_numeric_error": [
                "Check rounding and precision.",
                "Verify the calculation logic.",
            ],
            "numeric_error": [
                "Review the calculation approach.",
                "Check for off-by-one errors or missing factors.",
            ],
            "large_numeric_error": [
                "Fundamentally reconsider the approach.",
                "Check units and scale of values.",
            ],
            "string_mismatch": [
                "Check for exact string formatting.",
                "Verify case sensitivity and whitespace.",
            ],
            "length_mismatch": [
                "Check the number of elements expected.",
                "Verify loop bounds and filters.",
            ],
        }
        return hints_map.get(error_type, ["Review the approach and try again."])

    def _analyze_patterns(
        self, history: list[tuple]
    ) -> list[str]:
        """Analyze patterns in attempt history."""
        hints = []

        # Check if scores are improving
        scores = [eval.score for _, eval in history]
        if len(scores) >= 2:
            if scores[-1] > scores[-2]:
                hints.append("Making progress - continue this direction.")
            elif scores[-1] < scores[-2]:
                hints.append("Score decreased - reconsider last changes.")
            else:
                hints.append("Score unchanged - try a different approach.")

        # Check for repeated error types
        error_types = [eval.error_type for _, eval in history if eval.error_type]
        if error_types and len(set(error_types)) == 1:
            hints.append(f"Repeated error: {error_types[0]}. Fundamental change needed.")

        return hints
