"""No-improvement baseline strategy."""

from src.core.types import Task, ImprovementContext


class NoImprovementStrategy:
    """
    Baseline strategy that does nothing.

    Essential for measuring whether other strategies actually help.
    With this strategy, the agent will always get the same result
    regardless of how many attempts it makes.
    """

    def improve(self, context: ImprovementContext) -> dict:
        """Return empty improvements - no learning."""
        return {}

    def persist(self, context: ImprovementContext) -> None:
        """No persistence - nothing to save."""
        pass

    def load_priors(self, task: Task) -> dict:
        """No priors - start fresh every time."""
        return {}
