"""Core types and protocols for the agent framework."""

from .types import (
    Task,
    ExecutionResult,
    Evaluation,
    ImprovementContext,
    TaskResult,
    StrategyMetrics,
    ComparisonReport,
)
from .protocols import (
    Executor,
    Evaluator,
    ImprovementStrategy,
    Storage,
    MetricsCollector,
)

__all__ = [
    "Task",
    "ExecutionResult",
    "Evaluation",
    "ImprovementContext",
    "TaskResult",
    "StrategyMetrics",
    "ComparisonReport",
    "Executor",
    "Evaluator",
    "ImprovementStrategy",
    "Storage",
    "MetricsCollector",
]
