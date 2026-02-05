"""Core dataclasses for the agent framework."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import Executor, Evaluator, ImprovementStrategy
    from ..benchmark.model_config import ModelConfig


@dataclass(frozen=True)
class Task:
    """Immutable task definition."""

    id: str
    query: str
    expected_answer: Any | None = None
    difficulty: str = "medium"
    tags: tuple[str, ...] = ()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    max_attempts: int = 5
    strategies: dict[str, "ImprovementStrategy"] = field(default_factory=dict)
    task_suite: list[Task] = field(default_factory=list)
    models: list["ModelConfig"] = field(
        default_factory=list
    )  # Model configurations for multi-model benchmarking
    evaluator: "Evaluator | None" = None
    executor: "Executor | None" = None
    seed: int | None = None  # For reproducibility


@dataclass
class ExecutionResult:
    """Result of executing a task."""

    output: Any
    trajectory: list[dict] = field(default_factory=list)
    code_generated: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Evaluation:
    """Evaluation of an execution result."""

    score: float  # 0.0 - 1.0
    passed: bool
    feedback: str
    criteria_scores: dict[str, float] = field(default_factory=dict)
    error_type: str | None = None  # e.g., "logic_error", "data_misunderstanding"
    hints: list[str] = field(default_factory=list)  # Actionable hints for fixing errors


@dataclass
class ImprovementContext:
    """Context provided to improvement strategies."""

    task: Task
    result: ExecutionResult
    evaluation: Evaluation
    attempt_number: int
    history: list[tuple[ExecutionResult, Evaluation]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of running a task through the agent loop."""

    task: Task
    passed: bool
    attempts: int
    final_score: float
    score_progression: list[float] = field(default_factory=list)
    final_result: ExecutionResult | None = None
    final_evaluation: Evaluation | None = None


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy across all tasks."""

    strategy_name: str
    pass_rate: float
    avg_attempts_to_pass: float
    avg_final_score: float
    improvement_curve: list[float] = field(default_factory=list)
    per_task_results: list[TaskResult] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison of multiple strategies."""

    strategies: dict[str, StrategyMetrics] = field(default_factory=dict)
    best_strategy: str = ""
    improvement_over_baseline: float = 0.0
    summary: str = ""


@dataclass
class BenchmarkRunRecord:
    """Persistent record of a benchmark run."""

    run_id: str
    timestamp: str  # ISO 8601 datetime string
    version: str  # Schema version for compatibility
    config: BenchmarkConfig
    results: dict[str, StrategyMetrics]
    metadata: dict[str, Any] = field(default_factory=dict)
