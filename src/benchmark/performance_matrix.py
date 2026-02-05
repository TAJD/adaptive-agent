"""Performance matrix for strategy x model analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.types import BenchmarkRunRecord, StrategyMetrics


@dataclass
class PerformanceCell:
    """Performance data for a strategy-model combination."""

    strategy: str
    model: str
    pass_rate: float
    avg_attempts_to_pass: float
    avg_final_score: float
    improvement_curve: list[float]
    num_runs: int  # Number of runs aggregated
    run_ids: list[str]  # IDs of runs included


class PerformanceMatrix:
    """
    Matrix for analyzing performance across strategies and models.

    Provides aggregated performance metrics for each strategy-model pair.
    """

    def __init__(self, records: list[BenchmarkRunRecord]) -> None:
        self._records = records
        self._matrix: dict[str, dict[str, PerformanceCell]] = {}
        self._strategies: set[str] = set()
        self._models: set[str] = set()
        self._build_matrix(records)

    def _build_matrix(self, records: list[BenchmarkRunRecord]) -> None:
        """Build the performance matrix from benchmark records."""
        # Group records by strategy and model
        grouped: dict[str, dict[str, list[BenchmarkRunRecord]]] = {}

        for record in records:
            # Extract model from metadata or config (assuming metadata has "model")
            model = record.metadata.get("model", "unknown")
            for strategy_name, metrics in record.results.items():
                if strategy_name not in grouped:
                    grouped[strategy_name] = {}
                if model not in grouped[strategy_name]:
                    grouped[strategy_name][model] = []
                grouped[strategy_name][model].append(record)

        # Aggregate for each strategy-model pair
        for strategy, models in grouped.items():
            self._strategies.add(strategy)
            for model, recs in models.items():
                self._models.add(model)
                if strategy not in self._matrix:
                    self._matrix[strategy] = {}

                # Aggregate metrics across runs
                pass_rates = []
                avg_attempts = []
                avg_scores = []
                improvement_curves = []
                run_ids = []

                for rec in recs:
                    metrics = rec.results[strategy]
                    pass_rates.append(metrics.pass_rate)
                    avg_attempts.append(metrics.avg_attempts_to_pass)
                    avg_scores.append(metrics.avg_final_score)
                    improvement_curves.append(metrics.improvement_curve)
                    run_ids.append(rec.run_id)

                # Average the metrics
                avg_pass_rate = sum(pass_rates) / len(pass_rates)
                avg_attempts_val = sum(avg_attempts) / len(avg_attempts)
                avg_score = sum(avg_scores) / len(avg_scores)

                # Average improvement curves (element-wise)
                if improvement_curves:
                    max_len = max(len(curve) for curve in improvement_curves)
                    avg_curve = []
                    for i in range(max_len):
                        values = [
                            curve[i] for curve in improvement_curves if i < len(curve)
                        ]
                        avg_curve.append(sum(values) / len(values) if values else 0.0)
                else:
                    avg_curve = []

                cell = PerformanceCell(
                    strategy=strategy,
                    model=model,
                    pass_rate=avg_pass_rate,
                    avg_attempts_to_pass=avg_attempts_val,
                    avg_final_score=avg_score,
                    improvement_curve=avg_curve,
                    num_runs=len(recs),
                    run_ids=run_ids,
                )

                self._matrix[strategy][model] = cell

    @property
    def strategies(self) -> list[str]:
        """Get list of strategies in the matrix."""
        return sorted(self._strategies)

    @property
    def models(self) -> list[str]:
        """Get list of models in the matrix."""
        return sorted(self._models)

    def get_cell(self, strategy: str, model: str) -> PerformanceCell | None:
        """Get performance cell for a strategy-model pair."""
        return self._matrix.get(strategy, {}).get(model)

    def get_strategy_performance(self, strategy: str) -> dict[str, PerformanceCell]:
        """Get performance across all models for a strategy."""
        return self._matrix.get(strategy, {})

    def get_model_performance(self, model: str) -> dict[str, PerformanceCell]:
        """Get performance across all strategies for a model."""
        result = {}
        for strategy, models in self._matrix.items():
            if model in models:
                result[strategy] = models[model]
        return result

    def get_best_strategy_for_model(
        self, model: str, metric: str = "pass_rate"
    ) -> str | None:
        """Get the best strategy for a given model by metric."""
        performances = self.get_model_performance(model)
        if not performances:
            return None

        return max(performances.keys(), key=lambda s: getattr(performances[s], metric))

    def get_best_model_for_strategy(
        self, strategy: str, metric: str = "pass_rate"
    ) -> str | None:
        """Get the best model for a given strategy by metric."""
        performances = self.get_strategy_performance(strategy)
        if not performances:
            return None

        return max(performances.keys(), key=lambda m: getattr(performances[m], metric))

    def get_overall_best(self, metric: str = "pass_rate") -> tuple[str, str] | None:
        """Get the overall best strategy-model pair by metric."""
        best_value = float("-inf")
        best_pair = None

        for strategy, models in self._matrix.items():
            for model, cell in models.items():
                value = getattr(cell, metric)
                if value > best_value:
                    best_value = value
                    best_pair = (strategy, model)

        return best_pair

    def to_csv(self, path: str, metric: str = "pass_rate") -> None:
        """Export the matrix to CSV for the given metric."""
        import csv

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            header = ["Strategy"] + sorted(self._models)
            writer.writerow(header)

            # Rows
            for strategy in sorted(self._strategies):
                row = [strategy]
                for model in sorted(self._models):
                    cell = self.get_cell(strategy, model)
                    value = getattr(cell, metric) if cell else ""
                    row.append(value)
                writer.writerow(row)

    def get_strategy_trend(
        self, strategy: str, model: str, metric: str = "pass_rate"
    ) -> list[dict]:
        """
        Get historical trend for a strategy-model pair.

        Returns list of dicts with timestamp and metric value, sorted by time.
        """
        # Group records by timestamp for this strategy/model
        trend_data = []
        for record in self._records:  # Need to store records
            if strategy in record.results and record.metadata.get("model") == model:
                metrics = record.results[strategy]
                value = getattr(metrics, metric)
                trend_data.append(
                    {
                        "timestamp": record.timestamp,
                        "value": value,
                        "run_id": record.run_id,
                    }
                )

        # Sort by timestamp
        trend_data.sort(key=lambda x: x["timestamp"])
        return trend_data

    def get_overall_trends(self, metric: str = "pass_rate") -> dict[str, list[dict]]:
        """
        Get trends for all strategy-model combinations.

        Returns dict with keys like "strategy_model" and values as trend lists.
        """
        trends = {}
        for strategy in self._strategies:
            for model in self._models:
                key = f"{strategy}_{model}"
                trend = self.get_strategy_trend(strategy, model, metric)
                if trend:  # Only include if there's data
                    trends[key] = trend
        return trends

    def compare_runs_over_time(
        self, metric: str = "pass_rate"
    ) -> dict[str, list[float]]:
        """
        Compare performance across runs chronologically.

        Returns dict with strategy keys and lists of metric values over time.
        """
        # Group records by strategy, sort by time
        strategy_runs = {}
        for record in self._records:
            for strategy_name, metrics in record.results.items():
                if strategy_name not in strategy_runs:
                    strategy_runs[strategy_name] = []
                strategy_runs[strategy_name].append(
                    {
                        "timestamp": record.timestamp,
                        "value": getattr(metrics, metric),
                        "run_id": record.run_id,
                    }
                )

        # Sort each strategy's runs by time
        for strategy in strategy_runs:
            strategy_runs[strategy].sort(key=lambda x: x["timestamp"])

        # Extract just the values
        result = {}
        for strategy, runs in strategy_runs.items():
            result[strategy] = [run["value"] for run in runs]

        return result

    def get_improvement_trends(self) -> dict[str, list[float]]:
        """
        Get trends in improvement over time for each strategy.

        Returns dict with strategy keys and lists of average improvement rates.
        """
        trends = {}
        for strategy in self._strategies:
            improvements = []
            runs = [r for r in self._records if strategy in r.results]
            runs.sort(key=lambda x: x.timestamp)

            for run in runs:
                metrics = run.results[strategy]
                # Calculate improvement as average score increase per attempt
                if len(metrics.improvement_curve) > 1:
                    improvements.append(
                        metrics.improvement_curve[-1] - metrics.improvement_curve[0]
                    )
                else:
                    improvements.append(0.0)

            trends[strategy] = improvements

        return trends
