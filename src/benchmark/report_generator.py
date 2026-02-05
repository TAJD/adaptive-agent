"""Summary report generator for benchmark results."""

from typing import Any

from src.core.types import BenchmarkRunRecord
from src.benchmark.performance_matrix import PerformanceMatrix


class SummaryReportGenerator:
    """Generates human-readable summary reports from benchmark results."""

    def generate_run_summary(self, record: BenchmarkRunRecord) -> str:
        """Generate a summary report for a single benchmark run."""
        lines = []
        lines.append(f"# Benchmark Run Summary: {record.run_id}")
        lines.append("")
        lines.append(f"**Timestamp:** {record.timestamp}")
        lines.append(f"**Version:** {record.version}")
        lines.append(f"**Tasks:** {len(record.config.task_suite)}")
        lines.append(f"**Max Attempts:** {record.config.max_attempts}")
        lines.append("")

        if record.metadata:
            lines.append("## Metadata")
            for key, value in record.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("## Strategy Results")
        lines.append("")
        lines.append("| Strategy | Pass Rate | Avg Attempts | Avg Score | Tasks |")
        lines.append("|----------|-----------|--------------|-----------|-------|")

        for strategy_name, metrics in sorted(record.results.items()):
            pass_rate = f"{metrics.pass_rate:.1%}"
            avg_attempts = f"{metrics.avg_attempts_to_pass:.1f}"
            avg_score = f"{metrics.avg_final_score:.2f}"
            num_tasks = len(metrics.per_task_results)
            lines.append(
                f"| {strategy_name} | {pass_rate} | {avg_attempts} | {avg_score} | {num_tasks} |"
            )

        lines.append("")
        return "\n".join(lines)

    def generate_comparison_summary(self, records: list[BenchmarkRunRecord]) -> str:
        """Generate a comparison summary across multiple runs."""
        if not records:
            return "# No Records to Compare"

        lines = []
        lines.append("# Benchmark Comparison Summary")
        lines.append("")
        lines.append(f"**Total Runs:** {len(records)}")
        lines.append(
            f"**Date Range:** {min(r.timestamp for r in records)} to {max(r.timestamp for r in records)}"
        )
        lines.append("")

        # Create performance matrix
        matrix = PerformanceMatrix(records)

        lines.append("## Strategies")
        for strategy in sorted(matrix.strategies):
            lines.append(f"- {strategy}")
        lines.append("")

        lines.append("## Models")
        for model in sorted(matrix.models):
            lines.append(f"- {model}")
        lines.append("")

        lines.append("## Performance Matrix (Pass Rate)")
        lines.append("")
        lines.append(
            "| Strategy \\ Model | " + " | ".join(sorted(matrix.models)) + " |"
        )
        lines.append(
            "|-------------------|" + "|".join("---" for _ in matrix.models) + "|"
        )

        for strategy in sorted(matrix.strategies):
            row = [f"{strategy}"]
            for model in sorted(matrix.models):
                cell = matrix.get_cell(strategy, model)
                value = f"{cell.pass_rate:.1%}" if cell else "-"
                row.append(value)
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        lines.append("## Best Performers")
        best_overall = matrix.get_overall_best("pass_rate")
        if best_overall:
            strategy, model = best_overall
            cell = matrix.get_cell(strategy, model)
            lines.append(
                f"- **Overall Best:** {strategy} on {model} ({cell.pass_rate:.1%} pass rate)"
            )
        lines.append("")

        return "\n".join(lines)

    def generate_matrix_summary(self, matrix: PerformanceMatrix) -> str:
        """Generate a summary report from a performance matrix."""
        lines = []
        lines.append("# Performance Matrix Summary")
        lines.append("")
        lines.append(f"**Strategies:** {len(matrix.strategies)}")
        lines.append(f"**Models:** {len(matrix.models)}")
        lines.append(
            f"**Total Combinations:** {len(matrix.strategies) * len(matrix.models)}"
        )
        lines.append("")

        lines.append("## Strategies")
        for strategy in sorted(matrix.strategies):
            perf = matrix.get_strategy_performance(strategy)
            avg_pass = (
                sum(c.pass_rate for c in perf.values()) / len(perf) if perf else 0
            )
            lines.append(f"- {strategy}: {avg_pass:.1%} avg pass rate")
        lines.append("")

        lines.append("## Models")
        for model in sorted(matrix.models):
            perf = matrix.get_model_performance(model)
            avg_pass = (
                sum(c.pass_rate for c in perf.values()) / len(perf) if perf else 0
            )
            lines.append(f"- {model}: {avg_pass:.1%} avg pass rate")
        lines.append("")

        best = matrix.get_overall_best("pass_rate")
        if best:
            strategy, model = best
            cell = matrix.get_cell(strategy, model)
            lines.append(f"## Best Combination")
            lines.append(
                f"**{strategy}** on **{model}**: {cell.pass_rate:.1%} pass rate, {cell.avg_attempts_to_pass:.1f} avg attempts"
            )
        lines.append("")

        return "\n".join(lines)

    def generate_trend_summary(self, matrix: PerformanceMatrix) -> str:
        """Generate a trend analysis summary."""
        lines = []
        lines.append("# Trend Analysis Summary")
        lines.append("")

        overall_trends = matrix.get_overall_trends("pass_rate")
        if overall_trends:
            lines.append("## Performance Trends")
            for key, trend in overall_trends.items():
                if len(trend) > 1:
                    first = trend[0]["value"]
                    last = trend[-1]["value"]
                    change = last - first
                    direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    lines.append(
                        f"- {key}: {direction} {change:+.1%} ({first:.1%} → {last:.1%})"
                    )
            lines.append("")

        improvement_trends = matrix.get_improvement_trends()
        if improvement_trends:
            lines.append("## Improvement Trends")
            for strategy, improvements in improvement_trends.items():
                if improvements:
                    avg_improvement = sum(improvements) / len(improvements)
                    lines.append(
                        f"- {strategy}: {avg_improvement:.2f} avg improvement per run"
                    )
            lines.append("")

        return "\n".join(lines)
