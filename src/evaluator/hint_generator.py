"""Smart hint generation based on error patterns."""

from dataclasses import dataclass
from typing import Any


@dataclass
class HintResult:
    """Result of hint generation."""

    hints: list[str]
    confidence: float  # 0.0 - 1.0, how confident we are in the diagnosis
    error_pattern: str | None  # Identified error pattern
    suggested_fix_type: str | None  # Type of fix likely needed


class HintGenerator:
    """
    Generates actionable hints based on error analysis.

    This class analyzes evaluation results and generates specific,
    actionable hints rather than generic suggestions.
    """

    # Common error patterns and their hints
    NUMERIC_PATTERNS = {
        "much_larger": {
            "threshold": 3.0,  # actual > expected * 3
            "hints": [
                "Result is much larger than expected - check if you're missing filters",
                "Verify you're filtering by all required dimensions (time period, product, region)",
                "Make sure you're not summing across all rows when you should filter first",
            ],
            "fix_type": "add_filters",
        },
        "much_smaller": {
            "threshold": 0.33,  # actual < expected * 0.33
            "hints": [
                "Result is much smaller than expected - check if filters are too restrictive",
                "Verify filter values match the data exactly (case sensitivity, spelling)",
                "Check if you're filtering on a column that doesn't exist",
            ],
            "fix_type": "fix_filters",
        },
        "off_by_factor": {
            "factors": [10, 100, 1000, 1000000],
            "hints": [
                "Result is off by a power of 10 - check units (thousands vs actual values)",
                "Verify you're using the correct column (Amount in USD vs Local Currency)",
            ],
            "fix_type": "fix_units",
        },
        "negative_vs_positive": {
            "hints": [
                "Sign is wrong - check if you need to negate the result",
                "Verify whether the calculation should use subtraction or addition",
            ],
            "fix_type": "fix_sign",
        },
        "close_but_wrong": {
            "threshold": 0.1,  # within 10%
            "hints": [
                "Result is close but not exact - check rounding",
                "Verify you're including/excluding the correct boundary values",
            ],
            "fix_type": "fix_rounding",
        },
    }

    STRUCTURAL_PATTERNS = {
        "none_result": {
            "hints": [
                "Result is None - check if 'result' variable is assigned",
                "Verify your filter returns rows (df may be empty after filtering)",
                "Check for NaN values that might cause aggregation to return None",
            ],
            "fix_type": "fix_assignment",
        },
        "type_mismatch": {
            "hints": [
                "Result type doesn't match expected - check if you need type conversion",
                "If expecting a number, ensure the result is numeric not a string",
            ],
            "fix_type": "fix_type",
        },
        "empty_result": {
            "hints": [
                "Result is empty - filter conditions may be too strict",
                "Check column names match exactly (case-sensitive)",
                "Verify the values you're filtering on exist in the data",
            ],
            "fix_type": "fix_filters",
        },
    }

    QUERY_PATTERNS = {
        # Keywords that suggest specific filter requirements
        "time_keywords": {
            "keywords": ["q1", "q2", "q3", "q4", "quarter", "2020", "2021", "2022", "2023", "2024", "year"],
            "hints": ["Query mentions time period - ensure you filter by Fiscal Year and/or Fiscal Quarter"],
        },
        "product_keywords": {
            "keywords": ["product a", "product b", "product c", "product d", "product"],
            "hints": ["Query mentions product - ensure you filter by Product column"],
        },
        "country_keywords": {
            "keywords": ["australia", "canada", "germany", "japan", "uk", "united kingdom", "us", "united states"],
            "hints": ["Query mentions country - ensure you filter by Country column"],
        },
        "aggregation_keywords": {
            "keywords": ["total", "sum", "average", "mean", "count", "max", "min"],
            "hints": ["Query asks for aggregation - verify you're using the correct aggregation function"],
        },
    }

    def generate_hints(
        self,
        expected: Any,
        actual: Any,
        error_type: str | None,
        query: str | None = None,
    ) -> HintResult:
        """
        Generate hints based on the error analysis.

        Args:
            expected: The expected answer
            actual: The actual answer from execution
            error_type: The classified error type from evaluator
            query: The original query (for context-aware hints)

        Returns:
            HintResult with hints, confidence, and diagnosis
        """
        hints = []
        confidence = 0.5
        error_pattern = error_type
        fix_type = None

        # Analyze None/empty results
        if actual is None:
            pattern_info = self.STRUCTURAL_PATTERNS["none_result"]
            hints.extend(pattern_info["hints"])
            fix_type = pattern_info["fix_type"]
            confidence = 0.8
            error_pattern = "none_result"

        # Analyze numeric errors
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            numeric_hints = self._analyze_numeric_error(expected, actual)
            hints.extend(numeric_hints["hints"])
            if numeric_hints["pattern"]:
                error_pattern = numeric_hints["pattern"]
                fix_type = numeric_hints["fix_type"]
                confidence = numeric_hints["confidence"]

        # Analyze type mismatches
        elif error_type == "type_mismatch":
            pattern_info = self.STRUCTURAL_PATTERNS["type_mismatch"]
            hints.extend(pattern_info["hints"])
            fix_type = pattern_info["fix_type"]
            confidence = 0.7

        # Add query-based hints
        if query:
            query_hints = self._analyze_query(query)
            hints.extend(query_hints)

        # Deduplicate hints while preserving order
        seen = set()
        unique_hints = []
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                unique_hints.append(hint)

        return HintResult(
            hints=unique_hints[:5],  # Limit to 5 most relevant
            confidence=confidence,
            error_pattern=error_pattern,
            suggested_fix_type=fix_type,
        )

    def _analyze_numeric_error(self, expected: float, actual: float) -> dict:
        """Analyze numeric errors and return pattern-specific hints."""
        if expected == 0:
            return {"hints": ["Expected value is 0 - verify this is correct"], "pattern": None, "fix_type": None, "confidence": 0.5}

        ratio = actual / expected if expected != 0 else float("inf")

        # Check for sign error
        if expected > 0 and actual < 0 or expected < 0 and actual > 0:
            pattern = self.NUMERIC_PATTERNS["negative_vs_positive"]
            return {
                "hints": pattern["hints"],
                "pattern": "negative_vs_positive",
                "fix_type": pattern["fix_type"],
                "confidence": 0.9,
            }

        # Check for much larger
        if ratio > self.NUMERIC_PATTERNS["much_larger"]["threshold"]:
            pattern = self.NUMERIC_PATTERNS["much_larger"]
            return {
                "hints": pattern["hints"],
                "pattern": "much_larger",
                "fix_type": pattern["fix_type"],
                "confidence": 0.85,
            }

        # Check for much smaller
        if ratio < self.NUMERIC_PATTERNS["much_smaller"]["threshold"]:
            pattern = self.NUMERIC_PATTERNS["much_smaller"]
            return {
                "hints": pattern["hints"],
                "pattern": "much_smaller",
                "fix_type": pattern["fix_type"],
                "confidence": 0.85,
            }

        # Check for factor-of-10 errors
        for factor in self.NUMERIC_PATTERNS["off_by_factor"]["factors"]:
            if 0.9 < ratio / factor < 1.1 or 0.9 < ratio * factor < 1.1:
                pattern = self.NUMERIC_PATTERNS["off_by_factor"]
                return {
                    "hints": pattern["hints"] + [f"Result appears to be off by a factor of {factor}"],
                    "pattern": "off_by_factor",
                    "fix_type": pattern["fix_type"],
                    "confidence": 0.9,
                }

        # Check for close but wrong
        rel_diff = abs(expected - actual) / abs(expected) if expected != 0 else 0
        if rel_diff < self.NUMERIC_PATTERNS["close_but_wrong"]["threshold"]:
            pattern = self.NUMERIC_PATTERNS["close_but_wrong"]
            return {
                "hints": pattern["hints"],
                "pattern": "close_but_wrong",
                "fix_type": pattern["fix_type"],
                "confidence": 0.7,
            }

        # Generic numeric error
        return {
            "hints": ["Check your calculation logic and verify all filters are correct"],
            "pattern": "numeric_error",
            "fix_type": "fix_calculation",
            "confidence": 0.5,
        }

    def _analyze_query(self, query: str) -> list[str]:
        """Extract hints based on query keywords."""
        hints = []
        query_lower = query.lower()

        for pattern_name, pattern_info in self.QUERY_PATTERNS.items():
            for keyword in pattern_info["keywords"]:
                if keyword in query_lower:
                    hints.extend(pattern_info["hints"])
                    break  # Only add hints once per pattern

        return hints


def generate_hints_for_evaluation(
    expected: Any,
    actual: Any,
    error_type: str | None,
    query: str | None = None,
) -> list[str]:
    """
    Convenience function to generate hints.

    Args:
        expected: Expected answer
        actual: Actual answer
        error_type: Error type from evaluator
        query: Original query text

    Returns:
        List of hint strings
    """
    generator = HintGenerator()
    result = generator.generate_hints(expected, actual, error_type, query)
    return result.hints
