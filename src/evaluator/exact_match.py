"""Exact match evaluator with numeric tolerance and smart hints."""

import math
from typing import Any

from src.core.types import Task, ExecutionResult, Evaluation
from src.evaluator.hint_generator import HintGenerator, HintResult


class ExactMatchEvaluator:
    """
    Evaluator that checks for exact matches with numeric tolerance.

    Supports:
    - Exact string matching
    - Numeric matching with configurable tolerance (absolute and relative)
    - None/null handling
    - Error classification
    """

    def __init__(
        self,
        numeric_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-4,
        case_sensitive: bool = True,
        generate_hints: bool = True,
    ) -> None:
        self.numeric_tolerance = numeric_tolerance
        self.relative_tolerance = relative_tolerance
        self.case_sensitive = case_sensitive
        self.generate_hints = generate_hints
        self._hint_generator = HintGenerator() if generate_hints else None

    def evaluate(self, task: Task, result: ExecutionResult) -> Evaluation:
        """Evaluate an execution result against the task."""
        expected = task.expected_answer
        actual = result.output

        # Handle case where no expected answer is defined
        if expected is None:
            return Evaluation(
                score=1.0 if actual is not None else 0.0,
                passed=actual is not None,
                feedback="No expected answer defined; checked output is not None.",
                criteria_scores={"has_output": 1.0 if actual is not None else 0.0},
            )

        # Compare values
        match, score, error_type = self._compare(expected, actual)

        if match:
            return Evaluation(
                score=1.0,
                passed=True,
                feedback="Output matches expected answer.",
                criteria_scores={"exact_match": 1.0},
            )
        else:
            feedback = self._generate_feedback(expected, actual, error_type)

            # Generate smart hints if enabled
            hints = []
            hint_result = None
            if self._hint_generator:
                hint_result = self._hint_generator.generate_hints(
                    expected=expected,
                    actual=actual,
                    error_type=error_type,
                    query=task.query,
                )
                hints = hint_result.hints
                if hints:
                    feedback += " Hints: " + "; ".join(hints[:2])

            return Evaluation(
                score=score,
                passed=False,
                feedback=feedback,
                criteria_scores={
                    "exact_match": score,
                    "hint_confidence": hint_result.confidence if hint_result else 0.0,
                },
                error_type=error_type,
                hints=hints,
            )

    def _compare(self, expected: Any, actual: Any) -> tuple[bool, float, str | None]:
        """
        Compare expected and actual values.

        Returns: (match, partial_score, error_type)
        """
        # Handle None
        if actual is None:
            return False, 0.0, "no_output"

        # Numeric comparison
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return self._compare_numeric(expected, actual)

        # String comparison
        if isinstance(expected, str) and isinstance(actual, str):
            return self._compare_string(expected, actual)

        # List/sequence comparison
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            return self._compare_sequence(expected, actual)

        # Dict comparison
        if isinstance(expected, dict) and isinstance(actual, dict):
            return self._compare_dict(expected, actual)

        # Type mismatch - try string coercion
        if str(expected) == str(actual):
            return True, 1.0, None

        return False, 0.0, "type_mismatch"

    def _compare_numeric(
        self, expected: float, actual: float
    ) -> tuple[bool, float, str | None]:
        """Compare numeric values with tolerance."""
        # Handle special cases
        if math.isnan(expected) and math.isnan(actual):
            return True, 1.0, None
        if math.isinf(expected) and math.isinf(actual):
            return expected == actual, 1.0 if expected == actual else 0.0, "sign_error"

        # Absolute tolerance
        abs_diff = abs(expected - actual)
        if abs_diff <= self.numeric_tolerance:
            return True, 1.0, None

        # Relative tolerance
        if expected != 0:
            rel_diff = abs_diff / abs(expected)
            if rel_diff <= self.relative_tolerance:
                return True, 1.0, None

            # Partial score based on how close
            if rel_diff < 0.1:  # Within 10%
                return False, 0.9, "small_numeric_error"
            elif rel_diff < 0.5:  # Within 50%
                return False, 0.5, "numeric_error"

        return False, 0.0, "large_numeric_error"

    def _compare_string(
        self, expected: str, actual: str
    ) -> tuple[bool, float, str | None]:
        """Compare string values."""
        if self.case_sensitive:
            if expected == actual:
                return True, 1.0, None
        else:
            if expected.lower() == actual.lower():
                return True, 1.0, None

        # Check for whitespace differences
        if expected.strip() == actual.strip():
            return True, 0.95, "whitespace_difference"

        # Partial match score
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        if expected_words and actual_words:
            overlap = len(expected_words & actual_words)
            union = len(expected_words | actual_words)
            jaccard = overlap / union if union > 0 else 0
            if jaccard > 0.5:
                return False, jaccard, "partial_string_match"

        return False, 0.0, "string_mismatch"

    def _compare_sequence(
        self, expected: list | tuple, actual: list | tuple
    ) -> tuple[bool, float, str | None]:
        """Compare sequence values."""
        if len(expected) != len(actual):
            return False, 0.0, "length_mismatch"

        if len(expected) == 0:
            return True, 1.0, None

        matches = 0
        for e, a in zip(expected, actual):
            match, _, _ = self._compare(e, a)
            if match:
                matches += 1

        score = matches / len(expected)
        if score == 1.0:
            return True, 1.0, None
        return False, score, "sequence_partial_match"

    def _compare_dict(
        self, expected: dict, actual: dict
    ) -> tuple[bool, float, str | None]:
        """Compare dict values."""
        if set(expected.keys()) != set(actual.keys()):
            return False, 0.0, "key_mismatch"

        if len(expected) == 0:
            return True, 1.0, None

        matches = 0
        for key in expected:
            match, _, _ = self._compare(expected[key], actual[key])
            if match:
                matches += 1

        score = matches / len(expected)
        if score == 1.0:
            return True, 1.0, None
        return False, score, "dict_partial_match"

    def _generate_feedback(
        self, expected: Any, actual: Any, error_type: str | None
    ) -> str:
        """Generate human-readable feedback."""
        base = f"Expected {repr(expected)}, got {repr(actual)}."

        feedback_by_type = {
            "no_output": "No output was produced.",
            "type_mismatch": f"{base} Types don't match.",
            "small_numeric_error": f"{base} Close but not within tolerance.",
            "numeric_error": f"{base} Numeric value is significantly off.",
            "large_numeric_error": f"{base} Numeric value is very different.",
            "whitespace_difference": f"{base} Check for extra whitespace.",
            "partial_string_match": f"{base} Partial overlap in words.",
            "string_mismatch": f"{base} Strings don't match.",
            "length_mismatch": f"{base} Sequence lengths differ.",
            "sequence_partial_match": f"{base} Some elements match.",
            "key_mismatch": f"{base} Dictionary keys differ.",
            "dict_partial_match": f"{base} Some values match.",
            "sign_error": f"{base} Check the sign.",
        }

        return feedback_by_type.get(error_type or "", base)
