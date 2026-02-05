"""LLM-based evaluator using Claude to assess answer correctness."""

import json
from typing import Any

from src.core.types import Task, ExecutionResult, Evaluation
from src.llm.protocol import LLMClient


# Tool definition for structured evaluation output
EVALUATION_TOOL = {
    "name": "submit_evaluation",
    "description": "Submit the evaluation results for the agent's answer",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_correct": {
                "type": "boolean",
                "description": "Whether the answer is correct (True) or incorrect (False)",
            },
            "score": {
                "type": "number",
                "description": "Score from 0.0 to 1.0, where 1.0 is perfectly correct",
            },
            "error_type": {
                "type": "string",
                "enum": [
                    "none",
                    "logic_error",
                    "data_misunderstanding",
                    "calculation_error",
                    "filter_error",
                    "aggregation_error",
                    "type_error",
                    "edge_case",
                    "no_output",
                    "other",
                ],
                "description": "Classification of the error type if incorrect",
            },
            "feedback": {
                "type": "string",
                "description": "Detailed feedback explaining why the answer is correct or incorrect",
            },
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning for the evaluation",
            },
        },
        "required": ["is_correct", "score", "error_type", "feedback", "reasoning"],
    },
}


SYSTEM_PROMPT = """You are an expert evaluator for a data analysis agent.
Your task is to assess whether the agent's answer is correct.

You will be given:
1. The original question/query
2. The expected answer
3. The agent's output
4. The code the agent generated (if any)

Evaluate the answer carefully:
- For numeric answers, allow for small rounding differences (within 0.01%)
- For string answers, check for semantic equivalence
- Consider edge cases and error messages

Error Type Classifications:
- logic_error: Fundamental misunderstanding of the problem
- data_misunderstanding: Wrong interpretation of what data represents
- calculation_error: Math mistake in the calculation
- filter_error: Wrong filtering criteria (wrong product, country, time period)
- aggregation_error: Wrong aggregation (sum vs mean, missing groups)
- type_error: Wrong output type (number vs string)
- edge_case: Failed to handle edge case or missing data
- no_output: No answer was produced
- other: Other type of error

Use the submit_evaluation tool to provide your assessment."""


class LLMEvaluator:
    """
    Evaluator that uses an LLM (Claude) to assess answer correctness.

    This evaluator provides:
    - Semantic comparison of answers
    - Detailed error classification
    - Natural language feedback
    - Flexible handling of edge cases

    More nuanced than exact matching, but requires LLM calls.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        strict_numeric: bool = False,
        include_code_in_eval: bool = True,
    ) -> None:
        """
        Initialize the LLM evaluator.

        Args:
            llm_client: LLM client for evaluation.
            strict_numeric: If True, require exact numeric matches.
            include_code_in_eval: If True, include generated code in evaluation.
        """
        self._llm = llm_client
        self._strict_numeric = strict_numeric
        self._include_code = include_code_in_eval

    def evaluate(self, task: Task, result: ExecutionResult) -> Evaluation:
        """
        Evaluate an execution result against the task using LLM.

        Args:
            task: The task that was executed.
            result: The execution result to evaluate.

        Returns:
            Evaluation with score, pass/fail, feedback, and error type.
        """
        # Handle no output case directly
        if result.output is None:
            return Evaluation(
                score=0.0,
                passed=False,
                feedback="No output was produced by the agent.",
                error_type="no_output",
            )

        # Handle no expected answer case
        if task.expected_answer is None:
            return Evaluation(
                score=1.0 if result.output is not None else 0.0,
                passed=result.output is not None,
                feedback="No expected answer defined; output was produced.",
            )

        # Build evaluation prompt
        prompt = self._build_prompt(task, result)

        # Call LLM with tool
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._llm.complete_with_tools(
                messages=messages,
                tools=[EVALUATION_TOOL],
                system=SYSTEM_PROMPT,
            )

            # Extract evaluation from tool call
            return self._parse_response(response)

        except Exception as e:
            # Fallback to basic comparison if LLM fails
            return self._fallback_evaluate(task, result, str(e))

    def _build_prompt(self, task: Task, result: ExecutionResult) -> str:
        """Build the evaluation prompt."""
        parts = [
            "Please evaluate the following agent response:",
            "",
            f"**Question:** {task.query}",
            "",
            f"**Expected Answer:** {repr(task.expected_answer)}",
            "",
            f"**Agent's Output:** {repr(result.output)}",
        ]

        if self._include_code and result.code_generated:
            parts.extend([
                "",
                "**Generated Code:**",
                f"```python\n{result.code_generated}\n```",
            ])

        if self._strict_numeric:
            parts.extend([
                "",
                "Note: Use STRICT numeric comparison. Small differences are NOT acceptable.",
            ])

        return "\n".join(parts)

    def _parse_response(self, response: dict) -> Evaluation:
        """Parse the LLM response into an Evaluation."""
        tool_calls = response.get("tool_calls", [])

        if not tool_calls:
            # No tool call - try to extract from text
            return Evaluation(
                score=0.5,
                passed=False,
                feedback="Evaluation incomplete - no structured response.",
                error_type="other",
            )

        # Get the evaluation tool call
        eval_call = tool_calls[0]
        args = eval_call.get("arguments", {})

        # Handle string arguments (may need to parse JSON)
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return Evaluation(
                    score=0.5,
                    passed=False,
                    feedback="Failed to parse evaluation response.",
                    error_type="other",
                )

        is_correct = args.get("is_correct", False)
        score = float(args.get("score", 0.0))
        error_type = args.get("error_type", "other")
        feedback = args.get("feedback", "No feedback provided.")

        # Map "none" error type to None
        if error_type == "none":
            error_type = None

        return Evaluation(
            score=score,
            passed=is_correct,
            feedback=feedback,
            error_type=error_type,
            criteria_scores={
                "llm_assessment": score,
            },
        )

    def _fallback_evaluate(
        self, task: Task, result: ExecutionResult, error: str
    ) -> Evaluation:
        """Fallback evaluation when LLM fails."""
        expected = task.expected_answer
        actual = result.output

        # Simple comparison
        if expected == actual:
            return Evaluation(
                score=1.0,
                passed=True,
                feedback="Exact match (fallback evaluation).",
            )

        # Numeric comparison with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            rel_diff = abs(expected - actual) / max(abs(expected), 1e-10)
            if rel_diff < 0.0001:  # 0.01% tolerance
                return Evaluation(
                    score=1.0,
                    passed=True,
                    feedback="Numeric match within tolerance (fallback).",
                )
            elif rel_diff < 0.1:
                return Evaluation(
                    score=0.5,
                    passed=False,
                    feedback=f"Close but not exact: expected {expected}, got {actual}. (Fallback: {error})",
                    error_type="calculation_error",
                )

        return Evaluation(
            score=0.0,
            passed=False,
            feedback=f"Values don't match: expected {expected}, got {actual}. (Fallback: {error})",
            error_type="other",
        )
