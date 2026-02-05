"""LLM-based executor that generates and runs code."""

import re
from typing import Any

from src.core.types import Task, ExecutionResult
from src.llm.protocol import LLMClient
from src.executor.code_runner import CodeRunner


class LLMExecutor:
    """
    Executor that uses an LLM to generate code for tasks.

    Generates Python code via the LLM, executes it safely,
    and returns the result.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a Python code generator for data analysis tasks.
When given a task, generate Python code that:
1. Solves the problem step by step
2. Stores the final answer in a variable called 'result'
3. Uses only standard library modules and pandas if needed

Return ONLY the Python code wrapped in ```python and ``` markers.
Do not include explanations outside the code block."""

    def __init__(
        self,
        llm_client: LLMClient,
        code_runner: CodeRunner | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize the LLM executor.

        Args:
            llm_client: LLM client for code generation.
            code_runner: Code executor. Creates default if not provided.
            system_prompt: Custom system prompt for code generation.
        """
        self._llm = llm_client
        self._code_runner = code_runner or CodeRunner()
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def execute(self, task: Task, context: dict) -> ExecutionResult:
        """
        Execute a task by generating and running code.

        Args:
            task: The task to execute.
            context: Execution context with optional keys:
                - hints: List of hints for the LLM
                - constraints: List of constraints
                - examples: List of example solutions
                - data: Data to inject into code execution

        Returns:
            ExecutionResult with the output and trajectory.
        """
        trajectory: list[dict] = []

        # Build the prompt
        prompt = self._build_prompt(task, context)
        trajectory.append({"step": "prompt_built", "prompt": prompt})

        # Generate code
        messages = [{"role": "user", "content": prompt}]
        response = self._llm.complete(messages, system=self._system_prompt)
        trajectory.append({"step": "llm_response", "response": response})

        # Extract code from response
        code = self._extract_code(response)
        if not code:
            return ExecutionResult(
                output=None,
                trajectory=trajectory,
                code_generated=None,
                metadata={"error": "No code block found in LLM response"},
            )
        trajectory.append({"step": "code_extracted", "code": code})

        # Prepare execution globals
        globals_dict = self._prepare_globals(context)

        # Execute the code
        exec_result = self._code_runner.execute(code, globals_dict)
        trajectory.append({"step": "code_executed", "result": exec_result})

        if not exec_result["success"]:
            return ExecutionResult(
                output=None,
                trajectory=trajectory,
                code_generated=code,
                metadata={"error": exec_result["error"]},
            )

        return ExecutionResult(
            output=exec_result["result"],
            trajectory=trajectory,
            code_generated=code,
            metadata={
                "stdout": exec_result["stdout"],
                "locals": {k: str(v) for k, v in exec_result["locals"].items()},
            },
        )

    def _build_prompt(self, task: Task, context: dict) -> str:
        """Build the prompt for code generation."""
        parts = [f"Task: {task.query}"]

        # If data is provided, tell the LLM about it
        if "data" in context:
            data = context["data"]
            parts.append("\nData available:")
            parts.append("- A pandas DataFrame called 'data' is already loaded and available")
            if hasattr(data, "columns"):
                parts.append(f"- Columns: {list(data.columns)}")
            if hasattr(data, "shape"):
                parts.append(f"- Shape: {data.shape[0]} rows x {data.shape[1]} columns")
            parts.append("- DO NOT read from CSV files - use the 'data' variable directly")

        if hints := context.get("hints"):
            parts.append("\nHints:")
            for hint in hints:
                parts.append(f"- {hint}")

        if constraints := context.get("constraints"):
            parts.append("\nConstraints:")
            for constraint in constraints:
                parts.append(f"- {constraint}")

        if examples := context.get("examples"):
            parts.append("\nExamples of similar solutions:")
            for i, example in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"```python\n{example}\n```")

        return "\n".join(parts)

    def _extract_code(self, response: str) -> str | None:
        """Extract Python code from markdown code blocks."""
        # Try to find ```python ... ``` blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try generic ``` ... ``` blocks
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        return None

    def _prepare_globals(self, context: dict) -> dict[str, Any]:
        """Prepare global variables for code execution."""
        globals_dict: dict[str, Any] = {}

        # Inject any data from context
        data = context.get("data")
        if data is not None:
            globals_dict["data"] = data

        return globals_dict
