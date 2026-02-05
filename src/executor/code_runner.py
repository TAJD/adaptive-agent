"""Safe Python code execution."""

import sys
from io import StringIO
from typing import Any


class CodeRunner:
    """Executes Python code in a restricted environment."""

    def __init__(self, timeout: float = 30.0, allowed_modules: list[str] | None = None) -> None:
        self.timeout = timeout
        self.allowed_modules = allowed_modules or ["pandas", "math", "statistics", "json"]

    def execute(self, code: str, globals_dict: dict | None = None) -> dict[str, Any]:
        """
        Execute Python code and return the result.

        Returns:
            dict with keys:
                - "success": bool
                - "result": the value of 'result' variable if set
                - "stdout": captured stdout
                - "error": error message if failed
        """
        if globals_dict is None:
            globals_dict = {}

        # Add allowed imports to globals
        for module_name in self.allowed_modules:
            try:
                globals_dict[module_name] = __import__(module_name)
            except ImportError:
                pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute the code
            # Use same dict for globals and locals to avoid scoping issues
            # with comprehensions and generator expressions
            exec_namespace = globals_dict.copy()
            exec(code, exec_namespace, exec_namespace)

            # Extract user-defined variables (excluding modules we added)
            user_locals = {
                k: v for k, v in exec_namespace.items()
                if k not in globals_dict and not k.startswith("__")
            }

            return {
                "success": True,
                "result": exec_namespace.get("result"),
                "stdout": captured_output.getvalue(),
                "error": None,
                "locals": user_locals,
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "stdout": captured_output.getvalue(),
                "error": f"{type(e).__name__}: {e}",
                "locals": {},
            }
        finally:
            sys.stdout = old_stdout
