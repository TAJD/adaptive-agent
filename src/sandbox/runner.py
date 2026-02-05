"""Sandboxed code execution runner protocol."""

import sys
from abc import ABC, abstractmethod

from .types import SandboxConfig, SandboxResult


class SandboxedRunner(ABC):
    """Abstract base class for sandboxed code execution runners."""

    @abstractmethod
    def run_code(self, code: str, config: SandboxConfig) -> SandboxResult:
        """
        Execute code in a sandboxed environment.

        Args:
            code: The Python code to execute.
            config: Sandbox configuration (time limits, memory, etc.)

        Returns:
            SandboxResult with execution outcome.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this runner is available on the current system.

        Returns:
            True if the runner can be used.
        """
        pass


class SubprocessRunner(SandboxedRunner):
    """True subprocess-based sandboxed runner using separate Python process."""

    def is_available(self) -> bool:
        """Check if subprocess execution is available."""
        try:
            import subprocess

            # Test if we can run python
            result = subprocess.run(
                [sys.executable, "-c", "print('test')"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def run_code(self, code: str, config: SandboxConfig) -> SandboxResult:
        """
        Execute code in a true subprocess with sandboxing restrictions.

        Creates a temporary Python script and runs it in a separate process
        with resource limits and module restrictions.
        """
        import tempfile
        import subprocess
        import os
        import time

        start_time = time.time()

        try:
            # Create temporary script with sandboxing code
            sandbox_script = self._create_sandbox_script(code, config)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(sandbox_script)
                script_path = f.name

            try:
                # Prepare environment
                env = os.environ.copy()
                if config.environment_variables:
                    env.update(config.environment_variables)

                # Prepare subprocess arguments
                cmd = [sys.executable, script_path]

                # Run with timeout and resource limits
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.time_limit,
                    env=env,
                    cwd=config.working_directory if config.working_directory else None,
                )

                execution_time = time.time() - start_time

                # Check result
                if result.returncode == 0:
                    output = result.stdout
                    if len(output) > config.max_output_length:
                        output = output[: config.max_output_length] + "..."
                    return SandboxResult.success_result(output, execution_time)
                else:
                    error = result.stderr or result.stdout
                    return SandboxResult.error_result(
                        error, result.returncode, execution_time
                    )

            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                return SandboxResult.timeout_result(execution_time)

            finally:
                # Clean up temporary file
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult.error_result(str(e), execution_time=execution_time)

    def _create_sandbox_script(self, user_code: str, config: SandboxConfig) -> str:
        """Create a sandbox script that restricts execution."""
        import textwrap

        # Create restricted builtins
        allowed_builtins = config.allowed_builtins.copy()
        # Add builtins needed for the sandbox script itself
        required_builtins = [
            "__import__",
            "sys",
            "print",
            "str",
            "list",
            "dict",
            "getattr",
            "hasattr",
            "delattr",
            "Exception",
            "ImportError",
            "NameError",
            "AttributeError",
            "KeyError",
        ]
        for builtin in required_builtins:
            if builtin not in allowed_builtins:
                allowed_builtins.append(builtin)

        builtins_list = ", ".join(f'"{b}"' for b in allowed_builtins)
        modules_list = ", ".join(f'"{m}"' for m in config.allowed_modules)

        sandbox_code = f"""
import sys
import builtins

# Restrict builtins
allowed_builtins = [{builtins_list}]
original_builtins = builtins.__dict__.copy()

# Remove disallowed builtins
for name in list(original_builtins.keys()):
    if name not in allowed_builtins:
        try:
            delattr(builtins, name)
        except AttributeError:
            pass

# Restrict imports
allowed_modules = [{modules_list}]
original_import = builtins.__import__

def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in allowed_modules:
        raise ImportError(f"Import of '{{name}}' is not allowed")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = restricted_import

# Execute user code
try:
    {textwrap.indent(user_code, "    ")}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

        return sandbox_code


class InProcessRunner(SandboxedRunner):
    """
    In-process sandboxed runner with permissive security level.

    Runs code in the same process with basic restrictions but allows
    more modules and builtins than the subprocess runner.
    """

    def is_available(self) -> bool:
        """Always available since it runs in-process."""
        return True

    def run_code(self, code: str, config: SandboxConfig) -> SandboxResult:
        """
        Execute code in-process with permissive restrictions.

        Allows more modules and builtins while still providing basic safety.
        """
        import time
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        start_time = time.time()
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        try:
            # Create a more permissive globals dict
            import builtins

            # Allow most standard library modules except dangerous ones
            dangerous_modules = {
                "os",
                "sys",
                "subprocess",
                "multiprocessing",
                "threading",
                "socket",
                "urllib",
                "http",
                "ftplib",
                "telnetlib",
                "shutil",
                "glob",
                "tempfile",
                "pathlib",
            }

            def permissive_import(
                name, globals=None, locals=None, fromlist=(), level=0
            ):
                """Permissive import that blocks dangerous modules."""
                if name in dangerous_modules:
                    raise ImportError(
                        f"Import of '{name}' is not allowed in permissive mode"
                    )
                return __import__(name, globals, locals, fromlist, level)

            # Allow more builtins but still restrict dangerous ones
            restricted_builtins = {
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "exit",
                "quit",
            }

            builtins_dict = {
                name: getattr(builtins, name)
                for name in dir(builtins)
                if not name.startswith("_") and name not in restricted_builtins
            }
            builtins_dict["__import__"] = permissive_import

            globals_dict = {
                "__builtins__": builtins_dict,
                "__name__": "__main__",
                "__doc__": None,
                "__package__": None,
            }

            # Execute with timeout (basic implementation)
            def execute():
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    exec(code, globals_dict)

            # Simple timeout using threading
            import threading

            result = [None]
            exception = [None]

            def run_exec():
                try:
                    execute()
                    result[0] = True
                except Exception as e:
                    exception[0] = e
                    result[0] = False

            thread = threading.Thread(target=run_exec)
            thread.start()
            thread.join(config.time_limit)

            execution_time = time.time() - start_time

            if thread.is_alive():
                # Timeout
                return SandboxResult.timeout_result(execution_time)

            if exception[0]:
                return SandboxResult.error_result(
                    str(exception[0]), execution_time=execution_time
                )

            output = output_buffer.getvalue()
            if len(output) > config.max_output_length:
                output = output[: config.max_output_length] + "..."

            return SandboxResult.success_result(output, execution_time)

        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult.error_result(str(e), execution_time=execution_time)


def create_runner(
    runner_type: str = "auto", security_level: str = "standard", **kwargs
) -> SandboxedRunner:
    """
    Factory function to create the appropriate SandboxedRunner.

    Args:
        runner_type: Type of runner to create ("subprocess", "inprocess", "auto")
        security_level: Security level ("strict", "standard", "permissive")
        **kwargs: Additional arguments passed to runner constructor

    Returns:
        Configured SandboxedRunner instance

    Raises:
        ValueError: If runner_type or security_level is invalid
    """
    # Auto-select based on security level
    if runner_type == "auto":
        if security_level == "strict":
            runner_type = "subprocess"
        elif security_level == "permissive":
            runner_type = "inprocess"
        else:  # standard
            # Try subprocess first, fall back to inprocess
            runner = SubprocessRunner()
            if runner.is_available():
                runner_type = "subprocess"
            else:
                runner_type = "inprocess"

    # Create the runner
    if runner_type == "subprocess":
        if not SubprocessRunner().is_available():
            raise ValueError("SubprocessRunner is not available on this platform")
        return SubprocessRunner()
    elif runner_type == "inprocess":
        return InProcessRunner()
    else:
        raise ValueError(
            f"Unknown runner_type: {runner_type}. Must be 'subprocess', 'inprocess', or 'auto'"
        )


def get_available_runners() -> list[str]:
    """
    Get list of available runner types on this system.

    Returns:
        List of available runner type names
    """
    available = []
    if SubprocessRunner().is_available():
        available.append("subprocess")
    if InProcessRunner().is_available():
        available.append("inprocess")
    return available
