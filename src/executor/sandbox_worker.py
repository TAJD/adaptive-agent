#!/usr/bin/env python3
"""
Standalone sandbox worker script for subprocess-based code execution.

This script is designed to be run in a subprocess with restricted permissions.
It reads execution parameters from stdin as JSON, executes the code in a
sandboxed environment, and writes results to stdout as JSON.

Input JSON format:
{
    "code": "python_code_to_execute",
    "globals_serialized": {"key": "serialized_value", ...},  # optional
    "allowed_modules": ["module1", "module2", ...],
    "allowed_builtins": ["builtin1", "builtin2", ...],  # optional
    "timeout": 30.0,  # seconds
    "working_directory": "/path/to/cwd",  # optional
    "max_output_length": 10000  # optional
}

Output JSON format:
{
    "success": true,
    "result": "serialized_result",  # if code produces a result
    "stdout": "captured_stdout",
    "stderr": "captured_stderr",
    "error": "error_message",  # if success is false
    "execution_time": 1.23,
    "timed_out": false,
    "resource_usage": {
        "cpu_time": 1.0,
        "memory_peak": 1024000  # bytes
    }
}
"""

import json
import sys
import signal
import time
import io
import os
import builtins
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional


def serialize_result(obj: Any) -> str:
    """Serialize execution result to JSON-compatible format."""
    try:
        # Handle pandas DataFrames specially
        if hasattr(obj, "to_json") and hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            if class_name == "DataFrame":
                return obj.to_json(orient="records")
            elif class_name == "Series":
                return obj.to_json()

        # Handle numpy arrays
        if hasattr(obj, "tolist") and hasattr(obj, "__array__"):
            return json.dumps(obj.tolist())

        # Standard JSON serialization
        return json.dumps(obj, default=str)
    except Exception as e:
        return f"<unserializable: {type(obj).__name__}: {e}>"


def deserialize_globals(globals_serialized: Dict[str, Any]) -> Dict[str, Any]:
    """Deserialize globals from JSON-compatible format."""
    result = {}

    for key, value in globals_serialized.items():
        try:
            # Try to deserialize pandas DataFrames
            if isinstance(value, str) and value.startswith("[") and "}" in value:
                # Likely a JSON string representation of DataFrame
                try:
                    import pandas as pd

                    result[key] = pd.read_json(io.StringIO(value), orient="records")
                    continue
                except Exception:
                    pass

            # Try to deserialize as JSON
            if isinstance(value, str):
                try:
                    result[key] = json.loads(value)
                    continue
                except Exception:
                    pass

            # Keep as-is
            result[key] = value

        except Exception:
            result[key] = value

    return result


def create_restricted_environment(
    allowed_modules: List[str], allowed_builtins: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create a restricted execution environment."""

    # Default allowed builtins if not specified
    if allowed_builtins is None:
        allowed_builtins = [
            "__import__",
            "print",
            "len",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "tuple",
            "set",
            "range",
            "enumerate",
            "zip",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "sorted",
            "reversed",
            "Exception",
            "ValueError",
            "TypeError",
            "KeyError",
            "IndexError",
            "AttributeError",
            "NameError",
            "ImportError",
        ]

    # Create restricted builtins
    restricted_builtins = {}
    for builtin_name in allowed_builtins:
        if hasattr(builtins, builtin_name):
            restricted_builtins[builtin_name] = getattr(builtins, builtin_name)

    # Dangerous modules to block
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
        "importlib",
        "inspect",
        "pickle",
        "shelve",
        "dbm",
        "sqlite3",
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "tarfile",
        "csv",
        "configparser",
        "netrc",
        "xdrlib",
        "plistlib",
        "hashlib",
        "hmac",
        "secrets",
        "ssl",
        "socketserver",
        "http.server",
        "xmlrpc",
        "xmlrpc.server",
        "webbrowser",
        "cgi",
        "cgitb",
        "wsgiref",
        "xdrlib",
        "plistlib",
    }

    # Filter out dangerous modules from allowed_modules
    safe_allowed_modules = []
    for module in allowed_modules:
        if module not in dangerous_modules:
            safe_allowed_modules.append(module)

    # Create restricted import function
    original_import = builtins.__import__

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in safe_allowed_modules:
            raise ImportError(f"Import of '{name}' is not allowed")
        return original_import(name, globals, locals, fromlist, level)

    restricted_builtins["__import__"] = restricted_import

    return restricted_builtins


def execute_code(
    code: str,
    globals_dict: Dict[str, Any],
    timeout: float,
    max_output_length: int = 10000,
) -> Dict[str, Any]:
    """Execute code with basic monitoring (timeout enforced by parent process)."""

    start_time = time.time()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        "success": False,
        "result": None,
        "stdout": "",
        "stderr": "",
        "error": None,
        "execution_time": 0.0,
        "timed_out": False,  # This will be set by parent if subprocess times out
        "resource_usage": {"cpu_time": 0.0, "memory_peak": 0},
    }

    try:
        # Execute code with captured output
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals_dict)

        execution_time = time.time() - start_time

        # Get captured output
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()

        # Truncate output if too long
        if len(stdout) > max_output_length:
            stdout = stdout[:max_output_length] + "..."
        if len(stderr) > max_output_length:
            stderr = stderr[:max_output_length] + "..."

        # Try to get the result (last expression if any)
        result_value = None
        if "_result" in globals_dict:
            result_value = globals_dict["_result"]

        result.update(
            {
                "success": True,
                "result": serialize_result(result_value)
                if result_value is not None
                else None,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time,
            }
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e}"
        if len(error_msg) > max_output_length:
            error_msg = error_msg[:max_output_length] + "..."

        result.update(
            {
                "error": error_msg,
                "execution_time": execution_time,
            }
        )

    return result


def main():
    """Main entry point for the sandbox worker."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)

        # Extract parameters
        code = input_data.get("code", "")
        globals_serialized = input_data.get("globals_serialized", {})
        allowed_modules = input_data.get("allowed_modules", [])
        allowed_builtins = input_data.get("allowed_builtins")
        timeout = input_data.get("timeout", 30.0)
        working_directory = input_data.get("working_directory")
        max_output_length = input_data.get("max_output_length", 10000)

        # Change working directory if specified
        if working_directory and os.path.isdir(working_directory):
            os.chdir(working_directory)

        # Deserialize globals
        globals_dict = deserialize_globals(globals_serialized)

        # Create restricted environment
        restricted_builtins = create_restricted_environment(
            allowed_modules, allowed_builtins
        )

        # Set up globals
        globals_dict["__builtins__"] = restricted_builtins
        globals_dict.setdefault("__name__", "__main__")
        globals_dict.setdefault("__doc__", None)
        globals_dict.setdefault("__package__", None)

        # Execute code
        result = execute_code(code, globals_dict, timeout, max_output_length)

        # Output result as JSON
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")

    except Exception as e:
        # Handle any uncaught exceptions
        error_result = {
            "success": False,
            "error": f"Worker error: {e}",
            "execution_time": 0.0,
            "timed_out": False,
            "resource_usage": {"cpu_time": 0.0, "memory_peak": 0},
        }
        json.dump(error_result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
