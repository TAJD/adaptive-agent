"""Tests for sandbox worker script."""

import json
import subprocess
import sys
import pytest
from pathlib import Path

from src.executor.sandbox_worker import serialize_result, deserialize_globals


class TestSandboxWorker:
    """Tests for the standalone sandbox worker script."""

    def test_serialize_result_basic_types(self) -> None:
        """Test serialization of basic Python types."""
        assert serialize_result(42) == "42"
        assert serialize_result("hello") == '"hello"'
        assert serialize_result([1, 2, 3]) == "[1, 2, 3]"
        assert serialize_result({"key": "value"}) == '{"key": "value"}'

    def test_serialize_result_pandas_dataframe(self) -> None:
        """Test serialization of pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        result = serialize_result(df)
        # Should be JSON representation
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["A"] == 1
        assert parsed[0]["B"] == "x"

    def test_serialize_result_numpy_array(self) -> None:
        """Test serialization of numpy array."""
        pytest.importorskip("numpy")
        import numpy as np

        arr = np.array([1, 2, 3])
        result = serialize_result(arr)
        assert result == "[1, 2, 3]"

    def test_serialize_result_unserializable(self) -> None:
        """Test serialization of unserializable objects."""

        class Unserializable:
            pass

        obj = Unserializable()
        result = serialize_result(obj)
        assert "unserializable" in result.lower()

    def test_deserialize_globals_basic(self) -> None:
        """Test deserialization of basic globals."""
        serialized = {"x": 42, "y": "hello"}
        result = deserialize_globals(serialized)
        assert result["x"] == 42
        assert result["y"] == "hello"

    def test_deserialize_globals_pandas(self) -> None:
        """Test deserialization of pandas DataFrames."""
        pytest.importorskip("pandas")
        import pandas as pd

        # Create test DataFrame
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        json_str = df.to_json(orient="records")

        serialized = {"df": json_str}
        result = deserialize_globals(serialized)

        assert isinstance(result["df"], pd.DataFrame)
        assert result["df"].equals(df)

    def test_worker_basic_execution(self) -> None:
        """Test basic code execution via worker script."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "result = 2 + 3\nprint(f'Result: {result}')",
            "allowed_modules": ["math"],
            "timeout": 5.0,
        }

        # Run worker script
        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert process.returncode == 0
        result = json.loads(process.stdout)

        assert result["success"] is True
        assert "Result: 5" in result["stdout"]
        assert result["execution_time"] >= 0
        assert result["timed_out"] is False

    def test_worker_with_math_module(self) -> None:
        """Test execution with math module."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "import math\nresult = math.sqrt(16)\nprint(result)",
            "allowed_modules": ["math"],
            "timeout": 5.0,
        }

        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert process.returncode == 0
        result = json.loads(process.stdout)

        assert result["success"] is True
        assert "4.0" in result["stdout"]

    def test_worker_restricted_module(self) -> None:
        """Test that restricted modules are blocked."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "import os\nprint('Should not reach here')",
            "allowed_modules": ["math"],  # os not allowed
            "timeout": 5.0,
        }

        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=15,  # Give more time
        )

        print(f"Return code: {process.returncode}")
        print(f"Stdout: {repr(process.stdout)}")
        print(f"Stderr: {repr(process.stderr)}")

        if process.returncode != 0:
            # Worker crashed
            assert False, (
                f"Worker crashed with return code {process.returncode}, stderr: {process.stderr}"
            )

        if not process.stdout.strip():
            # No output
            assert False, f"No output from worker, stderr: {process.stderr}"

        result = json.loads(process.stdout)

        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_worker_timeout(self) -> None:
        """Test timeout handling (enforced by parent subprocess timeout)."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "import time\ntime.sleep(2)",  # Code that takes longer than subprocess timeout
            "allowed_modules": ["time"],
            "timeout": 5.0,  # Worker timeout (not enforced by worker itself)
        }

        # Use subprocess timeout shorter than code execution time
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [sys.executable, str(worker_path)],
                input=json.dumps(input_data),
                text=True,
                capture_output=True,
                timeout=1.0,  # Kill subprocess after 1 second
            )

    def test_worker_error_handling(self) -> None:
        """Test error handling in code execution."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "raise ValueError('test error')",
            "allowed_modules": [],
            "timeout": 5.0,
        }

        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert process.returncode == 0
        result = json.loads(process.stdout)

        assert result["success"] is False
        assert "ValueError" in result["error"]
        assert "test error" in result["error"]

    def test_worker_with_globals(self) -> None:
        """Test execution with provided globals."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "result = x + y\nprint(f'Sum: {result}')",
            "globals_serialized": {"x": 10, "y": 20},
            "allowed_modules": [],
            "timeout": 5.0,
        }

        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert process.returncode == 0
        result = json.loads(process.stdout)

        assert result["success"] is True
        assert "Sum: 30" in result["stdout"]

    def test_worker_output_truncation(self) -> None:
        """Test output truncation."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        input_data = {
            "code": "print('x' * 100)",
            "allowed_modules": [],
            "timeout": 5.0,
            "max_output_length": 20,
        }

        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input=json.dumps(input_data),
            text=True,
            capture_output=True,
            timeout=10,
        )

        assert process.returncode == 0
        result = json.loads(process.stdout)

        assert result["success"] is True
        assert len(result["stdout"]) <= 23  # 20 + "..."
        assert result["stdout"].endswith("...")

    def test_worker_invalid_input(self) -> None:
        """Test handling of invalid input."""
        worker_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "executor"
            / "sandbox_worker.py"
        )

        # Send invalid JSON
        process = subprocess.run(
            [sys.executable, str(worker_path)],
            input="invalid json",
            text=True,
            capture_output=True,
            timeout=10,
        )

        # Worker should handle the error gracefully
        assert process.returncode == 1
        result = json.loads(process.stdout)
        assert result["success"] is False
        assert "Worker error" in result["error"]
