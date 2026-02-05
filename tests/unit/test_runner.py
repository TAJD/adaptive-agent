"""Tests for sandbox runner."""

import time
import pytest

from src.sandbox.types import SandboxConfig, SandboxResult
from src.sandbox.runner import (
    SandboxedRunner,
    SubprocessRunner,
    InProcessRunner,
    create_runner,
    get_available_runners,
)


class TestSubprocessRunner:
    """Tests for SubprocessRunner."""

    def test_is_available(self) -> None:
        """Test runner availability."""
        runner = SubprocessRunner()
        assert runner.is_available()

    def test_run_simple_code(self) -> None:
        """Test running simple code."""
        runner = SubprocessRunner()
        config = SandboxConfig()

        result = runner.run_code("print('Hello')", config)

        assert result.success
        assert "Hello" in result.output
        assert result.execution_time >= 0

    def test_run_code_with_result(self) -> None:
        """Test code that produces a result."""
        runner = SubprocessRunner()
        config = SandboxConfig()

        code = """
result = 2 + 3
print(f"Result: {result}")
"""
        result = runner.run_code(code, config)

        assert result.success
        assert "Result: 5" in result.output

    def test_run_code_with_error(self) -> None:
        """Test code that raises an error."""
        runner = SubprocessRunner()
        config = SandboxConfig()

        result = runner.run_code("raise ValueError('test error')", config)

        assert not result.success
        assert result.error is not None
        assert "test error" in result.error

    def test_run_code_with_timeout(self) -> None:
        """Test code that times out."""
        runner = SubprocessRunner()
        config = SandboxConfig(time_limit=0.1)

        code = """
import time
time.sleep(1)
"""
        result = runner.run_code(code, config)

        assert not result.success
        assert result.timed_out
        assert result.execution_time >= 0.1

    def test_restricted_modules(self) -> None:
        """Test that restricted modules are not available."""
        runner = SubprocessRunner()
        config = SandboxConfig(allowed_modules=["math"])  # No 'os'

        result = runner.run_code("import os; print('imported')", config)

        assert not result.success
        assert result.error is not None
        assert "not allowed" in result.error

    def test_allowed_modules(self) -> None:
        """Test that allowed modules work."""
        runner = SubprocessRunner()
        config = SandboxConfig(allowed_modules=["math"])

        result = runner.run_code("import math; print(math.sqrt(4))", config)

        assert result.success
        assert "2.0" in result.output

    def test_output_truncation(self) -> None:
        """Test output truncation."""
        runner = SubprocessRunner()
        config = SandboxConfig(max_output_length=10)

        code = "print('very long output that should be truncated')"
        result = runner.run_code(code, config)

        assert result.success
        assert len(result.output) <= 10 + 3  # +3 for "..."
        assert result.output.endswith("...")

    def test_execution_time_tracking(self) -> None:
        """Test that execution time is tracked."""
        runner = SubprocessRunner()
        config = SandboxConfig()

        code = """
import time
time.sleep(0.01)
"""
        result = runner.run_code(code, config)

        assert result.success
        assert result.execution_time >= 0.01

    def test_subprocess_isolation(self) -> None:
        """Test that subprocess provides isolation."""
        runner = SubprocessRunner()
        config = SandboxConfig(allowed_modules=["math"])  # No 'os'

        # Try to access restricted module
        code = """
import os
print("Should not reach here")
"""
        result = runner.run_code(code, config)

        assert not result.success
        assert result.error is not None
        assert "not allowed" in result.error

    def test_subprocess_timeout(self) -> None:
        """Test subprocess timeout handling."""
        runner = SubprocessRunner()
        config = SandboxConfig(time_limit=0.1)

        code = """
import time
time.sleep(1)  # Sleep longer than timeout
"""
        result = runner.run_code(code, config)

        assert not result.success
        assert result.timed_out

    def test_subprocess_output_limits(self) -> None:
        """Test output length limits in subprocess."""
        runner = SubprocessRunner()
        config = SandboxConfig(max_output_length=20)

        code = """
print("This is a very long output that should be truncated")
"""
        result = runner.run_code(code, config)

        assert result.success
        assert len(result.output) <= 23  # 20 + "..."
        assert result.output.endswith("...")


class TestInProcessRunner:
    """Tests for InProcessRunner."""

    def test_is_available(self) -> None:
        """Test runner availability."""
        runner = InProcessRunner()
        assert runner.is_available()

    def test_run_simple_code(self) -> None:
        """Test running simple code."""
        runner = InProcessRunner()
        config = SandboxConfig()

        result = runner.run_code("print('Hello')", config)

        assert result.success
        assert "Hello" in result.output
        assert result.execution_time >= 0

    def test_permissive_mode_allows_math(self) -> None:
        """Test that permissive mode allows math module."""
        runner = InProcessRunner()
        config = SandboxConfig()

        code = """
import math
result = math.sqrt(16)
print(result)
"""
        result = runner.run_code(code, config)

        assert result.success
        assert "4.0" in result.output

    def test_permissive_mode_blocks_dangerous_modules(self) -> None:
        """Test that permissive mode blocks dangerous modules like os."""
        runner = InProcessRunner()
        config = SandboxConfig()

        code = """
import os
print("Should not reach here")
"""
        result = runner.run_code(code, config)

        assert not result.success
        assert result.error is not None
        assert "not allowed in permissive mode" in result.error

    def test_permissive_mode_allows_safe_modules(self) -> None:
        """Test that permissive mode allows safe modules."""
        runner = InProcessRunner()
        config = SandboxConfig()

        code = """
import json
data = {"key": "value"}
result = json.dumps(data)
print(result)
"""
        result = runner.run_code(code, config)

        assert result.success
        assert '"key": "value"' in result.output

    def test_timeout_still_works(self) -> None:
        """Test that timeout still works in permissive mode."""
        runner = InProcessRunner()
        config = SandboxConfig(time_limit=0.1)

        code = """
import time
time.sleep(1)
"""
        result = runner.run_code(code, config)

        assert not result.success
        assert result.timed_out


class TestCreateRunner:
    """Tests for create_runner factory function."""

    def test_create_subprocess_runner(self) -> None:
        """Test creating subprocess runner explicitly."""
        runner = create_runner("subprocess")

        assert isinstance(runner, SubprocessRunner)
        assert runner.is_available()

    def test_create_inprocess_runner(self) -> None:
        """Test creating inprocess runner explicitly."""
        runner = create_runner("inprocess")

        assert isinstance(runner, InProcessRunner)
        assert runner.is_available()

    def test_create_auto_runner_strict_security(self) -> None:
        """Test auto-selection with strict security level."""
        runner = create_runner("auto", "strict")

        assert isinstance(runner, SubprocessRunner)

    def test_create_auto_runner_permissive_security(self) -> None:
        """Test auto-selection with permissive security level."""
        runner = create_runner("auto", "permissive")

        assert isinstance(runner, InProcessRunner)

    def test_create_auto_runner_standard_security(self) -> None:
        """Test auto-selection with standard security level."""
        runner = create_runner("auto", "standard")

        # Should prefer subprocess if available, otherwise inprocess
        if SubprocessRunner().is_available():
            assert isinstance(runner, SubprocessRunner)
        else:
            assert isinstance(runner, InProcessRunner)

    def test_create_runner_with_invalid_type(self) -> None:
        """Test that invalid runner type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown runner_type"):
            create_runner("invalid")

    def test_create_subprocess_when_not_available(self) -> None:
        """Test that requesting unavailable subprocess raises error."""
        # Mock SubprocessRunner to be unavailable
        original_is_available = SubprocessRunner.is_available

        def mock_is_available(self):
            return False

        SubprocessRunner.is_available = mock_is_available

        try:
            with pytest.raises(ValueError, match="SubprocessRunner is not available"):
                create_runner("subprocess")
        finally:
            # Restore original method
            SubprocessRunner.is_available = original_is_available

    def test_create_runner_with_kwargs(self) -> None:
        """Test that kwargs are accepted (though not used in current impl)."""
        runner = create_runner("inprocess", some_kwarg="value")

        assert isinstance(runner, InProcessRunner)


class TestGetAvailableRunners:
    """Tests for get_available_runners function."""

    def test_get_available_runners(self) -> None:
        """Test getting list of available runners."""
        available = get_available_runners()

        assert isinstance(available, list)
        assert "inprocess" in available  # Always available

        # Subprocess may or may not be available depending on platform
        if SubprocessRunner().is_available():
            assert "subprocess" in available
        else:
            assert "subprocess" not in available

    def test_get_available_runners_comprehensive(self) -> None:
        """Test that available runners are actually available."""
        available = get_available_runners()

        for runner_type in available:
            runner = create_runner(runner_type)
            assert runner.is_available()

    def test_get_available_runners_on_windows(self) -> None:
        """Test available runners on Windows platform."""
        import sys

        if sys.platform == "win32":
            available = get_available_runners()
            # On Windows, subprocess should be available
            assert "subprocess" in available
            assert "inprocess" in available
