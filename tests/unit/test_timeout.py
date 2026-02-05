"""Tests for timeout functionality."""

import os
import pytest

from src.sandbox.timeout import (
    SubprocessTimeoutError,
    run_with_timeout,
    run_with_timeout_graceful,
    TimeoutManager,
)


class TestTimeoutUnix:
    """Tests for Unix signal-based timeout (skipped on Windows)."""

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific signal handling")
    def test_run_with_timeout_success(self) -> None:
        """Test successful completion within timeout."""
        import subprocess

        result = run_with_timeout(
            ["echo", "hello"], timeout=5.0, capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific signal handling")
    def test_run_with_timeout_times_out(self) -> None:
        """Test timeout raises exception."""
        with pytest.raises(SubprocessTimeoutError):
            run_with_timeout(["sleep", "10"], timeout=0.1)

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific signal handling")
    def test_run_with_timeout_graceful(self) -> None:
        """Test graceful timeout handling."""
        result = run_with_timeout_graceful(
            ["echo", "test"], timeout=5.0, capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "test" in result.stdout

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific signal handling")
    def test_timeout_manager_context(self) -> None:
        """Test TimeoutManager context manager."""
        with TimeoutManager(timeout=5.0) as manager:
            result = manager.run(["echo", "managed"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "managed" in result.stdout

    @pytest.mark.skipif(os.name != "posix", reason="Unix-specific signal handling")
    def test_timeout_manager_timeout(self) -> None:
        """Test TimeoutManager raises exception on timeout."""
        with pytest.raises(SubprocessTimeoutError):
            with TimeoutManager(timeout=0.1) as manager:
                manager.run(["sleep", "1"])


class TestTimeoutWindows:
    """Tests for timeout on Windows (should raise NotImplementedError)."""

    @pytest.mark.skipif(os.name == "posix", reason="Windows-specific behavior")
    def test_run_with_timeout_not_implemented(self) -> None:
        """Test that run_with_timeout raises NotImplementedError on Windows."""
        with pytest.raises(NotImplementedError, match="only supported on Unix"):
            run_with_timeout(["echo", "test"], timeout=1.0)

    @pytest.mark.skipif(os.name == "posix", reason="Windows-specific behavior")
    def test_run_with_timeout_graceful_not_implemented(self) -> None:
        """Test that run_with_timeout_graceful raises NotImplementedError on Windows."""
        with pytest.raises(NotImplementedError, match="only supported on Unix"):
            run_with_timeout_graceful(["echo", "test"], timeout=1.0)

    @pytest.mark.skipif(os.name == "posix", reason="Windows-specific behavior")
    def test_timeout_manager_not_implemented(self) -> None:
        """Test that TimeoutManager raises NotImplementedError on Windows."""
        with pytest.raises(NotImplementedError, match="only supported on Unix"):
            TimeoutManager(timeout=1.0)
