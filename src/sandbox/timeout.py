"""Signal-based timeout handling for Unix subprocesses."""

import os
import signal
import subprocess
import time
from typing import Any


class SubprocessTimeoutError(Exception):
    """Exception raised when a subprocess times out."""

    pass


def run_with_timeout(
    args: list[str], timeout: float, **kwargs: Any
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with a timeout using signals (Unix only).

    Args:
        args: Command and arguments to run
        timeout: Timeout in seconds
        **kwargs: Additional arguments for subprocess.run

    Returns:
        CompletedProcess instance

    Raises:
        SubprocessTimeoutError: If the process times out
        NotImplementedError: If not on Unix
    """
    if os.name != "posix":
        raise NotImplementedError(
            "Signal-based timeout is only supported on Unix systems"
        )

    # Start the process
    process = subprocess.Popen(args, **kwargs)

    def timeout_handler(signum: int, frame: Any) -> None:
        """Handle timeout by terminating the process."""
        try:
            # Send SIGTERM first
            process.terminate()
            # Wait a bit for graceful shutdown
            time.sleep(0.1)
            if process.poll() is None:
                # If still running, force kill
                process.kill()
        except ProcessLookupError:
            # Process already exited
            pass

    # Set up the timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        # Wait for the process to complete
        return_code = process.wait()
        # Cancel the alarm
        signal.alarm(0)

        # Create a CompletedProcess-like object
        return subprocess.CompletedProcess(
            args=args,
            returncode=return_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    except KeyboardInterrupt:
        # Timeout occurred
        signal.alarm(0)  # Cancel alarm
        raise SubprocessTimeoutError(f"Process timed out after {timeout} seconds")
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGALRM, old_handler)


def run_with_timeout_graceful(
    args: list[str], timeout: float, grace_period: float = 1.0, **kwargs: Any
) -> subprocess.CompletedProcess:
    """
    Run a subprocess with graceful timeout handling.

    First sends SIGTERM, waits for grace_period, then sends SIGKILL if needed.

    Args:
        args: Command and arguments to run
        timeout: Total timeout in seconds
        grace_period: Time to wait after SIGTERM before SIGKILL
        **kwargs: Additional arguments for subprocess.run

    Returns:
        CompletedProcess instance

    Raises:
        SubprocessTimeoutError: If the process times out
        NotImplementedError: If not on Unix
    """
    if os.name != "posix":
        raise NotImplementedError(
            "Signal-based timeout is only supported on Unix systems"
        )

    process = subprocess.Popen(args, **kwargs)
    start_time = time.time()

    def terminate_process() -> None:
        """Terminate the process gracefully."""
        try:
            process.terminate()
            # Wait for graceful shutdown
            end_grace = time.time() + grace_period
            while time.time() < end_grace and process.poll() is None:
                time.sleep(0.01)
            # If still running, force kill
            if process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    def timeout_handler(signum: int, frame: Any) -> None:
        """Handle timeout."""
        terminate_process()

    # Set up the timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        return_code = process.wait()
        signal.alarm(0)  # Cancel alarm

        return subprocess.CompletedProcess(
            args=args,
            returncode=return_code,
            stdout=process.stdout,
            stderr=process.stderr,
        )

    except KeyboardInterrupt:
        signal.alarm(0)
        raise SubprocessTimeoutError(
            f"Process timed out after {time.time() - start_time:.2f} seconds"
        )
    finally:
        signal.signal(signal.SIGALRM, old_handler)


class TimeoutManager:
    """Context manager for subprocess timeout handling."""

    def __init__(self, timeout: float, grace_period: float = 1.0):
        """
        Initialize timeout manager.

        Args:
            timeout: Timeout in seconds
            grace_period: Grace period after SIGTERM before SIGKILL
        """
        if os.name != "posix":
            raise NotImplementedError(
                "TimeoutManager is only supported on Unix systems"
            )

        self.timeout = timeout
        self.grace_period = grace_period
        self.process: subprocess.Popen | None = None
        self.timed_out = False

    def __enter__(self) -> "TimeoutManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.process and self.timed_out:
            self._terminate_process()

    def run(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """
        Run a subprocess with timeout management.

        Args:
            args: Command and arguments
            **kwargs: subprocess.Popen arguments

        Returns:
            CompletedProcess instance

        Raises:
            SubprocessTimeoutError: If timeout occurs
        """
        self.process = subprocess.Popen(args, **kwargs)
        start_time = time.time()

        def alarm_handler(signum: int, frame: Any) -> None:
            self.timed_out = True

        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(int(self.timeout))

        try:
            return_code = self.process.wait()
            signal.alarm(0)

            return subprocess.CompletedProcess(
                args=args,
                returncode=return_code,
                stdout=self.process.stdout,
                stderr=self.process.stderr,
            )

        except KeyboardInterrupt:
            signal.alarm(0)
            self._terminate_process()
            raise SubprocessTimeoutError(
                f"Process timed out after {time.time() - start_time:.2f} seconds"
            )
        finally:
            signal.signal(signal.SIGALRM, old_handler)

    def _terminate_process(self) -> None:
        """Terminate the managed process."""
        if not self.process:
            return

        try:
            self.process.terminate()
            # Wait for graceful shutdown
            end_grace = time.time() + self.grace_period
            while time.time() < end_grace and self.process.poll() is None:
                time.sleep(0.01)
            # Force kill if needed
            if self.process.poll() is None:
                self.process.kill()
        except ProcessLookupError:
            pass
