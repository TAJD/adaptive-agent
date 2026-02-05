"""Sandbox configuration and result types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SandboxConfig:
    """
    Configuration for code execution sandbox.

    Defines limits and permissions for safe code execution.
    """

    time_limit: float = 30.0  # seconds
    memory_limit: int = 100 * 1024 * 1024  # 100MB in bytes
    allowed_modules: list[str] = field(
        default_factory=lambda: ["math", "random", "datetime", "time"]
    )
    allowed_builtins: list[str] = field(
        default_factory=lambda: [
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "filter",
            "float",
            "int",
            "len",
            "list",
            "map",
            "max",
            "min",
            "print",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "type",
            "zip",
            "ValueError",
            "TypeError",
            "RuntimeError",
            "Exception",
        ]
    )
    max_output_length: int = 10 * 1024  # 10KB
    working_directory: str | Path | None = None
    environment_variables: dict[str, str] = field(default_factory=dict)
    enable_network: bool = False
    enable_filesystem: bool = False
    allowed_paths: list[str | Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "time_limit": self.time_limit,
            "memory_limit": self.memory_limit,
            "allowed_modules": self.allowed_modules.copy(),
            "allowed_builtins": self.allowed_builtins.copy(),
            "max_output_length": self.max_output_length,
            "working_directory": str(self.working_directory)
            if self.working_directory
            else None,
            "environment_variables": self.environment_variables.copy(),
            "enable_network": self.enable_network,
            "enable_filesystem": self.enable_filesystem,
            "allowed_paths": [str(p) for p in self.allowed_paths],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SandboxResult:
    """
    Result of sandboxed code execution.

    Contains execution outcome, output, and resource usage.
    """

    success: bool
    output: str
    error: str | None = None
    exit_code: int = 0
    execution_time: float = 0.0
    memory_used: int = 0
    timed_out: bool = False
    killed: bool = False
    truncated_output: bool = False

    @property
    def failed(self) -> bool:
        """Check if execution failed."""
        return not self.success

    @property
    def has_error(self) -> bool:
        """Check if there was an error."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "timed_out": self.timed_out,
            "killed": self.killed,
            "truncated_output": self.truncated_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxResult":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def success_result(
        cls, output: str, execution_time: float = 0.0, memory_used: int = 0
    ) -> "SandboxResult":
        """Create a successful result."""
        return cls(
            success=True,
            output=output,
            execution_time=execution_time,
            memory_used=memory_used,
        )

    @classmethod
    def error_result(
        cls, error: str, exit_code: int = 1, execution_time: float = 0.0
    ) -> "SandboxResult":
        """Create an error result."""
        return cls(
            success=False,
            output="",
            error=error,
            exit_code=exit_code,
            execution_time=execution_time,
        )

    @classmethod
    def timeout_result(cls, execution_time: float) -> "SandboxResult":
        """Create a timeout result."""
        return cls(
            success=False,
            output="",
            error="Execution timed out",
            timed_out=True,
            execution_time=execution_time,
        )
