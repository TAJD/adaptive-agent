"""Tests for sandbox types."""

from pathlib import Path

from src.sandbox.types import SandboxConfig, SandboxResult


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self) -> None:
        """Test default SandboxConfig values."""
        config = SandboxConfig()

        assert config.time_limit == 30.0
        assert config.memory_limit == 100 * 1024 * 1024  # 100MB
        assert "math" in config.allowed_modules
        assert "print" in config.allowed_builtins
        assert config.max_output_length == 10 * 1024  # 10KB
        assert config.working_directory is None
        assert config.environment_variables == {}
        assert not config.enable_network
        assert not config.enable_filesystem
        assert config.allowed_paths == []

    def test_custom_config(self) -> None:
        """Test custom SandboxConfig."""
        config = SandboxConfig(
            time_limit=60.0,
            memory_limit=50 * 1024 * 1024,
            allowed_modules=["math", "numpy"],
            allowed_builtins=["abs", "len"],
            max_output_length=5 * 1024,
            working_directory="/tmp",
            environment_variables={"HOME": "/home/user"},
            enable_network=True,
            enable_filesystem=True,
            allowed_paths=["/tmp"],
        )

        assert config.time_limit == 60.0
        assert config.memory_limit == 50 * 1024 * 1024
        assert config.allowed_modules == ["math", "numpy"]
        assert config.allowed_builtins == ["abs", "len"]
        assert config.max_output_length == 5 * 1024
        assert config.working_directory == "/tmp"
        assert config.environment_variables == {"HOME": "/home/user"}
        assert config.enable_network
        assert config.enable_filesystem
        assert config.allowed_paths == ["/tmp"]

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization of SandboxConfig."""
        config = SandboxConfig(
            time_limit=45.0,
            allowed_modules=["math", "random"],
            environment_variables={"TEST": "value"},
            allowed_paths=[Path("/tmp")],
        )

        data = config.to_dict()
        restored = SandboxConfig.from_dict(data)

        assert restored.time_limit == 45.0
        assert restored.allowed_modules == ["math", "random"]
        assert restored.environment_variables == {"TEST": "value"}
        assert restored.allowed_paths == [str(Path("/tmp"))]  # Path converted to str


class TestSandboxResult:
    """Tests for SandboxResult."""

    def test_success_result_creation(self) -> None:
        """Test creating a successful result."""
        result = SandboxResult(success=True, output="Hello World")

        assert result.success
        assert not result.failed
        assert result.output == "Hello World"
        assert result.error is None
        assert result.exit_code == 0
        assert not result.timed_out
        assert not result.killed
        assert not result.truncated_output

    def test_error_result_creation(self) -> None:
        """Test creating an error result."""
        result = SandboxResult(
            success=False, output="", error="SyntaxError: invalid syntax", exit_code=1
        )

        assert not result.success
        assert result.failed
        assert result.has_error
        assert result.error == "SyntaxError: invalid syntax"
        assert result.exit_code == 1

    def test_timeout_result_creation(self) -> None:
        """Test creating a timeout result."""
        result = SandboxResult(
            success=False,
            output="",
            error="Execution timed out",
            timed_out=True,
            execution_time=30.0,
        )

        assert not result.success
        assert result.failed
        assert result.timed_out
        assert result.execution_time == 30.0

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization of SandboxResult."""
        result = SandboxResult(
            success=False,
            output="Some output",
            error="Error occurred",
            exit_code=2,
            execution_time=5.5,
            memory_used=1024,
            timed_out=True,
            killed=False,
            truncated_output=True,
        )

        data = result.to_dict()
        restored = SandboxResult.from_dict(data)

        assert restored.success == result.success
        assert restored.output == result.output
        assert restored.error == result.error
        assert restored.exit_code == result.exit_code
        assert restored.execution_time == result.execution_time
        assert restored.memory_used == result.memory_used
        assert restored.timed_out == result.timed_out
        assert restored.killed == result.killed
        assert restored.truncated_output == result.truncated_output

    def test_class_method_constructors(self) -> None:
        """Test class method constructors."""
        # Success result
        success = SandboxResult.success_result("Output", 1.5, 2048)
        assert success.success
        assert success.output == "Output"
        assert success.execution_time == 1.5
        assert success.memory_used == 2048

        # Error result
        error = SandboxResult.error_result("Error msg", 1, 2.0)
        assert not error.success
        assert error.error == "Error msg"
        assert error.exit_code == 1
        assert error.execution_time == 2.0

        # Timeout result
        timeout = SandboxResult.timeout_result(10.0)
        assert not timeout.success
        assert timeout.timed_out
        assert timeout.execution_time == 10.0
        assert timeout.error == "Execution timed out"

    def test_property_methods(self) -> None:
        """Test property methods."""
        success_result = SandboxResult(success=True, output="ok")
        assert not success_result.failed
        assert not success_result.has_error

        error_result = SandboxResult(success=False, output="", error="error")
        assert error_result.failed
        assert error_result.has_error

        no_error_result = SandboxResult(success=False, output="")
        assert no_error_result.failed
        assert not no_error_result.has_error
