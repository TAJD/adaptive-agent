# Sandbox Architecture Design

## Overview

This document defines the sandbox architecture for secure code execution in the agent framework. The current `CodeRunner` uses Python's `exec()` with a module whitelist, which provides minimal security. This design introduces a `SandboxedRunner` protocol with configurable security levels.

## Current State

The existing `CodeRunner` (`src/executor/code_runner.py`) has these characteristics:

- **Execution**: Uses `exec()` with separate global/local scope
- **Module control**: Whitelist of allowed modules (pandas, math, statistics, json)
- **Timeout**: Parameter exists but not enforced
- **Isolation**: None - shares process space with host

**Security gaps**:
- No process isolation
- No memory/CPU limits
- No filesystem restrictions
- `__import__()` can bypass module whitelist
- No timeout enforcement

## Security Levels

### Level 1: Permissive

**Use case**: Development, trusted code, interactive exploration

| Aspect | Restriction |
|--------|-------------|
| Modules | Configurable whitelist (default: pandas, math, statistics, json, numpy, datetime) |
| Filesystem | Read-only access to data directory |
| Network | Blocked |
| Timeout | 60 seconds |
| Memory | 1 GB |
| CPU | No limit |

**Implementation**: Enhanced `exec()` with import hooks and resource limits via `resource` module (Unix) or job objects (Windows).

### Level 2: Standard (Default)

**Use case**: Production workloads, semi-trusted code

| Aspect | Restriction |
|--------|-------------|
| Modules | Strict whitelist (pandas, math, statistics, json only) |
| Filesystem | No access (data passed via serialization) |
| Network | Blocked |
| Timeout | 30 seconds |
| Memory | 512 MB |
| CPU | Single core |

**Implementation**: Subprocess with restricted environment and resource limits.

### Level 3: Strict

**Use case**: Untrusted code, multi-tenant environments

| Aspect | Restriction |
|--------|-------------|
| Modules | Minimal whitelist (math, statistics, json) |
| Filesystem | No access |
| Network | Blocked |
| Timeout | 10 seconds |
| Memory | 256 MB |
| CPU | Single core, 50% limit |

**Implementation**: Docker container with seccomp profile and dropped capabilities.

## Implementation Approaches

### Option A: Subprocess Isolation

**How it works**: Spawn a child Python process, communicate via stdin/stdout with pickled data.

```
Parent Process                    Child Process
     |                                 |
     |-- pickle(code, data) --------->|
     |                                 |-- exec(code)
     |<-------- pickle(result) -------|
     |                                 |
```

**Pros**:
- No external dependencies
- Works on all platforms
- Easy to implement
- Good resource control via `resource` module (Unix)

**Cons**:
- Limited isolation (same user, shared kernel)
- Windows resource limits require different approach
- Pickle security concerns (mitigated by controlling both ends)

**Resource control**:
```python
# Unix
import resource
resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time, max_cpu_time))

# Windows
import subprocess
CREATE_SUSPENDED = 0x00000004
# Use job objects via ctypes or pywin32
```

### Option B: Docker Containers

**How it works**: Run code in ephemeral Docker containers with strict resource limits.

```
Host                              Container
  |                                   |
  |-- docker run (volume mount) ---->|
  |                                   |-- python exec
  |<-------- stdout/file ------------|
  |                                   |
  |-- docker rm -f ----------------->|
```

**Pros**:
- Strong isolation (namespaces, cgroups)
- Consistent resource limits across platforms
- Network isolation built-in
- Can use seccomp for syscall filtering

**Cons**:
- Requires Docker installation
- Higher startup latency (~200-500ms)
- Image management overhead
- Not available in all environments

**Configuration**:
```yaml
# Container limits
memory: 256m
cpus: 0.5
network: none
read_only: true
security_opt:
  - no-new-privileges:true
  - seccomp:sandbox-profile.json
cap_drop:
  - ALL
```

### Option C: RestrictedPython

**How it works**: Compile Python code with restricted AST transformations that prevent dangerous operations.

**Pros**:
- No process overhead
- Fine-grained control over allowed operations
- Well-tested in Zope/Plone ecosystem

**Cons**:
- Complex to configure correctly
- May break legitimate pandas operations
- Limited to Python syntax restrictions
- Doesn't prevent resource exhaustion
- Maintenance burden

**Assessment**: Not recommended for this project due to pandas compatibility issues and incomplete protection.

## Recommendation

**Primary**: Subprocess isolation (Option A) for Standard level
**Secondary**: Docker containers (Option B) for Strict level

Rationale:
1. Subprocess provides adequate isolation for the data analysis use case
2. Docker adds complexity but enables multi-tenant deployments
3. RestrictedPython is incompatible with pandas workflows
4. Hybrid approach allows deployment flexibility

## SandboxedRunner Protocol

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SecurityLevel(Enum):
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandbox execution."""

    security_level: SecurityLevel = SecurityLevel.STANDARD
    timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    allowed_modules: tuple[str, ...] = ("pandas", "math", "statistics", "json")

    @classmethod
    def for_level(cls, level: SecurityLevel) -> "SandboxConfig":
        """Factory method for standard configurations."""
        configs = {
            SecurityLevel.PERMISSIVE: cls(
                security_level=level,
                timeout_seconds=60.0,
                max_memory_mb=1024,
                allowed_modules=("pandas", "math", "statistics", "json", "numpy", "datetime"),
            ),
            SecurityLevel.STANDARD: cls(
                security_level=level,
                timeout_seconds=30.0,
                max_memory_mb=512,
                allowed_modules=("pandas", "math", "statistics", "json"),
            ),
            SecurityLevel.STRICT: cls(
                security_level=level,
                timeout_seconds=10.0,
                max_memory_mb=256,
                allowed_modules=("math", "statistics", "json"),
            ),
        }
        return configs[level]


@dataclass(frozen=True)
class SandboxResult:
    """Result of sandboxed code execution."""

    success: bool
    result: Any
    stdout: str
    stderr: str
    error: str | None
    execution_time_ms: float
    memory_used_mb: float | None


class SandboxedRunner(ABC):
    """Protocol for sandboxed code execution."""

    @abstractmethod
    def __init__(self, config: SandboxConfig) -> None:
        """Initialize with configuration."""
        ...

    @abstractmethod
    def execute(self, code: str, context: dict[str, Any]) -> SandboxResult:
        """
        Execute code in sandbox with given context.

        Args:
            code: Python code to execute
            context: Variables to inject (e.g., {"df": dataframe})

        Returns:
            SandboxResult with execution outcome
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources held by the sandbox."""
        ...

    def __enter__(self) -> "SandboxedRunner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()
```

## Implementation Classes

```python
class SubprocessRunner(SandboxedRunner):
    """Subprocess-based sandbox for Standard security level."""

    def execute(self, code: str, context: dict[str, Any]) -> SandboxResult:
        # 1. Serialize context to temp file or pipe
        # 2. Spawn subprocess with resource limits
        # 3. Wait for completion or timeout
        # 4. Deserialize result
        # 5. Clean up
        ...


class DockerRunner(SandboxedRunner):
    """Docker-based sandbox for Strict security level."""

    def execute(self, code: str, context: dict[str, Any]) -> SandboxResult:
        # 1. Write code and context to temp volume
        # 2. docker run with limits
        # 3. Read result from volume
        # 4. docker rm container
        ...


class InProcessRunner(SandboxedRunner):
    """In-process execution for Permissive level (development only)."""

    def execute(self, code: str, context: dict[str, Any]) -> SandboxResult:
        # Enhanced version of current CodeRunner
        # with import hooks and signal-based timeout
        ...
```

## Factory Pattern

```python
def create_runner(
    level: SecurityLevel = SecurityLevel.STANDARD,
    config: SandboxConfig | None = None,
) -> SandboxedRunner:
    """
    Create appropriate sandbox runner for security level.

    Falls back gracefully if Docker unavailable for STRICT level.
    """
    if config is None:
        config = SandboxConfig.for_level(level)

    if level == SecurityLevel.STRICT:
        if _docker_available():
            return DockerRunner(config)
        else:
            # Fall back to subprocess with warning
            logger.warning("Docker unavailable, using subprocess for STRICT level")
            return SubprocessRunner(config)

    elif level == SecurityLevel.STANDARD:
        return SubprocessRunner(config)

    else:  # PERMISSIVE
        return InProcessRunner(config)
```

## Integration with Existing Code

The `LLMExecutor` will be updated to use `SandboxedRunner`:

```python
class LLMExecutor(Executor):
    def __init__(
        self,
        llm: LLMClient,
        sandbox_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> None:
        self.llm = llm
        self.runner = create_runner(sandbox_level)

    def execute(self, task: Task, context: ImprovementContext | None) -> ExecutionResult:
        code = self._generate_code(task, context)

        # Use sandboxed execution
        result = self.runner.execute(code, {"df": task.dataframe})

        return ExecutionResult(
            task_id=task.id,
            output=result.result,
            trajectory=[...],
            generated_code=code,
        )
```

## Security Considerations

### Subprocess Security
- Use `subprocess.Popen` with `shell=False`
- Set `env` explicitly, don't inherit
- Close unnecessary file descriptors
- Use `preexec_fn` for Unix resource limits

### Docker Security
- Use official Python slim image
- Run as non-root user
- Mount volumes read-only where possible
- Use `--rm` to auto-cleanup
- Apply seccomp profile
- Drop all capabilities

### Data Serialization
- Use pickle only for trusted data structures
- Consider JSON for simple types
- Validate deserialized data types

## Testing Strategy

1. **Unit tests**: Each runner implementation in isolation
2. **Security tests**: Verify resource limits enforced
3. **Integration tests**: Full execution flow with LLMExecutor
4. **Escape tests**: Attempt sandbox escapes (fuzzing, known bypasses)

```python
def test_memory_limit_enforced():
    runner = SubprocessRunner(SandboxConfig(max_memory_mb=50))
    result = runner.execute("x = 'a' * (100 * 1024 * 1024)", {})
    assert not result.success
    assert "MemoryError" in result.error or "killed" in result.error.lower()

def test_timeout_enforced():
    runner = SubprocessRunner(SandboxConfig(timeout_seconds=1))
    result = runner.execute("import time; time.sleep(10)", {})
    assert not result.success
    assert result.execution_time_ms < 2000

def test_network_blocked():
    runner = SubprocessRunner(SandboxConfig())
    result = runner.execute("import urllib.request; urllib.request.urlopen('http://example.com')", {})
    assert not result.success
```

## Migration Path

1. **Phase 1**: Implement `InProcessRunner` as drop-in for current `CodeRunner`
2. **Phase 2**: Implement `SubprocessRunner` with full resource limits
3. **Phase 3**: Implement `DockerRunner` for strict environments
4. **Phase 4**: Update `LLMExecutor` to use `create_runner()` factory
5. **Phase 5**: Add configuration via environment variables / config file

## Open Questions

1. **Windows support**: `resource` module unavailable - use job objects or skip limits?
2. **Pandas serialization**: Pickle vs Parquet vs Arrow for dataframe transfer?
3. **Container reuse**: Pool warm containers for lower latency in strict mode?
4. **Metrics**: Track execution time, memory usage for optimization insights?

## Appendix: Seccomp Profile for Docker

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "open", "close", "stat", "fstat", "mmap", "mprotect", "munmap", "brk", "exit_group", "arch_prctl", "set_tid_address", "set_robust_list"],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```
