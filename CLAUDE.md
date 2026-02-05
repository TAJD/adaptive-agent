# Claude Code Instructions

## Project Overview

This is a self-improving data analysis agent framework. The agent generates Python code to answer queries about financial data, evaluates results, and learns from failures using episodic memory.

### Architecture

```
Agent Runner (Orchestrator)
    ├── Executor → Generates and executes code via LLM
    │   └── Sandbox → Safe code execution with configurable security levels
    ├── Evaluator → Validates results against expected answers
    ├── ImprovementStrategy → Learns from failures (reflection, episodic memory)
    │   └── Versioning → Track improvement evolution and effectiveness
    ├── Storage → Persists learnings across sessions
    └── Benchmark → Multi-model performance testing and analysis
        ├── MatrixRunner → Strategy × Model benchmarking
        └── ResultsStore → Persistent results with export
```

### Key Modules

**Core**
- `src/core/protocols.py` - Interface definitions (ABCs for Executor, Evaluator, etc.)
- `src/core/types.py` - Data types (Task, ExecutionResult, Evaluation)

**Execution**
- `src/executor/llm.py` - LLM-based code generation
- `src/executor/code_runner.py` - Python code execution
- `src/sandbox/` - Sandboxed execution (types.py, runner.py, timeout.py)

**Learning**
- `src/strategies/episodic_memory.py` - Cross-session learning
- `src/strategies/reflection.py` - Self-reflection on failures
- `src/versioning/` - Track improvement evolution over time

**Benchmark**
- `src/benchmark/runner.py` - Single-model benchmark runner
- `src/benchmark/matrix_runner.py` - Strategy × Model matrix benchmarks
- `src/benchmark/results_store.py` - Persistent results storage
- `src/benchmark/model_config.py` - Model configuration for multi-model runs

**LLM**
- `src/llm/protocol.py` - LLM client interface
- `src/llm/claude.py` - Claude API client
- `src/llm/usage.py` - Token usage tracking

**Agent**
- `src/agent/runner.py` - Main orchestration loop

## Python Environment

**Always use `uv` to run Python.** Do not use `python` or `python3` directly.

```bash
uv run python script.py    # Run a script
uv run pytest              # Run tests
uv run python -c "..."     # Run inline code
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for LLM features

## Running Tests

**Always use haiku agents (Task tool with `model: "haiku"`) to run tests.** This minimizes cost and latency for test execution.

```bash
uv run pytest                      # All tests
uv run pytest tests/unit           # Unit tests only
uv run pytest tests/integration    # Integration tests
uv run pytest -k "test_name"       # Specific test
```

## Running Scripts

```bash
uv run python scripts/demo.py                 # Basic demo
uv run python scripts/demo_cross_session.py   # Cross-session learning demo
uv run python scripts/demo_system.py          # Full system demonstration
uv run python scripts/chat.py                 # Interactive CLI
uv run python scripts/run_benchmark.py        # Run single-model benchmarks
uv run python scripts/run_matrix_benchmark.py # Run strategy × model matrix
uv run python scripts/analyze_benchmark.py    # Analyze benchmark results
```

## Development Patterns

- **ABCs for interfaces**: Build implementations around Abstract Base Classes (ABCs) where appropriate to scale interfaces. Define ABCs in `src/core/protocols.py` for any component with multiple implementations (executors, evaluators, strategies, storage backends, benchmark runners). This enables proper type checking, clear contracts, and easier testing with mocks.
- **Immutable types**: Use frozen dataclasses from `src/core/types.py`
- **Dependency injection**: Components receive dependencies via constructor
- **Strategy pattern**: Improvement strategies are pluggable

## Sandbox Security Levels

See `docs/sandbox-design.md` for the sandbox architecture. Three levels:

- **Permissive**: Development use, enhanced exec() with import hooks
- **Standard**: Production, subprocess isolation with resource limits
- **Strict**: Untrusted code, Docker containers with seccomp
