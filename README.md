# Adaptive Agent: Self-Improving Data Analysis Agent

A framework for building AI agents that learn from mistakes and improve over time, designed for data analysis tasks.

## Overview

This project implements a self-improving agent framework that:

1. **Generates code** from natural language queries about data
2. **Executes and evaluates** the generated code against expected answers
3. **Learns from failures** by storing episodes (query + failed code + fix)
4. **Applies learnings** to similar future queries via keyword similarity matching

The key innovation is **cross-session learning**: when the agent encounters a similar query in a new session, it retrieves relevant past experiences and uses them to generate better code on the first attempt.

## Architecture

```
src/
  core/           # Core types and protocols (ABCs)
  agent/          # Agent runner orchestration
  executor/       # Code generation and execution
  evaluator/      # Answer evaluation (exact match + LLM-based)
  sandbox/        # Sandboxed code execution (permissive/standard/strict)
  strategies/     # Improvement strategies
    - none.py           # Baseline (no improvement)
    - reflection.py     # In-session reflection
    - episodic_memory.py # Cross-session learning
  storage/        # Persistence (in-memory + file-based)
  versioning/     # Strategy version tracking and snapshots
  llm/            # LLM client abstraction (Claude)
  benchmark/      # Benchmarking framework
  data/           # Dataset loading and validation
  prompts/        # LLM prompt templates

scripts/
  chat.py                    # Interactive CLI (recommended starting point)
  demo.py                    # Basic framework demo (mock)
  demo_cross_session.py      # Cross-session learning demo
  demo_cogs_learning.py      # COGS percentage learning demo (real LLM)
  demo_system.py             # Full system demo with real LLM
  seed_demo.py               # Pre-seed episodes for reliable demos
  run_learning_benchmark.py  # Prove learning works (recommended benchmark)
  run_benchmark.py           # Single-model benchmarks
  run_matrix_benchmark.py    # Strategy × Model matrix
  run_multi_session_benchmark.py  # Multi-session comparison
  analyze_benchmark.py       # Detailed single-task analysis

data/
  FUN_company_pl_actuals_dataset.csv  # P&L financial data
  sales_data.csv                      # Sales data
  order_data.csv                      # Order data
  employee_data.csv                   # Employee data
  challenging_data.csv                # Edge case test data
```

See [architecture.md](architecture.md) for diagrams and [docs/design-document.md](docs/design-document.md) for the full design document.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd adaptive-agent

# Install dependencies
uv sync

# Verify installation
uv run pytest
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your-api-key
```

## Quick Start

### Interactive CLI (Recommended)

The interactive CLI is the best way to experience the self-improving agent:

```bash
uv run python scripts/chat.py --persist-default
```

**Commands:**
| Command | Description |
|---------|-------------|
| `/schema` | View dataset structure |
| `/sample` | See sample data rows |
| `/correct` | Mark last answer as correct (saves pattern for learning) |
| `/wrong` | Mark last answer as wrong (saves failure for learning) |
| `/episodes` | List all stored learning episodes |
| `/stats` | Show learning statistics |
| `/clear` | Clear all learning history |
| `/quit` | Exit |

**Interactive Demo Story (Step-by-Step):**

This walkthrough shows the self-improving agent in action:

```bash
# 1. Start the CLI with persistent learning
uv run python scripts/chat.py --persist-default

# 2. Clear any previous learning (fresh start)
/clear

# 3. Ask a financial question the agent might get wrong initially
What was the FX impact for Product C in 2024?

# 4. If the answer is correct, mark it to save the pattern
/correct

# 5. Ask a SIMILAR question (same pattern, different product)
What was the FX impact for Product A in 2024?

# 6. Watch for "✨ LEARNING APPLIED ✨" banner - this shows the agent
#    retrieved the pattern from step 4 and used it!

# 7. View all stored learning episodes
/episodes

# 8. See learning statistics
/stats

# 9. Try more variations to see learning compound:
What was the total COGS for Product B in 2023?
/correct
What was the total COGS for Product D in 2024?
# Should show learning applied again!
```

**Key observations:**
- First query: Agent figures it out (may take multiple attempts)
- Mark as `/correct`: Pattern is saved with keywords
- Similar query: Agent retrieves the saved pattern and applies it
- The "LEARNING APPLIED" banner confirms cross-session learning is working

### Demo Scripts

#### 1. COGS Learning Demo (Real LLM + Real Data)

Shows the agent learning to calculate COGS as a percentage of Gross Revenue:

```bash
uv run python scripts/demo_cogs_learning.py
```

#### 2. Cross-Session Learning Demo

Demonstrates how learning persists across sessions:

```bash
# Optional: Seed with pre-computed episodes for reliable demo
uv run python scripts/seed_demo.py

# Run the cross-session demo
uv run python scripts/demo_cross_session.py
```

#### 3. Full System Demo

Tests all strategies with real Claude API calls:

```bash
uv run python scripts/demo_system.py
```

#### 4. Basic Framework Demo

Simple mock-based demo of the framework:

```bash
uv run python scripts/demo.py
```

### Benchmarking

Compare strategy effectiveness with real LLM calls:

```bash
# RECOMMENDED: Learning benchmark (proves episodic memory works)
uv run python scripts/run_learning_benchmark.py --mode pairs

# Full P&L suite benchmark
uv run python scripts/run_learning_benchmark.py --mode suite

# Both learning pairs and full suite
uv run python scripts/run_learning_benchmark.py --mode both --include-hard

# Analyze single task in detail (see LLM outputs)
uv run python scripts/analyze_benchmark.py

# Single-model benchmark across task suite
uv run python scripts/run_benchmark.py

# Strategy × Model matrix comparison
uv run python scripts/run_matrix_benchmark.py

# Multi-session learning comparison
uv run python scripts/run_multi_session_benchmark.py --sessions 5 --models haiku sonnet
```

**Learning Benchmark Output:**
```
ANALYSIS
======================================================================

✅ LEARNING PROVEN: Episodic memory improved learner task performance
   - Without learning: 1/3 passed
   - With episodic memory: 3/3 passed
   - Improvement: +2 tasks (66.7%)

✅ EFFICIENCY GAIN: Episodic memory reduced attempts on learner tasks
   - Without learning: 2.00 avg attempts
   - With episodic memory: 1.33 avg attempts
```

### Run Tests

```bash
uv run pytest                           # All tests
uv run pytest tests/unit                # Unit tests only
uv run pytest tests/integration         # Integration tests (real LLM)
uv run pytest -v                        # Verbose output
uv run pytest -k "learning"             # Learning-related tests only
```

## How Cross-Session Learning Works

### The Problem

Without learning, an agent makes the same mistakes repeatedly:

```
Query: "What was the Gross Revenue for Product A in Q1 2023?"

Attempt 1 (Wrong):
  df[df['Product'] == 'Product A']['Amount in USD'].sum()
  # Returns ALL revenue for Product A (missing quarter/year filter)

Attempt 2 (Correct):
  df[(df['Product'] == 'Product A') &
     (df['FSLine Statement L2'] == 'Gross Revenue') &
     (df['Fiscal Quarter'] == 'Q1') &
     (df['Fiscal Year'] == 2023)]['Amount in USD'].sum()
```

### The Solution: Episodic Memory

The `EpisodicMemoryStrategy` stores each failure as an "episode":

```python
Episode(
    query="What was the Gross Revenue for Product A in Q1 2023?",
    failed_code="df[df['Product'] == 'Product A']['Amount in USD'].sum()",
    error_message="Value too high, missing time filters",
    fixed_code="df[(df['Product'] == 'Product A') & ...]...",
    keywords=["revenue", "gross revenue", "product a", "q1", "2023"]
)
```

When a similar query appears later, the agent:

1. **Extracts keywords** from the new query
2. **Finds similar episodes** via Jaccard similarity on keywords
3. **Injects the fixed code** as an example in the prompt
4. **Generates better code** informed by past experience

## Improvement Strategies

| Strategy | Learning Type | Cross-Session | Best For |
|----------|---------------|---------------|----------|
| `NoImprovementStrategy` | None | No | Baseline comparison |
| `ReflectionStrategy` | Error-type hints | Optional | In-session iteration |
| `EpisodicMemoryStrategy` | Concrete examples | Yes | Similar query patterns |

### Strategy Details

**NoImprovementStrategy** (Baseline)
- Does nothing - each attempt is independent
- Essential for measuring whether other strategies actually help
- Use this as the control group in benchmarks

**ReflectionStrategy** (In-Session)
- Provides hints based on error types (e.g., "check rounding", "verify filters")
- Tracks patterns within a session (improving/declining scores)
- Can optionally persist learnings by tag/task for future sessions
- Good for iterative refinement within a single problem

**EpisodicMemoryStrategy** (Cross-Session Learning)
- Stores concrete failure/fix pairs as "episodes"
- When a similar query appears, retrieves matching episodes via keyword similarity
- Injects working code as examples into the prompt
- Tracks effectiveness scores: which fixes actually help?
- This is the key innovation - learns from past experience

### How to Compare Strategies

Run the learning benchmark to see quantitative proof:

```bash
uv run python scripts/run_learning_benchmark.py --mode pairs
```

Expected output shows episodic memory outperforming baseline on "learner" tasks (queries similar to previously seen ones).

## Benchmarking

Compare strategies using the benchmark framework:

```python
from src.benchmark.runner import BenchmarkRunner
from src.core.types import BenchmarkConfig
from src.strategies import NoImprovementStrategy, EpisodicMemoryStrategy

config = BenchmarkConfig(
    strategies={
        "baseline": NoImprovementStrategy(),
        "episodic": EpisodicMemoryStrategy(storage=storage),
    },
    task_suite=tasks,
    max_attempts=3,
)

runner = BenchmarkRunner(config)
results = runner.run()
report = runner.compare(results)
print(report.summary)
```

### Multi-Session Benchmarks

Measure how strategies improve over multiple sessions:

```bash
uv run python scripts/run_multi_session_benchmark.py \
  --sessions 5 \
  --models haiku sonnet \
  --strategies none reflection episodic \
  --tasks 5 \
  --difficulty medium
```

## Sandbox Security Levels

The sandbox provides three security levels for code execution:

| Level | Isolation | Use Case |
|-------|-----------|----------|
| Permissive | In-process with import hooks | Development/testing |
| Standard | Subprocess with resource limits | Production |
| Strict | Docker container with seccomp | Untrusted code |

See [docs/sandbox-design.md](docs/sandbox-design.md) for details.

## Design Principles

- **Protocol-based**: All major components implement protocols (ABCs) for pluggability
- **Immutable types**: Core data types are frozen dataclasses
- **Dependency injection**: Components receive dependencies via constructor
- **Strategy pattern**: Learning approaches are interchangeable

## License

MIT
