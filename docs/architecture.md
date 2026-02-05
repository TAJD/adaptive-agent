# Architecture

Self-Improving Data Analysis Agent Framework

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              AGENT RUNNER (Orchestrator)                          │
│                                src/agent/runner.py                                │
│                                                                                   │
│   Task → [Load Priors] → [Execute] → [Evaluate] → Pass? ──→ Return Result        │
│                             ↑                       │ No                          │
│                             └── [Improve] ← [Persist] ← Last? ──→ Return Failed  │
└───────────┬────────────────────┬───────────────────┬────────────────┬────────────┘
            │                    │                   │                │
            ▼                    ▼                   ▼                ▼
┌───────────────────┐ ┌───────────────────┐ ┌──────────────┐ ┌─────────────────────┐
│     EXECUTOR      │ │     EVALUATOR     │ │   STRATEGY   │ │      STORAGE        │
│  src/executor/    │ │   src/evaluator/  │ │ src/strategies│ │    src/storage/     │
├───────────────────┤ ├───────────────────┤ ├──────────────┤ ├─────────────────────┤
│ LLMExecutor       │ │ LLMEvaluator      │ │ Episodic     │ │ FileStorage (JSON)  │
│  ├─ Build prompt  │ │  ├─ Compare ans   │ │  Memory      │ │ MemoryStorage       │
│  ├─ Call LLM      │ │  ├─ Classify err  │ │  ├─ Episodes │ ├─────────────────────┤
│  ├─ Extract code  │ │  └─ Gen feedback  │ │  ├─ Similarity│ │ VersionedStorage   │
│  └─ Run in sandbox│ │                   │ │  └─ Fixes    │ │  └─ Snapshots      │
├───────────────────┤ │ ExactMatch        │ │              │ │  └─ Rollback       │
│ CodeRunner        │ │  └─ Direct compare│ │ Reflection   │ └─────────────────────┘
│  └─ Safe exec     │ └───────────────────┘ │  └─ In-sess  │
└─────────┬─────────┘                       │     hints    │
          │                                 │              │
          ▼                                 │ NoneStrategy │
┌───────────────────┐                       │  └─ Baseline │
│      SANDBOX      │                       └──────────────┘
│   src/sandbox/    │
├───────────────────┤
│ Permissive        │
│  └─ In-process    │
│ Standard          │
│  └─ Subprocess    │
│ Strict            │
│  └─ Docker        │
└───────────────────┘
```

## LLM Integration

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                  LLM INTEGRATION                                  │
│                                    src/llm/                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│  LLMClient Protocol ◄── ClaudeClient (src/llm/claude.py)                         │
│       │                      ├─ Anthropic API                                     │
│       └─ complete()          ├─ Tool support                                      │
│       └─ complete_with_tools()└─ Retry logic                                      │
│                                                                                   │
│  UsageTracker (src/llm/usage.py) ─ Token counting + cost tracking                │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Benchmark System

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               BENCHMARK SYSTEM                                    │
│                                src/benchmark/                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────┐         ┌────────────────────────────────────────┐      │
│  │  BenchmarkRunner    │         │       MatrixBenchmarkRunner            │      │
│  │  (Single Model)     │         │       (Strategy × Model Grid)          │      │
│  ├─────────────────────┤         ├────────────────────────────────────────┤      │
│  │ Config:             │         │                                        │      │
│  │  ├─ task_suite      │    ┌────┼─→ For each Model (Haiku, Sonnet, Opus) │      │
│  │  ├─ strategies      │    │    │    For each Strategy                   │      │
│  │  ├─ executor        │    │    │      → Run BenchmarkRunner             │      │
│  │  └─ evaluator       │────┘    │      → Collect MatrixResult            │      │
│  │                     │         │                                        │      │
│  │ Output:             │         │ Output:                                │      │
│  │  └─ StrategyMetrics │         │  └─ strategy × model grid              │      │
│  └─────────────────────┘         └────────────────────────────────────────┘      │
│                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐       │
│  │  Supporting Modules                                                    │       │
│  ├───────────────────────────────────────────────────────────────────────┤       │
│  │  model_config.py   │ ModelConfig + LLM parameters                     │       │
│  │  metrics.py        │ StrategyMetrics collection + aggregation         │       │
│  │  tasks.py          │ Task suite definitions + loading                 │       │
│  │  data_loader.py    │ Real data context injection                      │       │
│  │  results_store.py  │ Persistent results storage                       │       │
│  │  report_generator  │ Generate benchmark reports                       │       │
│  └───────────────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Core Types & Protocols

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            CORE TYPES & PROTOCOLS                                 │
│                                 src/core/                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  PROTOCOLS (src/core/protocols.py)          DATA TYPES (src/core/types.py)       │
│  ────────────────────────────               ─────────────────────────────        │
│  • Executor                                 • Task (query, expected_answer)       │
│  • Evaluator                                • ExecutionResult (code, output)      │
│  • ImprovementStrategy                      • Evaluation (score, error_type)      │
│  • Storage                                  • ImprovementContext                  │
│  • MetricsCollector                         • TaskResult (pass/fail, attempts)    │
│                                             • StrategyMetrics                     │
│                                             • BenchmarkConfig                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
    Task (query + expected answer)
           │
           ▼
    ┌──────────────────┐
    │  Load Priors     │◄────── Storage (past episodes, fixes)
    │  (strategy)      │
    └────────┬─────────┘
             │ context: {hints, examples, constraints}
             ▼
    ┌──────────────────┐        ┌─────────────┐        ┌──────────────┐
    │    EXECUTE       │───────►│   SANDBOX   │───────►│    OUTPUT    │
    │  (LLMExecutor)   │ code   │  (isolated) │ result │   + CODE     │
    └────────┬─────────┘        └─────────────┘        └──────┬───────┘
             │                                                 │
             │ ExecutionResult                                 │
             ▼                                                 │
    ┌──────────────────┐                                       │
    │    EVALUATE      │◄──────────────────────────────────────┘
    │  (LLMEvaluator)  │
    └────────┬─────────┘
             │ Evaluation {score, passed, error_type, feedback}
             ▼
        ┌────┴────┐
        │ PASSED? │
        └────┬────┘
       Yes   │   No
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
 ┌──────┐      ┌────────────┐
 │ DONE │      │  IMPROVE   │◄────── Similar episodes from storage
 └──────┘      │ (strategy) │
               └─────┬──────┘
                     │ new hints + examples
                     ▼
               ┌────────────┐
               │  PERSIST   │──────► Storage (save episode)
               └─────┬──────┘
                     │
                     ▼
               [Retry with enhanced context]
```

## Key Design Principles

- **Protocol-based**: All major components implement protocols (ABCs) for pluggability
- **Immutable types**: Core data types are frozen dataclasses
- **Dependency injection**: Components receive dependencies via constructor
- **Strategy pattern**: Learning approaches are interchangeable
- **Cross-session learning**: Episodes persist to storage for future sessions

## Module Reference

| Module | Location | Purpose |
|--------|----------|---------|
| Agent Runner | `src/agent/runner.py` | Main orchestration loop |
| LLM Executor | `src/executor/llm.py` | Prompt building + code generation |
| Code Runner | `src/executor/code_runner.py` | Basic Python execution |
| Sandbox | `src/sandbox/` | Isolated code execution |
| LLM Evaluator | `src/evaluator/llm.py` | Semantic evaluation with error classification |
| Episodic Memory | `src/strategies/episodic_memory.py` | Cross-session learning |
| Reflection | `src/strategies/reflection.py` | In-session self-reflection |
| File Storage | `src/storage/file.py` | JSON-based persistence |
| Versioned Storage | `src/versioning/versioned_storage.py` | Storage with snapshots + rollback |
| Benchmark Runner | `src/benchmark/runner.py` | Single-model benchmarks |
| Matrix Runner | `src/benchmark/matrix_runner.py` | Strategy × Model grid benchmarks |
| Claude Client | `src/llm/claude.py` | Anthropic API integration |
