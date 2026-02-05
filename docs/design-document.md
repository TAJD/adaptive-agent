# Design Document

## 1. Architecture Overview

The system in this repo implements a benchmarking system for evaluating strategies for improving agent responses between sessions.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER / BENCHMARK SYSTEM                            │
│                                                                             │
│   Task: "What was Gross Revenue for Product A in Q1 2023?"                  │
│   Expected: 125000.00                                                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT RUNNER (Orchestrator)                          │
│                           src/agent/runner.py                                │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Load Priors │───►│   Execute   │───►│  Evaluate   │───►│   Improve   │  │
│  │ (Strategy)  │    │ (LLM+Code)  │    │ (LLM/Exact) │    │ (Strategy)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│         ▲                                                        │         │
│         │                    RETRY LOOP (max 5)                  │         │
│         └────────────────────────────────────────────────────────┘         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│    EXECUTOR     │      │    EVALUATOR    │      │    STRATEGY     │
│                 │      │                 │      │                 │
│ LLMExecutor     │      │ LLMEvaluator    │      │ EpisodicMemory  │
│  ├─ Prompt      │      │  ├─ Compare     │      │  ├─ Episodes    │
│  ├─ Claude API  │      │  ├─ Classify    │      │  ├─ Similarity  │
│  └─ Code Run    │      │  └─ Feedback    │      │  └─ Fixes       │
│                 │      │                 │      │                 │
│ CodeRunner      │      │ ExactMatch      │      │ Reflection      │
│  └─ Sandbox     │      │  └─ Tolerance   │      │  └─ Hints       │
└────────┬────────┘      └─────────────────┘      └────────┬────────┘
         │                                                  │
         ▼                                                  ▼
┌─────────────────┐                              ┌─────────────────┐
│    SANDBOX      │                              │    STORAGE      │
│                 │                              │                 │
│ Permissive      │                              │ FileStorage     │
│ Standard        │                              │  └─ JSON files  │
│ Strict          │                              │                 │
│                 │                              │ MemoryStorage   │
│ Resource limits │                              │  └─ In-memory   │
│ Import control  │                              │                 │
└─────────────────┘                              └─────────────────┘
```

### How the Agentic Loop Works

The agent operates in a **generate-evaluate-improve** loop:

```python
def run(task, context) -> TaskResult:
    # 1. Load prior learnings for this task type
    priors = strategy.load_priors(task)
    context = merge(context, priors)

    for attempt in range(1, max_attempts + 1):
        # 2. Generate and execute code
        result = executor.execute(task, context)

        # 3. Evaluate the result
        evaluation = evaluator.evaluate(task, result)

        # 4. Check if passed
        if evaluation.passed:
            strategy.persist_success(task, result)
            return TaskResult(passed=True, ...)

        # 5. If failed, improve context for next attempt
        improvements = strategy.improve(ImprovementContext(
            task=task,
            result=result,
            evaluation=evaluation,
            attempt_number=attempt,
            history=previous_attempts
        ))

        # 6. Persist the failure for future learning
        strategy.persist(improvement_context)

        # 7. Update context with improvements
        context = merge(context, improvements)

    return TaskResult(passed=False, ...)
```

**Key characteristics:**
- Maximum 5 attempts per task (configurable)
- Each attempt builds on learnings from previous failures
- Improvements accumulate within a session AND across sessions
- The loop terminates early on success

### Where Improvements Are Stored and Retrieved

**Storage Location:** File-based JSON storage in configurable directories

```
storage/
  episodes/
    task_001/
      a1b2c3d4.json    # Episode for query hash a1b2c3d4
      e5f6g7h8.json    # Another episode
    task_002/
      ...
  reflection/
    errors/
      numeric_error.json
      filter_error.json
    tasks/
      task_001.json
```

**Retrieval Process:**

1. **On task start:** `strategy.load_priors(task)` searches for similar episodes
2. **Similarity matching:** Jaccard similarity on extracted keywords
3. **Injection:** Retrieved episodes' `fixed_code` becomes examples in the prompt

```python
# Simplified retrieval flow
def load_priors(task):
    keywords = extract_keywords(task.query)
    similar = find_similar_episodes(keywords, threshold=0.3)

    examples = []
    for episode in similar:
        if episode.fixed_code:
            examples.append({
                "query": episode.query,
                "code": episode.fixed_code
            })

    return {"examples": examples, "hints": [...]}
```

---

## 2. Self-Improvement Mechanism

### What Triggers Improvement Creation?

Improvements are created when:

1. **Task fails evaluation** - The generated code produces wrong output
2. **Error is classified** - The evaluator categorizes the error type
3. **Context is rich enough** - We have the query, code, error, and feedback

```python
# Trigger conditions in agent loop
if not evaluation.passed:
    improvement_context = ImprovementContext(
        task=task,
        result=result,  # Contains generated code
        evaluation=evaluation,  # Contains error_type, feedback
        attempt_number=attempt,
        history=previous_attempts
    )
    strategy.persist(improvement_context)  # Creates episode
```

### How Are Improvements Represented?

Improvements are represented as **Episodes** - structured data objects:

```python
@dataclass
class Episode:
    # The original query that failed
    query: str

    # The code that produced wrong results
    failed_code: str

    # Error message from evaluator
    error_message: str

    # Working code (filled when task eventually passes)
    fixed_code: str | None

    # Error classification for pattern matching
    error_type: str  # "numeric_error", "filter_error", "aggregation_error"

    # Extracted keywords for similarity search
    keywords: list[str]  # ["revenue", "product a", "q1", "2023"]

    # Effectiveness tracking
    effectiveness_score: float  # EMA of success rate
    times_applied: int
    times_succeeded: int
```

**Why this representation?**

- **Structured data, not code:** Episodes are data that inform prompt construction, not executable patches
- **Human-readable:** Easy to inspect, debug, and manually curate
- **Versioned:** Can track effectiveness over time and deprecate bad episodes
- **Searchable:** Keywords enable fast similarity matching

### Where Are They Stored? (Justification)

**Choice: File-based JSON storage**

```python
class FileStorage(Storage):
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def save(self, key: str, data: Any) -> None:
        path = self.base_path / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

**Justification:**

| Factor | File-based JSON | Database | Vector DB |
|--------|-----------------|----------|-----------|
| Simplicity | Excellent | Moderate | Complex |
| Portability | Copy folder | Export/import | Complex migration |
| Debugging | Open in editor | Query needed | Specialized tools |
| Version control | Git-friendly | Not practical | Not practical |
| Scalability | ~10K episodes | Millions | Millions + semantic |
| Dependencies | None | DB server | Embedding model |

For a demonstration/research system, file-based storage is optimal because:
1. **Transparency:** Episodes are human-readable JSON files
2. **Reproducibility:** Entire learning state can be version-controlled
3. **Debugging:** Easy to inspect what the agent learned
4. **Portability:** Copy a folder to transfer learnings

In production for thousands or millions of users, a vector database would be the scalable approach.

### How Are They Applied in New Sessions?

**Application flow:**

```
Session 2 starts
       │
       ▼
┌──────────────────────────────────────┐
│  task = Task("Gross Revenue for      │
│              Product B in Q2 2024")  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  keywords = extract(task.query)      │
│  = ["revenue", "gross revenue",      │
│     "product b", "q2", "2024"]       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Search stored episodes              │
│                                      │
│  Episode from Session 1:             │
│    query: "Gross Revenue Product A   │
│            in Q1 2023"               │
│    keywords: ["revenue", "gross      │
│      revenue", "product a", "q1",    │
│      "2023"]                         │
│                                      │
│  Jaccard similarity = 0.4 > 0.3      │
│  MATCH!                              │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Inject into prompt:                 │
│                                      │
│  "Here's a similar query that was    │
│   solved before:                     │
│                                      │
│   Query: Gross Revenue Product A...  │
│   Working code:                      │
│   df[(df['Product']=='Product A') &  │
│      (df['FSLine']=='Gross Rev') &   │
│      (df['Quarter']=='Q1') &         │
│      (df['Year']==2023)]             │
│      ['Amount'].sum()                │
│                                      │
│   Apply similar filtering for your   │
│   query."                            │
└──────────────────────────────────────┘
```

**Effectiveness tracking:**

When an episode is applied:
- `times_applied += 1`
- If task passes: `times_succeeded += 1`
- Update effectiveness: `score = 0.7 * old_score + 0.3 * (1 if passed else 0)`

Low-effectiveness episodes are deprioritized in retrieval.

---

## 3. Code Execution Strategy

### Approach Chosen

**Hybrid sandbox with three security levels:**

1. **Permissive (Development):** In-process `exec()` with import hooks
2. **Standard (Production):** Subprocess isolation with resource limits
3. **Strict (Untrusted):** Docker container with seccomp profiles

```python
class SandboxConfig:
    time_limit: float = 30.0          # Max execution time
    memory_limit: int = 100_000_000   # 100MB
    allowed_modules: list[str] = ["pandas", "numpy", "math", "statistics"]
    allowed_builtins: list[str] = ["len", "sum", "min", "max", "range", ...]
    enable_network: bool = False
    enable_filesystem: bool = False
```

### Why This Approach?

**Requirements analysis:**

| Requirement | Solution |
|-------------|----------|
| Execute LLM-generated Python | `exec()` or subprocess |
| Prevent infinite loops | Timeout enforcement |
| Prevent resource exhaustion | Memory limits |
| Prevent malicious imports | Module whitelist |
| Allow pandas/numpy | Selective import hooks |

**Why not alternatives?**

| Alternative | Rejection Reason |
|-------------|------------------|
| Docker only | Too slow for iteration (2-3s startup) |
| WebAssembly | No pandas/numpy support |
| Restricted Python (RestrictedPython) | Too restrictive for data analysis |
| No sandbox | Unacceptable security risk |

This is ultimatately an example. In production the sandbox should be a function of what the code being written can do.

### How Safety is Ensured

**Layer 1: Import Control**
```python
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "requests",
    "shutil", "pathlib", "importlib", "__builtins__"
}

ALLOWED_MODULES = {
    "pandas", "numpy", "math", "statistics",
    "datetime", "collections", "itertools"
}
```

**Layer 2: Builtin Restrictions**
```python
SAFE_BUILTINS = {
    "len", "sum", "min", "max", "abs", "round",
    "sorted", "reversed", "enumerate", "zip",
    "range", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "type"
}
# Notably excluded: eval, exec, compile, open, input, __import__
```

**Layer 3: Resource Limits (Standard mode)**
```python
def run_in_subprocess(code: str, context: dict) -> SandboxResult:
    process = subprocess.Popen(
        ["python", "-c", wrapper_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(timeout=config.time_limit)
    except subprocess.TimeoutExpired:
        process.kill()
        return SandboxResult(success=False, error="Timeout exceeded")
```

**Layer 4: Docker Isolation (Strict mode)**
```python
def run_in_docker(code: str, context: dict) -> SandboxResult:
    container = docker.run(
        "python:3.11-slim",
        command=["python", "-c", code],
        mem_limit="100m",
        cpu_period=100000,
        cpu_quota=50000,  # 50% CPU
        network_disabled=True,
        read_only=True,
        security_opt=["no-new-privileges", "seccomp=default.json"]
    )
```

### Trade-offs

| Trade-off | My Choice | Consequence |
|-----------|------------|-------------|
| Speed vs Security | Permissive for dev, Standard for prod | Dev is fast, prod is safe |
| Functionality vs Safety | Allow pandas/numpy | Larger attack surface |
| Simplicity vs Isolation | Subprocess over Docker default | Faster but less isolated |
| Flexibility vs Control | Whitelist approach | May block legitimate code |

**Known limitations:**
- Permissive mode is NOT production-safe
- Some pandas operations can be slow (no per-operation timeout)
- Memory limits are advisory in subprocess mode

---

## 4. Evaluation Strategy

### How We Measure Improvement Effectiveness

**Per-episode tracking:**

```python
class Episode:
    effectiveness_score: float  # Exponential moving average
    times_applied: int          # How many times retrieved
    times_succeeded: int        # How many times led to success
```

**Calculation:**
```python
def update_effectiveness(episode: Episode, succeeded: bool):
    episode.times_applied += 1
    if succeeded:
        episode.times_succeeded += 1

    # EMA with alpha=0.3 (recent results weighted more)
    outcome = 1.0 if succeeded else 0.0
    episode.effectiveness_score = (
        0.7 * episode.effectiveness_score + 0.3 * outcome
    )
```

**Aggregate metrics:**

```python
@dataclass
class StrategyMetrics:
    pass_rate: float                    # Tasks passed / total
    avg_attempts_to_pass: float         # Mean attempts when successful
    avg_final_score: float              # Mean evaluation score
    improvement_curve: list[float]      # Score by attempt number
    per_task_results: list[TaskResult]  # Detailed results
```

**Benchmark comparison:**

```python
# Compare strategies over same task suite
results = {
    "baseline": run_benchmark(NoImprovementStrategy()),
    "reflection": run_benchmark(ReflectionStrategy()),
    "episodic": run_benchmark(EpisodicMemoryStrategy()),
}

# Key metrics
for name, metrics in results.items():
    print(f"{name}: {metrics.pass_rate:.1%} pass rate, "
          f"{metrics.avg_attempts_to_pass:.1f} avg attempts")
```

### Preventing Bad Improvements

**Problem:** An episode might encode a wrong pattern that hurts future tasks.

**Solution 1: Effectiveness threshold**
```python
def retrieve_episodes(keywords: list[str]) -> list[Episode]:
    candidates = find_similar(keywords)

    # Filter out ineffective episodes
    return [
        ep for ep in candidates
        if ep.effectiveness_score > 0.3  # Threshold
        or ep.times_applied < 3           # Not enough data yet
    ]
```

**Solution 2: Decay over time**
```python
def decay_old_episodes():
    for episode in all_episodes():
        age_days = (now() - episode.created_at).days
        if age_days > 30:
            episode.effectiveness_score *= 0.95  # Gradual decay
```

**Solution 3: Negative feedback**
```python
# If applying an episode makes things worse
if new_score < previous_score and episode_was_applied:
    episode.effectiveness_score *= 0.5  # Penalize
```

**Solution 4: Human curation**
```
storage/episodes/
  task_001/
    a1b2c3d4.json         # Active
    e5f6g7h8.deprecated   # Manually disabled
```

**Solution 5: Versioned snapshots**
```python
# Can rollback to known-good state
storage.create_snapshot("before_experiment")
# ... run experiment ...
if results_bad:
    storage.restore_snapshot("before_experiment")
```

---

## 5. Production Considerations

### What Would Need to Change

| Component | Current State | Production Requirement |
|-----------|---------------|------------------------|
| Storage | File JSON | PostgreSQL + Redis cache |
| Sandbox | Subprocess | Kubernetes jobs / Firecracker |
| Auth | None | API keys + rate limiting |
| Logging | Print statements | Structured logging + APM |
| Monitoring | None | Prometheus + Grafana |
| Episodes | Local files | Centralized episode store |
| LLM calls | Direct API | Queue + retry + circuit breaker |

**Architecture changes:**

```
Current (Monolithic):
  User → AgentRunner → [all components in-process]

Production (Distributed):
  User → API Gateway → Task Queue → Worker Pool
                           │
                           ├── Code Executor (isolated pods)
                           ├── Episode Store (PostgreSQL)
                           ├── LLM Gateway (rate-limited)
                           └── Metrics Collector (Prometheus)
```

### Scalability Concerns

**1. Episode storage scaling:**
```
Problem: 10K+ episodes = slow keyword search
Solution:
  - PostgreSQL with GIN indexes on keywords
  - OR vector embeddings + approximate nearest neighbor
  - OR Redis for hot episodes, cold storage for archive
```

**2. LLM API rate limits:**
```
Problem: Concurrent users hit API limits
Solution:
  - Request queue with backpressure
  - Multiple API keys in rotation
  - Caching of common prompts/responses
```

**3. Code execution isolation:**
```
Problem: Many concurrent code executions
Solution:
  - Kubernetes with pod-per-execution
  - Firecracker microVMs (AWS Lambda style)
  - Pre-warmed container pool
```

**4. Cross-session learning at scale:**
```
Problem: User A's learnings shouldn't leak to User B
Solution:
  - Tenant-isolated episode stores
  - OR: Global episodes with privacy filtering
  - OR: Federated learning (aggregate patterns, not specific data)
```

### Security Considerations

**1. Code execution risks:**
```
Threat: LLM generates malicious code
Mitigations:
  - Never run in host process (always sandbox)
  - Network disabled by default
  - No filesystem access beyond data
  - Resource limits enforced at OS level
  - Audit logging of all executed code
```

**2. Prompt injection:**
```
Threat: Malicious data in CSV triggers harmful code gen
Mitigations:
  - Sanitize data before including in prompts
  - Separate user data from system prompts
  - Monitor for unusual code patterns
```

**3. Episode poisoning:**
```
Threat: Attacker creates episodes with malicious patterns
Mitigations:
  - Effectiveness threshold filters bad episodes
  - Human review for high-impact episodes
  - Anomaly detection on episode content
```

**4. Data privacy:**
```
Threat: Episodes contain sensitive data from queries
Mitigations:
  - PII detection and scrubbing
  - Episode retention policies
  - Encryption at rest
  - Access controls per tenant
```

---

## 6. AI Tool Usage

### Tools Used

**1. Claude (via Anthropic API)**
- **Code generation:** LLMExecutor uses Claude to generate Python code from queries
- **Evaluation:** LLMEvaluator uses Claude to assess correctness and classify errors
- **Hint generation:** Evaluator generates actionable hints for improvement

**2. Claude Code (CLI)**
- **Development assistance:** Used throughout development for code generation, debugging, and refactoring
- **Architecture exploration:** Used to understand and navigate the codebase
- **Documentation:** Assisted in writing this design document

### How AI Helped

**Accelerated development:**
- Boilerplate code generation (dataclasses, protocols)
- Test case generation
- Documentation writing
- Code review and bug detection

**Pattern recognition:**
- Suggested design patterns (Strategy, Protocol)
- Identified missing edge cases
- Proposed evaluation metrics

**Debugging assistance:**
- Traced through execution flow
- Identified type mismatches
- Suggested fixes for test failures

### AI Tool Limitations

**Over-engineering:**
- Initial suggestions included unnecessary abstractions
- Some generated code required simplification

**Context limits:**
- Large codebase required chunked exploration
- Some suggestions missed cross-file dependencies

**Hallucinated APIs:**
- Occasionally suggested non-existent library functions
- Required verification against actual documentation

This project was developed using a combination of AI-assisted development tools. Different strategies for improvement between sessions could be explored more thoroughly with additional compute resources.

### Implementation Breakdown

| Component | Implementation Approach | Notes |
|-----------|------------------------|-------|
| Core architecture | Manual design | Agent loop, strategy pattern |
| Type definitions | AI-assisted | Refined through iteration |
| LLMExecutor | AI-generated, refined | Prompt engineering was iterative |
| Sandbox | Manual design | Security-critical component |
| Episode similarity | AI-suggested, implemented | Keyword extraction refined |
| Benchmark framework | Collaborative | Structure designed manually, details by AI |
| Tests | AI-generated | Reviewed for correctness |
| Documentation | Collaborative | Structure designed manually, content by AI |

**Key insight:** AI excels at generating boilerplate and suggesting patterns, but security-critical components (sandbox, code execution) and architectural decisions benefited from manual oversight.

---

## Appendix: Key Files Reference

| File | Purpose |
|------|---------|
| `src/agent/runner.py` | Main agentic loop orchestration |
| `src/executor/llm.py` | LLM-based code generation |
| `src/evaluator/llm.py` | LLM-based evaluation with error classification |
| `src/strategies/episodic_memory.py` | Cross-session learning implementation |
| `src/sandbox/runner.py` | Code execution sandbox |
| `src/storage/file.py` | File-based persistence |
| `src/benchmark/runner.py` | Benchmark execution framework |
| `src/core/types.py` | Core data types |
| `src/core/protocols.py` | Interface definitions (ABCs) |
