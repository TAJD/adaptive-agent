# Project Development Log

This document tracks the development of the self-improving agent framework.

## Research Phase

Researched existing self-evolving agent architectures and patterns from academic papers and industry implementations.

## Game Plan

- Keep system diagrams
- Meta problem: create evaluation harness for self-improving agents with ability to swap out strategies for improvement
- Implement two simple self-improvement strategies between sessions and then review
- Use AI-assisted development tools for agent orchestration

Stretch - abstract on top of models:

  Current design: - Executor protocol: Takes Task → returns ExecutionResult (includes code generation + execution)

  Proposed abstraction:
  ┌─────────────────────────────────────────────────┐
  │           LLMClient Protocol                    │
  │  (abstracts chat completions across providers)  │
  ├─────────────────────────────────────────────────┤
  │  OllamaClient │ ClaudeClient │ OpenAIClient     │
  └─────────────────────────────────────────────────┘
                        │
                        ▼
  ┌─────────────────────────────────────────────────┐
  │              LLMExecutor                        │
  │  (uses any LLMClient to generate & run code)    │
  └─────────────────────────────────────────────────┘

---

## Final Implementation Architecture

### Implemented Components

```
src/
├── core/
│   ├── types.py          # Task, ExecutionResult, Evaluation, ImprovementContext
│   └── protocols.py      # Executor, Evaluator, ImprovementStrategy, Storage protocols
├── data/
│   └── dataset.py        # DatasetLoader with schema validation
├── prompts/
│   └── data_analysis.py  # Configurable prompt template for financial queries
├── executor/
│   ├── code_runner.py    # Safe Python execution with sandboxing
│   ├── llm.py            # LLMExecutor: prompt → code → execution
│   └── mock.py           # MockExecutor for testing
├── evaluator/
│   ├── exact_match.py    # Numeric tolerance, type coercion
│   └── llm.py            # LLMEvaluator with error classification
├── strategies/
│   ├── none.py           # NoImprovementStrategy (baseline)
│   ├── reflection.py     # ReflectionStrategy (in-session)
│   └── episodic_memory.py # EpisodicMemoryStrategy (cross-session)
├── storage/
│   ├── memory.py         # InMemoryStorage
│   └── file.py           # FileStorage (JSON-based persistence)
├── llm/
│   ├── protocol.py       # LLMClient protocol
│   ├── mock.py           # MockLLMClient for testing
│   └── claude.py         # ClaudeClient (Anthropic API)
├── agent/
│   └── runner.py         # AgentRunner orchestration
└── benchmark/
    ├── runner.py         # BenchmarkRunner, BenchmarkConfig
    ├── metrics.py        # MetricsCollector
    └── tasks.py          # Task suite management
```

### Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  AgentRunner    │
│  (orchestrator) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  load_priors()  │────▶│ ImprovementStrategy │
│  (if available) │     │ (episodic_memory)   │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│    Executor     │────▶│  CodeRunner      │
│  (LLM → code)   │     │  (sandboxed exec)│
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Evaluator     │────▶│  Score + Error   │
│ (compare answer)│     │  Classification  │
└────────┬────────┘     └──────────────────┘
         │
    Pass? ──Yes──▶ Done
         │
        No
         │
         ▼
┌─────────────────┐
│   improve()     │──────▶ Retry (up to max_attempts)
│ (get hints/examples)
└────────┬────────┘
         │
         ▼
    persist() ──────▶ Store episode for future sessions
```

### Self-Improvement Mechanism: EpisodicMemoryStrategy

The key innovation is storing concrete failure/fix pairs rather than abstract patterns.

**Episode Structure:**
```python
@dataclass
class Episode:
    query: str              # "What was Gross Revenue for Product A in Q1 2023?"
    failed_code: str        # df[df['Product'] == 'Product A'].sum()
    error_message: str      # "Value too high, missing time filters"
    fixed_code: str | None  # df[(Product A) & (Q1) & (2023) & (Gross Revenue)].sum()
    keywords: list[str]     # ["revenue", "gross revenue", "product a", "q1", "2023"]
    task_id: str
```

**Keyword-Based Similarity:**
- Extract financial terms: revenue, cogs, opex, margin, etc.
- Extract time terms: q1-q4, 2020-2024, year-over-year
- Extract entity terms: product a-d, country names
- Extract aggregation terms: total, sum, average, max
- Jaccard similarity on keyword sets

**Learning Cycle:**
1. Agent fails on query → create Episode with failed_code
2. Agent succeeds → update Episode with fixed_code
3. New query arrives → extract keywords → find similar episodes
4. Inject fixed_code as example in prompt → better first attempt

### Code Execution Strategy

**Sandboxed Python Execution:**
- CodeRunner executes in isolated namespace
- Allowed modules: pandas, math, statistics, json
- Pre-injected: DataFrame `df` with financial data
- Output captured via `result` variable

**Why Python (not SQL):**
- More expressive for complex calculations (margins, YoY growth)
- Easier to debug and explain generated logic
- Natural fit for pandas-based data manipulation
- Can handle edge cases with Python conditionals

### Evaluation Strategy

**Two-Tier Approach:**

1. **ExactMatchEvaluator** (fast, deterministic)
   - Numeric tolerance: 1e-6 absolute, 1e-4 relative
   - String normalization: whitespace, case
   - Type coercion: str(expected) == str(actual)
   - Error classification: no_output, type_mismatch, numeric_error, etc.

2. **LLMEvaluator** (nuanced, uses Claude)
   - Semantic comparison for edge cases
   - Detailed error classification via tool calling
   - Handles "equivalent but different" answers
   - Fallback to exact match if API fails

### Production Considerations

**1. Scalability**
- FileStorage works for single-user, local persistence
- For multi-user: swap to Redis/PostgreSQL via Storage protocol
- Episode retrieval is O(n) - use vector DB for large episode counts

**2. Safety**
- Code execution is sandboxed (limited imports)
- No file system access from generated code
- Timeout on code execution (30s default)
- LLM output is never executed without extraction

**3. Cost Management**
- EpisodicMemory reduces LLM calls by improving first-attempt success
- ExactMatchEvaluator avoids LLM call when answer is clearly right/wrong
- Configurable max_attempts limits retry costs

**4. Observability**
- All execution steps recorded in trajectory
- MetricsCollector tracks pass rates, attempt counts, scores
- BenchmarkRunner enables A/B testing of strategies

**5. Limitations**
- Keyword similarity is simple - could miss semantic relationships
- No embedding-based retrieval (would require vector DB)
- Single-turn queries only (no multi-turn conversation state)

### Test Coverage

```
141 tests across:
- Unit tests: types, storage, strategies, evaluators, prompts
- Integration tests: agent loop with mock executor
- E2E tests: cross-session learning, benchmark runner
```

### Demo Scripts

| Script | Purpose |
|--------|---------|
| `demo.py` | Basic framework demonstration |
| `demo_cross_session.py` | Cross-session learning proof |
| `chat.py` | Interactive CLI chatbot |

### Scaling Analysis: `load_priors` and `persist`

**How `persist` saves data:**
```
persist(context)
  └─► _save_episode(episode)
        └─► storage.save(key, episode.to_dict())
              └─► FileStorage: write JSON to disk
                  Key format: "episodes/{task_id}/{query_hash}.json"
```
- O(1) operation - single file write
- ~1KB per episode (query + code + keywords)
- Scales linearly with disk space

**How `load_priors` retrieves data:**
```
load_priors(task)
  └─► _find_similar_episodes(pseudo_episode)
        ├─► storage.list_keys("episodes")     # O(n) - glob all files
        └─► for each key:                      # O(n) iterations
              ├─► storage.load(key)            # O(1) - read JSON file
              ├─► Episode.from_dict(data)      # O(1) - deserialize
              └─► compute_similarity()         # O(k) - k=keyword count
```

**Complexity:** O(n × k) where n=episodes, k=keywords per episode

**When scaling breaks down:**

| Episodes | Latency (est.) | Status |
|----------|----------------|--------|
| ~100     | <50ms          | ✅ Fine |
| ~1,000   | 100-500ms      | ⚠️ Noticeable |
| ~10,000  | 1-5s           | ❌ Problematic |
| ~100,000 | 10-60s         | ❌ Unusable |

**Bottlenecks:**
1. **File system glob** - `list_keys()` scans entire directory
2. **No indexing** - brute force similarity on all episodes
3. **No caching** - episodes loaded fresh every call
4. **Single directory** - file systems slow down with 10K+ files

**Why keyword matching doesn't scale semantically:**
- "revenue" won't match "income" (no synonyms)
- "Q1 2023" won't match "first quarter of twenty-three"
- Jaccard similarity requires exact keyword overlap

**Production solutions:**
1. **Vector database** (Pinecone, Chroma, pgvector) - O(log n) similarity search
2. **Embedding-based retrieval** - semantic similarity, not keyword matching
3. **In-memory index** - load episodes once at startup
4. **Sharded storage** - partition by time or keyword cluster
5. **LRU cache** - cache recent episode lookups

**Current design is appropriate for:**
- Single user, local development
- <1000 episodes
- Proof of concept / demo

### Future Extensions

1. **Embedding-based retrieval** - Replace keyword matching with semantic embeddings
2. **Multi-turn conversations** - Add conversation history to context
3. **Prompt evolution** - Implement PromptEvolutionStrategy from design
4. **Skill extraction** - Identify reusable code patterns
5. **A/B testing infrastructure** - Production traffic splitting between strategies
