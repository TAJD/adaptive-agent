"""Episodic memory strategy for cross-session learning.

This strategy stores specific failure episodes (query + failed code + fix)
and retrieves relevant past experiences when encountering similar queries.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from src.core.types import Task, ImprovementContext
from src.core.protocols import Storage


@dataclass
class Episode:
    """A single learning episode from a failure and its resolution."""

    query: str
    failed_code: str
    error_message: str
    fixed_code: str | None = None
    error_type: str | None = None
    keywords: list[str] = field(default_factory=list)
    task_id: str = ""
    timestamp: str = ""

    # Versioning metadata
    version: str = "1.0"
    parent_version: str | None = None
    change_description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Effectiveness tracking
    effectiveness_score: float = (
        0.0  # Exponential moving average: 0.7*old + 0.3*(1.0 if success else 0.0)
    )
    times_applied: int = 0
    times_succeeded: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create from dictionary with backward compatibility for older episodes."""
        # Ensure backward compatibility with older stored episodes
        data.setdefault("fixed_code", None)
        data.setdefault("error_type", None)
        data.setdefault("keywords", [])
        data.setdefault("task_id", "")
        data.setdefault("timestamp", "")
        data.setdefault("version", "1.0")
        data.setdefault("parent_version", None)
        data.setdefault("change_description", "")
        data.setdefault("author", "")
        data.setdefault("tags", [])
        data.setdefault("metadata", {})
        data.setdefault("effectiveness_score", 0.0)
        data.setdefault("times_applied", 0)
        data.setdefault("times_succeeded", 0)
        return cls(**data)


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from query text for similarity matching.

    Focuses on:
    - Financial terms (revenue, COGS, margin, etc.)
    - Product/country/time references
    - Aggregation terms (total, sum, average, etc.)
    - Comparison terms (growth, change, vs, between)
    - Falls back to word-based matching if no domain keywords found
    """
    # Lowercase for matching
    text_lower = text.lower()

    # Financial terms
    financial_terms = [
        "revenue",
        "gross revenue",
        "net revenue",
        "cogs",
        "cost of goods",
        "opex",
        "operating expense",
        "margin",
        "profit",
        "income",
        "expense",
        "labor",
        "materials",
        "overhead",
        "marketing",
        "r&d",
        "sales",
        "g&a",
        "administrative",
        "headcount",
        "interest",
        "foreign exchange",
        "fx",
    ]

    # Time terms
    time_terms = [
        "q1",
        "q2",
        "q3",
        "q4",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "year",
        "quarter",
        "month",
        "yoy",
        "year-over-year",
    ]

    # Product/country terms
    entity_terms = [
        "product a",
        "product b",
        "product c",
        "product d",
        "australia",
        "canada",
        "germany",
        "japan",
        "united kingdom",
        "united states",
        "global",
        "all countries",
        "all products",
    ]

    # Aggregation terms
    agg_terms = [
        "total",
        "sum",
        "average",
        "mean",
        "count",
        "max",
        "min",
        "highest",
        "lowest",
        "top",
        "bottom",
    ]

    # Comparison terms
    comp_terms = [
        "growth",
        "change",
        "increase",
        "decrease",
        "compare",
        "vs",
        "versus",
        "between",
        "difference",
        "ratio",
        "percentage",
    ]

    all_terms = financial_terms + time_terms + entity_terms + agg_terms + comp_terms

    keywords = []
    for term in all_terms:
        if term in text_lower:
            keywords.append(term)

    # If no domain keywords found, fall back to extracting individual words (min 4 chars)
    if not keywords:
        words = re.findall(r'\b\w{4,}\b', text_lower)
        keywords = list(set(words))  # Deduplicate

    return keywords


def compute_similarity(keywords1: list[str], keywords2: list[str]) -> float:
    """Compute Jaccard similarity between two keyword sets."""
    if not keywords1 or not keywords2:
        return 0.0

    set1 = set(keywords1)
    set2 = set(keywords2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


class EpisodicMemoryStrategy:
    """
    Learning strategy that stores and retrieves failure episodes.

    This strategy implements true cross-session learning by:
    1. Storing each failure as an "episode" with query, code, and error
    2. When the same task succeeds, updating the episode with the fix
    3. On new queries, finding similar past episodes via keyword matching
    4. Injecting relevant fixes as examples into the prompt

    Unlike ReflectionStrategy which stores abstract patterns,
    this stores concrete examples for more precise learning.
    """

    STORAGE_PREFIX = "episodes"
    MAX_EPISODES_TO_RETRIEVE = 3
    MIN_SIMILARITY_THRESHOLD = 0.2

    def __init__(self, storage: Storage | None = None) -> None:
        """
        Initialize the episodic memory strategy.

        Args:
            storage: Storage for cross-session persistence.
                    Required for this strategy to be effective.
        """
        self.storage = storage
        self._session_episodes: dict[str, Episode] = {}
        self._pending_failures: dict[str, Episode] = {}
        self._applied_episodes: dict[
            str, list[Episode]
        ] = {}  # Track episodes applied in current session

    def improve(self, context: ImprovementContext) -> dict:
        """
        Generate improvements based on the current failure and past episodes.

        Returns hints, constraints, and examples from similar past failures.
        Uses multiple matching strategies:
        1. Keyword-based similarity (query content)
        2. Error-type matching (same type of mistake)
        3. Hints from the evaluator
        """
        hints: list[str] = []
        examples: list[str] = []

        # Create episode from current failure
        current_episode = self._create_episode(context)
        self._pending_failures[context.task.id] = current_episode

        # Include hints from the evaluator if available
        if context.evaluation.hints:
            hints.extend(context.evaluation.hints[:2])

        # Try to find similar past episodes by query similarity
        if self.storage:
            similar_episodes = self._find_similar_episodes(current_episode)

            for episode in similar_episodes:
                if episode.fixed_code:
                    # Add the fixed code as an example
                    examples.append(episode.fixed_code)
                    hints.append(
                        f"A similar query '{episode.query[:50]}...' was fixed using this approach."
                    )
                else:
                    # Even without a fix, we know what didn't work
                    hints.append(
                        f"Similar query failed with: {episode.error_message[:100]}"
                    )

            # Also find episodes with same error type (different matching strategy)
            if context.evaluation.error_type:
                error_matches = self._find_similar_by_error(context.evaluation.error_type)
                for episode in error_matches:
                    if episode.fixed_code and episode.fixed_code not in examples:
                        examples.append(episode.fixed_code)
                        hints.append(
                            f"Same error type was fixed before (effectiveness: {episode.effectiveness_score:.0%})."
                        )

        # Add error-specific hints (if not already covered by evaluator hints)
        if context.evaluation.error_type and not context.evaluation.hints:
            hints.extend(self._hints_for_error(context.evaluation.error_type))

        # Track score progression
        if context.history:
            last_score = context.history[-1][1].score
            if context.evaluation.score > last_score:
                hints.append("Progress detected - continue this direction.")
            elif len(context.history) >= 2:
                hints.append("Consider a different approach.")

        # Deduplicate hints while preserving order
        seen = set()
        unique_hints = []
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                unique_hints.append(hint)

        return {
            "hints": unique_hints[:6],  # Limit to 6 hints
            "examples": examples[:3],  # Limit to 3 examples
            "attempt": context.attempt_number + 1,
            "previous_score": context.evaluation.score,
            "error_type": context.evaluation.error_type,
        }

    def persist(self, context: ImprovementContext) -> None:
        """
        Persist the episode for cross-session learning.

        If the task passed, we update any pending failure with the fix
        and update effectiveness scores for applied episodes.
        If the task failed, we store the failure episode.
        """
        if self.storage is None:
            return

        task_id = context.task.id

        # Check if this attempt succeeded
        if context.evaluation.passed:
            # Update any pending failure with the successful code
            if task_id in self._pending_failures:
                episode = self._pending_failures[task_id]
                episode.fixed_code = context.result.code_generated
                self._save_episode(episode)
                del self._pending_failures[task_id]

            # Update effectiveness scores for applied episodes
            self._update_episode_effectiveness(task_id, success=True)
        else:
            # Store the failure episode
            episode = self._create_episode(context)
            self._save_episode(episode)

            # Update effectiveness scores for applied episodes (failure)
            self._update_episode_effectiveness(task_id, success=False)

    def load_priors(self, task: Task) -> dict:
        """
        Load prior learnings relevant to this task.

        Searches for episodes with similar queries and returns
        any fixes as examples. Records applied episodes for effectiveness tracking.
        """
        if self.storage is None:
            return {}

        # Create a pseudo-episode for similarity matching
        query_keywords = extract_keywords(task.query)
        pseudo_episode = Episode(
            query=task.query,
            failed_code="",
            error_message="",
            keywords=query_keywords,
            task_id=task.id,
        )

        similar_episodes = self._find_similar_episodes(pseudo_episode)

        hints: list[str] = []
        examples: list[str] = []
        applied_episodes: list[Episode] = []

        for episode in similar_episodes:
            if episode.fixed_code:
                examples.append(episode.fixed_code)
                hints.append(
                    f"Similar past query was solved. Study the example pattern."
                )
                applied_episodes.append(episode)

        # Record applied episodes for effectiveness tracking
        self._applied_episodes[task.id] = applied_episodes

        return {
            "hints": hints[:5],
            "examples": examples[:3],
        }

    def _create_episode(self, context: ImprovementContext) -> Episode:
        """Create an episode from the current failure context."""
        keywords = extract_keywords(context.task.query)

        return Episode(
            query=context.task.query,
            failed_code=context.result.code_generated or "",
            error_message=context.evaluation.feedback,
            error_type=context.evaluation.error_type,
            keywords=keywords,
            task_id=context.task.id,
            timestamp=str(int(time.time() * 1000)),  # Millisecond timestamp for uniqueness
        )

    def _save_episode(self, episode: Episode) -> None:
        """Save an episode to storage."""
        if self.storage is None:
            return

        # Generate a unique key based on query hash and timestamp
        # Including timestamp ensures multiple episodes from different runs don't overwrite
        query_hash = hashlib.md5(episode.query.encode()).hexdigest()[:8]
        timestamp = episode.timestamp or str(int(time.time() * 1000))
        key = f"{self.STORAGE_PREFIX}/{episode.task_id}/{query_hash}/{timestamp}"

        self.storage.save(key, episode.to_dict())

    def _find_similar_episodes(self, target: Episode) -> list[Episode]:
        """Find episodes similar to the target based on keyword overlap."""
        if self.storage is None:
            return []

        # Get all episode keys
        all_keys = self.storage.list_keys(self.STORAGE_PREFIX)

        candidates: list[tuple[float, Episode]] = []

        for key in all_keys:
            data = self.storage.load(key)
            if not data:
                continue

            episode = Episode.from_dict(data)

            # Skip if it's the same task (we don't want to match ourselves)
            # Use equality check instead of substring matching to avoid false positives
            # (e.g., "task-1" incorrectly matching "task-10")
            if target.task_id and episode.task_id == target.task_id:
                continue

            # Compute similarity
            similarity = compute_similarity(target.keywords, episode.keywords)

            if similarity >= self.MIN_SIMILARITY_THRESHOLD:
                candidates.append((similarity, episode))

        # Sort by similarity (descending) and return top matches
        candidates.sort(key=lambda x: x[0], reverse=True)

        return [ep for _, ep in candidates[: self.MAX_EPISODES_TO_RETRIEVE]]

    def _hints_for_error(self, error_type: str) -> list[str]:
        """Generate hints based on error type."""
        hints_map = {
            "no_output": [
                "Ensure the code sets a 'result' variable.",
                "Check for syntax errors that prevent execution.",
            ],
            "none_result": [
                "Result is None - check if 'result' variable is assigned.",
                "Verify your filter returns rows (df may be empty after filtering).",
            ],
            "type_mismatch": [
                "Verify the expected output type (number vs string).",
                "Check if you're returning a DataFrame instead of a value.",
            ],
            "numeric_error": [
                "Double-check the filtering conditions.",
                "Verify you're summing the correct column.",
            ],
            "small_numeric_error": [
                "Result is close but not exact - check rounding.",
                "Verify boundary values are correctly included/excluded.",
            ],
            "large_numeric_error": [
                "Review which rows are being included.",
                "Check if you need 'Amount in USD' vs 'Amount in Local Currency'.",
            ],
            "much_larger": [
                "Result is much larger than expected - check if you're missing filters.",
                "Verify you're filtering by all required dimensions (time, product, region).",
            ],
            "much_smaller": [
                "Result is much smaller than expected - filters may be too restrictive.",
                "Check filter values match the data exactly (case sensitivity, spelling).",
            ],
            "off_by_factor": [
                "Result appears off by a power of 10 - check units.",
                "Verify you're using the correct column (USD vs Local Currency).",
            ],
            "negative_vs_positive": [
                "Sign is wrong - check if you need to negate the result.",
                "Verify whether the calculation should use subtraction or addition.",
            ],
            "string_mismatch": [
                "Check exact string formatting and case.",
                "Verify you're returning the right column value.",
            ],
            "execution_error": [
                "Code failed to execute - check for syntax errors.",
                "Verify column names and variable references are correct.",
            ],
        }
        return hints_map.get(error_type, ["Review the approach carefully."])

    def _find_similar_by_error(self, error_type: str) -> list[Episode]:
        """Find episodes with the same error type that were successfully fixed."""
        if self.storage is None or not error_type:
            return []

        all_keys = self.storage.list_keys(self.STORAGE_PREFIX)
        matches: list[Episode] = []

        for key in all_keys:
            data = self.storage.load(key)
            if not data:
                continue

            episode = Episode.from_dict(data)

            # Only include if same error type AND has a fix
            if episode.error_type == error_type and episode.fixed_code:
                matches.append(episode)

        # Sort by effectiveness score (most effective fixes first)
        matches.sort(key=lambda e: e.effectiveness_score, reverse=True)
        return matches[:2]  # Return top 2 matches

    def get_episode_count(self) -> int:
        """Get the number of stored episodes."""
        if self.storage is None:
            return 0
        return len(self.storage.list_keys(self.STORAGE_PREFIX))

    def _update_episode_effectiveness(self, task_id: str, success: bool) -> None:
        """
        Update effectiveness scores for episodes applied to this task.

        Uses exponential moving average: new_score = 0.7 * old_score + 0.3 * (1.0 if success else 0.0)
        """
        if self.storage is None or task_id not in self._applied_episodes:
            return

        applied_episodes = self._applied_episodes[task_id]

        for episode in applied_episodes:
            # Generate the key for this episode (must include timestamp for new format)
            query_hash = hashlib.md5(episode.query.encode()).hexdigest()[:8]
            # Use episode's timestamp if available, otherwise search for matching episode
            if episode.timestamp:
                key = f"{self.STORAGE_PREFIX}/{episode.task_id}/{query_hash}/{episode.timestamp}"
                data = self.storage.load(key)
            else:
                # Fallback: search for episode by prefix (for backward compatibility)
                prefix = f"{self.STORAGE_PREFIX}/{episode.task_id}/{query_hash}"
                matching_keys = self.storage.list_keys(prefix)
                data = None
                key = None
                for k in matching_keys:
                    data = self.storage.load(k)
                    if data:
                        key = k
                        break

            if not data or not key:
                continue

            # Update effectiveness metrics
            current_score = data.get("effectiveness_score", 0.0)
            current_applied = data.get("times_applied", 0)
            current_succeeded = data.get("times_succeeded", 0)

            # Update using exponential moving average
            success_value = 1.0 if success else 0.0
            new_score = 0.7 * current_score + 0.3 * success_value

            # Update counters
            data["effectiveness_score"] = new_score
            data["times_applied"] = current_applied + 1
            if success:
                data["times_succeeded"] = current_succeeded + 1

            # Save updated episode
            self.storage.save(key, data)

        # Clean up applied episodes for this task
        del self._applied_episodes[task_id]

    def clear_session_state(self) -> None:
        """Clear session-only state between tasks to prevent cross-task contamination."""
        self._session_episodes.clear()
        self._pending_failures.clear()
        self._applied_episodes.clear()

    def clear_episodes(self) -> None:
        """Clear all stored episodes (useful for testing)."""
        if self.storage is None:
            return

        for key in self.storage.list_keys(self.STORAGE_PREFIX):
            self.storage.delete(key)
