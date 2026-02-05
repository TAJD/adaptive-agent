"""Versioning system for improvement strategies."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List

from .episodic_memory import Episode


@dataclass
class ImprovementVersion:
    """
    A snapshot/version of an improvement strategy at a point in time.

    Captures the complete state of a strategy including configuration,
    learned episodes, and performance metrics.
    """

    # Identity
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Versioning
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    parent_version_id: str | None = None
    tags: List[str] = field(default_factory=list)

    # Strategy state
    strategy_type: str = ""  # e.g., "episodic_memory", "reflection"
    strategy_config: dict[str, Any] = field(default_factory=dict)
    episodes: List[Episode] = field(default_factory=list)

    # Performance metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def episode_count(self) -> int:
        """Get the number of episodes in this version."""
        return len(self.episodes)

    @property
    def has_parent(self) -> bool:
        """Check if this version has a parent."""
        return self.parent_version_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version_id": self.version_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "author": self.author,
            "parent_version_id": self.parent_version_id,
            "tags": self.tags.copy(),
            "strategy_type": self.strategy_type,
            "strategy_config": self.strategy_config.copy(),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "metrics": self.metrics.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementVersion":
        """Create from dictionary."""
        # Convert episode dicts back to Episode objects
        episodes = [Episode.from_dict(ep) for ep in data.get("episodes", [])]

        return cls(
            version_id=data["version_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"],
            author=data["author"],
            parent_version_id=data.get("parent_version_id"),
            tags=data.get("tags", []),
            strategy_type=data["strategy_type"],
            strategy_config=data.get("strategy_config", {}),
            episodes=episodes,
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create_snapshot(
        cls,
        strategy_type: str,
        strategy_config: dict[str, Any],
        episodes: List[Episode],
        metrics: dict[str, Any] | None = None,
        parent_version_id: str | None = None,
        **kwargs: Any,
    ) -> "ImprovementVersion":
        """
        Create a new version snapshot.

        Args:
            strategy_type: Type of strategy (e.g., "episodic_memory")
            strategy_config: Current strategy configuration
            episodes: List of learned episodes
            metrics: Performance metrics
            parent_version_id: ID of parent version
            **kwargs: Additional fields (name, description, author, etc.)
        """
        return cls(
            strategy_type=strategy_type,
            strategy_config=strategy_config,
            episodes=episodes,
            metrics=metrics or {},
            parent_version_id=parent_version_id,
            **kwargs,
        )

    def create_child_version(
        self,
        episodes: List[Episode] | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ImprovementVersion":
        """
        Create a new version based on this one.

        Args:
            episodes: Updated episodes (defaults to current)
            metrics: Updated metrics (defaults to current)
            **kwargs: Fields to override
        """
        return self.create_snapshot(
            strategy_type=self.strategy_type,
            strategy_config=self.strategy_config,
            episodes=episodes if episodes is not None else self.episodes,
            metrics=metrics if metrics is not None else self.metrics,
            parent_version_id=self.version_id,
            **kwargs,
        )

    def get_episode_by_query(self, query: str) -> Episode | None:
        """Find an episode by query."""
        for episode in self.episodes:
            if episode.query == query:
                return episode
        return None

    def get_episodes_by_tag(self, tag: str) -> List[Episode]:
        """Get episodes with a specific tag."""
        return [ep for ep in self.episodes if tag in ep.tags]

    def summarize(self) -> dict[str, Any]:
        """Get a summary of this version."""
        return {
            "version_id": self.version_id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "episode_count": self.episode_count,
            "created_at": self.created_at,
            "author": self.author,
            "parent_version": self.parent_version_id,
            "tags": self.tags,
            "metrics": self.metrics,
        }
