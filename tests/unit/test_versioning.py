"""Tests for ImprovementVersion dataclass."""

from src.strategies.versioning import ImprovementVersion
from src.strategies.episodic_memory import Episode


class TestImprovementVersion:
    """Tests for ImprovementVersion."""

    def test_creation_defaults(self) -> None:
        """Test creating a version with defaults."""
        version = ImprovementVersion()

        assert version.version_id
        assert version.name == ""
        assert version.created_at
        assert version.author == ""
        assert version.parent_version_id is None
        assert version.tags == []
        assert version.strategy_type == ""
        assert version.strategy_config == {}
        assert version.episodes == []
        assert version.metrics == {}
        assert version.metadata == {}

    def test_creation_with_data(self) -> None:
        """Test creating a version with data."""
        episode = Episode(
            query="Test query", failed_code="bad", error_message="error", version="1.0"
        )

        version = ImprovementVersion(
            name="Test Version",
            description="A test version",
            author="tester",
            strategy_type="episodic_memory",
            strategy_config={"param": "value"},
            episodes=[episode],
            metrics={"accuracy": 0.95},
            tags=["test", "improvement"],
            metadata={"priority": "high"},
        )

        assert version.name == "Test Version"
        assert version.description == "A test version"
        assert version.author == "tester"
        assert version.strategy_type == "episodic_memory"
        assert version.strategy_config == {"param": "value"}
        assert len(version.episodes) == 1
        assert version.metrics == {"accuracy": 0.95}
        assert version.tags == ["test", "improvement"]
        assert version.metadata == {"priority": "high"}

    def test_properties(self) -> None:
        """Test computed properties."""
        episode1 = Episode(query="q1", failed_code="f1", error_message="e1")
        episode2 = Episode(query="q2", failed_code="f2", error_message="e2")

        version = ImprovementVersion(episodes=[episode1, episode2])
        assert version.episode_count == 2

        version_no_parent = ImprovementVersion()
        assert not version_no_parent.has_parent

        version_with_parent = ImprovementVersion(parent_version_id="parent-1")
        assert version_with_parent.has_parent

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization."""
        episode = Episode(
            query="Test query",
            failed_code="bad code",
            error_message="error msg",
            version="1.0",
            tags=["test"],
        )

        original = ImprovementVersion(
            name="Test Version",
            description="Description",
            author="author",
            strategy_type="episodic_memory",
            strategy_config={"key": "value"},
            episodes=[episode],
            metrics={"score": 0.9},
            tags=["tag1"],
            metadata={"meta": "data"},
        )

        data = original.to_dict()
        restored = ImprovementVersion.from_dict(data)

        assert restored.version_id == original.version_id
        assert restored.name == "Test Version"
        assert restored.description == "Description"
        assert restored.author == "author"
        assert restored.strategy_type == "episodic_memory"
        assert restored.strategy_config == {"key": "value"}
        assert len(restored.episodes) == 1
        assert restored.episodes[0].query == "Test query"
        assert restored.episodes[0].tags == ["test"]
        assert restored.metrics == {"score": 0.9}
        assert restored.tags == ["tag1"]
        assert restored.metadata == {"meta": "data"}

    def test_create_snapshot(self) -> None:
        """Test creating a snapshot."""
        episodes = [
            Episode(query="q1", failed_code="f1", error_message="e1"),
            Episode(query="q2", failed_code="f2", error_message="e2"),
        ]

        version = ImprovementVersion.create_snapshot(
            strategy_type="episodic_memory",
            strategy_config={"threshold": 0.8},
            episodes=episodes,
            metrics={"accuracy": 0.85},
            name="Snapshot 1",
            author="system",
        )

        assert version.strategy_type == "episodic_memory"
        assert version.strategy_config == {"threshold": 0.8}
        assert len(version.episodes) == 2
        assert version.metrics == {"accuracy": 0.85}
        assert version.name == "Snapshot 1"
        assert version.author == "system"
        assert version.parent_version_id is None

    def test_create_child_version(self) -> None:
        """Test creating a child version."""
        parent = ImprovementVersion(
            version_id="parent-123",
            strategy_type="reflection",
            episodes=[Episode(query="q", failed_code="f", error_message="e")],
            metrics={"score": 0.8},
        )

        child = parent.create_child_version(
            metrics={"score": 0.9}, name="Child Version"
        )

        assert child.parent_version_id == "parent-123"
        assert child.strategy_type == "reflection"
        assert child.episodes == parent.episodes  # Same episodes
        assert child.metrics == {"score": 0.9}  # Updated metrics
        assert child.name == "Child Version"
        assert child.version_id != parent.version_id

    def test_get_episode_by_query(self) -> None:
        """Test finding episodes by query."""
        episodes = [
            Episode(query="What is revenue?", failed_code="f1", error_message="e1"),
            Episode(query="What is profit?", failed_code="f2", error_message="e2"),
        ]

        version = ImprovementVersion(episodes=episodes)

        found = version.get_episode_by_query("What is revenue?")
        assert found is not None
        assert found.failed_code == "f1"

        not_found = version.get_episode_by_query("Unknown query")
        assert not_found is None

    def test_get_episodes_by_tag(self) -> None:
        """Test filtering episodes by tag."""
        episodes = [
            Episode(
                query="q1", failed_code="f1", error_message="e1", tags=["math", "basic"]
            ),
            Episode(query="q2", failed_code="f2", error_message="e2", tags=["finance"]),
            Episode(
                query="q3",
                failed_code="f3",
                error_message="e3",
                tags=["math", "advanced"],
            ),
        ]

        version = ImprovementVersion(episodes=episodes)

        math_episodes = version.get_episodes_by_tag("math")
        assert len(math_episodes) == 2
        assert math_episodes[0].query == "q1"
        assert math_episodes[1].query == "q3"

        finance_episodes = version.get_episodes_by_tag("finance")
        assert len(finance_episodes) == 1
        assert finance_episodes[0].query == "q2"

        unknown_episodes = version.get_episodes_by_tag("unknown")
        assert unknown_episodes == []

    def test_summarize(self) -> None:
        """Test getting a summary."""
        version = ImprovementVersion(
            version_id="v123",
            name="Test Version",
            author="tester",
            strategy_type="episodic_memory",
            episodes=[Episode(query="q", failed_code="f", error_message="e")],
            metrics={"accuracy": 0.95},
            tags=["test"],
        )

        summary = version.summarize()

        assert summary["version_id"] == "v123"
        assert summary["name"] == "Test Version"
        assert summary["strategy_type"] == "episodic_memory"
        assert summary["episode_count"] == 1
        assert summary["author"] == "tester"
        assert summary["tags"] == ["test"]
        assert summary["metrics"] == {"accuracy": 0.95}
