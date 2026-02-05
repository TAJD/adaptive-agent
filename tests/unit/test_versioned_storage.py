"""Tests for versioned storage system."""

import pytest
import tempfile
from pathlib import Path

from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage
from src.versioning.versioned_storage import VersionedStorageWrapper


class TestVersionedStorageWrapper:
    """Tests for VersionedStorageWrapper."""

    def test_save_and_load_version_memory_storage(self) -> None:
        """Test basic save and load with version using in-memory storage."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save data with version
        data = {"key": "value", "number": 42}
        version_id = versioned.save_with_version(
            "test_key", data, {"name": "test version", "description": "A test version"}
        )

        # Load the version
        loaded = versioned.load_version("test_key", version_id)
        assert loaded == data

        # Check current version
        current = versioned.get_current_version("test_key")
        assert current == version_id

    def test_list_versions(self) -> None:
        """Test listing versions."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save multiple versions
        data1 = {"version": 1}
        data2 = {"version": 2}

        import time

        v1 = versioned.save_with_version("test_key", data1, {"name": "v1"})
        time.sleep(0.01)  # Ensure different timestamps
        v2 = versioned.save_with_version("test_key", data2, {"name": "v2"})

        versions = versioned.list_versions("test_key")
        assert len(versions) == 2

        # Should be sorted by creation time (newest first)
        # Find versions by their IDs
        v1_info = next(v for v in versions if v["version_id"] == v1)
        v2_info = next(v for v in versions if v["version_id"] == v2)

        # v2 should be newer (higher index in sorted list)
        v1_index = next(i for i, v in enumerate(versions) if v["version_id"] == v1)
        v2_index = next(i for i, v in enumerate(versions) if v["version_id"] == v2)

        assert v2_index < v1_index  # v2 should be first (newest)

        # Check metadata
        assert v2_info["metadata"]["name"] == "v2"
        assert v1_info["metadata"]["name"] == "v1"

    def test_rollback_to_version(self) -> None:
        """Test rolling back to a previous version."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save versions
        data1 = {"version": 1}
        data2 = {"version": 2}

        v1 = versioned.save_with_version("test_key", data1, {"name": "v1"})
        v2 = versioned.save_with_version("test_key", data2, {"name": "v2"})

        # Current data should be v2
        current_data = storage.load("test_key")
        assert current_data == data2

        # Rollback to v1
        versioned.rollback_to("test_key", v1)

        # Current data should now be v1
        current_data = storage.load("test_key")
        assert current_data == data1

    def test_load_nonexistent_version(self) -> None:
        """Test loading a version that doesn't exist."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        with pytest.raises(KeyError):
            versioned.load_version("test_key", "nonexistent-version")

    def test_get_current_version_no_versions(self) -> None:
        """Test getting current version when no versions exist."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        current = versioned.get_current_version("test_key")
        assert current is None

    def test_list_versions_empty(self) -> None:
        """Test listing versions when none exist."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        versions = versioned.list_versions("test_key")
        assert versions == []

    def test_create_snapshot(self) -> None:
        """Test creating a snapshot as ImprovementVersion."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save some data that looks like a strategy
        strategy_data = {
            "strategy_type": "episodic_memory",
            "config": {"max_episodes": 10},
            "episodes": [{"id": "1", "task": "test"}],
        }

        version_id = versioned.save_with_version(
            "strategy_data", strategy_data, {"author": "test_user"}
        )

        # Create snapshot
        snapshot = versioned.create_snapshot(
            "strategy_data",
            name="Test Strategy",
            description="A test strategy snapshot",
            author="test_user",
            tags=["test", "episodic"],
        )

        assert snapshot.version_id == version_id
        assert snapshot.name == "Test Strategy"
        assert snapshot.description == "A test strategy snapshot"
        assert snapshot.author == "test_user"
        assert snapshot.tags == ["test", "episodic"]
        assert snapshot.strategy_type == "episodic_memory"
        assert snapshot.strategy_config == {"max_episodes": 10}
        assert len(snapshot.episodes) == 1
        assert "source_key" in snapshot.metadata

    def test_create_snapshot_no_versions(self) -> None:
        """Test creating snapshot when no versions exist."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        with pytest.raises(KeyError):
            versioned.create_snapshot("nonexistent_key")

    def test_file_storage_integration(self) -> None:
        """Test with file storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(temp_dir)
            versioned = VersionedStorageWrapper(storage)

            # Save data
            data = {"test": "data"}
            version_id = versioned.save_with_version(
                "test_key", data, {"name": "file test"}
            )

            # Load version
            loaded = versioned.load_version("test_key", version_id)
            assert loaded == data

            # Check that we can list versions
            versions = versioned.list_versions("test_key")
            assert len(versions) == 1
            assert versions[0]["version_id"] == version_id

    def test_multiple_keys_isolation(self) -> None:
        """Test that different keys are properly isolated."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save data for different keys
        v1 = versioned.save_with_version("key1", {"data": 1}, {})
        v2 = versioned.save_with_version("key2", {"data": 2}, {})

        # Check versions are separate
        versions1 = versioned.list_versions("key1")
        versions2 = versioned.list_versions("key2")

        assert len(versions1) == 1
        assert len(versions2) == 1
        assert versions1[0]["version_id"] == v1
        assert versions2[0]["version_id"] == v2

        # Load correct data
        assert versioned.load_version("key1", v1) == {"data": 1}
        assert versioned.load_version("key2", v2) == {"data": 2}

    def test_version_metadata_preservation(self) -> None:
        """Test that version metadata is properly stored and retrieved."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        metadata = {
            "name": "Test Version",
            "description": "A detailed description",
            "author": "test@example.com",
            "tags": ["important", "reviewed"],
            "custom_field": "custom_value",
        }

        version_id = versioned.save_with_version(
            "test_key", {"data": "value"}, metadata
        )

        versions = versioned.list_versions("test_key")
        assert len(versions) == 1

        version_info = versions[0]
        assert version_info["version_id"] == version_id
        assert "created_at" in version_info
        assert version_info["metadata"] == metadata

    def test_rollback_updates_current_data(self) -> None:
        """Test that rollback properly updates the current data in storage."""
        storage = InMemoryStorage()
        versioned = VersionedStorageWrapper(storage)

        # Save initial data
        data1 = {"version": 1}
        v1 = versioned.save_with_version("test_key", data1, {"name": "v1"})

        # Modify and save again
        data2 = {"version": 2}
        v2 = versioned.save_with_version("test_key", data2, {"name": "v2"})

        # Verify current data is v2
        assert storage.load("test_key") == data2

        # Rollback to v1
        versioned.rollback_to("test_key", v1)

        # Verify current data is now v1
        assert storage.load("test_key") == data1
