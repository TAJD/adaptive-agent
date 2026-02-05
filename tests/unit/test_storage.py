"""Tests for storage implementations."""

import pytest
from pathlib import Path

from src.storage.memory import InMemoryStorage
from src.storage.file import FileStorage


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.fixture
    def storage(self) -> InMemoryStorage:
        return InMemoryStorage()

    def test_save_and_load(self, storage: InMemoryStorage) -> None:
        """Test basic save and load."""
        storage.save("key1", {"value": 42})
        result = storage.load("key1")
        assert result == {"value": 42}

    def test_load_nonexistent(self, storage: InMemoryStorage) -> None:
        """Test loading nonexistent key returns None."""
        result = storage.load("nonexistent")
        assert result is None

    def test_delete(self, storage: InMemoryStorage) -> None:
        """Test delete functionality."""
        storage.save("key1", "value")
        assert storage.delete("key1") is True
        assert storage.load("key1") is None
        assert storage.delete("key1") is False  # Already deleted

    def test_list_keys(self, storage: InMemoryStorage) -> None:
        """Test listing keys."""
        storage.save("a/1", "v1")
        storage.save("a/2", "v2")
        storage.save("b/1", "v3")

        all_keys = storage.list_keys()
        assert set(all_keys) == {"a/1", "a/2", "b/1"}

        a_keys = storage.list_keys("a/")
        assert set(a_keys) == {"a/1", "a/2"}

    def test_query(self, storage: InMemoryStorage) -> None:
        """Test query functionality."""
        storage.save("k1", {"type": "error", "count": 5})
        storage.save("k2", {"type": "error", "count": 3})
        storage.save("k3", {"type": "success", "count": 1})

        results = storage.query({"type": "error"})
        assert len(results) == 2
        assert all(r["type"] == "error" for r in results)

    def test_clear(self, storage: InMemoryStorage) -> None:
        """Test clear functionality."""
        storage.save("k1", "v1")
        storage.save("k2", "v2")
        storage.clear()
        assert storage.list_keys() == []


class TestFileStorage:
    """Tests for FileStorage."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> FileStorage:
        return FileStorage(tmp_path / "test_storage")

    def test_save_and_load(self, storage: FileStorage) -> None:
        """Test basic save and load."""
        storage.save("key1", {"value": 42})
        result = storage.load("key1")
        assert result == {"value": 42}

    def test_load_nonexistent(self, storage: FileStorage) -> None:
        """Test loading nonexistent key returns None."""
        result = storage.load("nonexistent")
        assert result is None

    def test_delete(self, storage: FileStorage) -> None:
        """Test delete functionality."""
        storage.save("key1", "value")
        assert storage.delete("key1") is True
        assert storage.load("key1") is None
        assert storage.delete("key1") is False

    def test_list_keys(self, storage: FileStorage) -> None:
        """Test listing keys."""
        storage.save("a/1", "v1")
        storage.save("a/2", "v2")
        storage.save("b/1", "v3")

        all_keys = storage.list_keys()
        assert len(all_keys) == 3

    def test_persistence(self, tmp_path: Path) -> None:
        """Test data persists across instances."""
        path = tmp_path / "persist_test"

        # Write with one instance
        storage1 = FileStorage(path)
        storage1.save("key", {"persistent": True})

        # Read with another instance
        storage2 = FileStorage(path)
        result = storage2.load("key")
        assert result == {"persistent": True}

    def test_handles_special_characters(self, storage: FileStorage) -> None:
        """Test handling of special characters in keys."""
        storage.save("path/to/key", {"nested": "value"})
        result = storage.load("path/to/key")
        assert result == {"nested": "value"}

    def test_clear(self, storage: FileStorage) -> None:
        """Test clear functionality."""
        storage.save("k1", "v1")
        storage.save("k2", "v2")
        storage.clear()
        assert storage.list_keys() == []
