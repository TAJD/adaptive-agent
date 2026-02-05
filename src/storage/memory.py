"""In-memory storage implementation for testing."""

from typing import Any


class InMemoryStorage:
    """In-memory storage for testing and ephemeral use."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> None:
        """Save data under a key."""
        self._data[key] = data

    def load(self, key: str) -> Any | None:
        """Load data by key. Returns None if not found."""
        return self._data.get(key)

    def query(self, filter_dict: dict) -> list[Any]:
        """
        Query for items matching the filter.

        Simple implementation: checks if stored dicts contain all filter key-values.
        """
        results = []
        for value in self._data.values():
            if isinstance(value, dict):
                if all(value.get(k) == v for k, v in filter_dict.items()):
                    results.append(value)
        return results

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        if not prefix:
            return list(self._data.keys())
        return [k for k in self._data.keys() if k.startswith(prefix)]

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted, False if not found."""
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._data.clear()
