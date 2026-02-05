"""File-based JSON storage implementation."""

import json
from pathlib import Path
from typing import Any


class FileStorage:
    """File-based JSON storage for persistence."""

    def __init__(self, base_path: Path | str) -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a key to a file path."""
        # Replace problematic characters for filenames
        safe_key = key.replace("/", "__").replace("\\", "__")
        return self._base_path / f"{safe_key}.json"

    def save(self, key: str, data: Any) -> None:
        """Save data under a key."""
        path = self._key_to_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, key: str) -> Any | None:
        """Load data by key. Returns None if not found."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def query(self, filter_dict: dict) -> list[Any]:
        """
        Query for items matching the filter.

        Loads all JSON files and checks if they match the filter.
        """
        results = []
        for path in self._base_path.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if all(data.get(k) == v for k, v in filter_dict.items()):
                        results.append(data)
            except (json.JSONDecodeError, IOError):
                continue
        return results

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        keys = []
        for path in self._base_path.glob("*.json"):
            # Convert path back to key
            key = path.stem.replace("__", "/")
            if not prefix or key.startswith(prefix):
                keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted, False if not found."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        for path in self._base_path.glob("*.json"):
            path.unlink()
