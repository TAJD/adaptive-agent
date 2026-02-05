"""Versioned storage system for improvement strategies."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from pathlib import Path

from ..core.protocols import Storage
from ..strategies.versioning import ImprovementVersion


class VersionedStorage(Protocol):
    """Protocol for versioned storage with snapshot capabilities."""

    @abstractmethod
    def save_with_version(
        self, key: str, data: Any, version_metadata: Dict[str, Any]
    ) -> str:
        """
        Save data and create a new version.

        Args:
            key: The data key
            data: The data to save
            version_metadata: Metadata for the version (name, description, etc.)

        Returns:
            The version ID of the created version
        """
        ...

    @abstractmethod
    def load_version(self, key: str, version_id: str) -> Any:
        """
        Load a specific version of data.

        Args:
            key: The data key
            version_id: The version ID to load

        Returns:
            The data for that version

        Raises:
            KeyError: If the version doesn't exist
        """
        ...

    @abstractmethod
    def list_versions(self, key: str) -> List[Dict[str, Any]]:
        """
        List all versions for a key.

        Args:
            key: The data key

        Returns:
            List of version metadata dicts, sorted by creation time (newest first)
        """
        ...

    @abstractmethod
    def get_current_version(self, key: str) -> Optional[str]:
        """
        Get the current (latest) version ID for a key.

        Args:
            key: The data key

        Returns:
            The latest version ID, or None if no versions exist
        """
        ...

    @abstractmethod
    def rollback_to(self, key: str, version_id: str) -> None:
        """
        Rollback to a specific version (make it the current version).

        Args:
            key: The data key
            version_id: The version ID to rollback to

        Raises:
            KeyError: If the version doesn't exist
        """
        ...

    @abstractmethod
    def create_snapshot(
        self,
        key: str,
        name: str = "",
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> ImprovementVersion:
        """
        Create a snapshot of the current state as an ImprovementVersion.

        Args:
            key: The data key to snapshot
            name: Optional name for the snapshot
            description: Optional description
            author: Optional author
            tags: Optional tags

        Returns:
            An ImprovementVersion representing the snapshot

        Raises:
            KeyError: If no current version exists for the key
        """
        ...


class VersionedStorageWrapper:
    """
    Wrapper that adds versioning capabilities to any Storage implementation.

    Stores versions in a versions/ subdirectory with a versions.json index.
    """

    def __init__(self, storage: Storage, versions_dir: str = "versions"):
        """
        Initialize the versioned storage wrapper.

        Args:
            storage: The underlying storage implementation
            versions_dir: Directory name for storing versions (relative to storage root if applicable)
        """
        self.storage = storage
        self.versions_dir = versions_dir
        self._versions_index_key = f"{versions_dir}/versions.json"

        # Initialize versions index if it doesn't exist
        self._ensure_versions_index()

    def _ensure_versions_index(self) -> None:
        """Ensure the versions index exists."""
        if self.storage.load(self._versions_index_key) is None:
            self.storage.save(self._versions_index_key, {})

    def _load_versions_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load the versions index."""
        index = self.storage.load(self._versions_index_key)
        return index if index is not None else {}

    def _save_versions_index(self, index: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save the versions index."""
        self.storage.save(self._versions_index_key, index)

    def _get_version_key(self, key: str, version_id: str) -> str:
        """Get the storage key for a specific version."""
        return f"{self.versions_dir}/{key}/{version_id}.json"

    def save_with_version(
        self, key: str, data: Any, version_metadata: Dict[str, Any]
    ) -> str:
        """
        Save data and create a new version.

        Args:
            key: The data key
            data: The data to save
            version_metadata: Metadata for the version

        Returns:
            The version ID of the created version
        """
        import uuid
        from datetime import datetime

        version_id = str(uuid.uuid4())

        # Create version entry
        version_entry = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "metadata": version_metadata,
        }

        # Save the version data
        version_data = {
            "version_id": version_id,
            "data": data,
            "metadata": version_metadata,
            "created_at": version_entry["created_at"],
        }
        self.storage.save(self._get_version_key(key, version_id), version_data)

        # Update the versions index
        index = self._load_versions_index()
        if key not in index:
            index[key] = []

        index[key].append(version_entry)

        # Sort by creation time (newest first)
        index[key].sort(key=lambda x: x["created_at"], reverse=True)

        self._save_versions_index(index)

        # Also save the current data to the original key
        self.storage.save(key, data)

        return version_id

    def load_version(self, key: str, version_id: str) -> Any:
        """
        Load a specific version of data.

        Args:
            key: The data key
            version_id: The version ID to load

        Returns:
            The data for that version

        Raises:
            KeyError: If the version doesn't exist
        """
        version_data = self.storage.load(self._get_version_key(key, version_id))
        if version_data is None:
            raise KeyError(f"Version {version_id} not found for key {key}")

        return version_data["data"]

    def list_versions(self, key: str) -> List[Dict[str, Any]]:
        """
        List all versions for a key.

        Args:
            key: The data key

        Returns:
            List of version metadata dicts, sorted by creation time (newest first)
        """
        index = self._load_versions_index()
        return index.get(key, [])

    def get_current_version(self, key: str) -> Optional[str]:
        """
        Get the current (latest) version ID for a key.

        Args:
            key: The data key

        Returns:
            The latest version ID, or None if no versions exist
        """
        versions = self.list_versions(key)
        return versions[0]["version_id"] if versions else None

    def rollback_to(self, key: str, version_id: str) -> None:
        """
        Rollback to a specific version (make it the current version).

        Args:
            key: The data key
            version_id: The version ID to rollback to

        Raises:
            KeyError: If the version doesn't exist
        """
        # Load the version data
        data = self.load_version(key, version_id)

        # Save it as the current data
        self.storage.save(key, data)

    def create_snapshot(
        self,
        key: str,
        name: str = "",
        description: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
    ) -> ImprovementVersion:
        """
        Create a snapshot of the current state as an ImprovementVersion.

        Args:
            key: The data key to snapshot
            name: Optional name for the snapshot
            description: Optional description
            author: Optional author
            tags: Optional tags

        Returns:
            An ImprovementVersion representing the snapshot

        Raises:
            KeyError: If no current version exists for the key
        """
        current_version_id = self.get_current_version(key)
        if current_version_id is None:
            raise KeyError(f"No versions exist for key {key}")

        # Load the current data
        data = self.storage.load(key)
        if data is None:
            raise KeyError(f"No data found for key {key}")

        # Load version metadata
        versions = self.list_versions(key)
        current_version = next(
            (v for v in versions if v["version_id"] == current_version_id), None
        )
        if current_version is None:
            raise KeyError(f"Current version {current_version_id} metadata not found")

        # Create ImprovementVersion
        from ..strategies.versioning import ImprovementVersion

        # Try to infer strategy type from data
        strategy_type = ""
        strategy_config = {}
        episodes = []

        if isinstance(data, dict):
            strategy_type = data.get("strategy_type", "")
            strategy_config = data.get("config", {})
            episodes = data.get("episodes", [])

        snapshot = ImprovementVersion(
            version_id=current_version_id,
            name=name or f"Snapshot of {key}",
            description=description,
            author=author,
            tags=tags or [],
            strategy_type=strategy_type,
            strategy_config=strategy_config,
            episodes=episodes,
            metadata={
                "source_key": key,
                "snapshot_created_at": ImprovementVersion().created_at,  # current time
                **current_version.get("metadata", {}),
            },
        )

        return snapshot
