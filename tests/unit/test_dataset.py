"""Tests for the dataset loader."""

import pytest
from pathlib import Path

from src.data import DatasetLoader, DatasetValidationError


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_default_path(self) -> None:
        """Test loading from default path."""
        loader = DatasetLoader()
        data = loader.load()

        assert len(data) == 21600
        assert "Fiscal Year" in loader.columns
        assert "Amount in USD" in loader.columns

    def test_data_types_converted(self) -> None:
        """Test that numeric columns are converted."""
        loader = DatasetLoader()
        data = loader.load()

        assert isinstance(data[0]["Fiscal Year"], int)
        assert isinstance(data[0]["Amount in USD"], float)
        assert isinstance(data[0]["Amount in Local Currency"], float)

    def test_get_unique_values(self) -> None:
        """Test getting unique column values."""
        loader = DatasetLoader()
        loader.load()

        products = loader.get_unique_values("Product")
        assert products == ["Product A", "Product B", "Product C", "Product D"]

        years = loader.get_unique_values("Fiscal Year")
        assert years == [2020, 2021, 2022, 2023, 2024]

    def test_get_unique_values_invalid_column(self) -> None:
        """Test error on invalid column."""
        loader = DatasetLoader()
        loader.load()

        with pytest.raises(ValueError, match="not found"):
            loader.get_unique_values("Invalid Column")

    def test_get_summary(self) -> None:
        """Test summary generation."""
        loader = DatasetLoader()
        loader.load()

        summary = loader.get_summary()

        assert summary["row_count"] == 21600
        assert len(summary["products"]) == 4
        assert len(summary["countries"]) == 6
        assert len(summary["currencies"]) == 6

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        loader = DatasetLoader(tmp_path / "nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_validation_runs_by_default(self) -> None:
        """Test that validation runs on load."""
        loader = DatasetLoader()
        # Should not raise
        data = loader.load(validate=True)
        assert len(data) > 0
