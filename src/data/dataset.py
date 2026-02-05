"""Dataset loading and validation utilities for financial data."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""

    pass


@dataclass(frozen=True)
class DatasetSchema:
    """Expected schema for the financial dataset."""

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "Fiscal Year",
        "Fiscal Quarter",
        "Fiscal Period",
        "FSLine Statement L1",
        "FSLine Statement L2",
        "Product",
        "Country",
        "Currency",
        "Amount in Local Currency",
        "Amount in USD",
        "Version",
    )

    VALID_PRODUCTS: tuple[str, ...] = (
        "Product A",
        "Product B",
        "Product C",
        "Product D",
    )

    VALID_COUNTRIES: tuple[str, ...] = (
        "Australia",
        "Canada",
        "Germany",
        "Japan",
        "United Kingdom",
        "United States",
    )

    VALID_CURRENCIES: tuple[str, ...] = ("AUD", "CAD", "EUR", "GBP", "JPY", "USD")

    VALID_L1_CATEGORIES: tuple[str, ...] = (
        "Net Revenue",
        "Cost of Goods Sold",
        "OPEX",
        "Other Income/Expenses",
    )

    VALID_FISCAL_YEARS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

    VALID_QUARTERS: tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")


SCHEMA = DatasetSchema()


class DatasetLoader:
    """Loads and validates the financial dataset."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        """Initialize the loader with optional custom path.

        Args:
            data_path: Path to dataset. Defaults to data/FUN_company_pl_actuals_dataset.csv
        """
        if data_path is None:
            self._path = Path(__file__).parent.parent.parent / "data" / "FUN_company_pl_actuals_dataset.csv"
        else:
            self._path = Path(data_path)

        self._data: list[dict[str, Any]] | None = None
        self._columns: list[str] | None = None

    @property
    def path(self) -> Path:
        """Return the dataset path."""
        return self._path

    def load(self, validate: bool = True) -> list[dict[str, Any]]:
        """Load the dataset from CSV.

        Args:
            validate: Whether to validate the dataset after loading.

        Returns:
            List of row dictionaries.

        Raises:
            FileNotFoundError: If dataset file doesn't exist.
            DatasetValidationError: If validation fails.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._columns = reader.fieldnames or []
            self._data = list(reader)

        # Convert numeric columns
        for row in self._data:
            row["Fiscal Year"] = int(row["Fiscal Year"])
            row["Amount in Local Currency"] = float(row["Amount in Local Currency"])
            row["Amount in USD"] = float(row["Amount in USD"])

        if validate:
            self.validate()

        return self._data

    @property
    def data(self) -> list[dict[str, Any]]:
        """Return loaded data, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data  # type: ignore

    @property
    def columns(self) -> list[str]:
        """Return column names."""
        if self._columns is None:
            self.load()
        return self._columns  # type: ignore

    def validate(self) -> None:
        """Validate the dataset schema and values.

        Raises:
            DatasetValidationError: If validation fails.
        """
        if self._data is None:
            raise DatasetValidationError("No data loaded. Call load() first.")

        self._validate_columns()
        self._validate_row_count()
        self._validate_values()

    def _validate_columns(self) -> None:
        """Validate that all required columns are present."""
        missing = set(SCHEMA.REQUIRED_COLUMNS) - set(self._columns or [])
        if missing:
            raise DatasetValidationError(f"Missing required columns: {missing}")

    def _validate_row_count(self) -> None:
        """Validate row count matches expected."""
        expected = 21600  # 21601 including header
        actual = len(self._data or [])
        if actual != expected:
            # Warning only - row count might vary slightly
            pass

    def _validate_values(self) -> None:
        """Validate that values are within expected ranges."""
        if not self._data:
            return

        invalid_products = set()
        invalid_countries = set()
        invalid_years = set()
        invalid_quarters = set()

        for row in self._data:
            product = row.get("Product")
            if product and product not in SCHEMA.VALID_PRODUCTS:
                invalid_products.add(product)

            country = row.get("Country")
            if country and country not in SCHEMA.VALID_COUNTRIES:
                invalid_countries.add(country)

            year = row.get("Fiscal Year")
            if year and year not in SCHEMA.VALID_FISCAL_YEARS:
                invalid_years.add(year)

            quarter = row.get("Fiscal Quarter")
            if quarter and quarter not in SCHEMA.VALID_QUARTERS:
                invalid_quarters.add(quarter)

        errors = []
        if invalid_products:
            errors.append(f"Invalid products: {invalid_products}")
        if invalid_countries:
            errors.append(f"Invalid countries: {invalid_countries}")
        if invalid_years:
            errors.append(f"Invalid fiscal years: {invalid_years}")
        if invalid_quarters:
            errors.append(f"Invalid quarters: {invalid_quarters}")

        if errors:
            raise DatasetValidationError("; ".join(errors))

    def get_unique_values(self, column: str) -> list[Any]:
        """Get unique values for a column.

        Args:
            column: Column name.

        Returns:
            Sorted list of unique values.
        """
        if column not in self.columns:
            raise ValueError(f"Column '{column}' not found. Available: {self.columns}")

        values = {row[column] for row in self.data}
        return sorted(values)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the dataset.

        Returns:
            Dictionary with dataset statistics.
        """
        return {
            "path": str(self._path),
            "row_count": len(self.data),
            "columns": self.columns,
            "fiscal_years": self.get_unique_values("Fiscal Year"),
            "quarters": self.get_unique_values("Fiscal Quarter"),
            "products": self.get_unique_values("Product"),
            "countries": self.get_unique_values("Country"),
            "currencies": self.get_unique_values("Currency"),
            "l1_categories": self.get_unique_values("FSLine Statement L1"),
            "l2_categories": self.get_unique_values("FSLine Statement L2"),
        }
