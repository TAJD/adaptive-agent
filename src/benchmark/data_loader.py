"""Data loading utilities for benchmark tasks."""

import os
from pathlib import Path
from typing import Any, Dict

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_csv_data(filename: str) -> Any:
    """
    Load CSV data for benchmark tasks.

    Args:
        filename: Name of the CSV file in the data directory

    Returns:
        pandas DataFrame if pandas is available, otherwise dict
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    file_path = data_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if PANDAS_AVAILABLE:
        return pd.read_csv(file_path)
    else:
        # Fallback to basic CSV reading
        import csv

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)


def create_task_context(task_id: str) -> Dict[str, Any]:
    """
    Create execution context for a specific task, including any required data.

    Args:
        task_id: The task identifier

    Returns:
        Context dictionary with data and hints
    """
    context = {}

    # Load data based on task requirements
    if task_id in [
        "sales_total_2024",
        "sales_growth_analysis",
        "regional_sales_ranking",
    ]:
        context["data"] = load_csv_data("sales_data.csv")
        context["hints"] = [
            "Use pandas operations like groupby, sum, etc.",
            "Access columns like 'Sales_2024', 'Sales_2023', 'Region'",
        ]

    elif task_id in [
        "top_performing_department",
        "performance_distribution",
        "high_earners_analysis",
    ]:
        context["data"] = load_csv_data("employee_data.csv")
        context["hints"] = [
            "Use pandas for data analysis",
            "Columns include: Employee_ID, Name, Department, Salary, Performance_Rating",
        ]

    elif task_id in ["order_revenue_calculation", "popular_product_region"]:
        context["data"] = load_csv_data("order_data.csv")
        context["hints"] = [
            "Use pandas for data manipulation",
            "Columns include: Order_ID, Customer_ID, Product, Quantity, Unit_Price, Region",
            "Calculate revenue as Quantity * Unit_Price",
        ]

    elif task_id.startswith("challenge_"):
        context["data"] = load_csv_data("challenging_data.csv")
        context["hints"] = [
            "Data may contain null/NaN values - handle them appropriately",
            "Columns: Product, Q1_Sales, Q2_Sales, Q3_Sales, Q4_Sales, Returns, Category, Launch_Date, Discontinued",
            "Discontinued column is boolean (True/False)",
            "Be careful with null vs zero - they are different!",
        ]

    # P&L Financial Data tasks (same dataset as CLI/demos)
    elif task_id.startswith("pl_"):
        context["data"] = load_csv_data("FUN_company_pl_actuals_dataset.csv")
        context["hints"] = [
            "This is P&L (Profit & Loss) financial data",
            "Key columns:",
            "  - Fiscal Year (int): 2020-2024",
            "  - Fiscal Quarter: Q1, Q2, Q3, Q4",
            "  - FSLine Statement L1: High-level category (Net Revenue, Cost of Goods Sold, OPEX, Other Income/Expenses)",
            "  - FSLine Statement L2: Detailed line item (Gross Revenue, Direct Labor, Marketing Expenses, etc.)",
            "  - Product: Product A, B, C, D",
            "  - Country: Australia, Canada, Germany, Japan, United Kingdom, United States",
            "  - Amount in USD: The monetary value to use for calculations",
            "",
            "Common patterns:",
            "  - Gross Revenue: FSLine Statement L2 == 'Gross Revenue'",
            "  - COGS: FSLine Statement L1 == 'Cost of Goods Sold'",
            "  - OPEX: FSLine Statement L1 == 'OPEX'",
            "  - FX Impact: FSLine Statement L2 == 'Foreign Exchange Gain/Loss'",
            "",
            "Always filter by the specific dimensions mentioned in the query!",
        ]

    # Add common constraints
    context["constraints"] = [
        "Store final result in variable named 'result'",
        "Use only standard library and pandas operations",
        "Return data in the expected format (dict, list, etc.)",
    ]

    return context


def get_available_data_files() -> list[str]:
    """Get list of available data files."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not data_dir.exists():
        return []

    return [f.name for f in data_dir.glob("*.csv")]
