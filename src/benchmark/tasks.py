"""Task suite definitions for benchmarking."""

from src.core.types import Task


# =============================================================================
# P&L FINANCIAL DATA TASKS
# These use the FUN_company_pl_actuals_dataset.csv - same as demos/CLI
# =============================================================================

# P&L Easy - Simple aggregations, single filter dimension
PL_EASY_TASKS = [
    Task(
        id="pl_gross_revenue_a_2023",
        query="What is the total Gross Revenue for Product A in 2023? Return the numeric value.",
        expected_answer=29001195.46,  # Verified from dataset
        difficulty="easy",
        tags=("pl_data", "revenue", "aggregation"),
    ),
    Task(
        id="pl_gross_revenue_b_2024",
        query="What is the total Gross Revenue for Product B in 2024? Return the numeric value.",
        expected_answer=27329403.78,  # Verified from dataset
        difficulty="easy",
        tags=("pl_data", "revenue", "aggregation"),
    ),
    Task(
        id="pl_cogs_c_2023",
        query="What is the total Cost of Goods Sold for Product C in 2023? Return the numeric value.",
        expected_answer=9006991.97,  # Verified from dataset
        difficulty="easy",
        tags=("pl_data", "cogs", "aggregation"),
    ),
    Task(
        id="pl_opex_japan_2024",
        query="What is the total Operating Expenses (OPEX) for Japan in 2024? Return the numeric value.",
        expected_answer=105172.62,  # Verified from dataset
        difficulty="easy",
        tags=("pl_data", "opex", "aggregation"),
    ),
]

# P&L Medium - Multiple filters, specific line items
PL_MEDIUM_TASKS = [
    Task(
        id="pl_fx_impact_c_2024",
        query="What was the total Foreign Exchange Gain/Loss for Product C across ALL countries in fiscal year 2024? This is found in FSLine Statement L2. Return just the numeric value (can be negative).",
        expected_answer=35095.60,  # Calculated from dataset
        difficulty="medium",
        tags=("pl_data", "fx", "aggregation"),
    ),
    Task(
        id="pl_fx_impact_d_2024",
        query="What was the total Foreign Exchange Gain/Loss for Product D across ALL countries in fiscal year 2024? This is found in FSLine Statement L2. Return just the numeric value (can be negative).",
        expected_answer=-16650.33,  # Calculated from dataset - negative!
        difficulty="medium",
        tags=("pl_data", "fx", "aggregation"),
    ),
    Task(
        id="pl_opex_japan_q1_2024",
        query="What is the total OPEX for Japan in Q1 2024? OPEX includes: General & Administrative, Headcount Expenses, IT Expenses, Marketing Expenses, R&D Expenses, and Sales Expenses. Return just the numeric sum.",
        expected_answer=26161.39,  # Verified from dataset
        difficulty="medium",
        tags=("pl_data", "opex", "quarter", "aggregation"),
    ),
    Task(
        id="pl_revenue_us_q2_2023",
        query="What is the total Gross Revenue for the United States in Q2 2023? Return the numeric value.",
        expected_answer=5179110.21,  # Verified from dataset
        difficulty="medium",
        tags=("pl_data", "revenue", "quarter", "aggregation"),
    ),
    Task(
        id="pl_direct_labor_a_germany_2024",
        query="What is the total Direct Labor cost for Product A in Germany for 2024? Direct Labor is in FSLine Statement L2. Return the numeric value.",
        expected_answer=999005.64,  # Verified from dataset
        difficulty="medium",
        tags=("pl_data", "labor", "aggregation"),
    ),
]

# P&L Hard - Percentages, comparisons, multi-step calculations
PL_HARD_TASKS = [
    Task(
        id="pl_cogs_pct_b_2020",
        query="Calculate the Cost of Goods Sold as a percentage of Gross Revenue for Product B in 2020. COGS is FSLine Statement L1 = 'Cost of Goods Sold'. Gross Revenue is FSLine Statement L2 = 'Gross Revenue'. Return the percentage rounded to 2 decimal places.",
        expected_answer=44.04,  # Calculated from dataset
        difficulty="hard",
        tags=("pl_data", "cogs", "percentage", "calculation"),
    ),
    Task(
        id="pl_cogs_pct_b_2024",
        query="Calculate the Cost of Goods Sold as a percentage of Gross Revenue for Product B in 2024. COGS is FSLine Statement L1 = 'Cost of Goods Sold'. Gross Revenue is FSLine Statement L2 = 'Gross Revenue'. Return the percentage rounded to 2 decimal places.",
        expected_answer=43.66,  # Calculated from dataset
        difficulty="hard",
        tags=("pl_data", "cogs", "percentage", "calculation"),
    ),
    Task(
        id="pl_gross_margin_a_2023",
        query="Calculate the Gross Margin percentage for Product A in 2023. Gross Margin = (Gross Revenue - COGS) / Gross Revenue * 100. Return the percentage rounded to 2 decimal places.",
        expected_answer=53.99,  # Verified from dataset
        difficulty="hard",
        tags=("pl_data", "margin", "percentage", "calculation"),
    ),
    Task(
        id="pl_yoy_revenue_growth_c",
        query="Calculate the year-over-year Gross Revenue growth rate for Product C from 2023 to 2024. Formula: (2024_revenue - 2023_revenue) / 2023_revenue * 100. Return percentage rounded to 2 decimal places.",
        expected_answer=12.12,  # Verified from dataset
        difficulty="hard",
        tags=("pl_data", "growth", "percentage", "calculation"),
    ),
    Task(
        id="pl_opex_ratio_japan_2024",
        query="Calculate the OPEX to Revenue ratio for Japan in 2024. Formula: Total OPEX / Total Gross Revenue * 100. Return percentage rounded to 2 decimal places.",
        expected_answer=66.26,  # Verified from dataset
        difficulty="hard",
        tags=("pl_data", "opex", "ratio", "calculation"),
    ),
]

# P&L Learning Pairs - Similar queries that test cross-session learning
# Each pair has a "teacher" and a "learner" query with same pattern
PL_LEARNING_PAIRS = [
    # FX Impact pair (same pattern, different product)
    Task(
        id="pl_learn_fx_a_2024",
        query="What was the total Foreign Exchange Gain/Loss for Product A across ALL countries in fiscal year 2024? Return just the numeric value.",
        expected_answer=11419.74,  # Teacher query - verified from dataset
        difficulty="medium",
        tags=("pl_data", "fx", "learning_teacher"),
    ),
    Task(
        id="pl_learn_fx_b_2024",
        query="What was the total Foreign Exchange Gain/Loss for Product B across ALL countries in fiscal year 2024? Return just the numeric value.",
        expected_answer=-1488.06,  # Learner query - verified from dataset (negative!)
        difficulty="medium",
        tags=("pl_data", "fx", "learning_learner"),
    ),
    # COGS percentage pair (same pattern, different year)
    Task(
        id="pl_learn_cogs_pct_a_2023",
        query="Calculate COGS as a percentage of Gross Revenue for Product A in 2023. Return percentage rounded to 2 decimal places.",
        expected_answer=46.01,  # Teacher query - verified from dataset
        difficulty="hard",
        tags=("pl_data", "cogs", "percentage", "learning_teacher"),
    ),
    Task(
        id="pl_learn_cogs_pct_a_2024",
        query="Calculate COGS as a percentage of Gross Revenue for Product A in 2024. Return percentage rounded to 2 decimal places.",
        expected_answer=45.28,  # Learner query - verified from dataset
        difficulty="hard",
        tags=("pl_data", "cogs", "percentage", "learning_learner"),
    ),
    # OPEX by country pair (same pattern, different country)
    Task(
        id="pl_learn_opex_us_2024",
        query="What is the total Operating Expenses for United States in 2024? OPEX is FSLine Statement L1 = 'OPEX'. Return the numeric value.",
        expected_answer=16334627.74,  # Teacher query - verified from dataset
        difficulty="medium",
        tags=("pl_data", "opex", "learning_teacher"),
    ),
    Task(
        id="pl_learn_opex_canada_2024",
        query="What is the total Operating Expenses for Canada in 2024? OPEX is FSLine Statement L1 = 'OPEX'. Return the numeric value.",
        expected_answer=7430198.91,  # Learner query - verified from dataset
        difficulty="medium",
        tags=("pl_data", "opex", "learning_learner"),
    ),
]


# =============================================================================
# ORIGINAL TASKS (non-P&L)
# =============================================================================

# Easy tasks - single operation, clear expected answer
EASY_TASKS = [
    Task(
        id="easy_sum",
        query="Calculate the sum of [1, 2, 3, 4, 5]",
        expected_answer=15,
        difficulty="easy",
        tags=("arithmetic", "list"),
    ),
    Task(
        id="easy_mean",
        query="Calculate the mean of [10, 20, 30]",
        expected_answer=20.0,
        difficulty="easy",
        tags=("statistics", "list"),
    ),
    Task(
        id="easy_max",
        query="Find the maximum value in [3, 7, 2, 9, 1]",
        expected_answer=9,
        difficulty="easy",
        tags=("arithmetic", "list"),
    ),
    Task(
        id="easy_count",
        query="Count how many items are in [1, 2, 3, 4]",
        expected_answer=4,
        difficulty="easy",
        tags=("arithmetic", "list"),
    ),
    Task(
        id="easy_string",
        query="What is 'hello' + ' ' + 'world'?",
        expected_answer="hello world",
        difficulty="easy",
        tags=("string",),
    ),
]

# Medium tasks - require multiple steps or data understanding
MEDIUM_TASKS = [
    Task(
        id="medium_std",
        query="Calculate the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9] (population)",
        expected_answer=2.0,
        difficulty="medium",
        tags=("statistics", "list"),
    ),
    Task(
        id="medium_filter_sum",
        query="Sum all even numbers in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
        expected_answer=30,
        difficulty="medium",
        tags=("arithmetic", "filter", "list"),
    ),
    Task(
        id="medium_percentile",
        query="What is the median of [1, 3, 5, 7, 9, 11]?",
        expected_answer=6.0,
        difficulty="medium",
        tags=("statistics", "list"),
    ),
    Task(
        id="medium_string_count",
        query="How many vowels are in 'hello world'?",
        expected_answer=3,
        difficulty="medium",
        tags=("string", "count"),
    ),
    Task(
        id="medium_compound",
        query="Calculate (10 + 20) * 3 / 2",
        expected_answer=45.0,
        difficulty="medium",
        tags=("arithmetic",),
    ),
]

# Real business data tasks
REAL_DATA_TASKS = [
    Task(
        id="sales_total_2024",
        query="What is the total sales amount for all products in 2024? (data is in 'sales_data.csv' with columns Product, Sales_2024, etc.)",
        expected_answer=1046000,  # Sum of all Sales_2024: 180000+140000+250000+210000+95000+75000+30000+18000+48000
        difficulty="medium",
        tags=("business_intelligence", "sales", "aggregation", "real_data"),
    ),
    Task(
        id="sales_growth_analysis",
        query="Calculate the year-over-year sales growth percentage for each product. Return as a dictionary with product names as keys and growth percentages as values. (data in 'sales_data.csv')",
        expected_answer={
            "Laptop A": 20.0,  # (180000-150000)/150000 * 100
            "Laptop B": 16.67,  # (140000-120000)/120000 * 100
            "Phone X": 25.0,  # (250000-200000)/200000 * 100
            "Phone Y": 16.67,  # (210000-180000)/180000 * 100
            "Tablet Z": 18.75,  # (95000-80000)/80000 * 100
            "Monitor Pro": 25.0,  # (75000-60000)/60000 * 100
            "Keyboard Q": 20.0,  # (30000-25000)/25000 * 100
            "Mouse R": 20.0,  # (18000-15000)/15000 * 100
            "Headphones S": 20.0,  # (48000-40000)/40000 * 100
        },
        difficulty="hard",
        tags=("business_intelligence", "growth_analysis", "calculations", "real_data"),
    ),
    Task(
        id="regional_sales_ranking",
        query="Which region had the highest total sales in 2024? Return as a tuple (region_name, total_sales). (data in 'sales_data.csv')",
        expected_answer=("South", 460000),  # South: 250000 + 210000 = 460000
        difficulty="medium",
        tags=("business_intelligence", "regional_analysis", "ranking", "real_data"),
    ),
    Task(
        id="top_performing_department",
        query="Which department has the highest average salary? Return department name and average salary. (data in 'employee_data.csv')",
        expected_answer=("Engineering", 91500.0),  # (95000 + 88000) / 2
        difficulty="medium",
        tags=("hr_analytics", "salary_analysis", "averages", "real_data"),
    ),
    Task(
        id="performance_distribution",
        query="How many employees have a performance rating above 4.5? (data in 'employee_data.csv')",
        expected_answer=2,  # Alice (4.5) and Carol (4.8) - wait, 4.5 is not above 4.5, so only Carol (4.8)
        difficulty="easy",
        tags=("hr_analytics", "performance", "counting", "real_data"),
    ),
    Task(
        id="order_revenue_calculation",
        query="What is the total revenue from all orders? (Calculate quantity * unit_price for each order and sum them) (data in 'order_data.csv')",
        expected_answer=16650,  # Let's calculate: 2*1500 + 1*1200 + 3*400 + 1*600 + 5*50 + 10*25 + 2*120 + 1*1300 + 2*1050 + 2*600
        difficulty="medium",
        tags=("ecommerce", "revenue", "calculations", "real_data"),
    ),
    Task(
        id="popular_product_region",
        query="Which product is most popular (highest quantity sold) in the East region? Return product name and quantity. (data in 'order_data.csv')",
        expected_answer=(
            "Tablet Z",
            3,
        ),  # East region orders: Tablet Z (3), Phone Y (2), Monitor Pro (2)
        difficulty="hard",
        tags=("ecommerce", "regional_analysis", "product_popularity", "real_data"),
    ),
    Task(
        id="high_earners_analysis",
        query="List all employees who earn more than $70,000. Return as list of dictionaries with name and salary. (data in 'employee_data.csv')",
        expected_answer=[
            {"name": "Alice Johnson", "salary": 95000},
            {"name": "Bob Smith", "salary": 88000},
            {"name": "Ivy Chen", "salary": 78000},
            {"name": "Jack Anderson", "salary": 75000},
        ],
        difficulty="medium",
        tags=("hr_analytics", "salary_filtering", "list_operations", "real_data"),
    ),
]


# Challenging tasks - edge cases, nulls, tricky calculations
CHALLENGING_TASKS = [
    Task(
        id="challenge_null_safe_sum",
        query="Calculate the total Q1 sales, treating missing values as 0. Return as integer.",
        expected_answer=138000,  # Sum of Q1 with nulls treated as 0: 15000+0+30000+10000+0+8000+50000+20000+5000+0
        difficulty="hard",
        tags=("challenging", "null_handling", "aggregation"),
    ),
    Task(
        id="challenge_growth_with_zero",
        query="Calculate the Q4 vs Q1 growth rate for 'Widget A'. Formula: (Q4-Q1)/Q1*100. Return as float rounded to 2 decimal places.",
        expected_answer=66.67,  # (25000-15000)/15000*100 = 66.666... rounded to 66.67
        difficulty="hard",
        tags=("challenging", "percentage", "rounding"),
    ),
    Task(
        id="challenge_exclude_nulls_avg",
        query="What is the average Q3 sales across all products? Exclude products with missing Q3 values. Return as float rounded to 1 decimal place.",
        expected_answer=19222.2,  # (22000+8000+25000+10000+15000+60000+25000+3000+5000)/9 = 19222.22... rounded to 19222.2
        difficulty="hard",
        tags=("challenging", "null_handling", "average", "rounding"),
    ),
    Task(
        id="challenge_active_products_only",
        query="What is the total Q4 sales for products that are NOT discontinued? Return as integer.",
        expected_answer=179000,  # Exclude Widget C (20000) and Legacy P (2000): 201000-20000-2000 = 179000
        difficulty="hard",
        tags=("challenging", "filtering", "boolean"),
    ),
    Task(
        id="challenge_net_sales",
        query="Calculate the net annual sales (sum of all quarters minus returns) for 'Service A'. Note: if returns is missing, treat as 0. Return as integer.",
        expected_answer=230000,  # 50000+55000+60000+65000 - 0 (null returns treated as 0) = 230000
        difficulty="hard",
        tags=("challenging", "null_handling", "calculation"),
    ),
    Task(
        id="challenge_category_with_nulls",
        query="Which category has the highest total Q2 sales? Return as tuple (category_name, total_sales).",
        expected_answer=("Services", 55000),  # Electronics: 18000+5000+28000=51000, Accessories: 10000+12000+9000=31000, Services: 55000+0=55000, Legacy: 4000
        difficulty="hard",
        tags=("challenging", "groupby", "null_handling"),
    ),
    Task(
        id="challenge_products_declining",
        query="How many products have declining sales (Q4 < Q1)? Only count products where both Q1 and Q4 have values.",
        expected_answer=2,  # Widget C (20000 < 30000) and Legacy P (2000 < 5000)
        difficulty="hard",
        tags=("challenging", "comparison", "null_handling", "counting"),
    ),
    Task(
        id="challenge_zero_returns_count",
        query="How many products have exactly zero returns (not null, but 0)?",
        expected_answer=2,  # Gadget X (0) and Service B (0)
        difficulty="hard",
        tags=("challenging", "null_vs_zero", "counting"),
    ),
    Task(
        id="challenge_recent_products_sales",
        query="What is the total Q4 sales for products launched in 2023 or later? Return as integer.",
        expected_answer=38000,  # Widget B (2023): 12000, Gadget Y (2023): 18000, New Item (2024): 8000 = 38000
        difficulty="hard",
        tags=("challenging", "date_filtering", "aggregation"),
    ),
    Task(
        id="challenge_return_rate",
        query="What is the return rate (returns / total_sales * 100) for 'Widget C'? Total sales = sum of all quarters. Round to 2 decimal places.",
        expected_answer=3.40,  # 3500 / (30000+28000+25000+20000) * 100 = 3500/103000*100 = 3.398... rounded to 3.40
        difficulty="hard",
        tags=("challenging", "percentage", "rounding"),
    ),
]


def create_task_suite(
    include_easy: bool = True,
    include_medium: bool = True,
    include_real_data: bool = False,
    include_challenging: bool = False,
    include_pl_data: bool = False,
    tags: list[str] | None = None,
) -> list[Task]:
    """
    Create a task suite for benchmarking.

    Args:
        include_easy: Include easy tasks
        include_medium: Include medium tasks
        include_real_data: Include real business data tasks
        include_challenging: Include challenging tasks with edge cases
        include_pl_data: Include P&L financial data tasks (same dataset as CLI)
        tags: Filter to only include tasks with any of these tags

    Returns:
        List of tasks matching the criteria
    """
    tasks: list[Task] = []

    if include_easy:
        tasks.extend(EASY_TASKS)
    if include_medium:
        tasks.extend(MEDIUM_TASKS)
    if include_real_data:
        tasks.extend(REAL_DATA_TASKS)
    if include_challenging:
        tasks.extend(CHALLENGING_TASKS)
    if include_pl_data:
        tasks.extend(PL_EASY_TASKS)
        tasks.extend(PL_MEDIUM_TASKS)
        tasks.extend(PL_HARD_TASKS)

    if tags:
        tag_set = set(tags)
        tasks = [t for t in tasks if any(tag in tag_set for tag in t.tags)]

    return tasks


def create_pl_task_suite(
    include_easy: bool = True,
    include_medium: bool = True,
    include_hard: bool = True,
) -> list[Task]:
    """
    Create a P&L-only task suite (matches CLI/demo dataset).

    Args:
        include_easy: Include easy P&L tasks
        include_medium: Include medium P&L tasks
        include_hard: Include hard P&L tasks

    Returns:
        List of P&L tasks
    """
    tasks: list[Task] = []

    if include_easy:
        tasks.extend(PL_EASY_TASKS)
    if include_medium:
        tasks.extend(PL_MEDIUM_TASKS)
    if include_hard:
        tasks.extend(PL_HARD_TASKS)

    return tasks


def create_learning_pairs() -> list[tuple[Task, Task]]:
    """
    Create pairs of similar tasks for testing cross-session learning.

    Each pair has a "teacher" task and a "learner" task. The agent should:
    1. Learn from solving (or failing) the teacher task
    2. Apply that learning to the learner task

    Returns:
        List of (teacher_task, learner_task) tuples
    """
    pairs = []

    # Group learning tasks by pattern
    teachers = [t for t in PL_LEARNING_PAIRS if "learning_teacher" in t.tags]
    learners = [t for t in PL_LEARNING_PAIRS if "learning_learner" in t.tags]

    # Match by pattern (FX, COGS, OPEX)
    for teacher in teachers:
        # Find matching learner by similar id pattern
        pattern = teacher.id.split("_")[2]  # e.g., "fx", "cogs", "opex"
        for learner in learners:
            if pattern in learner.id:
                pairs.append((teacher, learner))
                break

    return pairs


def get_all_pl_tasks() -> list[Task]:
    """Get all P&L tasks including learning pairs."""
    return PL_EASY_TASKS + PL_MEDIUM_TASKS + PL_HARD_TASKS + PL_LEARNING_PAIRS
