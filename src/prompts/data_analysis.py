"""Data analysis prompt template for financial data queries."""

from dataclasses import dataclass, field


@dataclass
class DataAnalysisPrompt:
    """Configurable prompt template for financial data analysis."""

    include_schema: bool = True
    include_examples: bool = True
    include_domain_knowledge: bool = True
    include_error_guidance: bool = True

    # Customizable sections
    additional_constraints: list[str] = field(default_factory=list)
    additional_hints: list[str] = field(default_factory=list)

    def build(self) -> str:
        """Build the complete system prompt."""
        sections = [self._base_instructions()]

        if self.include_schema:
            sections.append(self._dataset_schema())

        if self.include_domain_knowledge:
            sections.append(self._financial_domain_knowledge())

        if self.include_examples:
            sections.append(self._example_patterns())

        if self.include_error_guidance:
            sections.append(self._error_guidance())

        sections.append(self._output_requirements())

        return "\n\n".join(sections)

    def _base_instructions(self) -> str:
        return """You are a Python code generator specialized in financial data analysis.
Your task is to write pandas code that answers questions about a P&L (Profit & Loss) dataset.

The data is pre-loaded as a pandas DataFrame called `df`. Do NOT load or read any files.
Generate code that queries `df` directly and stores the final answer in a variable called `result`."""

    def _dataset_schema(self) -> str:
        return """## Dataset Schema

The DataFrame `df` contains P&L (Profit & Loss) financial data with these columns:

| Column | Type | Description |
|--------|------|-------------|
| Fiscal Year | int | Year (2020-2024) |
| Fiscal Quarter | str | Quarter (Q1, Q2, Q3, Q4) |
| Fiscal Period | str | YYYY-MM format (e.g., "2020-01") |
| FSLine Statement L1 | str | High-level category |
| FSLine Statement L2 | str | Detailed line item |
| Product | str | Product name (Product A, B, C, D) |
| Country | str | Country name |
| Currency | str | Local currency code |
| Amount in Local Currency | float | Amount in local currency |
| Amount in USD | float | Amount converted to USD |
| Version | str | Always "Actuals" |

### Valid Values

**Products:** Product A, Product B, Product C, Product D

**Countries:** Australia, Canada, Germany, Japan, United Kingdom, United States

**Currencies:** AUD, CAD, EUR, GBP, JPY, USD

**Fiscal Years:** 2020, 2021, 2022, 2023, 2024

**Quarters:** Q1, Q2, Q3, Q4

### Financial Statement Structure

**L1 Categories (FSLine Statement L1):**
- Net Revenue
- Cost of Goods Sold
- OPEX
- Other Income/Expenses

**L2 Line Items (FSLine Statement L2):**
- Revenue: Gross Revenue, Returns and Refunds, Revenue Adjustment
- COGS: Direct Labor, Direct Materials, Manufacturing Overhead
- OPEX: Marketing Expenses, R&D Expenses, Sales Expenses, General & Administrative, IT Expenses, Headcount Expenses
- Other: Interest Income, Interest Expense, Foreign Exchange Gain/Loss"""

    def _financial_domain_knowledge(self) -> str:
        return """## Financial Domain Knowledge

### Key Calculations

**Net Revenue** = Gross Revenue + Returns and Refunds + Revenue Adjustment
(Note: Returns and Refunds are typically negative values)

**Cost of Goods Sold (COGS)** = Direct Labor + Direct Materials + Manufacturing Overhead

**OPEX (Operating Expenses)** = Marketing + R&D + Sales + G&A + IT + Headcount Expenses

**Gross Profit** = Net Revenue - COGS

**Operating Income** = Gross Profit - OPEX

**Operating Margin** = Operating Income / Net Revenue

**Gross Margin** = Gross Profit / Net Revenue

### Important Notes

- Always use "Amount in USD" for cross-country comparisons
- Returns and Refunds are negative values (reduce revenue)
- When calculating totals, sum all relevant L2 items within an L1 category
- Year-over-year (YoY) growth = (Current - Prior) / Prior * 100"""

    def _example_patterns(self) -> str:
        return """## Common Query Patterns

### Basic Filtering
```python
# Single product, country, quarter
result = df[
    (df['Product'] == 'Product A') &
    (df['Country'] == 'United States') &
    (df['Fiscal Quarter'] == 'Q1') &
    (df['Fiscal Year'] == 2020) &
    (df['FSLine Statement L2'] == 'Gross Revenue')
]['Amount in USD'].sum()
```

### Global Aggregation (all countries)
```python
# Total Marketing Expenses globally in Q2 2023
result = df[
    (df['FSLine Statement L2'] == 'Marketing Expenses') &
    (df['Fiscal Quarter'] == 'Q2') &
    (df['Fiscal Year'] == 2023)
]['Amount in USD'].sum()
```

### Category Aggregation (all L2 items in an L1)
```python
# Total OPEX for Q4 2023
result = df[
    (df['FSLine Statement L1'] == 'OPEX') &
    (df['Fiscal Quarter'] == 'Q4') &
    (df['Fiscal Year'] == 2023)
]['Amount in USD'].sum()
```

### Net Revenue Calculation
```python
# Net Revenue = Gross Revenue + Returns + Adjustments
net_revenue = df[
    (df['FSLine Statement L1'] == 'Net Revenue') &
    (df['Fiscal Quarter'] == 'Q4') &
    (df['Fiscal Year'] == 2023)
]['Amount in USD'].sum()
result = net_revenue
```

### Year-over-Year Comparison
```python
# YoY growth in OPEX
opex_2022 = df[
    (df['FSLine Statement L1'] == 'OPEX') &
    (df['Fiscal Quarter'] == 'Q1') &
    (df['Fiscal Year'] == 2022)
]['Amount in USD'].sum()

opex_2023 = df[
    (df['FSLine Statement L1'] == 'OPEX') &
    (df['Fiscal Quarter'] == 'Q1') &
    (df['Fiscal Year'] == 2023)
]['Amount in USD'].sum()

result = ((opex_2023 - opex_2022) / opex_2022) * 100  # Percentage
```

### Operating Margin by Product
```python
# For a specific quarter/year, per product
def calc_margin(product_df):
    revenue = product_df[product_df['FSLine Statement L1'] == 'Net Revenue']['Amount in USD'].sum()
    cogs = product_df[product_df['FSLine Statement L1'] == 'Cost of Goods Sold']['Amount in USD'].sum()
    opex = product_df[product_df['FSLine Statement L1'] == 'OPEX']['Amount in USD'].sum()
    if revenue == 0:
        return 0
    return (revenue - cogs - opex) / revenue

quarter_data = df[(df['Fiscal Quarter'] == 'Q3') & (df['Fiscal Year'] == 2023)]
margins = quarter_data.groupby('Product').apply(calc_margin)
result = margins.idxmax()  # Product with highest margin
```"""

    def _error_guidance(self) -> str:
        return """## Error Handling

### Data Validation
Before calculating, verify the data exists:
```python
# Check if a product exists
if 'Product E' not in df['Product'].unique():
    result = "Product E does not exist in the dataset"
```

### Common Mistakes to Avoid
1. **Don't confuse L1 and L2 categories** - "Headcount Expenses" is L2, not L1
2. **Don't assume columns exist** - Check the schema; there's no "Employee Headcount" column
3. **Use exact string matches** - "Product A" not "product a" or "ProductA"
4. **Remember negative values** - Returns and Refunds reduce revenue
5. **Use USD for comparisons** - Don't mix currencies"""

    def _output_requirements(self) -> str:
        constraints = [
            "Store the final answer in a variable called `result`",
            "Use the pre-loaded DataFrame `df` - do NOT read from files",
            "For monetary values, return the numeric value (float)",
            "For percentages, return as a number (e.g., 15.5 for 15.5%)",
            "For non-existent data, return a descriptive error string",
            "Round monetary results to 2 decimal places",
        ]
        constraints.extend(self.additional_constraints)

        hints = list(self.additional_hints)

        parts = ["## Output Requirements", ""]
        parts.extend(f"- {c}" for c in constraints)

        if hints:
            parts.append("\n### Additional Hints")
            parts.extend(f"- {h}" for h in hints)

        parts.append("""
Return ONLY Python code wrapped in ```python and ``` markers.
Do not include explanations outside the code block.""")

        return "\n".join(parts)


def build_data_analysis_prompt(
    include_schema: bool = True,
    include_examples: bool = True,
    include_domain_knowledge: bool = True,
    include_error_guidance: bool = True,
    additional_constraints: list[str] | None = None,
    additional_hints: list[str] | None = None,
) -> str:
    """
    Build a data analysis system prompt.

    Args:
        include_schema: Include dataset schema documentation.
        include_examples: Include example query patterns.
        include_domain_knowledge: Include financial domain knowledge.
        include_error_guidance: Include error handling guidance.
        additional_constraints: Extra constraints to add.
        additional_hints: Extra hints to add.

    Returns:
        Complete system prompt string.
    """
    prompt = DataAnalysisPrompt(
        include_schema=include_schema,
        include_examples=include_examples,
        include_domain_knowledge=include_domain_knowledge,
        include_error_guidance=include_error_guidance,
        additional_constraints=additional_constraints or [],
        additional_hints=additional_hints or [],
    )
    return prompt.build()
