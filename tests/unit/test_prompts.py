"""Tests for prompt templates."""

import pytest

from src.prompts import DataAnalysisPrompt, build_data_analysis_prompt


class TestDataAnalysisPrompt:
    """Tests for DataAnalysisPrompt."""

    def test_build_includes_base_instructions(self) -> None:
        """Test that base instructions are always included."""
        prompt = DataAnalysisPrompt()
        result = prompt.build()

        assert "pandas" in result
        assert "DataFrame" in result
        assert "`df`" in result
        assert "`result`" in result

    def test_build_includes_schema_by_default(self) -> None:
        """Test schema is included by default."""
        prompt = DataAnalysisPrompt()
        result = prompt.build()

        assert "Fiscal Year" in result
        assert "FSLine Statement L1" in result
        assert "Product A" in result
        assert "Amount in USD" in result

    def test_build_excludes_schema_when_disabled(self) -> None:
        """Test schema can be excluded."""
        prompt = DataAnalysisPrompt(include_schema=False)
        result = prompt.build()

        # Base instructions still mention df
        assert "`df`" in result
        # But detailed schema section is excluded
        assert "## Dataset Schema" not in result
        assert "| Column | Type |" not in result

    def test_build_includes_domain_knowledge(self) -> None:
        """Test financial domain knowledge is included."""
        prompt = DataAnalysisPrompt()
        result = prompt.build()

        assert "Net Revenue" in result
        assert "COGS" in result
        assert "Operating Margin" in result
        assert "Gross Profit" in result

    def test_build_excludes_domain_knowledge_when_disabled(self) -> None:
        """Test domain knowledge can be excluded."""
        prompt = DataAnalysisPrompt(include_domain_knowledge=False)
        result = prompt.build()

        # The domain knowledge section header should be excluded
        assert "## Financial Domain Knowledge" not in result
        assert "### Key Calculations" not in result

    def test_build_includes_examples(self) -> None:
        """Test example patterns are included."""
        prompt = DataAnalysisPrompt()
        result = prompt.build()

        assert "```python" in result
        assert "df[" in result
        assert ".sum()" in result

    def test_build_excludes_examples_when_disabled(self) -> None:
        """Test examples can be excluded."""
        prompt = DataAnalysisPrompt(include_examples=False)
        result = prompt.build()

        # Code blocks from examples should not appear
        # (but output requirements still have code markers)
        assert "Common Query Patterns" not in result

    def test_build_includes_error_guidance(self) -> None:
        """Test error handling guidance is included."""
        prompt = DataAnalysisPrompt()
        result = prompt.build()

        assert "Error Handling" in result
        assert "Data Validation" in result

    def test_build_with_additional_constraints(self) -> None:
        """Test additional constraints are included."""
        prompt = DataAnalysisPrompt(
            additional_constraints=["Always round to nearest integer"]
        )
        result = prompt.build()

        assert "Always round to nearest integer" in result

    def test_build_with_additional_hints(self) -> None:
        """Test additional hints are included."""
        prompt = DataAnalysisPrompt(
            additional_hints=["Consider edge cases with zero values"]
        )
        result = prompt.build()

        assert "Consider edge cases with zero values" in result
        assert "Additional Hints" in result


class TestBuildDataAnalysisPrompt:
    """Tests for the convenience function."""

    def test_returns_string(self) -> None:
        """Test function returns a string."""
        result = build_data_analysis_prompt()
        assert isinstance(result, str)
        assert len(result) > 1000  # Should be substantial

    def test_with_all_options_disabled(self) -> None:
        """Test minimal prompt with all options disabled."""
        result = build_data_analysis_prompt(
            include_schema=False,
            include_examples=False,
            include_domain_knowledge=False,
            include_error_guidance=False,
        )

        # Should still have base instructions and output requirements
        assert "`df`" in result
        assert "`result`" in result

    def test_with_custom_constraints_and_hints(self) -> None:
        """Test with custom constraints and hints."""
        result = build_data_analysis_prompt(
            additional_constraints=["Custom constraint"],
            additional_hints=["Custom hint"],
        )

        assert "Custom constraint" in result
        assert "Custom hint" in result

    def test_valid_countries_listed(self) -> None:
        """Test all valid countries are in the prompt."""
        result = build_data_analysis_prompt()

        countries = ["Australia", "Canada", "Germany", "Japan", "United Kingdom", "United States"]
        for country in countries:
            assert country in result

    def test_valid_products_listed(self) -> None:
        """Test all valid products are in the prompt."""
        result = build_data_analysis_prompt()

        for product in ["Product A", "Product B", "Product C", "Product D"]:
            assert product in result

    def test_l2_categories_listed(self) -> None:
        """Test key L2 categories are in the prompt."""
        result = build_data_analysis_prompt()

        categories = [
            "Gross Revenue",
            "Returns and Refunds",
            "Direct Labor",
            "Marketing Expenses",
            "Headcount Expenses",
        ]
        for cat in categories:
            assert cat in result
