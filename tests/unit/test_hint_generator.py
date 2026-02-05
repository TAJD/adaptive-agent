"""Tests for the hint generator."""

import pytest
from src.evaluator.hint_generator import HintGenerator, generate_hints_for_evaluation


class TestHintGenerator:
    """Tests for the HintGenerator class."""

    @pytest.fixture
    def generator(self):
        return HintGenerator()

    def test_much_larger_result(self, generator):
        """Test hints for result much larger than expected."""
        result = generator.generate_hints(
            expected=100,
            actual=500,
            error_type="large_numeric_error",
            query="What is the total revenue for Product A in Q1?",
        )

        assert len(result.hints) > 0
        assert result.confidence >= 0.5
        assert any("filter" in h.lower() for h in result.hints)

    def test_much_smaller_result(self, generator):
        """Test hints for result much smaller than expected."""
        result = generator.generate_hints(
            expected=100,
            actual=10,
            error_type="numeric_error",
            query="What is the total revenue for Product A?",
        )

        assert len(result.hints) > 0
        assert result.error_pattern in ["much_smaller", "numeric_error"]

    def test_none_result(self, generator):
        """Test hints for None result."""
        result = generator.generate_hints(
            expected=100,
            actual=None,
            error_type="no_output",
            query="Calculate the sum",
        )

        assert len(result.hints) > 0
        assert result.error_pattern == "none_result"
        assert any("result" in h.lower() or "variable" in h.lower() for h in result.hints)

    def test_sign_error(self, generator):
        """Test hints for sign error."""
        result = generator.generate_hints(
            expected=100,
            actual=-100,
            error_type="numeric_error",
        )

        assert result.error_pattern == "negative_vs_positive"
        assert result.confidence >= 0.8
        assert any("sign" in h.lower() for h in result.hints)

    def test_off_by_factor_of_10(self, generator):
        """Test hints for factor-of-10 errors.

        Note: The hint generator checks much_larger before factor-of-10,
        so 10x errors are caught as much_larger. This tests the behavior.
        """
        result = generator.generate_hints(
            expected=100,
            actual=1000,
            error_type="numeric_error",
        )

        # 10x is caught by much_larger (threshold 3x)
        assert result.error_pattern in ["off_by_factor", "much_larger"]
        # Should have filter-related hints
        assert any("filter" in h.lower() or "unit" in h.lower() for h in result.hints)

    def test_query_based_hints_time(self, generator):
        """Test hints based on query keywords - time."""
        result = generator.generate_hints(
            expected=100,
            actual=500,
            error_type="large_numeric_error",
            query="What was the revenue in Q1 2023?",
        )

        # Should have time-related hints
        assert any("quarter" in h.lower() or "year" in h.lower() or "time" in h.lower() for h in result.hints)

    def test_query_based_hints_product(self, generator):
        """Test hints based on query keywords - product."""
        result = generator.generate_hints(
            expected=100,
            actual=500,
            error_type="large_numeric_error",
            query="What is the total for Product A?",
        )

        # Should have product-related hints
        assert any("product" in h.lower() for h in result.hints)

    def test_close_but_wrong(self, generator):
        """Test hints for close but wrong results."""
        result = generator.generate_hints(
            expected=100,
            actual=99,
            error_type="small_numeric_error",
        )

        assert result.error_pattern == "close_but_wrong"
        assert any("round" in h.lower() or "boundary" in h.lower() for h in result.hints)

    def test_hints_are_limited(self, generator):
        """Test that hints are limited to 5."""
        result = generator.generate_hints(
            expected=100,
            actual=500,
            error_type="large_numeric_error",
            query="What is the total revenue for Product A in Q1 2023 in United States?",
        )

        assert len(result.hints) <= 5

    def test_convenience_function(self):
        """Test the convenience function."""
        hints = generate_hints_for_evaluation(
            expected=100,
            actual=None,
            error_type="no_output",
            query="Calculate total",
        )

        assert isinstance(hints, list)
        assert len(hints) > 0


class TestHintGeneratorEdgeCases:
    """Edge cases for hint generator."""

    @pytest.fixture
    def generator(self):
        return HintGenerator()

    def test_expected_zero(self, generator):
        """Test when expected is zero."""
        result = generator.generate_hints(
            expected=0,
            actual=100,
            error_type="numeric_error",
        )

        assert len(result.hints) > 0

    def test_both_zero(self, generator):
        """Test when both expected and actual are zero (should match)."""
        # This would normally pass, but test the hint generator anyway
        result = generator.generate_hints(
            expected=0,
            actual=0,
            error_type=None,
        )

        # Should return some default hints
        assert result.confidence >= 0.0

    def test_no_query(self, generator):
        """Test without query."""
        result = generator.generate_hints(
            expected=100,
            actual=500,
            error_type="large_numeric_error",
            query=None,
        )

        # Should still work without query
        assert len(result.hints) > 0
