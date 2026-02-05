"""Tests for ModelConfig."""

from src.benchmark.model_config import ModelConfig


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic ModelConfig."""
        config = ModelConfig(name="claude-sonnet-4-20250514", provider="anthropic")

        assert config.name == "claude-sonnet-4-20250514"
        assert config.provider == "anthropic"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.model_key == "anthropic/claude-sonnet-4-20250514"

    def test_full_configuration(self) -> None:
        """Test creating a fully configured ModelConfig."""
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            top_k=50,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop_sequences=["STOP", "END"],
            extra_kwargs={"logprobs": True},
        )

        assert config.name == "gpt-4"
        assert config.provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.1
        assert config.stop_sequences == ["STOP", "END"]
        assert config.extra_kwargs == {"logprobs": True}

    def test_to_llm_kwargs(self) -> None:
        """Test converting to LLM kwargs."""
        config = ModelConfig(
            name="claude-3",
            provider="anthropic",
            temperature=0.5,
            max_tokens=1024,
            top_p=0.95,
            stop_sequences=["END"],
            extra_kwargs={"system": "You are helpful"},
        )

        kwargs = config.to_llm_kwargs()

        expected = {
            "model": "claude-3",
            "temperature": 0.5,
            "max_tokens": 1024,
            "top_p": 0.95,
            "stop_sequences": ["END"],
            "system": "You are helpful",
        }

        assert kwargs == expected

    def test_to_llm_kwargs_excludes_defaults(self) -> None:
        """Test that default values are not included if they're zero/default."""
        config = ModelConfig(
            name="model1",
            provider="prov1",
            temperature=0.0,  # default
            frequency_penalty=0.0,  # default
            presence_penalty=0.0,  # default
        )

        kwargs = config.to_llm_kwargs()

        # Should not include frequency_penalty and presence_penalty since they're 0.0 (default)
        assert "frequency_penalty" not in kwargs
        assert "presence_penalty" not in kwargs
        assert kwargs["temperature"] == 0.0  # But temperature is always included

    def test_from_dict_and_to_dict(self) -> None:
        """Test serialization to/from dict."""
        original = ModelConfig(
            name="test-model",
            provider="test-provider",
            temperature=0.8,
            max_tokens=2048,
            stop_sequences=["STOP"],
            extra_kwargs={"custom": "value"},
        )

        data = original.to_dict()
        restored = ModelConfig.from_dict(data)

        assert restored == original

    def test_model_key_uniqueness(self) -> None:
        """Test model_key provides unique identification."""
        config1 = ModelConfig(name="model1", provider="prov1")
        config2 = ModelConfig(name="model2", provider="prov1")
        config3 = ModelConfig(name="model1", provider="prov2")

        assert config1.model_key == "prov1/model1"
        assert config2.model_key == "prov1/model2"
        assert config3.model_key == "prov2/model1"

        # Different keys
        keys = {config1.model_key, config2.model_key, config3.model_key}
        assert len(keys) == 3
